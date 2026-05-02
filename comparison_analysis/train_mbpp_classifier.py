#!/usr/bin/env python3
"""
Train MBPP semantic duplicate classifier with proper train/val split and comprehensive metrics.
Uses Tinker API to fine-tune Qwen3-30B-A3B-Instruct-2507.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json
from pathlib import Path
import argparse
from datetime import datetime

# Set API key via environment variable (TINKER_API_KEY must be set)
if not os.environ.get("TINKER_API_KEY"):
    raise ValueError("TINKER_API_KEY environment variable not set")

import tinker
from tinker import types


# Configuration
CONFIG = {
    "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "lora_rank": 32,
    "learning_rate": 5e-4,
    "num_epochs": 3,
    "batch_size": 128,
    "eval_loss_every": 10,
    "eval_accuracy_every": 25,
    "save_every": 25,
    "max_seq_length": 4096,
    "train_split": 0.8,  # 80% train, 20% val
    "val_samples_for_quick_eval": 50,
}


def format_example(row: Dict[str, Any]) -> Dict[str, str]:
    """Format a single example for instruction fine-tuning."""
    source = row.get('source', 'mbpp')

    test_text = str(row.get('test_text', ''))[:2000] if pd.notna(row.get('test_text')) else ''
    corpus_text = str(row.get('corpus_text', ''))[:2000] if pd.notna(row.get('corpus_text')) else ''

    if source == 'codeforces':
        user = f"""You are an expert competitive programmer analyzing potential semantic duplicates between programming problems.

## Task
Determine if the following two competitive programming problems are semantically related - meaning exposure to the corpus problem during training could help solve the test problem.

## Test Problem (from benchmark):
{test_text}

## Corpus Problem (from training data):
{corpus_text}

## Match Types:
- "exact": Nearly identical problem statements
- "equivalent": Different framing but identical algorithmic core
- "subset": Test is a special case of corpus
- "superset": Corpus is a special case of test
- "related": Corpus covers a component or shares key insight with test
- "unrelated": Different problems, or corpus data is unusable

Respond with valid JSON only."""
    else:
        user = f"""You are an expert programmer analyzing potential semantic duplicates between coding tasks.
## Task
Determine if the following two coding tasks are semantic duplicates - meaning they describe the same programming task, just potentially phrased differently.
## Test Task (from benchmark):
{test_text}
## Corpus Task (from training data):
{corpus_text}
## Guidelines:
1. **Focus on the TASK, not the solution** - ignore any code or solutions that may be present
2. **Mathematical equivalence counts as duplicate** - e.g., "sum 1 to n" and "sum n, n-1, ..., 1" are equivalent
3. **Corpus subsumes test = duplicate** - if the corpus task is strictly harder (asks for more), but solving it would trivially solve the test task, mark as duplicate
4. **Be calibrated** - use confidence primarily for ambiguous cases, tricky phrasing, or when you're uncertain
## Match Types:
- "exact": Nearly identical wording
- "equivalent": Different phrasing, same underlying task
- "subset": Corpus is a subset of test (test asks for more)
- "superset": Corpus is a superset of test (corpus asks for more, but solving it solves test)
- "unrelated": Different tasks entirely
Respond with valid JSON only."""

    is_sd = str(row['is_sd']).lower() == 'true'
    confidence = float(row['confidence']) if pd.notna(row['confidence']) else 1.0
    output = json.dumps({
        "is_duplicate": is_sd,
        "match_type": row['match_type'],
        "confidence": confidence,
        "reasoning": row['reasoning'][:500] if pd.notna(row['reasoning']) else ""
    }, indent=2)

    return {"user": user, "assistant": output}


def process_example_to_datum(example: Dict[str, str], tokenizer) -> types.Datum:
    """Convert a formatted example to a Tinker Datum for training."""
    prompt = f"""<|im_start|>user
{example['user']}<|im_end|>
<|im_start|>assistant
"""
    completion = f"""{example['assistant']}<|im_end|>"""

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)

    prompt_weights = [0] * len(prompt_tokens)
    completion_weights = [1] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    max_len = CONFIG["max_seq_length"]
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
        weights = weights[:max_len]

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )


def load_all_annotations(training_data_dir: str) -> pd.DataFrame:
    """Load all annotation CSVs with ground truth labels."""
    cols = ['test_text', 'corpus_text', 'is_sd', 'confidence', 'match_type', 'reasoning']
    all_dfs = []

    # Files with ground truth annotations - MBPP + Codeforces for training
    annotation_files = [
        ('mbpp_annotations_full(1).csv', 'mbpp'),  # 534 dupes, 5128 total
        ('codeforces_annotations.csv', 'codeforces'),  # 644 dupes, 26110 total
        # NOTE: mbpp_annotations_old_with_text.csv was removed — its non-unrelated
        # rows are mostly Gemini hallucinations on URL-only / image-only corpus_texts
        # (125 of 182 non-unrelated rows contain a URL). Including it poisoned the
        # classifier into labeling code-only and URL-only corpus_texts as duplicates.
    ]

    for filename, source in annotation_files:
        filepath = Path(training_data_dir) / filename
        if not filepath.exists():
            print(f"  Skipping {filename} (not found)")
            continue

        print(f"  Loading {filename}...")
        df = pd.read_csv(filepath)

        # Keep only required columns
        available_cols = [c for c in cols if c in df.columns]
        if 'test_text' not in available_cols or 'corpus_text' not in available_cols:
            print(f"    Skipping: missing required columns")
            continue

        df = df[available_cols].dropna(subset=['test_text', 'corpus_text'])
        df['source'] = source

        # Fill missing columns
        for col in cols:
            if col not in df.columns:
                df[col] = None

        # Ensure is_sd is boolean
        df['is_sd'] = df['is_sd'].astype(str).str.lower() == 'true'

        num_dupes = df['is_sd'].sum()
        print(f"    Loaded {len(df)} rows ({num_dupes} duplicates)")
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No valid annotation data found!")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal combined: {len(combined)} rows")
    return combined


def create_train_val_split(df: pd.DataFrame, val_ratio: float = 0.2, train_class_ratio: int = 10, val_class_ratio: int = 3, seed: int = 42) -> tuple:
    """Create train/val split. Train on all data (MBPP+Codeforces), val on MBPP only.

    Args:
        train_class_ratio: Max non-dupe:dupe ratio for training (default 10:1)
        val_class_ratio: Max non-dupe:dupe ratio for validation (default 3:1 for better eval signal)
    """
    np.random.seed(seed)

    # Separate by source
    mbpp_df = df[df['source'] == 'mbpp'].copy()
    codeforces_df = df[df['source'] == 'codeforces'].copy()

    print(f"\nData by source:")
    print(f"  MBPP: {len(mbpp_df)} ({mbpp_df['is_sd'].sum()} dupes)")
    print(f"  Codeforces: {len(codeforces_df)} ({codeforces_df['is_sd'].sum()} dupes)")

    # Split MBPP for val (stratified)
    mbpp_dupes = mbpp_df[mbpp_df['is_sd'] == True].sample(frac=1, random_state=seed)
    mbpp_non_dupes = mbpp_df[mbpp_df['is_sd'] == False].sample(frac=1, random_state=seed)

    n_val_dupes = max(1, int(len(mbpp_dupes) * val_ratio))
    train_mbpp_dupes = mbpp_dupes.iloc[n_val_dupes:]
    train_mbpp_non_dupes = mbpp_non_dupes  # All non-dupes go to train initially

    # Val set: MBPP only, balanced at val_class_ratio:1
    val_dupes = mbpp_dupes.iloc[:n_val_dupes]
    n_val_non_dupes = min(len(val_dupes) * val_class_ratio, len(mbpp_non_dupes) // 5)  # Take from pool
    val_non_dupes = mbpp_non_dupes.iloc[:n_val_non_dupes]
    train_mbpp_non_dupes = mbpp_non_dupes.iloc[n_val_non_dupes:]  # Rest goes to train

    val_df = pd.concat([val_dupes, val_non_dupes], ignore_index=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Train set: remaining MBPP + all Codeforces
    all_train = pd.concat([train_mbpp_dupes, train_mbpp_non_dupes, codeforces_df], ignore_index=True)

    # Apply class balancing for training (max train_class_ratio:1)
    train_dupes = all_train[all_train['is_sd'] == True].sample(frac=1, random_state=seed)
    train_non_dupes = all_train[all_train['is_sd'] == False].sample(frac=1, random_state=seed)

    print(f"\nBefore balancing:")
    print(f"  Train dupes: {len(train_dupes)}, non-dupes: {len(train_non_dupes)} (ratio {len(train_non_dupes)/max(1,len(train_dupes)):.1f}:1)")

    max_non_dupes = len(train_dupes) * train_class_ratio
    if len(train_non_dupes) > max_non_dupes:
        train_non_dupes = train_non_dupes.iloc[:max_non_dupes]
        print(f"  Downsampled train non-dupes to {len(train_non_dupes)} (ratio {train_class_ratio}:1)")

    train_df = pd.concat([train_dupes, train_non_dupes], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\nFinal split:")
    print(f"  Train: {len(train_df)} ({train_df['is_sd'].sum()} dupes, {(~train_df['is_sd']).sum()} non-dupes, ratio {(~train_df['is_sd']).sum()/max(1,train_df['is_sd'].sum()):.1f}:1)")
    print(f"  Val (MBPP only): {len(val_df)} ({val_df['is_sd'].sum()} dupes, {(~val_df['is_sd']).sum()} non-dupes, ratio {(~val_df['is_sd']).sum()/max(1,val_df['is_sd'].sum()):.1f}:1)")

    return train_df, val_df


def compute_batch_loss(fwdbwd_result, processed_batch: List[types.Datum]) -> float:
    """Compute weighted average loss per token."""
    logprobs = np.concatenate([
        output['logprobs'].tolist()
        for output in fwdbwd_result.loss_fn_outputs
    ])
    weights = np.concatenate([
        example.loss_fn_inputs['weights'].tolist()
        for example in processed_batch
    ])
    total_weight = weights.sum()
    if total_weight == 0:
        return 0.0
    return -np.dot(logprobs, weights) / total_weight


def compute_metrics(y_true: List[bool], y_pred: List[bool], y_conf: List[float] = None) -> Dict[str, Any]:
    """Compute comprehensive classification metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Confusion matrix elements
    tp = np.sum((y_true == True) & (y_pred == True))
    tn = np.sum((y_true == False) & (y_pred == False))
    fp = np.sum((y_true == False) & (y_pred == True))
    fn = np.sum((y_true == True) & (y_pred == False))

    total = len(y_true)

    # Basic metrics
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # Duplicate class metrics
    precision_dup = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_dup = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_dup = 2 * precision_dup * recall_dup / (precision_dup + recall_dup) if (precision_dup + recall_dup) > 0 else 0.0

    # Non-duplicate class metrics
    precision_nondup = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_nondup = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_nondup = 2 * precision_nondup * recall_nondup / (precision_nondup + recall_nondup) if (precision_nondup + recall_nondup) > 0 else 0.0

    # Macro F1
    macro_f1 = (f1_dup + f1_nondup) / 2

    # Matthews Correlation Coefficient
    mcc_num = (tp * tn) - (fp * fn)
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_num / mcc_denom if mcc_denom > 0 else 0.0

    # Balanced accuracy
    sensitivity = recall_dup  # TPR
    specificity = recall_nondup  # TNR
    balanced_accuracy = (sensitivity + specificity) / 2

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_dup": precision_dup,
        "recall_dup": recall_dup,
        "f1_dup": f1_dup,
        "precision_nondup": precision_nondup,
        "recall_nondup": recall_nondup,
        "f1_nondup": f1_nondup,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "confusion_matrix": {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        },
        "total": total,
    }

    return metrics


def evaluate_model(
    sampling_client,
    tokenizer,
    formatted_examples: List[Dict[str, str]],
    batch_size: int = 50
) -> Dict[str, Any]:
    """Evaluate model on validation set with comprehensive metrics."""

    print(f"\n{'='*60}")
    print(f"EVALUATING ON {len(formatted_examples)} EXAMPLES")
    print(f"{'='*60}")

    params = types.SamplingParams(
        max_tokens=512,
        temperature=0.0,
        stop=["<|im_end|>"]
    )

    y_true = []
    y_pred = []
    y_conf = []

    for batch_start in range(0, len(formatted_examples), batch_size):
        batch_end = min(batch_start + batch_size, len(formatted_examples))
        batch = formatted_examples[batch_start:batch_end]

        print(f"  Processing batch {batch_start//batch_size + 1}/{(len(formatted_examples) + batch_size - 1)//batch_size}...")

        # Fire off all requests in parallel
        futures = []
        for example in batch:
            prompt = f"""<|im_start|>user
{example['user']}<|im_end|>
<|im_start|>assistant
"""
            prompt_tokens = types.ModelInput.from_ints(tokenizer.encode(prompt))
            future = sampling_client.sample(
                prompt=prompt_tokens,
                sampling_params=params,
                num_samples=1
            )
            futures.append((future, example))

        # Collect results
        for future, example in futures:
            try:
                result = future.result()
                response = tokenizer.decode(result.sequences[0].tokens)

                # Parse expected
                expected = json.loads(example['assistant'])
                expected_is_dup = expected.get('is_duplicate', False)

                # Parse predicted
                try:
                    predicted = json.loads(response)
                    predicted_is_dup = predicted.get('is_duplicate', False)
                    predicted_conf = predicted.get('confidence', 1.0)
                except json.JSONDecodeError:
                    predicted_is_dup = '"is_duplicate": true' in response.lower()
                    predicted_conf = 0.5

                y_true.append(expected_is_dup)
                y_pred.append(predicted_is_dup)
                y_conf.append(predicted_conf)

            except Exception as e:
                print(f"    Warning: Failed to evaluate sample: {e}")
                continue

    metrics = compute_metrics(y_true, y_pred, y_conf)

    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\nOverall:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f} ({metrics['balanced_accuracy']*100:.2f}%)")
    print(f"  MCC:               {metrics['mcc']:.4f}")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f}")

    print(f"\nDuplicate Class:")
    print(f"  Precision: {metrics['precision_dup']:.4f}")
    print(f"  Recall:    {metrics['recall_dup']:.4f}")
    print(f"  F1:        {metrics['f1_dup']:.4f}")

    print(f"\nNon-Duplicate Class:")
    print(f"  Precision: {metrics['precision_nondup']:.4f}")
    print(f"  Recall:    {metrics['recall_nondup']:.4f}")
    print(f"  F1:        {metrics['f1_nondup']:.4f}")

    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"                      Predicted")
    print(f"                   Dup    Non-Dup")
    print(f"  Actual Dup       {cm['tp']:4d}    {cm['fn']:4d}")
    print(f"  Actual Non-Dup   {cm['fp']:4d}    {cm['tn']:4d}")
    print(f"{'='*60}\n")

    return metrics


def train(
    training_client,
    train_data: List[types.Datum],
    val_data: List[types.Datum],
    formatted_val_examples: List[Dict[str, str]],
    tokenizer,
    config: Dict[str, Any],
    output_dir: Path
):
    """Main training loop with periodic validation."""

    batch_size = config["batch_size"]
    num_examples = len(train_data)
    num_batches = (num_examples + batch_size - 1) // batch_size

    print(f"\n{'='*60}")
    print(f"STARTING TRAINING")
    print(f"  Model: {config['base_model']}")
    print(f"  Training examples: {len(train_data)}")
    print(f"  Validation examples: {len(val_data)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"{'='*60}\n")

    metrics_log = []
    step = 0
    best_f1 = 0.0

    for epoch in range(config["num_epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{config['num_epochs']} ---")

        indices = np.random.permutation(num_examples)

        pending_future = None
        pending_optim = None
        pending_batch = None

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_examples)
            batch_indices = indices[start_idx:end_idx]
            batch = [train_data[i] for i in batch_indices]

            fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
            optim_future = training_client.optim_step(
                types.AdamParams(learning_rate=config["learning_rate"])
            )

            if pending_future is not None:
                fwdbwd_result = pending_future.result()
                _ = pending_optim.result()
                train_loss = compute_batch_loss(fwdbwd_result, pending_batch)
                step += 1

                if step % 5 == 0:
                    print(f"Step {step}/{num_batches * config['num_epochs']}: train_loss={train_loss:.4f}")

                # Periodic checkpoint and evaluation
                if step % config["save_every"] == 0:
                    checkpoint_name = f"checkpoint-step{step}"
                    training_client.save_state(checkpoint_name)
                    print(f"  [Checkpoint] Saved: {checkpoint_name}")

                    metrics_log.append({
                        "step": step,
                        "epoch": epoch + 1,
                        "train_loss": train_loss
                    })

            pending_future = fwdbwd_future
            pending_optim = optim_future
            pending_batch = batch

        # Process final batch
        if pending_future is not None:
            fwdbwd_result = pending_future.result()
            _ = pending_optim.result()
            train_loss = compute_batch_loss(fwdbwd_result, pending_batch)
            step += 1
            print(f"Step {step}: train_loss={train_loss:.4f}")

        # End of epoch quick evaluation (subset of val set)
        quick_eval_size = min(200, len(formatted_val_examples))
        quick_eval_samples = formatted_val_examples[:quick_eval_size]
        print(f"\n[Epoch {epoch + 1}] Running quick validation ({quick_eval_size} samples)...")
        sampling_client = training_client.save_weights_and_get_sampling_client(
            name=f"epoch{epoch+1}-eval"
        )

        eval_metrics = evaluate_model(
            sampling_client,
            tokenizer,
            quick_eval_samples,
            batch_size=50
        )

        metrics_log.append({
            "step": step,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **eval_metrics
        })

        # Save best model
        if eval_metrics['macro_f1'] > best_f1:
            best_f1 = eval_metrics['macro_f1']
            print(f"  [Best] New best macro F1: {best_f1:.4f}")
            training_client.save_state("best-checkpoint")

        # Save metrics
        with open(output_dir / "training_metrics.jsonl", "w") as f:
            for m in metrics_log:
                f.write(json.dumps(m) + "\n")

    print(f"\nTraining complete. Best macro F1: {best_f1:.4f}")
    return metrics_log


def main():
    parser = argparse.ArgumentParser(description="Train MBPP classifier with validation metrics")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation on existing checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load for eval-only mode")
    args = parser.parse_args()

    CONFIG["num_epochs"] = args.epochs
    CONFIG["batch_size"] = args.batch_size
    CONFIG["train_split"] = args.train_split

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "outputs" / f"mbpp_classifier_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("MBPP SEMANTIC DUPLICATE CLASSIFIER TRAINING")
    print("="*60)

    # Load data
    training_data_dir = Path(__file__).parent / "training_data"
    df = load_all_annotations(str(training_data_dir))

    # Create train/val split
    train_df, val_df = create_train_val_split(df, val_ratio=1 - CONFIG["train_split"])

    # Save split info
    split_info = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "train_dupes": int(train_df['is_sd'].sum()),
        "val_dupes": int(val_df['is_sd'].sum()),
        "train_split": CONFIG["train_split"],
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Save validation set for reproducibility
    val_df.to_csv(output_dir / "val_set.csv", index=False)

    # Format examples
    print("\nFormatting training examples...")
    formatted_train = [format_example(row) for _, row in train_df.iterrows()]
    print(f"  Formatted {len(formatted_train)} training examples")

    print("Formatting validation examples...")
    formatted_val = [format_example(row) for _, row in val_df.iterrows()]
    print(f"  Formatted {len(formatted_val)} validation examples")

    # Connect to Tinker
    print(f"\nConnecting to Tinker API...")
    service_client = tinker.ServiceClient()

    print(f"Creating training client for {CONFIG['base_model']}...")
    training_client = service_client.create_lora_training_client(
        base_model=CONFIG["base_model"],
        rank=CONFIG["lora_rank"]
    )

    tokenizer = training_client.get_tokenizer()
    print(f"Tokenizer loaded")

    if args.eval_only:
        # Load checkpoint and evaluate
        if args.checkpoint:
            print(f"\nLoading checkpoint: {args.checkpoint}")
            training_client.load_state(args.checkpoint)

        sampling_client = training_client.save_weights_and_get_sampling_client(name="eval-only")

        eval_metrics = evaluate_model(
            sampling_client,
            tokenizer,
            formatted_val,
            batch_size=50
        )

        with open(output_dir / "eval_results.json", "w") as f:
            json.dump(eval_metrics, f, indent=2)

        print(f"\nResults saved to: {output_dir}")
        return

    # Convert to Datum format
    print("\nProcessing training examples...")
    train_data = []
    for i, example in enumerate(formatted_train):
        try:
            datum = process_example_to_datum(example, tokenizer)
            train_data.append(datum)
        except Exception as e:
            print(f"  Warning: Failed to process example {i}: {e}")
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(formatted_train)}")

    print(f"Successfully processed {len(train_data)} training examples")

    print("Processing validation examples...")
    val_data = []
    for i, example in enumerate(formatted_val):
        try:
            datum = process_example_to_datum(example, tokenizer)
            val_data.append(datum)
        except Exception as e:
            print(f"  Warning: Failed to process example {i}: {e}")
    print(f"Successfully processed {len(val_data)} validation examples")

    # Train
    metrics_log = train(
        training_client,
        train_data,
        val_data,
        formatted_val,
        tokenizer,
        CONFIG,
        output_dir
    )

    # Final evaluation on best checkpoint
    print("\n" + "="*60)
    print("FINAL EVALUATION ON BEST CHECKPOINT")
    print("="*60)

    training_client.load_state("best-checkpoint")
    sampling_client = training_client.save_weights_and_get_sampling_client(name="final-best")

    final_metrics = evaluate_model(
        sampling_client,
        tokenizer,
        formatted_val,
        batch_size=50
    )

    # Save final results
    final_results = {
        "config": CONFIG,
        "split_info": split_info,
        "final_metrics": final_metrics,
        "training_log": metrics_log
    }
    with open(output_dir / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    # Save summary text file for easy reference
    summary_text = f"""MBPP CLASSIFIER TRAINING SUMMARY
================================
Timestamp: {timestamp}
Output Directory: {output_dir}

MODEL INFO
----------
Base Model: {CONFIG['base_model']}
LoRA Rank: {CONFIG['lora_rank']}
Best Checkpoint: best-checkpoint
Sampler Name: final-best

To reload this model:
    training_client.load_state("best-checkpoint")
    sampling_client = training_client.save_weights_and_get_sampling_client(name="final-best")

TRAINING CONFIG
---------------
Epochs: {CONFIG['num_epochs']}
Batch Size: {CONFIG['batch_size']}
Learning Rate: {CONFIG['learning_rate']}
Max Seq Length: {CONFIG['max_seq_length']}

DATASET
-------
Train Size: {split_info['train_size']}
Train Duplicates: {split_info['train_dupes']}
Val Size: {split_info['val_size']}
Val Duplicates: {split_info['val_dupes']}

FINAL METRICS (on validation set)
---------------------------------
Accuracy: {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)
Balanced Accuracy: {final_metrics['balanced_accuracy']:.4f}
MCC: {final_metrics['mcc']:.4f}
Macro F1: {final_metrics['macro_f1']:.4f}

Duplicate Class:
  Precision: {final_metrics['precision_dup']:.4f}
  Recall: {final_metrics['recall_dup']:.4f}
  F1: {final_metrics['f1_dup']:.4f}

Non-Duplicate Class:
  Precision: {final_metrics['precision_nondup']:.4f}
  Recall: {final_metrics['recall_nondup']:.4f}
  F1: {final_metrics['f1_nondup']:.4f}

Confusion Matrix:
  TP: {final_metrics['confusion_matrix']['tp']}  FN: {final_metrics['confusion_matrix']['fn']}
  FP: {final_metrics['confusion_matrix']['fp']}  TN: {final_metrics['confusion_matrix']['tn']}
"""
    with open(output_dir / "training_summary.txt", "w") as f:
        f.write(summary_text)

    print(summary_text)
    print(f"\nAll results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
