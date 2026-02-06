#!/usr/bin/env python3
"""
Train Codeforces semantic duplicate classifier.

Fine-tunes GPT-OSS-20b on MBPP+Codeforces annotation data using Tinker API
with LoRA. Evaluates on Codeforces-only held-out set. Includes early stopping
and post-training classification of the full Codeforces dataset.

Training config:
- Train on MBPP + Codeforces (combined)
- Eval on Codeforces only
- 10:1 non-dupes vs dupes ratio for training
- 3:1 non-dupes vs dupes ratio for eval
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json
from pathlib import Path
from datetime import datetime

# Require API key from environment
if not os.environ.get("TINKER_API_KEY"):
    raise ValueError("TINKER_API_KEY environment variable not set")

import tinker
from tinker import types

from shared_utilities import compute_batch_loss, evaluate


# Configuration - matches MBPP classifier settings
CONFIG = {
    "base_model": "openai/gpt-oss-20b",
    "lora_rank": 32,  # Max for GPT-OSS-20b
    "learning_rate": 1e-4,
    "num_epochs": 10,
    "batch_size": 128,  # Match MBPP classifier
    "eval_loss_every": 10,
    "eval_accuracy_every": 25,
    "eval_accuracy_samples": 200,  # Quick eval subset size
    "save_every": 25,
    "max_seq_length": 4096,
    "train_split": 0.8,  # 80% train, 20% val
    "train_class_ratio": 10,  # 10:1 non-dupe:dupe for training
    "val_class_ratio": 3,  # 3:1 for balanced eval
}

# Classification config - runs after training
CLASSIFY_CONFIG = {
    "input_csv": str(Path(__file__).parent / "data" / "codeforces_top100_per_testid_dataset.csv"),
    "output_csv": str(Path(__file__).parent / "data" / "codeforces_top100_classified_gptoss.csv"),
    "batch_size": 512,
    "save_every": 10,
}

def load_all_annotations(training_data_dir: str) -> pd.DataFrame:
    """Load all annotation CSVs with ground truth labels (MBPP + Codeforces).

    Matches MBPP classifier data loading approach.
    """
    cols = ['test_text', 'corpus_text', 'is_sd', 'confidence', 'match_type', 'reasoning']
    all_dfs = []

    # Files with ground truth annotations - MBPP + Codeforces for training
    annotation_files = [
        ('mbpp_annotations_full(1).csv', 'mbpp'),  # 534 dupes, 5128 total
        ('mbpp_annotations_old_with_text.csv', 'mbpp'),  # 168 dupes, 18709 total
        ('codeforces_annotations.csv', 'codeforces'),  # 644 dupes, 26110 total
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
    """Create train/val split. Train on all data (MBPP+Codeforces), val on Codeforces only.

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

    # Split Codeforces for val (stratified)
    cf_dupes = codeforces_df[codeforces_df['is_sd'] == True].sample(frac=1, random_state=seed)
    cf_non_dupes = codeforces_df[codeforces_df['is_sd'] == False].sample(frac=1, random_state=seed)

    n_val_dupes = max(1, int(len(cf_dupes) * val_ratio))
    train_cf_dupes = cf_dupes.iloc[n_val_dupes:]
    train_cf_non_dupes = cf_non_dupes  # All non-dupes go to train initially

    # Val set: Codeforces only, balanced at val_class_ratio:1
    val_dupes = cf_dupes.iloc[:n_val_dupes]
    n_val_non_dupes = min(len(val_dupes) * val_class_ratio, len(cf_non_dupes) // 5)  # Take from pool
    val_non_dupes = cf_non_dupes.iloc[:n_val_non_dupes]
    train_cf_non_dupes = cf_non_dupes.iloc[n_val_non_dupes:]  # Rest goes to train

    val_df = pd.concat([val_dupes, val_non_dupes], ignore_index=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Train set: remaining Codeforces + all MBPP
    all_train = pd.concat([train_cf_dupes, train_cf_non_dupes, mbpp_df], ignore_index=True)

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
    print(f"  Val (Codeforces only): {len(val_df)} ({val_df['is_sd'].sum()} dupes, {(~val_df['is_sd']).sum()} non-dupes, ratio {(~val_df['is_sd']).sum()/max(1,val_df['is_sd'].sum()):.1f}:1)")

    return train_df, val_df


def format_example(row: Dict[str, Any]) -> Dict[str, str]:
    """Format a single example for instruction fine-tuning. Uses source-specific prompts."""
    source = row.get('source', 'codeforces')

    # Handle NaN/None values in text fields
    test_text = str(row.get('test_text', ''))[:2000] if pd.notna(row.get('test_text')) else ''
    corpus_text = str(row.get('corpus_text', ''))[:2000] if pd.notna(row.get('corpus_text')) else ''

    if source == 'mbpp':
        # MBPP prompt template
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
    else:
        # Codeforces prompt template
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

    # Generate expected output
    is_sd = str(row['is_sd']).lower() == 'true'
    confidence = float(row['confidence']) if pd.notna(row['confidence']) else 1.0
    match_type = str(row['match_type']).lower().strip() if pd.notna(row['match_type']) else 'unrelated'
    reasoning = str(row['reasoning'])[:500] if pd.notna(row['reasoning']) else ""

    output = json.dumps({
        "is_duplicate": is_sd,
        "match_type": match_type,
        "confidence": confidence,
        "reasoning": reasoning
    }, indent=2)

    return {
        "user": user,
        "assistant": output
    }


def process_example_to_datum(example: Dict[str, str], tokenizer) -> types.Datum:
    """Convert a formatted example to Tinker Datum format for training."""

    # GPT-OSS format - use tokenizer's chat template if available
    # Fallback to standard instruct format
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{example['user']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    completion = f"""{example['assistant']}<|eot_id|>"""

    # Tokenize separately
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)

    # Set weights: 0 for prompt (context), 1 for completion (to learn)
    prompt_weights = [0] * len(prompt_tokens)
    completion_weights = [1] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    # Truncate if too long
    max_len = CONFIG["max_seq_length"]
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
        weights = weights[:max_len]

    # Shift for next-token prediction
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )


def compute_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, Any]:
    """Compute comprehensive classification metrics (matches MBPP classifier)."""
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

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision_dup": float(precision_dup),
        "recall_dup": float(recall_dup),
        "f1_dup": float(f1_dup),
        "precision_nondup": float(precision_nondup),
        "recall_nondup": float(recall_nondup),
        "f1_nondup": float(f1_nondup),
        "macro_f1": float(macro_f1),
        "mcc": float(mcc),
        "confusion_matrix": {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        },
        "total": int(total),
    }


def train(
    training_client,
    train_data: List[types.Datum],
    eval_data: List[types.Datum],
    formatted_eval_examples: List[Dict[str, str]],
    tokenizer,
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Training loop with pipelined forward-backward and optimizer steps."""

    metrics = []
    best_f1 = 0.0
    best_f1_metrics = None
    best_f1_step = 0
    evals_without_improvement = 0
    early_stopping_patience = config.get("early_stopping_patience", 7)
    early_stopped = False

    num_batches = (len(train_data) + config["batch_size"] - 1) // config["batch_size"]
    total_steps = num_batches * config["num_epochs"]

    print(f"\n{'='*60}")
    print(f"TRAINING")
    print(f"{'='*60}")
    print(f"Training samples: {len(train_data)}")
    print(f"Eval samples: {len(eval_data)}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Num epochs: {config['num_epochs']}")
    print(f"Total steps: {total_steps}")
    print(f"Early stopping patience: {early_stopping_patience} evals")
    print(f"{'='*60}\n")

    step = 0

    for epoch in range(config["num_epochs"]):
        if early_stopped:
            break

        print(f"\n=== Epoch {epoch + 1}/{config['num_epochs']} ===")

        # Shuffle training data each epoch
        indices = np.random.permutation(len(train_data))
        shuffled_data = [train_data[i] for i in indices]

        pending_future = None
        pending_optim = None
        pending_batch = None

        for batch_idx in range(num_batches):
            if early_stopped:
                break

            start_idx = batch_idx * config["batch_size"]
            end_idx = min(start_idx + config["batch_size"], len(shuffled_data))
            batch = shuffled_data[start_idx:end_idx]

            # Start forward-backward for current batch
            fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")

            # If we have a pending result, process it
            if pending_future is not None:
                fwdbwd_result = pending_future.result()
                _ = pending_optim.result()

                train_loss = compute_batch_loss(fwdbwd_result, pending_batch)
                step += 1

                # Log progress
                if step % 5 == 0:
                    print(f"Step {step}/{total_steps}: train_loss={train_loss:.4f}")

                # Eval loss (fast)
                if step % config["eval_loss_every"] == 0:
                    eval_loss = evaluate(training_client, eval_data, config["batch_size"])
                    print(f"  [Eval] eval_loss={eval_loss:.4f}")

                # Accuracy eval (slower)
                if step % config["eval_accuracy_every"] == 0:
                    print(f"  [Accuracy Eval] Saving weights and generating {config['eval_accuracy_samples']} samples...")
                    acc_metrics = evaluate_accuracy(
                        training_client, tokenizer, formatted_eval_examples,
                        num_samples=config['eval_accuracy_samples']
                    )
                    print(f"  [Accuracy] acc={acc_metrics['accuracy']:.3f}, "
                          f"cat_acc={acc_metrics['category_accuracy']:.3f}, "
                          f"prec={acc_metrics['precision']:.3f}, "
                          f"rec={acc_metrics['recall']:.3f}, "
                          f"f1={acc_metrics['f1']:.3f}")

                    metrics.append({
                        "step": step,
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        **acc_metrics
                    })

                    # Track best model by F1
                    if acc_metrics['f1'] > best_f1:
                        best_f1 = acc_metrics['f1']
                        best_f1_metrics = acc_metrics.copy()
                        best_f1_step = step
                        evals_without_improvement = 0
                        print(f"  [Best] New best F1: {best_f1:.3f} (acc={acc_metrics['accuracy']:.3f}, prec={acc_metrics['precision']:.3f}, rec={acc_metrics['recall']:.3f})")
                        training_client.save_state("sd-gptoss-best-checkpoint")
                    else:
                        evals_without_improvement += 1
                        print(f"  [Early Stopping] No F1 improvement for {evals_without_improvement}/{early_stopping_patience} evals (best={best_f1:.3f}, current={acc_metrics['f1']:.3f})")

                        if evals_without_improvement >= early_stopping_patience:
                            print(f"\n{'='*60}")
                            print(f"EARLY STOPPING triggered at step {step}")
                            print(f"Best F1: {best_f1:.3f} at step {best_f1_step}")
                            print(f"{'='*60}")
                            early_stopped = True
                            break

                # Save checkpoint
                if step % config["save_every"] == 0:
                    checkpoint_name = f"sd-gptoss-checkpoint-step{step}"
                    training_client.save_state(checkpoint_name)
                    print(f"  [Checkpoint] Saved: {checkpoint_name}")

            # Start optimizer step for current batch
            optim_future = training_client.optim_step(
                types.AdamParams(learning_rate=config["learning_rate"])
            )

            # Store current as pending
            pending_future = fwdbwd_future
            pending_optim = optim_future
            pending_batch = batch

        # Process final batch (if not early stopped)
        if pending_future is not None and not early_stopped:
            fwdbwd_result = pending_future.result()
            _ = pending_optim.result()
            train_loss = compute_batch_loss(fwdbwd_result, pending_batch)
            step += 1
            print(f"Step {step}/{total_steps}: train_loss={train_loss:.4f}")

    if early_stopped:
        print(f"\nTraining stopped early at step {step}.")
    if best_f1_metrics:
        print(f"Best F1: {best_f1:.3f} at step {best_f1_step} (acc={best_f1_metrics['accuracy']:.3f}, prec={best_f1_metrics['precision']:.3f}, rec={best_f1_metrics['recall']:.3f})")
    else:
        print(f"\nTraining complete. No best checkpoint saved.")
    return metrics, best_f1_metrics


def evaluate_accuracy(
    training_client,
    tokenizer,
    formatted_eval_examples: List[Dict[str, str]],
    num_samples: int = 50
) -> Dict[str, float]:
    """Evaluate accuracy by generating responses and checking predictions."""

    sample_indices = np.random.choice(len(formatted_eval_examples), min(num_samples, len(formatted_eval_examples)), replace=False)
    samples = [formatted_eval_examples[i] for i in sample_indices]

    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=f"sd-scored-eval-temp-{np.random.randint(100000)}"
    )

    params = types.SamplingParams(
        max_tokens=512,
        temperature=0.0,
        stop=["<|eot_id|>"]
    )

    futures = []
    for example in samples:
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{example['user']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        prompt_tokens = types.ModelInput.from_ints(tokenizer.encode(prompt))
        future = sampling_client.sample(
            prompt=prompt_tokens,
            sampling_params=params,
            num_samples=1
        )
        futures.append((future, example))

    correct = 0
    category_correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for future, example in futures:
        try:
            result = future.result()
            response = tokenizer.decode(result.sequences[0].tokens)
            response = response.replace('<|eot_id|>', '').strip()

            expected = json.loads(example['assistant'])
            expected_is_dup = expected.get('is_duplicate', False)
            expected_category = expected.get('predicted_category', 'unrelated')

            try:
                predicted = json.loads(response)
                predicted_is_dup = predicted.get('is_duplicate', False)
                predicted_category = predicted.get('predicted_category', 'unrelated')
            except json.JSONDecodeError:
                predicted_is_dup = '"is_duplicate": true' in response.lower()
                predicted_category = None

            total += 1
            if expected_is_dup == predicted_is_dup:
                correct += 1

            if predicted_category and expected_category == predicted_category:
                category_correct += 1

            if expected_is_dup and predicted_is_dup:
                true_positives += 1
            elif not expected_is_dup and not predicted_is_dup:
                true_negatives += 1
            elif not expected_is_dup and predicted_is_dup:
                false_positives += 1
            else:
                false_negatives += 1

        except Exception as e:
            continue

    accuracy = correct / total if total > 0 else 0.0
    category_accuracy = category_correct / total if total > 0 else 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "category_accuracy": category_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": total,
        "correct": correct
    }


def main():
    training_data_dir = str(Path(__file__).parent.parent / "comparison_analysis" / "training_data")

    print("="*60)
    print("CODEFORCES SEMANTIC DUPLICATE CLASSIFIER TRAINING (GPT-OSS)")
    print("  - Train on: MBPP + Codeforces (combined)")
    print("  - Eval on: Codeforces only")
    print("="*60)

    # Load all annotation data
    df = load_all_annotations(training_data_dir)

    # Create train/val split
    train_df, val_df = create_train_val_split(
        df,
        val_ratio=1 - CONFIG["train_split"],
        train_class_ratio=CONFIG["train_class_ratio"],
        val_class_ratio=CONFIG["val_class_ratio"]
    )

    train_dupes_final = train_df['is_sd'].sum()
    train_non_dupes_final = (~train_df['is_sd']).sum()
    eval_dupes_count = val_df['is_sd'].sum()
    eval_non_dupes_count = (~val_df['is_sd']).sum()

    # Format examples
    print("\nFormatting training examples...")
    formatted_examples = [format_example(row.to_dict()) for _, row in train_df.iterrows()]

    print("Formatting eval examples...")
    formatted_eval_examples = [format_example(row.to_dict()) for _, row in val_df.iterrows()]

    # Create clients
    print(f"\nConnecting to Tinker API...")
    service_client = tinker.ServiceClient()

    print(f"Creating training client for {CONFIG['base_model']}...")
    training_client = service_client.create_lora_training_client(
        base_model=CONFIG["base_model"],
        rank=CONFIG["lora_rank"]
        # Note: dropout not supported by Tinker API
    )

    tokenizer = training_client.get_tokenizer()
    print(f"Tokenizer loaded")

    # Convert to Datum format
    print("Processing training examples to Datum format...")
    train_data = []
    for i, example in enumerate(formatted_examples):
        try:
            datum = process_example_to_datum(example, tokenizer)
            train_data.append(datum)
        except Exception as e:
            print(f"Warning: Failed to process example {i}: {e}")
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(formatted_examples)}")

    print(f"Successfully processed {len(train_data)} training examples")

    print("Processing eval examples to Datum format...")
    eval_data = []
    for example in formatted_eval_examples:
        try:
            datum = process_example_to_datum(example, tokenizer)
            eval_data.append(datum)
        except Exception as e:
            pass

    print(f"Successfully processed {len(eval_data)} eval examples")

    # Train
    metrics, best_f1_metrics = train(
        training_client,
        train_data,
        eval_data,
        formatted_eval_examples,
        tokenizer,
        CONFIG
    )

    # Save final checkpoint
    print("\nSaving final checkpoint...")
    training_client.save_state("sd-gptoss-final-checkpoint")

    # Load best checkpoint for final evaluation and classification
    print("\nLoading best F1 checkpoint for final evaluation...")
    training_client.load_state("sd-gptoss-best-checkpoint")

    # Final evaluation
    print("\nRunning final evaluation on best model...")
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name="sd-gptoss-best-sampler"
    )

    final_metrics = evaluate_full_accuracy(
        sampling_client, tokenizer, formatted_eval_examples
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "outputs" / f"codeforces_classifier_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save split info
    split_info = {
        "train_size": len(train_df),
        "eval_size": len(val_df),
        "train_dupes": int(train_dupes_final),
        "eval_dupes": int(eval_dupes_count),
    }

    # Save final results JSON
    final_results = {
        "config": CONFIG,
        "split_info": split_info,
        "final_metrics": final_metrics,
        "training_log": metrics
    }
    with open(output_dir / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    # Save training summary text file
    summary_text = f"""CODEFORCES CLASSIFIER TRAINING SUMMARY (GPT-OSS)
================================================
Timestamp: {timestamp}
Output Directory: {output_dir}

MODEL INFO
----------
Base Model: {CONFIG['base_model']}
LoRA Rank: {CONFIG['lora_rank']}
Best Checkpoint: sd-gptoss-best-checkpoint
Sampler Name: sd-gptoss-best-sampler

To reload this model:
    training_client.load_state("sd-gptoss-best-checkpoint")
    sampling_client = training_client.save_weights_and_get_sampling_client(name="sd-gptoss-best-sampler")

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
Eval Size: {split_info['eval_size']}
Eval Duplicates: {split_info['eval_dupes']}

FINAL METRICS (on eval set)
---------------------------
Accuracy: {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)
Balanced Accuracy: {final_metrics['balanced_accuracy']:.4f}
MCC: {final_metrics['mcc']:.4f}
Macro F1: {final_metrics['macro_f1']:.4f}
Category Accuracy: {final_metrics.get('category_accuracy', 0):.4f}

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

    # Save eval set for reproducibility
    val_df.to_csv(output_dir / "eval_set.csv", index=False)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(summary_text)
    print(f"\nAll results saved to: {output_dir}")

    # Classify CodeForces dataset using CLASSIFY_CONFIG
    print("\n" + "="*60)
    print("CLASSIFYING CODEFORCES DATASET")
    print(f"  Input:  {CLASSIFY_CONFIG['input_csv']}")
    print(f"  Output: {CLASSIFY_CONFIG['output_csv']}")
    print("="*60)

    classify_dataset(
        sampling_client,
        tokenizer,
        CLASSIFY_CONFIG['input_csv'],
        CLASSIFY_CONFIG['output_csv'],
        batch_size=CLASSIFY_CONFIG['batch_size'],
        save_every=CLASSIFY_CONFIG['save_every']
    )


def evaluate_full_accuracy(
    sampling_client,
    tokenizer,
    formatted_eval_examples: List[Dict[str, str]],
    batch_size: int = 50
) -> Dict[str, Any]:
    """Full accuracy evaluation on all eval examples with comprehensive metrics."""

    print(f"\n{'='*60}")
    print(f"FULL EVALUATION ON {len(formatted_eval_examples)} EXAMPLES")
    print(f"{'='*60}")

    params = types.SamplingParams(
        max_tokens=512,
        temperature=0.0,
        stop=["<|eot_id|>"]
    )

    y_true = []
    y_pred = []
    category_correct = 0

    for batch_start in range(0, len(formatted_eval_examples), batch_size):
        batch_end = min(batch_start + batch_size, len(formatted_eval_examples))
        batch = formatted_eval_examples[batch_start:batch_end]

        print(f"  Batch {batch_start//batch_size + 1}/{(len(formatted_eval_examples) + batch_size - 1)//batch_size}...")

        futures = []
        for example in batch:
            prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{example['user']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            prompt_tokens = types.ModelInput.from_ints(tokenizer.encode(prompt))
            future = sampling_client.sample(
                prompt=prompt_tokens,
                sampling_params=params,
                num_samples=1
            )
            futures.append((future, example))

        for future, example in futures:
            try:
                result = future.result()
                response = tokenizer.decode(result.sequences[0].tokens)
                response = response.replace('<|eot_id|>', '').strip()

                expected = json.loads(example['assistant'])
                expected_is_dup = expected.get('is_duplicate', False)
                expected_category = expected.get('predicted_category', 'unrelated')

                try:
                    predicted = json.loads(response)
                    predicted_is_dup = predicted.get('is_duplicate', False)
                    predicted_category = predicted.get('predicted_category', 'unrelated')
                except json.JSONDecodeError:
                    predicted_is_dup = '"is_duplicate": true' in response.lower()
                    predicted_category = None

                y_true.append(expected_is_dup)
                y_pred.append(predicted_is_dup)

                if predicted_category and expected_category == predicted_category:
                    category_correct += 1

            except Exception as e:
                print(f"    Warning: {e}")
                continue

    # Use unified compute_metrics function
    metrics = compute_metrics(y_true, y_pred)
    metrics['category_accuracy'] = category_correct / len(y_true) if y_true else 0.0

    # Print comprehensive results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\nOverall:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  MCC:               {metrics['mcc']:.4f}")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"  Category Accuracy: {metrics['category_accuracy']:.4f}")

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


def format_example_for_classification(row: Dict[str, Any]) -> Dict[str, str]:
    """Format a single example for classification - matches training format."""
    test_text = str(row.get('test_text', ''))[:2000] if pd.notna(row.get('test_text')) else ''
    corpus_text = str(row.get('corpus_text', ''))[:2000] if pd.notna(row.get('corpus_text')) else ''

    # Use same strict JSON prompt as training
    user = f"""Analyze if these two competitive programming problems are semantic duplicates.

TEST PROBLEM:
{test_text}

CORPUS PROBLEM:
{corpus_text}

Categories: exact (identical/verbatim substring), equivalent (same algorithmic core), subset (test is special case of corpus), superset (corpus is simpler), related (shared key insight), unrelated (different problems or unusable corpus)

is_duplicate = true if category is NOT "unrelated"

RESPOND WITH ONLY THIS JSON, NO OTHER TEXT:
{{"reasoning": "<brief analysis>", "predicted_category": "<category>", "is_duplicate": <true/false>, "confidence": <0.0-1.0>}}"""

    return {'user': user}


def classify_dataset(
    sampling_client,
    tokenizer,
    input_path: str,
    output_path: str,
    batch_size: int = 512,
    save_every: int = 10
):
    """Classify a dataset using the trained model."""

    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(df)} rows")

    print(f"\n{'='*60}")
    print(f"CLASSIFYING DATAFRAME ({len(df)} rows)")
    print(f"  Batch size: {batch_size}")
    print(f"  Save every: {save_every} batches")
    print(f"{'='*60}")

    params = types.SamplingParams(
        max_tokens=512,
        temperature=0.0,
        stop=["<|eot_id|>"]
    )

    results = []
    num_batches = (len(df) + batch_size - 1) // batch_size

    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        batch_num = batch_start // batch_size + 1

        print(f"  Processing batch {batch_num}/{num_batches} ({batch_start}-{batch_end})...")

        # Fire off all sampling requests in parallel
        futures = []
        for idx, row in batch_df.iterrows():
            row_dict = row.to_dict()
            example = format_example_for_classification(row_dict)

            prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{example['user']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            prompt_tokens = types.ModelInput.from_ints(tokenizer.encode(prompt))
            future = sampling_client.sample(
                prompt=prompt_tokens,
                sampling_params=params,
                num_samples=1
            )
            futures.append((future, idx, row))

        # Collect results
        for future, idx, row in futures:
            result_row = row.to_dict()

            try:
                result = future.result()
                response = tokenizer.decode(result.sequences[0].tokens)
                response = response.replace('<|eot_id|>', '').strip()

                # Try direct JSON parse first
                try:
                    predicted = json.loads(response)
                except json.JSONDecodeError:
                    # Fallback: extract JSON from response (in case of text before/after)
                    import re
                    json_match = re.search(r'\{[^{}]*"is_duplicate"[^{}]*\}', response, re.DOTALL)
                    if json_match:
                        try:
                            predicted = json.loads(json_match.group())
                        except json.JSONDecodeError:
                            predicted = None
                    else:
                        predicted = None

                if predicted is not None:
                    result_row['predicted_is_duplicate'] = predicted.get('is_duplicate', False)
                    result_row['predicted_confidence'] = predicted.get('confidence', None)
                    result_row['predicted_category'] = predicted.get('predicted_category', None)
                    result_row['predicted_reasoning'] = predicted.get('reasoning', None)
                    result_row['predicted_scores'] = json.dumps(predicted.get('scores', {}))
                    result_row['raw_response'] = response
                    result_row['parse_success'] = True
                else:
                    # Last resort fallback: string matching
                    result_row['predicted_is_duplicate'] = '"is_duplicate": true' in response.lower() or '"is_duplicate":true' in response.lower()
                    result_row['predicted_confidence'] = None
                    result_row['predicted_category'] = None
                    result_row['predicted_reasoning'] = None
                    result_row['predicted_scores'] = None
                    result_row['raw_response'] = response
                    result_row['parse_success'] = False

            except Exception as e:
                print(f"    Warning: Failed to classify row {idx}: {e}")
                result_row['predicted_is_duplicate'] = None
                result_row['predicted_confidence'] = None
                result_row['predicted_category'] = None
                result_row['predicted_reasoning'] = None
                result_row['predicted_scores'] = None
                result_row['raw_response'] = str(e)
                result_row['parse_success'] = False

            results.append(result_row)

        # Periodic save
        if batch_num % save_every == 0:
            temp_df = pd.DataFrame(results)
            temp_path = output_path.replace('.csv', '_checkpoint.csv')
            temp_df.to_csv(temp_path, index=False)
            print(f"    [Checkpoint] Saved {len(results)} rows to {temp_path}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)

    print(f"\n  Classification complete!")
    print(f"  Saved {len(result_df)} rows to {output_path}")

    # Print summary stats if ground truth available
    if 'is_sd' in result_df.columns:
        valid_preds = result_df[result_df['predicted_is_duplicate'].notna()]
        if len(valid_preds) > 0:
            correct = (valid_preds['predicted_is_duplicate'] == valid_preds['is_sd']).sum()
            accuracy = correct / len(valid_preds)
            print(f"  Accuracy (vs is_sd): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
