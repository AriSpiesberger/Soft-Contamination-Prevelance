#!/usr/bin/env python3
"""
Train MBPP semantic duplicate classifier.

Fine-tunes Qwen3-30B-A3B-Instruct-2507 on MBPP+Codeforces annotation data
using Tinker API with LoRA. Evaluates on held-out set, then classifies
remaining MBPP data if accuracy > 95%.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json
from pathlib import Path

# Require API key from environment
if not os.environ.get("TINKER_API_KEY"):
    raise ValueError("TINKER_API_KEY environment variable not set")

import tinker
from tinker import types

from shared_utilities import compute_batch_loss, evaluate


# Configuration
CONFIG = {
    "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "lora_rank": 32,
    "learning_rate": 5e-4,  # Tinker recommended: 5e-5 * 10 (LoRA multiplier)
    "num_epochs": 3,
    "batch_size": 128,  # Tinker recommended for fine-tuning
    "eval_loss_every": 10,  # Eval loss (fast, no weight save needed)
    "eval_accuracy_every": 25,  # Accuracy eval (slower, requires weight save)
    "save_every": 25,
    "max_seq_length": 4096,
    "log_path": "/tmp/tinker-sd-finetune",
    "train_split": 0.9,
    "eval_accuracy_samples": 50,  # Number of samples to generate for accuracy eval
}


def load_data(
    csv_path: str,
    additional_csv_path: str = None,
    additional_nondupe_ratio: int = 10,
    codeforces_csv_path: str = None
) -> pd.DataFrame:
    """Load and preprocess the CSV data.

    Args:
        csv_path: Primary CSV with 'success' column
        additional_csv_path: Optional secondary CSV (no 'success' column)
        additional_nondupe_ratio: For additional data, ratio of non-dupes to dupes (e.g., 10 means 10:1)
        codeforces_csv_path: Optional Codeforces CSV (uses different prompt template, takes all rows)
    """
    # Load primary data
    df = pd.read_csv(csv_path)
    # Filter to successful annotations only
    df = df[df['success'] == True].copy()
    # Keep relevant columns
    cols = ['test_text', 'corpus_text', 'is_sd', 'confidence', 'match_type', 'reasoning']
    df = df[cols].dropna()
    df['source'] = 'mbpp'  # Mark source for prompt selection

    # Load additional MBPP data if provided
    if additional_csv_path:
        df_add = pd.read_csv(additional_csv_path)
        df_add = df_add[cols].dropna()
        df_add['source'] = 'mbpp'

        # Split into dupes and non-dupes
        dupes = df_add[df_add['is_sd'] == True]
        non_dupes = df_add[df_add['is_sd'] == False]

        # Sample non-dupes at the specified ratio to dupes
        num_dupes = len(dupes)
        num_non_dupes_to_sample = min(num_dupes * additional_nondupe_ratio, len(non_dupes))

        if num_non_dupes_to_sample > 0:
            non_dupes_sampled = non_dupes.sample(n=num_non_dupes_to_sample, random_state=42)
        else:
            non_dupes_sampled = non_dupes

        # Combine
        df_add_balanced = pd.concat([dupes, non_dupes_sampled], ignore_index=True)
        print(f"Additional MBPP data: {len(dupes)} dupes + {len(non_dupes_sampled)} non-dupes = {len(df_add_balanced)} total")

        df = pd.concat([df, df_add_balanced], ignore_index=True)

    # Load Codeforces data if provided (uses different prompt template)
    # Takes ALL usable rows (no ratio sampling)
    # For Codeforces, is_sd = (match_type != 'unrelated')
    if codeforces_csv_path:
        df_cf = pd.read_csv(codeforces_csv_path)
        df_cf = df_cf[cols].dropna()  # Only keep rows with both texts
        df_cf['source'] = 'codeforces'

        # Fix is_sd based on match_type (non-unrelated = semantic duplicate)
        df_cf['is_sd'] = df_cf['match_type'] != 'unrelated'

        num_dupes = df_cf['is_sd'].sum()
        num_non_dupes = len(df_cf) - num_dupes
        print(f"Codeforces data: {num_dupes} dupes + {num_non_dupes} non-dupes = {len(df_cf)} total (all rows)")

        df = pd.concat([df, df_cf], ignore_index=True)

    return df


def format_example(row: Dict[str, Any]) -> Dict[str, str]:
    """Format a single example for instruction fine-tuning."""

    source = row.get('source', 'mbpp')

    # Handle NaN/None values in text fields
    test_text = str(row.get('test_text', ''))[:2000] if pd.notna(row.get('test_text')) else ''
    corpus_text = str(row.get('corpus_text', ''))[:2000] if pd.notna(row.get('corpus_text')) else ''

    if source == 'codeforces':
        # Codeforces prompt template (competitive programming)
        user = f"""You are an expert competitive programmer analyzing potential semantic duplicates between programming problems.

## Task
Determine if the following two competitive programming problems are semantically related - meaning exposure to the corpus problem during training could help solve the test problem.

## Test Problem (from benchmark):
{test_text}

## Corpus Problem (from training data):
{corpus_text}

## Analysis Steps:
1. **Check data quality first**: Is the corpus text a complete problem statement? If it's empty, fragmentary, or contains only code without a problem description, mark as "unrelated".
2. **Extract the core problem**: Strip away story/narrative framing. What is the actual computational task?
3. **Identify the key insight**: What algorithmic technique or observation is needed?
4. **Compare**: Is there meaningful overlap in what's being asked or how to solve it?

## Match Types:
- "exact": Nearly identical problem statements
- "equivalent": Different framing but identical algorithmic core
- "subset": Test is a special case of corpus
- "superset": Corpus is a special case of test
- "related": Corpus covers a component or shares key insight with test
- "unrelated": Different problems, or corpus data is unusable

## What counts as semantically related:
- Same computational task (any framing)
- One is a special case of the other
- Shared key insight or trick
- Corpus solves a significant component of test

## What is unrelated:
- Sharing only common techniques (DP, BFS) without structural similarity
- Unusable corpus data (empty, fragmentary, code-only)
- Genuinely different computational questions

Analyze the problems and provide your structured judgment."""
    else:
        # Default MBPP prompt template (coding tasks)
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
- "subset": Test task is a subset of corpus task (corpus is harder but solves test)
- "superset": Corpus task is a subset of test task (test is harder) - NOT a duplicate
- "unrelated": Different tasks entirely
Analyze the tasks and provide your structured judgment."""

    # Expected output
    is_sd = str(row['is_sd']).lower() == 'true'
    confidence = float(row['confidence']) if pd.notna(row['confidence']) else 1.0
    output = json.dumps({
        "is_duplicate": is_sd,
        "match_type": row['match_type'],
        "confidence": confidence,
        "reasoning": row['reasoning'][:500] if pd.notna(row['reasoning']) else ""
    }, indent=2)

    return {
        "user": user,
        "assistant": output
    }


def format_example_for_classification(row: Dict[str, Any]) -> str:
    """Format a single example for classification (prompt only, no expected output)."""

    source = row.get('source', 'mbpp')

    # Handle NaN/None values in text fields
    test_text = str(row.get('test_text', ''))[:2000] if pd.notna(row.get('test_text')) else ''
    corpus_text = str(row.get('corpus_text', ''))[:2000] if pd.notna(row.get('corpus_text')) else ''

    if source == 'codeforces':
        # Codeforces prompt template (competitive programming)
        user = f"""You are an expert competitive programmer analyzing potential semantic duplicates between programming problems.

## Task
Determine if the following two competitive programming problems are semantically related - meaning exposure to the corpus problem during training could help solve the test problem.

## Test Problem (from benchmark):
{test_text}

## Corpus Problem (from training data):
{corpus_text}

## Analysis Steps:
1. **Check data quality first**: Is the corpus text a complete problem statement? If it's empty, fragmentary, or contains only code without a problem description, mark as "unrelated".
2. **Extract the core problem**: Strip away story/narrative framing. What is the actual computational task?
3. **Identify the key insight**: What algorithmic technique or observation is needed?
4. **Compare**: Is there meaningful overlap in what's being asked or how to solve it?

## Match Types:
- "exact": Nearly identical problem statements
- "equivalent": Different framing but identical algorithmic core
- "subset": Test is a special case of corpus
- "superset": Corpus is a special case of test
- "related": Corpus covers a component or shares key insight with test
- "unrelated": Different problems, or corpus data is unusable

## What counts as semantically related:
- Same computational task (any framing)
- One is a special case of the other
- Shared key insight or trick
- Corpus solves a significant component of test

## What is unrelated:
- Sharing only common techniques (DP, BFS) without structural similarity
- Unusable corpus data (empty, fragmentary, code-only)
- Genuinely different computational questions

Analyze the problems and provide your structured judgment."""
    else:
        # Default MBPP prompt template (coding tasks)
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
- "subset": Test task is a subset of corpus task (corpus is harder but solves test)
- "superset": Corpus task is a subset of test task (test is harder) - NOT a duplicate
- "unrelated": Different tasks entirely
Analyze the tasks and provide your structured judgment."""

    return user


def process_example_to_datum(
    example: Dict[str, str],
    tokenizer
) -> types.Datum:
    """Convert a formatted example to a Tinker Datum for training."""
    
    # Build the chat-formatted prompt using Qwen chat template (no system)
    prompt = f"""<|im_start|>user
{example['user']}<|im_end|>
<|im_start|>assistant
"""
    completion = f"""{example['assistant']}<|im_end|>"""
    
    # Tokenize
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


def train(
    training_client,
    train_data: List[types.Datum],
    eval_data: List[types.Datum],
    formatted_eval_examples: List[Dict[str, str]],
    tokenizer,
    config: Dict[str, Any]
):
    """Main training loop."""

    batch_size = config["batch_size"]
    num_examples = len(train_data)
    num_batches = (num_examples + batch_size - 1) // batch_size

    print(f"\n{'='*60}")
    print(f"Starting training")
    print(f"  Model: {config['base_model']}")
    print(f"  Training examples: {len(train_data)}")
    print(f"  Eval examples: {len(eval_data)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num batches per epoch: {num_batches}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"{'='*60}\n")

    metrics = []
    step = 0
    best_accuracy = 0.0

    for epoch in range(config["num_epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{config['num_epochs']} ---")

        # Shuffle training data
        indices = np.random.permutation(num_examples)

        # Pipelined training: submit next batch before waiting for current
        # This prevents missing Tinker's ~10 second clock cycles
        pending_future = None
        pending_optim = None
        pending_batch = None

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_examples)
            batch_indices = indices[start_idx:end_idx]
            batch = [train_data[i] for i in batch_indices]

            # Submit current batch immediately
            fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
            optim_future = training_client.optim_step(
                types.AdamParams(learning_rate=config["learning_rate"])
            )

            # Now wait for PREVIOUS batch results (if any)
            if pending_future is not None:
                fwdbwd_result = pending_future.result()
                _ = pending_optim.result()  # Must wait for optim to complete
                train_loss = compute_batch_loss(fwdbwd_result, pending_batch)

                step += 1

                # Log progress
                if step % 1 == 0:  # Log every step since we have fewer batches now
                    print(f"Step {step}/{num_batches * config['num_epochs']}: train_loss={train_loss:.4f}")

                # Eval loss (fast - no weight save needed)
                if step % config["eval_loss_every"] == 0 and len(eval_data) > 0:
                    eval_loss = evaluate(training_client, eval_data, batch_size)
                    print(f"  [Eval] step={step}, eval_loss={eval_loss:.4f}")
                    metrics.append({
                        "step": step,
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "eval_loss": eval_loss
                    })

                # Accuracy evaluation (slower - requires weight save for sampling)
                if step % config["eval_accuracy_every"] == 0 and len(eval_data) > 0:
                    print(f"  [Accuracy Eval] Saving weights and generating {config['eval_accuracy_samples']} samples...")
                    acc_metrics = evaluate_accuracy(
                        training_client, tokenizer, formatted_eval_examples,
                        num_samples=config['eval_accuracy_samples']
                    )
                    print(f"  [Accuracy] acc={acc_metrics['accuracy']:.3f}, "
                          f"prec={acc_metrics['precision']:.3f}, "
                          f"rec={acc_metrics['recall']:.3f}, "
                          f"f1={acc_metrics['f1']:.3f}")

                    metrics.append({
                        "step": step,
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "eval_loss": eval_loss,
                        **acc_metrics
                    })

                    # Track best model
                    if acc_metrics['accuracy'] > best_accuracy:
                        best_accuracy = acc_metrics['accuracy']
                        print(f"  [Best] New best accuracy: {best_accuracy:.3f}")
                        training_client.save_state("sd-best-checkpoint")

                # Save checkpoint
                if step % config["save_every"] == 0:
                    checkpoint_name = f"sd-checkpoint-step{step}"
                    training_client.save_state(checkpoint_name)
                    print(f"  [Checkpoint] Saved: {checkpoint_name}")

            # Store current as pending for next iteration
            pending_future = fwdbwd_future
            pending_optim = optim_future
            pending_batch = batch

        # Process final batch
        if pending_future is not None:
            fwdbwd_result = pending_future.result()
            _ = pending_optim.result()  # Must wait for optim to complete
            train_loss = compute_batch_loss(fwdbwd_result, pending_batch)
            step += 1
            print(f"Step {step}/{num_batches * config['num_epochs']}: train_loss={train_loss:.4f}")
            metrics.append({"step": step, "epoch": epoch + 1, "train_loss": train_loss})

    print(f"\nTraining complete. Best accuracy: {best_accuracy:.3f}")
    return metrics


def evaluate_accuracy(
    training_client,
    tokenizer,
    formatted_eval_examples: List[Dict[str, str]],
    num_samples: int = 50
) -> Dict[str, float]:
    """Evaluate accuracy by generating responses and checking is_duplicate prediction."""

    # Sample a subset for efficiency
    sample_indices = np.random.choice(len(formatted_eval_examples), min(num_samples, len(formatted_eval_examples)), replace=False)
    samples = [formatted_eval_examples[i] for i in sample_indices]

    # Get sampling client from training client (requires saving weights)
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=f"sd-eval-temp-{np.random.randint(100000)}"
    )

    params = types.SamplingParams(
        max_tokens=512,
        temperature=0.0,
        stop=["<|im_end|>"]
    )

    # Fire off all sampling requests in parallel (async futures)
    futures = []
    for example in samples:
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
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for future, example in futures:
        try:
            result = future.result()
            response = tokenizer.decode(result.sequences[0].tokens)

            # Parse expected and predicted is_duplicate
            expected = json.loads(example['assistant'])
            expected_is_dup = expected.get('is_duplicate', False)

            try:
                predicted = json.loads(response)
                predicted_is_dup = predicted.get('is_duplicate', False)
            except json.JSONDecodeError:
                # If we can't parse, check for substring
                predicted_is_dup = '"is_duplicate": true' in response.lower()

            # Update metrics
            total += 1
            if predicted_is_dup == expected_is_dup:
                correct += 1
                if expected_is_dup:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                if predicted_is_dup:
                    false_positives += 1
                else:
                    false_negatives += 1

        except Exception as e:
            print(f"    Warning: Failed to evaluate sample: {e}")
            continue

    accuracy = correct / total if total > 0 else 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": total,
        "correct": correct
    }


def evaluate_full_accuracy(
    sampling_client,
    tokenizer,
    formatted_eval_examples: List[Dict[str, str]],
    batch_size: int = 50
) -> Dict[str, Any]:
    """Run full accuracy evaluation on ALL eval examples with detailed stats."""

    print(f"\n{'='*60}")
    print(f"FULL ACCURACY EVALUATION ON {len(formatted_eval_examples)} EXAMPLES")
    print(f"{'='*60}")

    params = types.SamplingParams(
        max_tokens=512,
        temperature=0.0,
        stop=["<|im_end|>"]
    )

    # Process in batches to avoid memory issues
    all_results = []

    for batch_start in range(0, len(formatted_eval_examples), batch_size):
        batch_end = min(batch_start + batch_size, len(formatted_eval_examples))
        batch = formatted_eval_examples[batch_start:batch_end]

        print(f"  Processing batch {batch_start//batch_size + 1}/{(len(formatted_eval_examples) + batch_size - 1)//batch_size} ({batch_start}-{batch_end})...")

        # Fire off all sampling requests in parallel
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
                except json.JSONDecodeError:
                    predicted_is_dup = '"is_duplicate": true' in response.lower()

                all_results.append({
                    'expected': expected_is_dup,
                    'predicted': predicted_is_dup,
                    'correct': expected_is_dup == predicted_is_dup
                })

            except Exception as e:
                print(f"    Warning: Failed to evaluate sample: {e}")
                continue

    # Calculate metrics
    total = len(all_results)
    correct = sum(1 for r in all_results if r['correct'])

    true_positives = sum(1 for r in all_results if r['expected'] and r['predicted'])
    true_negatives = sum(1 for r in all_results if not r['expected'] and not r['predicted'])
    false_positives = sum(1 for r in all_results if not r['expected'] and r['predicted'])
    false_negatives = sum(1 for r in all_results if r['expected'] and not r['predicted'])

    accuracy = correct / total if total > 0 else 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Print detailed results
    print(f"\n{'='*60}")
    print("FINAL MODEL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\nOverall Statistics:")
    print(f"  Total examples:     {total}")
    print(f"  Correct:            {correct}")
    print(f"  Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")

    print(f"\nConfusion Matrix:")
    print(f"                      Predicted")
    print(f"                   Dup    Non-Dup")
    print(f"  Actual Dup       {true_positives:4d}    {false_negatives:4d}")
    print(f"  Actual Non-Dup   {false_positives:4d}    {true_negatives:4d}")

    print(f"\nClass-wise Metrics:")
    print(f"  Precision (dup):    {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall (dup):       {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1 Score:           {f1:.4f} ({f1*100:.2f}%)")

    # Calculate non-duplicate metrics too
    nd_precision = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0.0
    nd_recall = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
    nd_f1 = 2 * nd_precision * nd_recall / (nd_precision + nd_recall) if (nd_precision + nd_recall) > 0 else 0.0

    print(f"\n  Precision (non-dup): {nd_precision:.4f} ({nd_precision*100:.2f}%)")
    print(f"  Recall (non-dup):    {nd_recall:.4f} ({nd_recall*100:.2f}%)")
    print(f"  F1 (non-dup):        {nd_f1:.4f} ({nd_f1*100:.2f}%)")

    print(f"\nClass Distribution in Eval Set:")
    actual_dups = true_positives + false_negatives
    actual_non_dups = true_negatives + false_positives
    print(f"  Actual duplicates:     {actual_dups} ({actual_dups/total*100:.1f}%)")
    print(f"  Actual non-duplicates: {actual_non_dups} ({actual_non_dups/total*100:.1f}%)")
    print(f"{'='*60}\n")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "nd_precision": nd_precision,
        "nd_recall": nd_recall,
        "nd_f1": nd_f1,
        "total": total,
        "correct": correct,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }


def classify_dataframe(
    sampling_client,
    tokenizer,
    df: pd.DataFrame,
    output_path: str,
    source: str = 'mbpp',
    batch_size: int = 50
) -> pd.DataFrame:
    """Classify an entire dataframe using the fine-tuned model.

    Args:
        sampling_client: Tinker sampling client
        tokenizer: Model tokenizer
        df: DataFrame with test_text and corpus_text columns
        output_path: Path to save the classified CSV
        source: 'mbpp' or 'codeforces' for prompt selection
        batch_size: Batch size for parallel inference

    Returns:
        DataFrame with added prediction columns
    """
    print(f"\n{'='*60}")
    print(f"CLASSIFYING DATAFRAME ({len(df)} rows)")
    print(f"{'='*60}")

    params = types.SamplingParams(
        max_tokens=512,
        temperature=0.0,
        stop=["<|im_end|>"]
    )

    # Prepare all rows
    results = []

    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]

        print(f"  Processing batch {batch_start//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size} ({batch_start}-{batch_end})...")

        # Fire off all sampling requests in parallel
        futures = []
        for idx, row in batch_df.iterrows():
            # Format the prompt for classification (no expected output needed)
            row_dict = row.to_dict()
            row_dict['source'] = source
            user_prompt = format_example_for_classification(row_dict)

            prompt = f"""<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
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

                # Parse predicted response
                try:
                    predicted = json.loads(response)
                    result_row['predicted_is_duplicate'] = predicted.get('is_duplicate', False)
                    result_row['predicted_confidence'] = predicted.get('confidence', None)
                    result_row['predicted_match_type'] = predicted.get('match_type', None)
                    result_row['predicted_reasoning'] = predicted.get('reasoning', None)
                    result_row['raw_response'] = response
                    result_row['parse_success'] = True
                except json.JSONDecodeError:
                    # Fallback: check for substring
                    result_row['predicted_is_duplicate'] = '"is_duplicate": true' in response.lower()
                    result_row['predicted_confidence'] = None
                    result_row['predicted_match_type'] = None
                    result_row['predicted_reasoning'] = None
                    result_row['raw_response'] = response
                    result_row['parse_success'] = False

            except Exception as e:
                print(f"    Warning: Failed to classify row {idx}: {e}")
                result_row['predicted_is_duplicate'] = None
                result_row['predicted_confidence'] = None
                result_row['predicted_match_type'] = None
                result_row['predicted_reasoning'] = None
                result_row['raw_response'] = str(e)
                result_row['parse_success'] = False

            results.append(result_row)

    # Create output dataframe
    result_df = pd.DataFrame(results)

    # Calculate stats if ground truth exists
    if 'is_sd' in result_df.columns:
        valid_preds = result_df[result_df['predicted_is_duplicate'].notna()]
        if len(valid_preds) > 0:
            correct = (valid_preds['predicted_is_duplicate'] == valid_preds['is_sd']).sum()
            accuracy = correct / len(valid_preds)
            print(f"\n  Classification complete!")
            print(f"  Valid predictions: {len(valid_preds)}/{len(result_df)}")
            print(f"  Accuracy (vs is_sd): {accuracy:.4f} ({accuracy*100:.2f}%)")
    else:
        print(f"\n  Classification complete!")
        print(f"  Total rows classified: {len(result_df)}")

    # Save to CSV
    result_df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    print(f"{'='*60}\n")

    return result_df


def test_sampling(sampling_client, tokenizer, test_examples: List[Dict[str, str]]):
    """Test the fine-tuned model on a few examples."""
    print("\n" + "="*60)
    print("Testing fine-tuned model")
    print("="*60)
    
    for i, example in enumerate(test_examples[:3]):
        prompt = f"""<|im_start|>user
{example['user']}<|im_end|>
<|im_start|>assistant
"""
        prompt_tokens = types.ModelInput.from_ints(tokenizer.encode(prompt))
        params = types.SamplingParams(
            max_tokens=512,
            temperature=0.0,
            stop=["<|im_end|>"]
        )
        
        result = sampling_client.sample(
            prompt=prompt_tokens,
            sampling_params=params,
            num_samples=1
        ).result()
        
        response = tokenizer.decode(result.sequences[0].tokens)
        
        print(f"\n--- Test Example {i+1} ---")
        print(f"Expected: {example['assistant'][:200]}...")
        print(f"Generated: {response[:200]}...")


def load_all_training_data(training_data_dir: str) -> pd.DataFrame:
    """Load all CSV files from training_data directory."""
    import glob

    all_dfs = []
    cols = ['test_text', 'corpus_text', 'is_sd', 'confidence', 'match_type', 'reasoning']

    csv_files = glob.glob(f"{training_data_dir}/*.csv")
    print(f"Found {len(csv_files)} CSV files in {training_data_dir}")

    for csv_path in csv_files:
        filename = csv_path.split('/')[-1].split('\\')[-1]

        # Skip partial classification files and files without annotations
        skip_patterns = ['_partial_', '_classified', '_to_classify', '_sampled']
        if any(p in filename for p in skip_patterns):
            print(f"  Skipping {filename} (not annotation data)")
            continue

        print(f"  Loading {filename}...")

        try:
            df = pd.read_csv(csv_path)

            # Check if it has a 'success' column (filter to successful only)
            if 'success' in df.columns:
                df = df[df['success'] == True].copy()

            # Keep only required columns
            available_cols = [c for c in cols if c in df.columns]
            if 'test_text' not in available_cols or 'corpus_text' not in available_cols:
                print(f"    Skipping {filename}: missing required columns")
                continue

            df = df[available_cols].dropna(subset=['test_text', 'corpus_text'])

            # Determine source based on filename
            if 'codeforces' in filename.lower():
                df['source'] = 'codeforces'
                # Fix is_sd based on match_type for codeforces
                if 'match_type' in df.columns:
                    df['is_sd'] = df['match_type'] != 'unrelated'
            else:
                df['source'] = 'mbpp'

            # Fill missing columns with defaults
            for col in cols:
                if col not in df.columns:
                    df[col] = None

            num_dupes = df['is_sd'].sum() if 'is_sd' in df.columns else 0
            print(f"    Loaded {len(df)} rows ({num_dupes} dupes)")
            all_dfs.append(df)

        except Exception as e:
            print(f"    Error loading {filename}: {e}")
            continue

    if not all_dfs:
        raise ValueError("No valid training data found!")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal combined: {len(combined)} rows")
    return combined


def main():
    # Load ALL training data from folder
    training_data_dir = str(Path(__file__).parent.parent / "comparison_analysis" / "training_data")

    print("="*60)
    print("LOADING ALL TRAINING DATA")
    print("="*60)

    df = load_all_training_data(training_data_dir)
    print(f"Source distribution: {df['source'].value_counts().to_dict()}")
    print(f"is_sd distribution: {df['is_sd'].value_counts().to_dict()}")

    # Create a small balanced eval set from ALL data (MBPP + Codeforces)
    # 30 semantic dupes + 300 non-dupes = 330 total
    print("\nCreating balanced eval set (30 dupes + 300 non-dupes)...")
    all_dupes = df[df['is_sd'] == True]
    all_non_dupes = df[df['is_sd'] == False]

    # Sample for eval (keep track of indices to exclude from training)
    eval_dupes = all_dupes.sample(n=min(30, len(all_dupes)), random_state=42)
    eval_non_dupes = all_non_dupes.sample(n=min(300, len(all_non_dupes)), random_state=42)
    eval_df = pd.concat([eval_dupes, eval_non_dupes], ignore_index=False)  # Keep original indices
    eval_indices = set(eval_df.index)
    eval_df = eval_df.sample(frac=1, random_state=42)  # Shuffle
    eval_dupes = (eval_df['is_sd'] == True).sum()
    eval_non_dupes = (eval_df['is_sd'] == False).sum()
    print(f"Eval data: {len(eval_df)} rows ({eval_dupes} dupes, {eval_non_dupes} non-dupes)")

    # Exclude eval samples from training data
    train_df = df[~df.index.isin(eval_indices)].copy()
    print(f"Training data (before balancing): {len(train_df)} rows (excluded {len(eval_indices)} eval samples)")
    train_dupes_count = (train_df['is_sd'] == True).sum()
    train_non_dupes_count = (train_df['is_sd'] == False).sum()
    print(f"  Training dupes: {train_dupes_count}, non-dupes: {train_non_dupes_count}")

    # Balance training data: 1:10 dupes to non-dupes ratio
    # 10 non-dupes for every 1 dupe
    train_dupes_df = train_df[train_df['is_sd'] == True]
    train_non_dupes_df = train_df[train_df['is_sd'] == False]

    # Sample non-dupes at 10:1 ratio to dupes (i.e., 10 non-dupes per 1 dupe)
    non_dupe_sample_size = min(len(train_dupes_df) * 10, len(train_non_dupes_df))
    train_non_dupes_sampled = train_non_dupes_df.sample(n=non_dupe_sample_size, random_state=42)

    train_df = pd.concat([train_dupes_df, train_non_dupes_sampled], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=42)  # Shuffle

    train_dupes_final = (train_df['is_sd'] == True).sum()
    train_non_dupes_final = (train_df['is_sd'] == False).sum()
    print(f"Training data (after balancing): {len(train_df)} rows")
    print(f"  Balanced training: {train_dupes_final} dupes, {train_non_dupes_final} non-dupes (ratio 1:{train_non_dupes_final/max(1,train_dupes_final):.1f})")

    # Format examples
    print("Formatting training examples...")
    formatted_examples = [format_example(row) for _, row in train_df.iterrows()]

    print("Formatting eval examples...")
    formatted_eval_examples = [format_example(row) for _, row in eval_df.iterrows()]
    
    # Create service client and training client
    print(f"\nConnecting to Tinker API...")
    service_client = tinker.ServiceClient()
    
    print(f"Creating training client for {CONFIG['base_model']}...")
    training_client = service_client.create_lora_training_client(
        base_model=CONFIG["base_model"],
        rank=CONFIG["lora_rank"]
    )
    
    # Get tokenizer
    tokenizer = training_client.get_tokenizer()
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    # Convert training data to Datum format
    print("Processing training examples to Datum format...")
    train_data = []
    for i, example in enumerate(formatted_examples):
        try:
            datum = process_example_to_datum(example, tokenizer)
            train_data.append(datum)
        except Exception as e:
            print(f"Warning: Failed to process training example {i}: {e}")

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(formatted_examples)} training examples")

    print(f"Successfully processed {len(train_data)} training examples")

    # Convert eval data to Datum format
    print("Processing eval examples to Datum format...")
    eval_data = []
    for i, example in enumerate(formatted_eval_examples):
        try:
            datum = process_example_to_datum(example, tokenizer)
            eval_data.append(datum)
        except Exception as e:
            print(f"Warning: Failed to process eval example {i}: {e}")

    print(f"Successfully processed {len(eval_data)} eval examples")

    # Shuffle training data
    train_indices = np.random.permutation(len(train_data)).tolist()
    train_data = [train_data[i] for i in train_indices]

    print(f"\n  Train: {len(train_data)} examples")
    print(f"  Eval: {len(eval_data)} examples ({eval_df['is_sd'].sum()} dupes)")

    # Create log directory
    log_path = Path(CONFIG["log_path"])
    log_path.mkdir(parents=True, exist_ok=True)

    # Train
    metrics = train(training_client, train_data, eval_data, formatted_eval_examples, tokenizer, CONFIG)
    
    # Save metrics
    metrics_path = log_path / "metrics.jsonl"
    with open(metrics_path, "w") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")
    print(f"\nMetrics saved to {metrics_path}")
    
    # Save final model and create sampling client
    print("\nSaving final model...")
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name="sd-detector-qwen3-30b-final"
    )

    # Test sampling
    test_sampling(sampling_client, tokenizer, formatted_examples[-5:])

    # Run full accuracy evaluation on final model
    # (Using final model since best checkpoint requires loading via training_client)
    print("\n" + "="*60)
    print("Running full evaluation on final model...")
    print("="*60)

    # Run full evaluation on entire eval set
    final_metrics = evaluate_full_accuracy(
        sampling_client,  # Use the final model's sampling client
        tokenizer,
        formatted_eval_examples,
        batch_size=50
    )

    # Save final evaluation results
    final_eval_path = log_path / "final_evaluation.json"
    with open(final_eval_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Final evaluation saved to {final_eval_path}")

    # Classify combined MBPP + Dolma dataset only if accuracy > 95%
    mbpp_csv_path = str(Path(__file__).parent.parent / "comparison_analysis" / "training_data" / "mbpp_to_classify_sampled.csv")
    dolma_csv_path = str(Path(__file__).parent / "data" / "dolma_mbpp_sample100.csv")
    combined_output_path = str(Path(__file__).parent.parent / "comparison_analysis" / "training_data" / "mbpp_dolma_combined_classified.csv")

    if final_metrics['accuracy'] < 0.95:
        print(f"\nSkipping classification - accuracy {final_metrics['accuracy']*100:.2f}% is below 95% threshold")
        print("Model needs more training or data to reach acceptable accuracy.")
    else:
        print("\n" + "="*60)
        print(f"Classifying combined MBPP + Dolma dataset (accuracy {final_metrics['accuracy']*100:.2f}% >= 95%)...")
        print("="*60)

        # Load and combine both CSVs
        dfs_to_combine = []

        if Path(mbpp_csv_path).exists():
            mbpp_df = pd.read_csv(mbpp_csv_path)
            print(f"Loaded {len(mbpp_df)} rows from MBPP sampled")
            dfs_to_combine.append(mbpp_df)
        else:
            print(f"Warning: MBPP file not found: {mbpp_csv_path}")

        if Path(dolma_csv_path).exists():
            dolma_df = pd.read_csv(dolma_csv_path)
            print(f"Loaded {len(dolma_df)} rows from Dolma")
            dfs_to_combine.append(dolma_df)
        else:
            print(f"Warning: Dolma file not found: {dolma_csv_path}")

        if dfs_to_combine:
            combined_df = pd.concat(dfs_to_combine, ignore_index=True)
            print(f"Combined total: {len(combined_df)} rows")

            # Run classification (large batch size for high throughput)
            classified_df = classify_dataframe(
                sampling_client,
                tokenizer,
                combined_df,
                output_path=combined_output_path,
                source='mbpp',
                batch_size=256  # Large batch for parallel inference
            )
        else:
            print("No data files found to classify!")

    print("\n" + "="*60)
    print("Training complete!")
    print("Final model: sd-detector-qwen3-30b-final")
    print(f"Logs: {CONFIG['log_path']}")
    print("="*60)


if __name__ == "__main__":
    main()
