#!/usr/bin/env python3
"""
Classification script for MBPP sample100 (0.1%) data using the fine-tuned Qwen3 model.
"""

import os
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any

# Set API key via environment variable (TINKER_API_KEY must be set)
if not os.environ.get("TINKER_API_KEY"):
    raise ValueError("TINKER_API_KEY environment variable not set")

import tinker
from tinker import types


def format_example(row: Dict[str, Any]) -> Dict[str, str]:
    """Format a single example for classification - MBPP prompt template."""

    # Handle NaN/None values in text fields
    test_text = str(row.get('test_text', ''))[:2000] if pd.notna(row.get('test_text')) else ''
    corpus_text = str(row.get('corpus_text', ''))[:2000] if pd.notna(row.get('corpus_text')) else ''

    # MBPP prompt template (same as training)
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

    return {'user': user}


def classify_dataframe(
    sampling_client,
    tokenizer,
    df: pd.DataFrame,
    output_path: str,
    batch_size: int = 512,
    save_every: int = 10
) -> pd.DataFrame:
    """Classify an entire dataframe using the fine-tuned model."""

    print(f"\n{'='*60}")
    print(f"CLASSIFYING DATAFRAME ({len(df)} rows)")
    print(f"  Batch size: {batch_size}")
    print(f"  Save every: {save_every} batches")
    print(f"{'='*60}")

    params = types.SamplingParams(
        max_tokens=512,
        temperature=0.0,
        stop=["<|im_end|>"]
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
            example = format_example(row_dict)

            # ChatML format for Qwen
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
            futures.append((future, idx, row))

        # Collect results
        for future, idx, row in futures:
            result_row = row.to_dict()

            try:
                result = future.result()
                response = tokenizer.decode(result.sequences[0].tokens)

                # Strip stop token if present
                response = response.replace('<|im_end|>', '').strip()

                try:
                    predicted = json.loads(response)
                    result_row['predicted_is_duplicate'] = predicted.get('is_duplicate', False)
                    result_row['predicted_confidence'] = predicted.get('confidence', None)
                    result_row['predicted_match_type'] = predicted.get('match_type', None)
                    result_row['predicted_reasoning'] = predicted.get('reasoning', None)
                    result_row['raw_response'] = response
                    result_row['parse_success'] = True
                except json.JSONDecodeError:
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

        # Periodic save
        if batch_num % save_every == 0:
            temp_df = pd.DataFrame(results)
            temp_path = output_path.replace('.csv', '_checkpoint.csv')
            temp_df.to_csv(temp_path, index=False)
            print(f"    [Checkpoint] Saved {len(results)} rows to {temp_path}")

    result_df = pd.DataFrame(results)

    # Stats
    parse_success_rate = result_df['parse_success'].mean() * 100
    dup_rate = result_df['predicted_is_duplicate'].mean() * 100 if result_df['predicted_is_duplicate'].notna().any() else 0

    print(f"\n  Classification complete!")
    print(f"  Total rows: {len(result_df)}")
    print(f"  Parse success rate: {parse_success_rate:.2f}%")
    print(f"  Duplicate rate: {dup_rate:.2f}%")

    if 'dataset' in result_df.columns:
        print(f"\n  Duplicate rate by dataset:")
        for ds in result_df['dataset'].unique():
            ds_df = result_df[result_df['dataset'] == ds]
            ds_dup_rate = ds_df['predicted_is_duplicate'].mean() * 100
            print(f"    {ds}: {ds_dup_rate:.2f}%")

    result_df.to_csv(output_path, index=False)
    print(f"\n  Saved to: {output_path}")
    print(f"{'='*60}\n")

    return result_df


def main():
    # Paths - MBPP sample100 (0.1% sample)
    script_dir = Path(__file__).parent
    input_csv_path = script_dir / "data" / "mbpp_sample100" / "mbpp_sample100_per_testid_dataset.csv"
    output_path = script_dir / "data" / "mbpp_sample100" / "mbpp_sample100_classified.csv"

    # MBPP classifier checkpoint (Qwen3-30B fine-tuned)
    training_checkpoint = "tinker://c38ec8a6-d887-54aa-b552-2af1c89da5af:train:0/weights/sd-best-checkpoint"
    base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    lora_rank = 32

    # Load the model
    print("="*60)
    print("LOADING MBPP CLASSIFIER (Qwen3-30B)")
    print("="*60)

    service_client = tinker.ServiceClient()

    # Create training client and restore from checkpoint
    print(f"Creating training client for {base_model}...")
    training_client = service_client.create_lora_training_client(
        base_model=base_model,
        rank=lora_rank
    )

    print(f"Loading checkpoint: {training_checkpoint}")
    training_client.load_state(training_checkpoint)

    # Save weights for sampler and create sampling client
    print("Saving weights for sampler...")
    sampling_path = training_client.save_weights_for_sampler(name="mbpp-sample-sampler-weights").result().path
    print(f"Sampling weights saved to: {sampling_path}")

    print("Creating sampling client...")
    sampling_client = service_client.create_sampling_client(model_path=sampling_path)

    tokenizer = sampling_client.get_tokenizer()
    print(f"Model loaded successfully!")

    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)

    df = pd.read_csv(input_csv_path, low_memory=False)
    print(f"Loaded {len(df):,} rows from {input_csv_path}")
    if 'dataset' in df.columns:
        print(f"Datasets: {df['dataset'].value_counts().to_dict()}")

    # Run classification
    classified_df = classify_dataframe(
        sampling_client,
        tokenizer,
        df,
        output_path=output_path,
        batch_size=512,
        save_every=10
    )

    print("\n" + "="*60)
    print("CLASSIFICATION COMPLETE!")
    print(f"Output saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
