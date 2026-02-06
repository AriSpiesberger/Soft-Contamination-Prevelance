#!/usr/bin/env python3
"""
Extract top 100 Dolma matches per test_id and classify using the finetuned GPT-OSS model.
"""

import os
import pandas as pd
import json
import urllib.request
import tarfile
from pathlib import Path
from typing import Dict, Any

# Set API key via environment variable (TINKER_API_KEY must be set)
if not os.environ.get("TINKER_API_KEY"):
    raise ValueError("TINKER_API_KEY environment variable not set")

import tinker
from tinker import types


def format_example(row: Dict[str, Any]) -> Dict[str, str]:
    """Format a single example for classification - matches GPT-OSS training format."""

    # Handle NaN/None values in text fields
    test_text = str(row.get('test_text', ''))[:2000] if pd.notna(row.get('test_text')) else ''
    corpus_text = str(row.get('corpus_text', ''))[:2000] if pd.notna(row.get('corpus_text')) else ''

    # CodeForces prompt template - STRICT JSON OUTPUT
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


def classify_dataframe(
    sampling_client,
    tokenizer,
    df: pd.DataFrame,
    output_path: str,
    batch_size: int = 512,
    save_every: int = 5
) -> pd.DataFrame:
    """Classify an entire dataframe using the fine-tuned GPT-OSS model."""

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
            example = format_example(row_dict)

            # GPT-OSS / Llama format
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

                # Strip stop token if present
                response = response.replace('<|eot_id|>', '').strip()

                try:
                    # Try direct JSON parse first
                    predicted = json.loads(response)
                except json.JSONDecodeError:
                    # Fallback: try to extract JSON from response (in case of text before/after)
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

        # Periodic save to avoid losing progress
        if batch_num % save_every == 0:
            temp_df = pd.DataFrame(results)
            temp_path = output_path.replace('.csv', '_checkpoint.csv')
            temp_df.to_csv(temp_path, index=False)
            print(f"    [Checkpoint] Saved {len(results)} rows to {temp_path}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    print(f"{'='*60}\n")

    return result_df


def main():
    # Input path - Dolma Codeforces matches (top 1000 per test_id)
    script_dir = Path(__file__).parent
    input_csv_path = script_dir / "data" / "codeforces_top100" / "all_top1000_matches.csv"

    # Output paths
    top100_output_path = script_dir / "data" / "codeforces_top100" / "codeforces_dolma_top100_for_classification.csv"
    classified_output_path = script_dir / "data" / "codeforces_top100" / "codeforces_dolma_top100_classified.csv"

    # GPT-OSS checkpoint from training (F1=0.815, acc=0.977, prec=0.917, rec=0.733)
    training_checkpoint = "sd-gptoss-checkpoint-step700"
    base_model = "openai/gpt-oss-20b"
    lora_rank = 32

    # === STEP 1: Extract top 100 per test_id ===
    print("="*60)
    print("STEP 1: EXTRACTING TOP 100 MATCHES PER TEST_ID")
    print("="*60)

    print(f"Loading {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    print(f"Total rows: {len(df)}")
    print(f"Unique test_ids: {df['test_id'].nunique()}")

    # Sort by test_id and rank (rank 1 is best), then take top 100 per test_id
    df = df.sort_values(['test_id', 'rank'])
    top100_df = df.groupby('test_id').head(100)
    print(f"After taking top 100 per test_id: {len(top100_df)} rows")

    # Rename columns to match expected format
    top100_df = top100_df.rename(columns={'score': 'similarity'})

    # Add required columns
    top100_df['dataset'] = 'dolma'
    top100_df['p999_threshold'] = 0.2299808349609389  # Use same value as other datasets

    # Select and reorder columns to match expected format
    output_columns = ['test_id', 'test_text', 'corpus_id', 'corpus_text', 'similarity',
                      'p999_threshold', 'dataset', 'elo_bin']
    top100_df = top100_df[output_columns]

    # Save the extracted top 100 for classification
    top100_df.to_csv(top100_output_path, index=False)
    print(f"Saved top 100 dataset to: {top100_output_path}")

    # === STEP 2: Load model and classify ===
    print("\n" + "="*60)
    print("STEP 2: LOADING GPT-OSS MODEL FOR INFERENCE")
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
    sampling_path = training_client.save_weights_for_sampler(name="sd-gptoss-sampler-weights").result().path
    print(f"Sampling weights saved to: {sampling_path}")

    print("Creating sampling client...")
    sampling_client = service_client.create_sampling_client(model_path=sampling_path)

    tokenizer = sampling_client.get_tokenizer()
    print(f"Model loaded successfully!")

    # === STEP 3: Run classification ===
    print("\n" + "="*60)
    print("STEP 3: CLASSIFYING DATA")
    print("="*60)

    classified_df = classify_dataframe(
        sampling_client,
        tokenizer,
        top100_df,
        output_path=classified_output_path,
        batch_size=512,
        save_every=5
    )

    print("\n" + "="*60)
    print("DONE!")
    print(f"Top 100 extracted: {top100_output_path}")
    print(f"Classified output: {classified_output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
