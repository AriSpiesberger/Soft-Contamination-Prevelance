#!/usr/bin/env python3
"""
Classify Codeforces semantic duplicate pairs using the fine-tuned GPT-OSS model.

Loads a trained checkpoint from Tinker and runs inference on Codeforces data.
No training - classification only. Uses regex JSON fallback parsing.

Usage:
    python classify_codeforces.py                          # default paths
    python classify_codeforces.py --input data/custom.csv  # custom input
"""

import os
import argparse
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any

# Require API key from environment
if not os.environ.get("TINKER_API_KEY"):
    raise ValueError("TINKER_API_KEY environment variable not set")

import tinker
from tinker import types

from shared_utilities import (
    download_checkpoint,
    parse_json_response,
    prepare_text_fields,
    LLAMA_CHAT_FORMAT,
    CODEFORCES_STRICT_JSON_PROMPT_TEMPLATE,
)


def format_example(row: Dict[str, Any]) -> Dict[str, str]:
    """Format a single example for Codeforces classification."""
    test_text, corpus_text = prepare_text_fields(row)
    user = CODEFORCES_STRICT_JSON_PROMPT_TEMPLATE.format(
        test_text=test_text, corpus_text=corpus_text
    )
    return {'user': user}


def classify_dataframe(
    sampling_client,
    tokenizer,
    df: pd.DataFrame,
    output_path: str,
    batch_size: int = 512,
    save_every: int = 10
) -> pd.DataFrame:
    """Classify an entire dataframe using the fine-tuned GPT-OSS model."""

    chat = LLAMA_CHAT_FORMAT

    print(f"\n{'='*60}")
    print(f"CLASSIFYING DATAFRAME ({len(df)} rows)")
    print(f"  Batch size: {batch_size}")
    print(f"  Save every: {save_every} batches")
    print(f"{'='*60}")

    params = types.SamplingParams(
        max_tokens=512,
        temperature=0.0,
        stop=[chat["stop_token"]]
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
            example = format_example(row.to_dict())
            prompt = chat["prompt_template"].format(user=example['user'])
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
                response = response.replace(chat["stop_token"], '').strip()

                predicted = parse_json_response(response)

                if predicted is not None:
                    result_row['predicted_is_duplicate'] = predicted.get('is_duplicate', False)
                    result_row['predicted_confidence'] = predicted.get('confidence', None)
                    result_row['predicted_category'] = predicted.get('predicted_category', None)
                    result_row['predicted_reasoning'] = predicted.get('reasoning', None)
                    result_row['predicted_scores'] = json.dumps(predicted.get('scores', {}))
                    result_row['raw_response'] = response
                    result_row['parse_success'] = True
                else:
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

    # Calculate stats if ground truth exists
    if 'is_sd' in result_df.columns:
        valid_preds = result_df[result_df['predicted_is_duplicate'].notna()]
        if len(valid_preds) > 0:
            correct = (valid_preds['predicted_is_duplicate'] == valid_preds['is_sd']).sum()
            accuracy = correct / len(valid_preds)

            tp = ((valid_preds['predicted_is_duplicate'] == True) & (valid_preds['is_sd'] == True)).sum()
            fp = ((valid_preds['predicted_is_duplicate'] == True) & (valid_preds['is_sd'] == False)).sum()
            fn = ((valid_preds['predicted_is_duplicate'] == False) & (valid_preds['is_sd'] == True)).sum()
            tn = ((valid_preds['predicted_is_duplicate'] == False) & (valid_preds['is_sd'] == False)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            print(f"\n  Classification complete!")
            print(f"  Valid predictions: {len(valid_preds)}/{len(result_df)}")
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1: {f1:.4f}")
            print(f"\n  Confusion Matrix:")
            print(f"                   Pred Dup   Pred Non-Dup")
            print(f"    Actual Dup       {tp:4d}        {fn:4d}")
            print(f"    Actual Non-Dup   {fp:4d}        {tn:4d}")
    else:
        print(f"\n  Classification complete!")
        print(f"  Total rows classified: {len(result_df)}")

    result_df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    print(f"{'='*60}\n")

    return result_df


def main():
    parser = argparse.ArgumentParser(description="Classify Codeforces semantic duplicate pairs")
    parser.add_argument("--input", default=str(Path(__file__).parent / "data" / "codeforces_sample100_per_testid_dataset.csv"),
                        help="Input CSV path")
    parser.add_argument("--output", default=str(Path(__file__).parent / "data" / "codeforces_sample100_classified_gptoss.csv"),
                        help="Output CSV path")
    parser.add_argument("--checkpoint", default="sd-gptoss-checkpoint-step700",
                        help="Tinker checkpoint name")
    parser.add_argument("--model", default="openai/gpt-oss-20b",
                        help="Base model name")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    args = parser.parse_args()

    # Load the model
    print("="*60)
    print("LOADING CODEFORCES CLASSIFIER (GPT-OSS-20b)")
    print("="*60)

    service_client = tinker.ServiceClient()

    print(f"Creating training client for {args.model}...")
    training_client = service_client.create_lora_training_client(
        base_model=args.model,
        rank=args.lora_rank
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    training_client.load_state(args.checkpoint)

    print("Saving weights for sampler...")
    sampling_path = training_client.save_weights_for_sampler(name="sd-gptoss-sampler-weights").result().path
    print(f"Sampling weights saved to: {sampling_path}")

    print("Creating sampling client...")
    sampling_client = service_client.create_sampling_client(model_path=sampling_path)

    tokenizer = sampling_client.get_tokenizer()
    print(f"Model loaded successfully!")

    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)

    df = pd.read_csv(args.input, low_memory=False)
    print(f"Loaded {len(df)} rows from {args.input}")
    if 'dataset' in df.columns:
        print(f"Datasets: {df['dataset'].value_counts().to_dict()}")

    # Run classification
    classify_dataframe(
        sampling_client,
        tokenizer,
        df,
        output_path=args.output,
        batch_size=args.batch_size,
        save_every=10
    )

    print("\n" + "="*60)
    print("Classification complete!")
    print(f"Output saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
