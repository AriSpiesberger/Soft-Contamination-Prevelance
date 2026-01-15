#!/usr/bin/env python3
"""
Fix MBPP training data by adding function signatures to prompts.

The current training data has prompts like:
  "Create a function that identifies elements common to both lists."

But the test harness expects specific function names. This script extracts
the function signature from the code and prepends it to the prompt:
  "def similar_elements(test_tup1, test_tup2): Create a function that identifies elements common to both lists."

This helps the model learn to use the correct function signature.
"""

import csv
import re
from pathlib import Path


def extract_function_signature(code: str) -> str:
    """Extract the function signature (def line) from code."""
    # Match 'def function_name(params):'
    match = re.search(r'def\s+\w+\s*\([^)]*\)\s*:', code)
    if match:
        return match.group(0)
    return None


def main():
    input_file = Path("/lambda/nfs/embeddings/SDTD_Main/mbpp_train.csv")
    output_file = Path("/lambda/nfs/embeddings/SDTD_Main/mbpp_train_fixed.csv")

    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")

    rows_processed = 0
    rows_with_signature = 0
    rows_without_signature = 0

    with open(input_file) as f_in, open(output_file, 'w', newline='') as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=['task_id', 'pair_num', 'prompt', 'code'])
        writer.writeheader()

        for row in reader:
            rows_processed += 1

            # Extract function signature from code
            signature = extract_function_signature(row['code'])

            if signature:
                # Prepend signature to prompt
                new_prompt = f"{signature} {row['prompt']}"
                rows_with_signature += 1
            else:
                # Keep original prompt if no signature found
                new_prompt = row['prompt']
                rows_without_signature += 1

            writer.writerow({
                'task_id': row['task_id'],
                'pair_num': row['pair_num'],
                'prompt': new_prompt,
                'code': row['code']
            })

            if rows_processed % 1000 == 0:
                print(f"Processed {rows_processed} rows...")

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    print(f"Total rows processed: {rows_processed}")
    print(f"Rows with signature added: {rows_with_signature}")
    print(f"Rows without signature: {rows_without_signature}")
    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()
