#!/usr/bin/env python3
"""
Filter training data to only include examples that pass their test cases.
"""
import csv
import subprocess
import tempfile
import os
import sys
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset


def load_test_cases():
    """Load test cases from HuggingFace MBPP dataset."""
    test_cases = {}

    print("Loading test cases from HuggingFace MBPP dataset...")
    for split in ['test', 'train', 'validation', 'prompt']:
        try:
            ds = load_dataset('mbpp', split=split)
            for item in ds:
                task_id = item['task_id']
                if task_id not in test_cases:
                    test_cases[task_id] = item['test_list']
        except Exception as e:
            print(f"Warning: Could not load split '{split}': {e}")

    print(f"Loaded test cases for {len(test_cases)} tasks")
    return test_cases


def run_code_with_tests(code: str, test_list: list, timeout: float = 5.0) -> bool:
    """Execute code with test assertions. Returns True if all tests pass."""
    # Build test script
    test_script = code + "\n\n"
    for test in test_list:
        test_script += test + "\n"

    # Run in subprocess
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            temp_path = f.name

        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        os.unlink(temp_path)
        return result.returncode == 0

    except:
        try:
            os.unlink(temp_path)
        except:
            pass
        return False


def main():
    input_file = Path("/lambda/nfs/embeddings/SDTD_Main/mbpp_train_fixed.csv")
    output_file = Path("/lambda/nfs/embeddings/SDTD_Main/mbpp_train_filtered.csv")

    # Load test cases
    test_cases = load_test_cases()

    # Load training data
    print(f"\nFiltering training data from: {input_file}")
    print(f"Output will be saved to: {output_file}")

    passing_rows = []
    failing_rows = []

    with open(input_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in tqdm(rows, desc="Testing examples"):
        task_id = int(row['task_id'])
        code = row['code']

        # Get test cases for this task
        if task_id not in test_cases:
            # No tests available, keep it (assume correct)
            passing_rows.append(row)
            continue

        # Run tests
        if run_code_with_tests(code, test_cases[task_id]):
            passing_rows.append(row)
        else:
            failing_rows.append(row)

    # Write filtered data
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task_id', 'pair_num', 'prompt', 'code'])
        writer.writeheader()
        writer.writerows(passing_rows)

    # Print summary
    print(f"\n{'='*70}")
    print("Filtering Complete")
    print(f"{'='*70}")
    print(f"Input examples: {len(rows)}")
    print(f"PASSED (kept): {len(passing_rows)} ({100*len(passing_rows)/len(rows):.1f}%)")
    print(f"FAILED (removed): {len(failing_rows)} ({100*len(failing_rows)/len(rows):.1f}%)")
    print(f"\nFiltered data saved to: {output_file}")
    print(f"{'='*70}\n")

    # Update the training script to use filtered data
    print("Next step: Update p2_train_mbpp.py to use the filtered data:")
    print(f"  Change: mbpp_train_fixed.csv → mbpp_train_filtered.csv")


if __name__ == "__main__":
    main()
