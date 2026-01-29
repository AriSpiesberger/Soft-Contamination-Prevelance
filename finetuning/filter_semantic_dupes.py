#!/usr/bin/env python3
"""Filter semantic duplicates to only keep code that passes tests."""

import csv
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

PWD = Path(__file__).parent
INPUT_CSV = PWD / "mbpp_data" / "mbpp_train.csv"
OUTPUT_CSV = PWD / "mbpp_data" / "mbpp_train_filtered.csv"


def load_test_cases():
    """Load test cases from MBPP dataset."""
    test_cases = {}
    for split in ['train', 'validation', 'test', 'prompt']:
        try:
            ds = load_dataset('mbpp', split=split)
            for item in ds:
                task_id = item['task_id']
                if task_id not in test_cases:
                    test_cases[task_id] = item['test_list']
        except:
            pass
    return test_cases


def run_code_with_tests(code: str, test_list: list, timeout: float = 5.0) -> bool:
    """Execute code with test assertions."""
    test_script = code + "\n\n" + "\n".join(test_list)

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            temp_path = f.name

        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True, text=True, timeout=timeout
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
    print("Loading test cases...")
    test_cases = load_test_cases()
    print(f"Loaded test cases for {len(test_cases)} tasks")

    print(f"\nReading {INPUT_CSV}...")
    rows = []
    with open(INPUT_CSV, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total rows: {len(rows)}")

    # Filter rows
    passed = []
    failed = 0
    no_tests = 0

    for row in tqdm(rows, desc="Filtering"):
        task_id = int(row['task_id'])
        code = row['code'].replace('\r\n', '\n').replace('\r', '\n').strip()

        if task_id not in test_cases:
            no_tests += 1
            continue

        if run_code_with_tests(code, test_cases[task_id]):
            passed.append(row)
        else:
            failed += 1

    print(f"\nResults:")
    print(f"  Passed: {len(passed)}")
    print(f"  Failed: {failed}")
    print(f"  No tests: {no_tests}")

    # Write filtered CSV
    print(f"\nWriting {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task_id', 'pair_num', 'prompt', 'code'])
        writer.writeheader()
        writer.writerows(passed)

    print(f"Done! Wrote {len(passed)} rows.")


if __name__ == "__main__":
    main()
