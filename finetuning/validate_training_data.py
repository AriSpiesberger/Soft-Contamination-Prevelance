#!/usr/bin/env python3
"""
Validate that semantic duplicate implementations in training data pass test cases.
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


def run_code_with_tests(code: str, test_list: list, timeout: float = 5.0) -> dict:
    """Execute code with test assertions."""
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

        if result.returncode == 0:
            return {'passed': True, 'error': None}
        else:
            return {'passed': False, 'error': result.stderr[:200]}

    except subprocess.TimeoutExpired:
        try:
            os.unlink(temp_path)
        except:
            pass
        return {'passed': False, 'error': 'Timeout'}

    except Exception as e:
        try:
            os.unlink(temp_path)
        except:
            pass
        return {'passed': False, 'error': str(e)[:200]}


def main():
    train_file = Path("/lambda/nfs/embeddings/SDTD_Main/mbpp_train_fixed.csv")

    # Load test cases
    test_cases = load_test_cases()

    # Load and validate training data
    print(f"\nValidating training data from: {train_file}")

    results = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'no_tests': 0,
        'by_task': {},
    }

    failures = []

    with open(train_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in tqdm(rows, desc="Validating"):
        task_id = int(row['task_id'])
        pair_num = int(row['pair_num'])
        code = row['code']

        results['total'] += 1

        # Get test cases for this task
        if task_id not in test_cases:
            results['no_tests'] += 1
            continue

        # Run tests
        result = run_code_with_tests(code, test_cases[task_id])

        # Track by task
        if task_id not in results['by_task']:
            results['by_task'][task_id] = {'passed': 0, 'failed': 0, 'total': 0}

        results['by_task'][task_id]['total'] += 1

        if result['passed']:
            results['passed'] += 1
            results['by_task'][task_id]['passed'] += 1
        else:
            results['failed'] += 1
            results['by_task'][task_id]['failed'] += 1

            # Save first few failures for inspection
            if len(failures) < 10:
                failures.append({
                    'task_id': task_id,
                    'pair_num': pair_num,
                    'error': result['error'],
                    'code': code[:200],
                })

    # Print summary
    print(f"\n{'='*70}")
    print("Training Data Validation Results")
    print(f"{'='*70}")
    print(f"Total examples: {results['total']}")
    print(f"Examples with test cases: {results['total'] - results['no_tests']}")
    print(f"")
    print(f"PASSED: {results['passed']} ({100*results['passed']/results['total']:.1f}%)")
    print(f"FAILED: {results['failed']} ({100*results['failed']/results['total']:.1f}%)")
    print(f"No tests available: {results['no_tests']}")

    # Check per-task consistency
    inconsistent_tasks = []
    for task_id, stats in results['by_task'].items():
        if stats['failed'] > 0 and stats['passed'] > 0:
            inconsistent_tasks.append((task_id, stats))

    if inconsistent_tasks:
        print(f"\n{'='*70}")
        print(f"Tasks with INCONSISTENT semantic duplicates:")
        print(f"(Some variants pass, some fail - this is BAD for training!)")
        print(f"{'='*70}")
        for task_id, stats in sorted(inconsistent_tasks)[:20]:
            print(f"Task {task_id}: {stats['passed']} passed, {stats['failed']} failed out of {stats['total']}")

    # Show example failures
    if failures:
        print(f"\n{'='*70}")
        print(f"Example Failures (first 5):")
        print(f"{'='*70}")
        for i, fail in enumerate(failures[:5]):
            print(f"\nFailure {i+1}:")
            print(f"  Task ID: {fail['task_id']}, Pair: {fail['pair_num']}")
            print(f"  Code: {fail['code']}...")
            print(f"  Error: {fail['error']}")

    # Summary conclusion
    print(f"\n{'='*70}")
    pass_rate = 100 * results['passed'] / (results['total'] - results['no_tests'])
    if pass_rate < 90:
        print(f"⚠️  WARNING: Only {pass_rate:.1f}% of training examples pass their tests!")
        print(f"    This means the semantic duplicates have bugs.")
        print(f"    The model is learning from INCORRECT code!")
    else:
        print(f"✓  Good: {pass_rate:.1f}% of training examples pass their tests.")

    if inconsistent_tasks:
        print(f"\n⚠️  WARNING: {len(inconsistent_tasks)} tasks have inconsistent duplicates.")
        print(f"    This confuses the model - same task, different correctness.")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
