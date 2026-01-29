#!/usr/bin/env python3
"""Verify that original MBPP code passes its own tests."""

import subprocess
import sys
import tempfile
import os
from datasets import load_dataset
from tqdm import tqdm


def run_code_with_tests(code: str, test_list: list, timeout: float = 5.0) -> tuple:
    """Execute code with test assertions. Returns (passed, error_msg)."""
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
        if result.returncode == 0:
            return True, None
        else:
            return False, result.stderr[:200]
    except Exception as e:
        try:
            os.unlink(temp_path)
        except:
            pass
        return False, str(e)[:200]


def main():
    print("Loading MBPP dataset...")
    test_cases = {}
    for split in ['train', 'validation', 'test', 'prompt']:
        try:
            ds = load_dataset('mbpp', split=split)
            for item in ds:
                task_id = item['task_id']
                if task_id not in test_cases:
                    test_cases[task_id] = {
                        'text': item['text'],
                        'code': item['code'],
                        'test_list': item['test_list'],
                    }
        except:
            pass

    print(f"Loaded {len(test_cases)} tasks")

    # Test each one
    passed = 0
    failed = 0
    failed_tasks = []

    for task_id in tqdm(sorted(test_cases.keys()), desc="Testing"):
        tc = test_cases[task_id]
        ok, err = run_code_with_tests(tc['code'], tc['test_list'])
        if ok:
            passed += 1
        else:
            failed += 1
            failed_tasks.append((task_id, err))

    print(f"\nResults:")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if failed_tasks:
        print(f"\nFirst 10 failed tasks:")
        for task_id, err in failed_tasks[:10]:
            print(f"  Task {task_id}: {err[:100]}")


if __name__ == "__main__":
    main()
