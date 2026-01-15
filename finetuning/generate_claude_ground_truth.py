"""
Generate Claude ground truth code solutions for MBPP.

Uses Claude API to generate high-quality reference solutions that OLMo
finetuned models will be compared against using similarity metrics.

Usage:
    python generate_claude_ground_truth.py --split eval
    python generate_claude_ground_truth.py --split train --verify-tests
"""

import json
import csv
import argparse
import subprocess
import tempfile
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import anthropic

# Configuration
TEST_TRAIN_HALF = Path(__file__).parent.parent / "mbpp_test_train_half.csv"
TEST_EVAL_HALF = Path(__file__).parent.parent / "mbpp_test_eval_half.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "claude_ground_truth"

SYSTEM_PROMPT = """You are an expert Python programmer. Generate complete, working Python code.

Rules:
1. Output ONLY the Python code - no explanations, no markdown formatting
2. Include all necessary imports at the top
3. Use clear variable names and standard Python idioms
4. The function name must match what the prompt asks for
5. Handle edge cases appropriately"""


def load_test_cases() -> dict:
    """Load test cases from HuggingFace MBPP dataset."""
    from datasets import load_dataset

    test_cases = {}
    print("Loading test cases from HuggingFace MBPP dataset...")

    for split in ['test', 'train', 'validation', 'prompt']:
        try:
            ds = load_dataset('mbpp', split=split)
            for item in ds:
                task_id = item['task_id']
                test_list = item.get('test_list', [])
                if task_id not in test_cases:
                    test_cases[task_id] = test_list
        except Exception as e:
            print(f"Warning: Could not load split '{split}': {e}")

    print(f"Loaded test cases for {len(test_cases)} tasks")
    return test_cases


def load_prompts(csv_path: str, test_cases: dict) -> list:
    """Load prompts from CSV."""
    prompts = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = int(row['task_id'])
            if task_id not in test_cases:
                continue

            prompts.append({
                'task_id': task_id,
                'prompt': row['original_text'],
                'gold_code': row['original_code'],
                'test_list': test_cases[task_id],
            })

    print(f"Loaded {len(prompts)} prompts from {csv_path}")
    return prompts


def run_tests(code: str, test_list: list, timeout: float = 10.0) -> dict:
    """Execute code with test assertions."""
    test_script = code + "\n\n"
    for test in test_list:
        test_script += test + "\n"

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
            return {'passed': False, 'error': result.stderr[:500]}

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
        return {'passed': False, 'error': str(e)[:500]}


def generate_with_claude(
    client: anthropic.Anthropic,
    prompt: str,
    model: str = "claude-opus-4-5-20251101",
    max_retries: int = 3,
) -> str:
    """Generate code using Claude API."""

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": f"{prompt}\n\nGenerate only the Python code."}
                ],
            )

            code = response.content[0].text.strip()

            # Clean markdown if present
            if code.startswith("```python"):
                code = code[9:]
            elif code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]

            return code.strip()

        except anthropic.RateLimitError:
            wait_time = 2 ** attempt
            print(f"Rate limited, waiting {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"API error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

    return None


def generate_ground_truth(
    prompts: list,
    api_key: str,
    model: str = "claude-opus-4-5-20251101",
    verify_tests: bool = True,
    output_path: Path = None,
) -> dict:
    """Generate Claude ground truth for all prompts."""

    client = anthropic.Anthropic(api_key=api_key)

    results = []
    passed = 0
    failed = 0

    pbar = tqdm(prompts, desc="Generating")

    for item in pbar:
        task_id = item['task_id']
        prompt = item['prompt']
        test_list = item['test_list']

        # Generate code
        code = generate_with_claude(client, prompt, model)

        if code is None:
            result = {
                'task_id': task_id,
                'prompt': prompt,
                'test_list': test_list,
                'claude_code': None,
                'gold_code': item['gold_code'],
                'tests_passed': False,
                'error': 'API failure',
            }
            failed += 1
        else:
            # Verify tests if requested
            if verify_tests:
                test_result = run_tests(code, test_list)
                tests_passed = test_result['passed']
                error = test_result['error']
            else:
                tests_passed = None
                error = None

            result = {
                'task_id': task_id,
                'prompt': prompt,
                'test_list': test_list,
                'claude_code': code,
                'gold_code': item['gold_code'],
                'tests_passed': tests_passed,
                'error': error,
            }

            if tests_passed:
                passed += 1
            elif tests_passed is False:
                failed += 1

        results.append(result)

        # Update progress
        if verify_tests:
            pbar.set_description(f"Gen | Pass: {passed}/{passed+failed} ({100*passed/(passed+failed) if (passed+failed) > 0 else 0:.1f}%)")

        # Save incrementally
        if output_path:
            with open(output_path, 'w') as f:
                json.dump({
                    'model': model,
                    'timestamp': datetime.now().isoformat(),
                    'stats': {
                        'total': len(results),
                        'passed': passed,
                        'failed': failed,
                    },
                    'results': results,
                }, f, indent=2)

    return {
        'model': model,
        'stats': {
            'total': len(results),
            'passed': passed,
            'failed': failed,
            'pass_rate': 100 * passed / len(results) if results else 0,
        },
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Claude ground truth for MBPP")
    parser.add_argument("--api-key", type=str, required=True, help="Anthropic API key")
    parser.add_argument("--split", type=str, default="all", choices=["train", "eval", "all"],
                        help="Which split to generate for (default: all)")
    parser.add_argument("--model", type=str, default="claude-opus-4-5-20251101",
                        help="Claude model to use")
    parser.add_argument("--verify-tests", action="store_true", default=True,
                        help="Verify generated code passes tests")
    parser.add_argument("--no-verify", dest="verify_tests", action="store_false",
                        help="Skip test verification")

    args = parser.parse_args()

    # Determine which splits to run
    if args.split == "all":
        splits = ["train", "eval"]
    else:
        splits = [args.split]

    # Load test cases once
    test_cases = load_test_cases()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = {}

    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*60}")

        # Select CSV
        csv_path = TEST_TRAIN_HALF if split == "train" else TEST_EVAL_HALF

        # Load prompts
        prompts = load_prompts(csv_path, test_cases)

        if not prompts:
            print(f"ERROR: No prompts found for {split}!")
            continue

        output_path = OUTPUT_DIR / f"claude_ground_truth_{split}_{timestamp}.json"

        print(f"\nGenerating Claude ground truth...")
        print(f"  Model: {args.model}")
        print(f"  Split: {split}")
        print(f"  Prompts: {len(prompts)}")
        print(f"  Verify tests: {args.verify_tests}")
        print(f"  Output: {output_path}\n")

        # Generate
        results = generate_ground_truth(
            prompts=prompts,
            api_key=args.api_key,
            model=args.model,
            verify_tests=args.verify_tests,
            output_path=output_path,
        )

        all_results[split] = results

        # Print summary
        print(f"\n{'-'*60}")
        print(f"{split.upper()} Split Complete")
        print(f"{'-'*60}")
        print(f"  Total: {results['stats']['total']}")
        if args.verify_tests:
            print(f"  Passed tests: {results['stats']['passed']}")
            print(f"  Failed tests: {results['stats']['failed']}")
            print(f"  Pass rate: {results['stats']['pass_rate']:.1f}%")
        print(f"  Output: {output_path}")

    # Final summary
    print(f"\n{'='*60}")
    print("Claude Ground Truth Generation Complete")
    print(f"{'='*60}")
    for split, results in all_results.items():
        print(f"  {split}: {results['stats']['total']} tasks, {results['stats']['pass_rate']:.1f}% pass rate")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
