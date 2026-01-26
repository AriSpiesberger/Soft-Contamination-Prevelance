"""
Prepare training data aligned with lm-eval-harness MBPP format.

Converts semantic duplicate data to match the evaluation format:
- Uses [BEGIN]/[DONE] delimiters
- Includes test cases in prompt
- Matches the 3-shot evaluation structure

Output format for each example:
    You are an expert Python programmer, and here is your task: {text}
    Your code should pass these tests:

    {test_cases}
    [BEGIN]
    {code}
    [DONE]
"""
import csv
import json
from pathlib import Path
from datasets import load_dataset

pwd = Path(__file__).parent

# Input files
SEMANTIC_DUPES_CSV = pwd.parent / "mbpp_train_filtered.csv"
ORIGINAL_MBPP_CSV = pwd.parent / "mbpp_train.csv"  # Has original prompts too

# Output files
SEMANTIC_ALIGNED_CSV = pwd / "mbpp_data" / "mbpp_train_semantic_aligned.csv"
EXACT_ALIGNED_CSV = pwd / "mbpp_data" / "mbpp_train_exact_5x_aligned.csv"

# lm-eval-harness format
PROMPT_TEMPLATE = """You are an expert Python programmer, and here is your task: {text}
Your code should pass these tests:

{test_cases}
[BEGIN]
"""

# 3-shot examples (same as evaluation)
FEWSHOT_EXAMPLES = [
    {
        "text": "Write a function to find the shared elements from the given two lists.",
        "code": "def similar_elements(test_tup1, test_tup2):\n    res = tuple(set(test_tup1) & set(test_tup2))\n    return res",
        "test_list": [
            "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
            "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
            "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"
        ]
    },
    {
        "text": "Write a python function to identify non-prime numbers.",
        "code": "import math\ndef is_not_prime(n):\n    result = False\n    for i in range(2,int(math.sqrt(n)) + 1):\n        if n % i == 0:\n            result = True\n    return result",
        "test_list": [
            "assert is_not_prime(2) == False",
            "assert is_not_prime(10) == True",
            "assert is_not_prime(35) == True"
        ]
    },
    {
        "text": "Write a function to find the n largest integers from a given list of numbers, returned in descending order.",
        "code": "import heapq as hq\ndef heap_queue_largest(nums,n):\n    largest_nums = hq.nlargest(n, nums)\n    return largest_nums",
        "test_list": [
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]"
        ]
    }
]


def load_test_cases_from_hf() -> dict:
    """Load test cases from HuggingFace MBPP dataset."""
    print("Loading test cases from HuggingFace MBPP dataset...")
    test_cases = {}

    for split in ['test', 'train', 'validation', 'prompt']:
        try:
            ds = load_dataset('mbpp', split=split, trust_remote_code=True)
            for item in ds:
                task_id = item['task_id']
                if task_id not in test_cases:
                    test_cases[task_id] = {
                        'test_list': item.get('test_list', []),
                        'text': item.get('text', ''),
                        'code': item.get('code', ''),
                    }
        except Exception as e:
            print(f"  Warning: Could not load split '{split}': {e}")

    print(f"  Loaded test cases for {len(test_cases)} tasks")
    return test_cases


def build_fewshot_prefix() -> str:
    """Build the 3-shot prefix matching evaluation."""
    prefix = ""
    for ex in FEWSHOT_EXAMPLES:
        test_cases_str = "\n".join(ex["test_list"])
        prefix += PROMPT_TEMPLATE.format(text=ex["text"], test_cases=test_cases_str)
        prefix += ex["code"] + "\n[DONE]\n\n"
    return prefix


def extract_task_description(prompt: str) -> str:
    """Extract task description from training prompt format.

    Training prompts look like:
        def func_name(args): Task description here.

    We want just: Task description here.
    """
    # Find the colon after the function signature
    if ': ' in prompt:
        # Split on first ': ' after def
        parts = prompt.split(': ', 1)
        if len(parts) > 1:
            return parts[1].strip()
    return prompt


def prepare_semantic_duplicates(test_cases: dict, output_path: str):
    """Prepare semantic duplicates with aligned format."""
    print(f"\nPreparing semantic duplicates...")

    fewshot_prefix = build_fewshot_prefix()

    with open(SEMANTIC_DUPES_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    output_rows = []
    skipped = 0

    for row in rows:
        task_id = int(row['task_id'])

        # Skip fewshot example tasks (2, 3, 4)
        if task_id in [2, 3, 4]:
            skipped += 1
            continue

        if task_id not in test_cases:
            skipped += 1
            continue

        tc = test_cases[task_id]
        test_list = tc['test_list']

        if not test_list:
            skipped += 1
            continue

        # Extract task description from the paraphrased prompt
        task_text = extract_task_description(row['prompt'])

        # Build the full prompt with fewshot examples
        test_cases_str = "\n".join(test_list)
        current_prompt = PROMPT_TEMPLATE.format(text=task_text, test_cases=test_cases_str)
        full_prompt = fewshot_prefix + current_prompt

        # Clean up code - remove excessive comments for cleaner output
        code = row['code'].strip()
        completion = code + "\n[DONE]"

        output_rows.append({
            'task_id': row['task_id'],
            'pair_num': row['pair_num'],
            'prompt': full_prompt,
            'completion': completion,
            'text': task_text,  # Keep original for reference
        })

    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task_id', 'pair_num', 'prompt', 'completion', 'text'])
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"  Input rows: {len(rows)}")
    print(f"  Output rows: {len(output_rows)}")
    print(f"  Skipped: {skipped} (missing test cases or fewshot tasks)")
    print(f"  Saved to: {output_path}")

    return output_rows


def prepare_exact_duplicates(test_cases: dict, output_path: str, num_copies: int = 5,
                             restrict_to_tasks: set = None):
    """Prepare exact duplicates with aligned format.

    Uses ORIGINAL MBPP prompts and code (not paraphrased) duplicated exactly.

    Args:
        restrict_to_tasks: If provided, only include these task IDs (for fair comparison)
    """
    print(f"\nPreparing exact duplicates ({num_copies}x)...")
    if restrict_to_tasks:
        print(f"  Restricting to {len(restrict_to_tasks)} tasks from semantic duplicates")

    fewshot_prefix = build_fewshot_prefix()

    output_rows = []
    skipped = 0

    # Use original MBPP data from test_cases (from HuggingFace)
    for task_id, tc in sorted(test_cases.items(), key=lambda x: x[0]):
        # Restrict to specified tasks if provided
        if restrict_to_tasks and task_id not in restrict_to_tasks:
            continue
        # Skip fewshot example tasks
        if task_id in [2, 3, 4]:
            skipped += 1
            continue

        test_list = tc['test_list']
        original_text = tc['text']
        original_code = tc['code']

        if not test_list or not original_text or not original_code:
            skipped += 1
            continue

        # Build the full prompt
        test_cases_str = "\n".join(test_list)
        current_prompt = PROMPT_TEMPLATE.format(text=original_text, test_cases=test_cases_str)
        full_prompt = fewshot_prefix + current_prompt

        completion = original_code.strip() + "\n[DONE]"

        # Create exact duplicates
        for copy_num in range(1, num_copies + 1):
            output_rows.append({
                'task_id': task_id,
                'pair_num': copy_num,
                'prompt': full_prompt,
                'completion': completion,
                'text': original_text,
            })

    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task_id', 'pair_num', 'prompt', 'completion', 'text'])
        writer.writeheader()
        writer.writerows(output_rows)

    unique_tasks = len(set(r['task_id'] for r in output_rows))
    print(f"  Unique tasks: {unique_tasks}")
    print(f"  Copies per task: {num_copies}")
    print(f"  Total rows: {len(output_rows)}")
    print(f"  Skipped: {skipped}")
    print(f"  Saved to: {output_path}")

    return output_rows


def get_semantic_task_ids() -> set:
    """Get task IDs from the semantic duplicates training set."""
    task_ids = set()
    with open(SEMANTIC_DUPES_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_ids.add(int(row['task_id']))
    return task_ids


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-only", action="store_true")
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument("--num-copies", type=int, default=5)
    parser.add_argument("--full-mbpp", action="store_true",
                        help="Use all MBPP tasks for exact duplicates (not just training tasks)")
    args = parser.parse_args()

    # Load test cases
    test_cases = load_test_cases_from_hf()

    # Get task IDs from semantic duplicates for fair comparison
    semantic_task_ids = get_semantic_task_ids()
    print(f"Semantic duplicates training set has {len(semantic_task_ids)} unique tasks")

    if not args.exact_only:
        prepare_semantic_duplicates(test_cases, SEMANTIC_ALIGNED_CSV)

    if not args.semantic_only:
        # Restrict exact duplicates to same tasks as semantic (unless --full-mbpp)
        restrict_tasks = None if args.full_mbpp else semantic_task_ids
        prepare_exact_duplicates(test_cases, EXACT_ALIGNED_CSV, args.num_copies,
                                restrict_to_tasks=restrict_tasks)

    print("\nDone!")


if __name__ == "__main__":
    main()
