"""
Generate semantic duplicates: varied prompts + same code.

Prepares prompts for LLM to paraphrase task descriptions.
Code stays the same for all paraphrases of each task.
"""
import csv
import json
from pathlib import Path
from datasets import load_dataset

pwd = Path(__file__).parent.parent
OUTPUT_PROMPTS = pwd / "mbpp_data" / "paraphrase_prompts.jsonl"
OUTPUT_TEMPLATE = pwd / "mbpp_data" / "mbpp_semantic_template.csv"

SYSTEM_PROMPT = """You are a helpful assistant that paraphrases programming task descriptions.
Given a task description, generate 5 different ways to describe the same task.
Each paraphrase should:
- Request the exact same functionality
- Use different wording/phrasing
- Be concise (1-2 sentences)
- Maintain technical accuracy

Return ONLY the 5 paraphrases, one per line, numbered 1-5. No other text."""

USER_TEMPLATE = """Paraphrase this programming task 5 different ways:

TASK: {description}

Generate 5 paraphrases:"""


def load_mbpp_tasks():
    """Load MBPP tasks with descriptions and code."""
    tasks = {}
    for split in ['test', 'train', 'validation', 'prompt']:
        try:
            ds = load_dataset('mbpp', split=split)
            for item in ds:
                tasks[item['task_id']] = {
                    'text': item['text'],
                    'code': item['code'],
                    'test_list': item['test_list'],
                }
        except:
            pass
    return tasks


def prepare_prompts_for_llm(tasks: dict, task_ids: list = None):
    """Prepare prompts for LLM to generate paraphrases."""

    if task_ids is None:
        task_ids = list(tasks.keys())

    prompts = []
    for task_id in task_ids:
        if task_id not in tasks:
            continue

        task = tasks[task_id]
        prompts.append({
            'task_id': task_id,
            'system': SYSTEM_PROMPT,
            'user': USER_TEMPLATE.format(description=task['text']),
            'original_text': task['text'],
            'original_code': task['code'],
        })

    return prompts


def save_prompts(prompts: list, output_path: str):
    """Save prompts to JSONL for batch processing."""
    with open(output_path, 'w') as f:
        for p in prompts:
            f.write(json.dumps(p) + '\n')
    print(f"Saved {len(prompts)} prompts to {output_path}")


def create_template_csv(tasks: dict, task_ids: list, output_path: str):
    """Create a template CSV that just needs paraphrases filled in."""
    rows = []
    for task_id in task_ids:
        if task_id not in tasks:
            continue
        task = tasks[task_id]
        for pair_num in range(1, 6):
            rows.append({
                'task_id': task_id,
                'pair_num': pair_num,
                'prompt': f"PARAPHRASE_{pair_num}_HERE",  # Placeholder
                'code': task['code'],
                'original_text': task['text'],
            })

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task_id', 'pair_num', 'prompt', 'code', 'original_text'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved template with {len(rows)} rows to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-ids", type=str, help="Comma-separated task IDs, or 'all'")
    parser.add_argument("--from-csv", type=str, help="Get task IDs from existing CSV")
    args = parser.parse_args()

    # Load MBPP
    print("Loading MBPP tasks...")
    tasks = load_mbpp_tasks()
    print(f"Loaded {len(tasks)} tasks")

    # Determine which task IDs to process
    if args.from_csv:
        with open(args.from_csv) as f:
            reader = csv.DictReader(f)
            task_ids = list(set(int(r['task_id']) for r in reader))
        print(f"Using {len(task_ids)} task IDs from {args.from_csv}")
    elif args.task_ids and args.task_ids != 'all':
        task_ids = [int(x) for x in args.task_ids.split(',')]
    else:
        task_ids = list(tasks.keys())

    # Prepare and save
    prompts = prepare_prompts_for_llm(tasks, task_ids)
    save_prompts(prompts, OUTPUT_PROMPTS)
    create_template_csv(tasks, task_ids, OUTPUT_TEMPLATE)

    print()
    print("Next steps:")
    print(f"1. Use {OUTPUT_PROMPTS} to generate paraphrases with your LLM")
    print(f"2. Fill in the paraphrases in {OUTPUT_TEMPLATE}")
    print("3. Run prepare_aligned_data.py to format for training")


if __name__ == "__main__":
    main()
