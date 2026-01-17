#!/usr/bin/env python3
"""
Create train and test sets for MBPP semantic duplicates experiment.

Train: Semantic duplicates (english_synonym_input -> python_semantic_output)
       for first half of task_ids

Test:  Original MBPP prompts (original_text) for both halves
       - test_train_half.csv: original prompts for train task_ids (contamination test)
       - test_eval_half.csv: original prompts for eval task_ids (generalization test)

Output:
- mbpp_train.csv: semantic pairs for training (first half task_ids)
- mbpp_test_train_half.csv: original prompts for train task_ids
- mbpp_test_eval_half.csv: original prompts for held-out task_ids
"""

import csv
from pathlib import Path

INPUT_CSV = Path(__file__).parent / "semantic_pairs_full.csv"
TRAIN_CSV = Path(__file__).parent / "mbpp_train.csv"
TEST_TRAIN_HALF = Path(__file__).parent / "mbpp_test_train_half.csv"
TEST_EVAL_HALF = Path(__file__).parent / "mbpp_test_eval_half.csv"


def main():
    # Read all rows
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Loaded {len(rows)} rows from {INPUT_CSV}")
    
    # Get unique task_ids and sort numerically
    task_ids = sorted(set(int(row['task_id']) for row in rows))
    print(f"Found {len(task_ids)} unique task_ids")
    print(f"Task ID range: {min(task_ids)} - {max(task_ids)}")
    
    # Split at median
    median_idx = len(task_ids) // 2
    train_task_ids = set(task_ids[:median_idx])
    eval_task_ids = set(task_ids[median_idx:])
    
    split_task_id = task_ids[median_idx]
    print(f"\nSplit at task_id {split_task_id}:")
    print(f"  Train half: {len(train_task_ids)} task_ids (< {split_task_id})")
    print(f"  Eval half:  {len(eval_task_ids)} task_ids (>= {split_task_id})")
    
    # === TRAIN CSV: semantic pairs for first half ===
    train_rows = []
    for row in rows:
        if int(row['task_id']) in train_task_ids:
            train_rows.append({
                'task_id': row['task_id'],
                'pair_num': row['pair_num'],
                'prompt': row['english_synonym_input'],  # Paraphrased prompt
                'code': row['python_semantic_output'],   # Alternative implementation
            })
    
    with open(TRAIN_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['task_id', 'pair_num', 'prompt', 'code'])
        writer.writeheader()
        writer.writerows(train_rows)
    print(f"\nTrain set: {len(train_rows)} semantic pairs -> {TRAIN_CSV}")
    
    # === TEST CSVs: original prompts for each half ===
    # Get unique original prompts per task_id (they're the same across all 5 pairs)
    task_originals = {}
    for row in rows:
        tid = int(row['task_id'])
        if tid not in task_originals:
            task_originals[tid] = {
                'task_id': row['task_id'],
                'original_text': row['original_text'],
                'original_code': row['original_code'],
            }
    
    # Test set for train half (contamination test)
    test_train_rows = [task_originals[tid] for tid in sorted(train_task_ids)]
    with open(TEST_TRAIN_HALF, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['task_id', 'original_text', 'original_code'])
        writer.writeheader()
        writer.writerows(test_train_rows)
    print(f"Test (train half): {len(test_train_rows)} original prompts -> {TEST_TRAIN_HALF}")
    
    # Test set for eval half (generalization test)
    test_eval_rows = [task_originals[tid] for tid in sorted(eval_task_ids)]
    with open(TEST_EVAL_HALF, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['task_id', 'original_text', 'original_code'])
        writer.writeheader()
        writer.writerows(test_eval_rows)
    print(f"Test (eval half): {len(test_eval_rows)} original prompts -> {TEST_EVAL_HALF}")
    
    print(f"\nDone! Created 3 files.")


if __name__ == "__main__":
    main()
