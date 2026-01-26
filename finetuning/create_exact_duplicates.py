"""
Create an exact duplicates dataset from MBPP training data.

Takes ONE canonical example per task (pair_num=1) and duplicates it 5 times.
This creates a training set of the same approximate size as the semantic duplicates
but with exact repetitions instead of paraphrases.
"""
import csv
from pathlib import Path

pwd = Path(__file__).parent
INPUT_CSV = pwd.parent / "mbpp_train_filtered.csv"
OUTPUT_CSV = pwd / "mbpp_data" / "mbpp_train_exact_5x.csv"


def create_exact_duplicates(input_path: str, output_path: str, num_copies: int = 5):
    """Create dataset with exact duplicates."""

    # Load original data
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    # Get unique tasks - take ONLY pair_num=1 or the first available
    task_to_row = {}
    for row in all_rows:
        task_id = row['task_id']
        pair_num = int(row['pair_num'])

        # Prefer pair_num=1, otherwise take the first one we see
        if task_id not in task_to_row or pair_num == 1:
            task_to_row[task_id] = row

    # Create output with exact duplicates
    output_rows = []
    for task_id, row in sorted(task_to_row.items(), key=lambda x: int(x[0])):
        for copy_num in range(1, num_copies + 1):
            new_row = row.copy()
            new_row['pair_num'] = copy_num  # Use pair_num to indicate copy number
            output_rows.append(new_row)

    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task_id', 'pair_num', 'prompt', 'code'])
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Created exact duplicates dataset:")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Unique tasks: {len(task_to_row)}")
    print(f"  Copies per task: {num_copies}")
    print(f"  Total rows: {len(output_rows)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-copies", type=int, default=5, help="Number of exact copies per task")
    parser.add_argument("-i", "--input", type=str, default=INPUT_CSV, help="Input CSV path")
    parser.add_argument("-o", "--output", type=str, default=OUTPUT_CSV, help="Output CSV path")
    args = parser.parse_args()

    create_exact_duplicates(args.input, args.output, args.num_copies)
