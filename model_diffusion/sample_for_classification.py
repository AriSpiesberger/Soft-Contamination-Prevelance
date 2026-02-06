"""
Sample 100 items per (test_id, dataset) combination from mbpp_all_datasets_combined.csv
for classification by the fine-tuned model.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def sample_for_classification(
    input_csv: str,
    output_csv: str,
    samples_per_group: int = 100,
    random_state: int = 42
):
    """
    Sample items for each unique (test_id, dataset) combination.

    Args:
        input_csv: Path to mbpp_all_datasets_combined.csv
        output_csv: Path to save sampled data
        samples_per_group: Number of samples per (test_id, dataset) group
        random_state: Random seed for reproducibility
    """
    print(f"Loading {input_csv}...")
    print("(This may take a while for large files)")

    # Load the full dataset
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df):,} rows")

    # Get unique combinations
    unique_combos = df.groupby(['test_id', 'dataset']).size().reset_index(name='count')
    print(f"\nUnique (test_id, dataset) combinations: {len(unique_combos)}")
    print(f"Datasets: {df['dataset'].unique().tolist()}")
    print(f"Unique test_ids: {df['test_id'].nunique()}")

    # Sample from each group
    print(f"\nSampling {samples_per_group} items per group...")

    sampled_dfs = []
    np.random.seed(random_state)

    for (test_id, dataset), group in df.groupby(['test_id', 'dataset']):
        if len(group) <= samples_per_group:
            # Take all if fewer than requested
            sampled_dfs.append(group)
        else:
            # Sample randomly
            sampled = group.sample(n=samples_per_group, random_state=random_state)
            sampled_dfs.append(sampled)

    # Combine all samples
    result_df = pd.concat(sampled_dfs, ignore_index=True)

    print(f"\nSampled {len(result_df):,} total rows")
    print(f"\nRows per dataset:")
    print(result_df['dataset'].value_counts())

    print(f"\nRows per test_id (sample):")
    test_counts = result_df.groupby('test_id').size()
    print(f"  Min: {test_counts.min()}, Max: {test_counts.max()}, Mean: {test_counts.mean():.1f}")

    # Save
    result_df.to_csv(output_csv, index=False)
    print(f"\nSaved to {output_csv}")

    return result_df


if __name__ == "__main__":
    input_csv = str(Path(__file__).parent / "data" / "mbpp_all_datasets_combined.csv")
    output_csv = str(Path(__file__).parent.parent / "comparison_analysis" / "training_data" / "mbpp_to_classify_sampled.csv")

    sample_for_classification(
        input_csv=input_csv,
        output_csv=output_csv,
        samples_per_group=100,
        random_state=42
    )
