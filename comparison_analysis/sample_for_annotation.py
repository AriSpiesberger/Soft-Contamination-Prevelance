"""
Sample rows from merged benchmark CSVs for annotation.

For each test_id:
1. Compute weight = score^2
2. Filter to rows in top N percentile by weight for that test_id
3. Sample up to K rows using weighted random sampling by weight
4. Output a CSV ready for annotation

Usage:
    python sample_for_annotation.py --benchmark mbpp --max-per-test 20 --top-percentile 20
    python sample_for_annotation.py --benchmark codeforces --max-per-test 20 --top-percentile 20
"""

import argparse
import csv
import random
from pathlib import Path
from collections import defaultdict
import json

# Paths
DATA_DIR = Path(__file__).parent / "data"
MERGED_FILES = {
    "mbpp": DATA_DIR / "merged_mbpp_all_top1000_matches.csv",
    "codeforces": DATA_DIR / "merged_codeforces_all_top1000_matches.csv",
}


def weighted_sample(items: list, weights: list, k: int) -> list:
    """Sample k items using weights, without replacement."""
    if k >= len(items):
        return items
    
    # Use random.choices with weights, but we need without replacement
    # So we'll do iterative sampling
    remaining_items = list(items)
    remaining_weights = list(weights)
    sampled = []
    
    for _ in range(k):
        if not remaining_items:
            break
        # Normalize weights
        total = sum(remaining_weights)
        if total == 0:
            break
        probs = [w / total for w in remaining_weights]
        
        # Sample one
        idx = random.choices(range(len(remaining_items)), weights=probs, k=1)[0]
        sampled.append(remaining_items[idx])
        
        # Remove from remaining
        remaining_items.pop(idx)
        remaining_weights.pop(idx)
    
    return sampled


def sample_benchmark(benchmark: str, max_per_test: int, top_percentile: int = None, seed: int = 42) -> Path:
    """
    Sample rows from a merged benchmark CSV.
    
    Args:
        benchmark: Which benchmark to sample
        max_per_test: Maximum rows to sample per test_id
        top_percentile: Filter to top N percentile by weight (e.g., 20 = top 20%)
                       If None, uses legacy "above mean" filter
        seed: Random seed
    
    Returns path to the output sampled CSV.
    """
    random.seed(seed)
    
    input_path = MERGED_FILES[benchmark]
    
    # Output filename includes percentile if specified
    if top_percentile:
        suffix = f"_top{top_percentile}pc"
        filter_desc = f"top {top_percentile}% by weight per test_id"
    else:
        suffix = ""
        filter_desc = "weight > mean(weight) per test_id"
    
    output_path = DATA_DIR / f"sampled_{benchmark}_for_annotation{suffix}.csv"
    
    print(f"Reading {input_path}...")
    
    # Group rows by test_id
    rows_by_test = defaultdict(list)
    fieldnames = None
    
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            test_id = row["test_id"]
            rows_by_test[test_id].append(row)
    
    print(f"Found {len(rows_by_test)} unique test_ids")
    print(f"Filter: {filter_desc}")
    
    # Output fieldnames: add weight column
    out_fieldnames = ["weight"] + list(fieldnames)
    
    sampled_rows = []
    stats = {
        "total_test_ids": len(rows_by_test),
        "total_rows_before_filter": 0,
        "total_rows_after_filter": 0,
        "total_sampled": 0,
        "test_ids_with_samples": 0,
    }
    
    for test_id, rows in rows_by_test.items():
        stats["total_rows_before_filter"] += len(rows)
        
        # Compute weight = score^2
        for row in rows:
            score = float(row["score"])
            row["_weight"] = score ** 2
        
        # Filter based on method
        if top_percentile:
            # Top N percentile filter
            # Sort by weight descending and take top N%
            sorted_rows = sorted(rows, key=lambda r: r["_weight"], reverse=True)
            cutoff_idx = max(1, int(len(sorted_rows) * top_percentile / 100))
            filtered = sorted_rows[:cutoff_idx]
        else:
            # Legacy: above mean filter
            weights = [r["_weight"] for r in rows]
            mean_weight = sum(weights) / len(weights) if weights else 0
            filtered = [r for r in rows if r["_weight"] > mean_weight]
        
        stats["total_rows_after_filter"] += len(filtered)
        
        if not filtered:
            continue
        
        # Sample up to max_per_test
        filtered_weights = [r["_weight"] for r in filtered]
        sampled = weighted_sample(filtered, filtered_weights, max_per_test)
        
        if sampled:
            stats["test_ids_with_samples"] += 1
            stats["total_sampled"] += len(sampled)
            
            for row in sampled:
                out_row = {"weight": row["_weight"]}
                for col in fieldnames:
                    out_row[col] = row[col]
                sampled_rows.append(out_row)
    
    # Write output
    print(f"Writing {len(sampled_rows)} sampled rows to {output_path}...")
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(sampled_rows)
    
    # Save sampling metadata
    metadata = {
        "benchmark": benchmark,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "max_per_test": max_per_test,
        "seed": seed,
        "sampling_method": "weighted_random_without_replacement",
        "weight_formula": "score^2",
        "filter": filter_desc,
        "top_percentile": top_percentile,
        "stats": stats,
    }
    
    meta_path = DATA_DIR / f"sampled_{benchmark}{suffix}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {meta_path}")
    print(f"\nSampling stats:")
    print(f"  Total test_ids: {stats['total_test_ids']}")
    print(f"  Rows before filter: {stats['total_rows_before_filter']:,}")
    print(f"  Rows after filter: {stats['total_rows_after_filter']:,}")
    print(f"  Test_ids with samples: {stats['test_ids_with_samples']}")
    print(f"  Total sampled rows: {stats['total_sampled']:,}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Sample rows for annotation")
    parser.add_argument(
        "--benchmark",
        choices=["mbpp", "codeforces"],
        required=True,
        help="Which benchmark to sample",
    )
    parser.add_argument(
        "--max-per-test",
        type=int,
        default=20,
        help="Maximum rows to sample per test_id (default: 20)",
    )
    parser.add_argument(
        "--top-percentile",
        type=int,
        default=None,
        help="Filter to top N percentile by weight (e.g., 20 = top 20%%). If not set, uses above-mean filter.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    
    output_path = sample_benchmark(
        benchmark=args.benchmark,
        max_per_test=args.max_per_test,
        top_percentile=args.top_percentile,
        seed=args.seed,
    )
    
    print(f"\nDone! Sampled CSV: {output_path}")


if __name__ == "__main__":
    main()
