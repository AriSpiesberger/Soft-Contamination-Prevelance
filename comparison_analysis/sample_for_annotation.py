"""
Sample rows from merged benchmark CSVs for annotation.

Sampling modes:
1. --top-k N: Simply take top N matches by score per test_id (deterministic, recommended)
2. --top-percentile P --max-per-test K: Filter to top P% by weight, then weighted sample K

Usage:
    # Recommended: Top 10 matches per test (deterministic)
    python sample_for_annotation.py --benchmark mbpp --top-k 10
    python sample_for_annotation.py --benchmark zebralogic --top-k 10
    
    # Legacy: Weighted sampling from top percentile
    python sample_for_annotation.py --benchmark mbpp --max-per-test 20 --top-percentile 20
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
    "zebralogic": DATA_DIR / "merged_zebralogic_all_top1000_matches.csv",
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


def _format_size(bytes_val: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}TB"


def _format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m{secs:.0f}s"


def sample_benchmark_top_k(benchmark: str, top_k: int) -> Path:
    """
    Sample top K matches by score per test_id (deterministic).
    
    This is the recommended sampling method - simple, reproducible, and
    focuses annotation effort on the highest-similarity matches.
    
    Uses a memory-efficient streaming approach with heaps - only keeps
    top_k items per test_id in memory at any time.
    
    Args:
        benchmark: Which benchmark to sample
        top_k: Number of top matches to take per test_id
    
    Returns path to the output sampled CSV.
    """
    import heapq
    import sys
    import time
    
    input_path = MERGED_FILES[benchmark]
    output_path = DATA_DIR / f"sampled_{benchmark}_for_annotation.csv"
    
    # Get file size for progress
    file_size = input_path.stat().st_size
    
    print(f"Reading {input_path.name} ({_format_size(file_size)})...")
    print(f"Sampling: top {top_k} by score per test_id\n")
    
    # Use min-heaps to efficiently track top-k per test_id
    # Heap contains (score, counter, row_dict) tuples - counter breaks ties for dict comparison
    top_k_by_test: dict[str, list] = defaultdict(list)
    fieldnames = None
    total_rows = 0
    row_counter = 0  # Unique counter to break ties in heap comparison
    
    start_time = time.time()
    last_print_time = start_time
    bar_width = 40
    
    # Estimate total rows from file size (rough: ~75 bytes per row average)
    est_total_rows = file_size // 75
    
    with open(input_path, "r", encoding="utf-8", newline="", buffering=8*1024*1024) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            total_rows += 1
            row_counter += 1
            test_id = row["test_id"]
            score = float(row["score"])
            
            heap = top_k_by_test[test_id]
            
            if len(heap) < top_k:
                # Haven't filled top-k yet, just add
                heapq.heappush(heap, (score, row_counter, row))
            elif score > heap[0][0]:
                # Score is better than current minimum in top-k
                heapq.heapreplace(heap, (score, row_counter, row))
            # else: score is worse than all top-k, skip
            
            # Progress update every 500k rows
            if total_rows % 500_000 == 0:
                now = time.time()
                
                # Estimate progress from row count
                pct = min(0.99, total_rows / est_total_rows) if est_total_rows > 0 else 0
                
                elapsed = now - start_time
                if pct > 0:
                    eta = (elapsed / pct) - elapsed
                    eta_str = _format_time(eta)
                else:
                    eta_str = "?"
                
                # Progress bar
                filled = int(bar_width * pct)
                bar = "#" * filled + "-" * (bar_width - filled)
                
                rate = total_rows / elapsed if elapsed > 0 else 0
                bytes_est = int(pct * file_size)
                
                line = f"\r[{bar}] {pct*100:5.1f}% | ~{_format_size(bytes_est)}/{_format_size(file_size)} | {total_rows/1e6:.1f}M rows | {rate/1e6:.2f}M/s | ETA: {eta_str}  "
                sys.stdout.write(line)
                sys.stdout.flush()
    
    elapsed = time.time() - start_time
    # Final progress line
    bar = "#" * bar_width
    print(f"\r[{bar}] 100.0% | {_format_size(file_size)} | {total_rows/1e6:.1f}M rows | Done in {_format_time(elapsed)}                    ")
    
    # Extract results from heaps
    print("Extracting top-k results...")
    sampled_rows = []
    
    for test_id, heap in top_k_by_test.items():
        # Sort by score descending (heap is min-heap, so reverse)
        sorted_items = sorted(heap, key=lambda x: x[0], reverse=True)
        for score, counter, row in sorted_items:
            out_row = {"weight": score ** 2}  # Keep weight for compatibility
            for col in fieldnames:
                out_row[col] = row[col]
            sampled_rows.append(out_row)
    
    stats = {
        "total_test_ids": len(top_k_by_test),
        "total_rows_input": total_rows,
        "total_sampled": len(sampled_rows),
        "test_ids_with_samples": len(top_k_by_test),
    }
    
    # Output fieldnames: add weight column for compatibility
    out_fieldnames = ["weight"] + list(fieldnames)
    
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
        "top_k": top_k,
        "sampling_method": "top_k_by_score",
        "weight_formula": "score^2 (for compatibility)",
        "stats": stats,
    }
    
    meta_path = DATA_DIR / f"sampled_{benchmark}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {meta_path}")
    print(f"\nSampling stats:")
    print(f"  Total test_ids: {stats['total_test_ids']}")
    print(f"  Total input rows: {stats['total_rows_input']:,}")
    print(f"  Test_ids with samples: {stats['test_ids_with_samples']}")
    print(f"  Total sampled rows: {stats['total_sampled']:,}")
    
    return output_path


def sample_benchmark_weighted(benchmark: str, max_per_test: int, top_percentile: int = None, seed: int = 42) -> Path:
    """
    Sample rows using weighted random sampling (legacy method).
    
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
        choices=["mbpp", "codeforces", "zebralogic"],
        required=True,
        help="Which benchmark to sample",
    )
    
    # Sampling mode: either --top-k OR (--max-per-test + optional --top-percentile)
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Take top K matches by score per test_id (recommended, deterministic)",
    )
    parser.add_argument(
        "--max-per-test",
        type=int,
        default=20,
        help="Maximum rows to sample per test_id for weighted sampling (default: 20)",
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
    
<<<<<<< HEAD
    # Choose sampling method
    if args.top_k is not None:
        # Recommended: deterministic top-k sampling
        output_path = sample_benchmark_top_k(
            benchmark=args.benchmark,
            top_k=args.top_k,
        )
    else:
        # Legacy: weighted random sampling
        output_path = sample_benchmark_weighted(
            benchmark=args.benchmark,
            max_per_test=args.max_per_test,
            top_percentile=args.top_percentile,
            seed=args.seed,
        )
=======
    output_path = sample_benchmark(
        benchmark=args.benchmark,
        max_per_test=args.max_per_test,
        top_percentile=args.top_percentile,
        seed=args.seed,
    )
>>>>>>> efd47b7aced91b17b8e2466090781f42adee4f1c
    
    print(f"\nDone! Sampled CSV: {output_path}")


if __name__ == "__main__":
    main()
