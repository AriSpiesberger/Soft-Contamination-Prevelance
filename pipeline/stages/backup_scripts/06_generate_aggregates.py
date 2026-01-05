#!/usr/bin/env python3
"""
Generate ONLY aggregate plots and CSVs - much faster!
Per-test plots can be generated separately later if needed.
"""

import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import gc
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import time


def load_benchmark(benchmark_name: str, mode: str):
    """Load benchmark data."""
    data = []

    if benchmark_name == 'musr':
        ds = load_dataset("TAUR-Lab/MuSR")
        split = list(ds.keys())[0]
        for item in ds[split]:
            task_id = str(item.get('task_id', f"musr_{len(data)}"))
            narrative = item.get('narrative', item.get('question', ''))
            answer = item.get('answer', '')
            data.append({'id': task_id, 'input': narrative, 'output': answer})

    elif benchmark_name == 'mbpp':
        try:
            ds = load_dataset("evalplus/mbpp", "mbpp")
        except:
            ds = load_dataset("google-research-datasets/mbpp", "sanitized")
        for item in ds['test']:
            task_id = str(item.get('task_id', f"mbpp_{len(data)}"))
            prompt = item.get('prompt', item.get('text', ''))
            solution = item.get('canonical_solution', item.get('code', ''))
            data.append({'id': task_id, 'input': prompt, 'output': solution})

    texts, ids = [], []
    for item in data:
        if benchmark_name == 'musr':
            texts.append(f"{item['input']}\n\n{item['output']}")
        elif benchmark_name == 'mbpp':
            if mode == 'input':
                texts.append(item['input'])
            elif mode == 'output':
                texts.append(item['output'])
            else:
                texts.append(f"{item['input']}\n\n{item['output']}")
        ids.append(item['id'])

    return texts, ids


def process_single_test_minimal(test_data, output_dir, world_size, num_bins=1000, hist_range=(-1.0, 1.0)):
    """
    Process a single test: load ALL similarities and compute histogram.
    """
    global_idx = test_data['global_idx']
    benchmark = test_data['benchmark']
    mode = test_data['mode']

    # Load all chunk files
    chunk_files = []
    for r in range(world_size):
        chunk_dir = output_dir / "temp_similarities" / f"rank_{r}" / f"test_{global_idx}"
        if chunk_dir.exists():
            chunk_files.extend(sorted(chunk_dir.glob("chunk_*.npy")))

    if not chunk_files:
        return None

    # Load chunks in parallel
    with ThreadPoolExecutor(max_workers=16) as executor:
        chunks = [np.load(f, mmap_mode='r') for f in chunk_files]

    all_similarities = np.concatenate(chunks)

    # Compute histogram using ALL similarities (no sampling!)
    hist_counts, _ = np.histogram(all_similarities, bins=num_bins, range=hist_range)

    result = {
        'test_id': test_data['test_id'],
        'benchmark': benchmark,
        'mode': mode,
        'hist_counts': hist_counts.tolist(),
        'num_similarities': len(all_similarities)
    }

    del chunks, all_similarities
    gc.collect()

    return result


def worker_process(rank, world_size, all_test_data, output_dir, args):
    """Worker to process subset of test points - minimal processing."""
    my_tests = [t for i, t in enumerate(all_test_data) if i % world_size == rank]

    print(f"[R{rank}] Processing {len(my_tests)} test points...")

    results = []
    for test_data in tqdm(my_tests, desc=f"[R{rank}]", position=rank):
        result = process_single_test_minimal(test_data, output_dir, args.source_world_size)
        if result:
            results.append(result)

    # Save worker results
    output_file = output_dir / f"agg_rank_{rank}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f)

    print(f"[R{rank}] ✅ Done!")


def generate_aggregates(output_dir, world_size, num_bins=1000, hist_range=(-1.0, 1.0)):
    """Generate aggregate plots by summing histogram counts from workers."""
    print("\n" + "="*80)
    print("Generating aggregate outputs (FULL DATA - NO SAMPLING)...")
    print("="*80)

    # Load all worker results
    all_results = []
    for rank in range(world_size):
        file = output_dir / f"agg_rank_{rank}.json"
        if file.exists():
            with open(file, 'r') as f:
                all_results.extend(json.load(f))

    print(f"Loaded {len(all_results)} test results with histogram data")

    # Also load top-100 data from merged files
    merged_results = []
    for rank in range(8):
        file = output_dir / f"merged_rank_{rank}.json"
        if file.exists():
            with open(file, 'r') as f:
                merged_results.extend(json.load(f))
    top100_map = {r['test_id']: r for r in merged_results}

    # Group results by benchmark/mode
    benchmark_groups = defaultdict(list)
    for result in all_results:
        key = (result['benchmark'], result['mode'])
        benchmark_groups[key].append(result)

    # Create bin edges for histogram
    bin_edges = np.linspace(hist_range[0], hist_range[1], num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for (benchmark, mode), results in benchmark_groups.items():
        print(f"\n{'='*60}")
        print(f"Processing {benchmark.upper()} - {mode.upper()}")
        print(f"Aggregating histograms from {len(results)} test points...")
        print('='*60)

        mode_dir = output_dir / f"{benchmark}_{mode}"
        mode_dir.mkdir(parents=True, exist_ok=True)

        # Aggregate histogram counts
        aggregate_hist = np.zeros(num_bins, dtype=np.int64)
        total_similarities = 0
        all_top_scores = []

        for result in tqdm(results, desc=f"Aggregating"):
            # Sum histogram counts
            hist_counts = np.array(result['hist_counts'], dtype=np.int64)
            aggregate_hist += hist_counts
            total_similarities += result['num_similarities']

            # Get top scores from merged data
            test_id = result['test_id']
            if test_id in top100_map:
                all_top_scores.extend(top100_map[test_id]['top_100_scores'][:10])

        print(f"Total similarities: {total_similarities:,}")

        # Generate aggregate plots from full histogram
        if aggregate_hist.sum() > 0:
            max_score = max(all_top_scores) if all_top_scores else 0

            # Plot 1: Histogram - Linear scale
            plt.figure(figsize=(14, 8))
            plt.bar(bin_centers, aggregate_hist, width=(bin_edges[1] - bin_edges[0]),
                   alpha=0.7, edgecolor='black', linewidth=0.5)
            plt.axvline(max_score, color='r', linestyle='--', linewidth=2, label=f'Max: {max_score:.4f}')
            plt.xlabel('Cosine Similarity', fontsize=13)
            plt.ylabel('Frequency', fontsize=13)
            plt.title(f'{benchmark.upper()} {mode.upper()} - Full Aggregate Distribution (Linear)\n{total_similarities:,} similarities from {len(results)} test points', fontsize=15, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(mode_dir / "aggregate_histogram_linear.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ Histogram (linear)")

            # Plot 2: Histogram - Log scale
            plt.figure(figsize=(14, 8))
            # Filter out zero bins for log scale
            nonzero_mask = aggregate_hist > 0
            plt.bar(bin_centers[nonzero_mask], aggregate_hist[nonzero_mask],
                   width=(bin_edges[1] - bin_edges[0]), alpha=0.7, edgecolor='black', linewidth=0.5)
            plt.axvline(max_score, color='r', linestyle='--', linewidth=2, label=f'Max: {max_score:.4f}')
            plt.xlabel('Cosine Similarity', fontsize=13)
            plt.ylabel('Frequency (log scale)', fontsize=13)
            plt.yscale('log')
            plt.title(f'{benchmark.upper()} {mode.upper()} - Full Aggregate Distribution (Log)\n{total_similarities:,} similarities from {len(results)} test points', fontsize=15, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(mode_dir / "aggregate_histogram_log.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ Histogram (log)")

            # Plot 3: CDF from histogram
            cumsum = np.cumsum(aggregate_hist)
            cdf = cumsum / cumsum[-1]  # Normalize

            plt.figure(figsize=(14, 8))
            plt.plot(bin_centers, cdf, linewidth=2.5, color='blue')
            plt.axvline(max_score, color='r', linestyle='--', linewidth=2, label=f'Max: {max_score:.4f}')
            plt.xlabel('Cosine Similarity', fontsize=13)
            plt.ylabel('Cumulative Probability', fontsize=13)
            plt.title(f'{benchmark.upper()} {mode.upper()} - Full Aggregate CDF\n{total_similarities:,} similarities from {len(results)} test points', fontsize=15, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(mode_dir / "aggregate_cdf.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ CDF")

        # Plot 4: Top-k across all tests
        if all_top_scores:
            sorted_top = np.sort(all_top_scores)[::-1][:1000]
            plt.figure(figsize=(14, 8))
            plt.plot(range(1, len(sorted_top) + 1), sorted_top, marker='o', markersize=3, linewidth=1.5)
            plt.xlabel('Rank', fontsize=13)
            plt.ylabel('Cosine Similarity', fontsize=13)
            plt.title(f'{benchmark.upper()} {mode.upper()} - Top Scores Across All Test Points', fontsize=15, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(mode_dir / "aggregate_topk.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ Top-K plot")

    print("\n" + "="*80)
    print("✅ All aggregate plots complete!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='contamination_results')
    parser.add_argument('--source-world-size', type=int, default=8)
    parser.add_argument('--world-size', type=int, default=8)
    parser.add_argument('--benchmarks', nargs='+', default=['musr', 'mbpp'])
    parser.add_argument('--modes', nargs='+', default=['input', 'output'])
    parser.add_argument('--rank', type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Load test data
    print("Loading test metadata...")
    all_test_data = []
    all_test_texts = []

    for benchmark in args.benchmarks:
        modes_to_process = ['input_output'] if benchmark == 'musr' else args.modes
        for mode in modes_to_process:
            test_texts, test_ids = load_benchmark(benchmark, mode)
            for text, test_id in zip(test_texts, test_ids):
                all_test_data.append({
                    'benchmark': benchmark,
                    'mode': mode,
                    'test_id': test_id,
                    'text': text,
                    'global_idx': len(all_test_texts)
                })
                all_test_texts.append(text)

    print(f"Found {len(all_test_data)} test points")

    benchmark_groups = defaultdict(list)
    for test_data in all_test_data:
        key = (test_data['benchmark'], test_data['mode'])
        benchmark_groups[key].append(test_data)

    # Histogram settings
    num_bins = 1000
    hist_range = (-1.0, 1.0)

    if args.rank is not None:
        # Worker mode
        worker_process(args.rank, args.world_size, all_test_data, output_dir, args)
    else:
        # Coordinator mode
        print("\nWaiting for workers...")
        while True:
            complete = sum(1 for r in range(args.world_size)
                          if (output_dir / f"agg_rank_{r}.json").exists())
            if complete == args.world_size:
                break
            print(f"  {complete}/{args.world_size} complete", end='\r')
            time.sleep(2)

        print(f"\n✅ All workers done!")
        generate_aggregates(output_dir, args.world_size, num_bins, hist_range)


if __name__ == "__main__":
    main()
