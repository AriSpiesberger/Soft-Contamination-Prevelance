#!/usr/bin/env python3
"""
Fast Parallel Merger for Contamination Analysis Results
Distributes merge work across 8 workers with parallel I/O
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
import sys
from collections import defaultdict
import time
import matplotlib.pyplot as plt


class StreamingStats:
    """Compute statistics in a streaming fashion without storing all values."""
    def __init__(self, sample_size=100000):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.sample_reservoir = []
        self.sample_size = sample_size

    def update_batch(self, values):
        values = np.asarray(values).flatten().astype(np.float64)
        n_new = len(values)
        if n_new == 0:
            return

        self.min_val = min(self.min_val, float(values.min()))
        self.max_val = max(self.max_val, float(values.max()))

        new_mean = float(values.mean())
        new_var = float(values.var()) if n_new > 1 else 0.0

        if self.n == 0:
            self.mean = new_mean
            self.M2 = new_var * n_new
            self.n = n_new
        else:
            delta = new_mean - self.mean
            total_n = self.n + n_new
            self.mean = self.mean + delta * n_new / total_n
            self.M2 = self.M2 + new_var * n_new + delta * delta * self.n * n_new / total_n
            self.n = total_n

        subsample_rate = max(1, n_new // 1000)
        for x in values[::subsample_rate]:
            if len(self.sample_reservoir) < self.sample_size:
                self.sample_reservoir.append(float(x))
            else:
                j = np.random.randint(0, len(self.sample_reservoir))
                self.sample_reservoir[j] = float(x)

    def get_stats(self):
        variance = self.M2 / self.n if self.n > 1 else 0.0
        std = np.sqrt(variance)
        if self.sample_reservoir:
            sorted_samples = np.sort(self.sample_reservoir)
            p99 = float(np.percentile(sorted_samples, 99))
            p95 = float(np.percentile(sorted_samples, 95))
        else:
            p99 = p95 = 0.0

        return {
            'count': int(self.n),
            'mean': float(self.mean),
            'std': float(std),
            'min': float(self.min_val),
            'max': float(self.max_val),
            'p95': float(p95),
            'p99': float(p99)
        }


def load_benchmark(benchmark_name: str, mode: str):
    """Load benchmark data - copied from original script."""
    data = []

    # Handle MuSR splits: musr_murder_mysteries, musr_object_placements, musr_team_allocation
    if benchmark_name.startswith('musr_'):
        split_name = benchmark_name.replace('musr_', '')  # e.g., 'object_placements'
        ds = load_dataset("TAUR-Lab/MuSR")
        if split_name not in ds:
            raise ValueError(f"Unknown MuSR split: {split_name}. Available: {list(ds.keys())}")
        for idx, item in enumerate(ds[split_name]):
            task_id = f"{benchmark_name}_{idx}"
            narrative = item.get('narrative', item.get('question', ''))
            answer = item.get('answer', '')
            data.append({'id': task_id, 'input': narrative, 'output': answer})

    # Legacy 'musr' name - for backwards compatibility
    elif benchmark_name == 'musr':
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
        if benchmark_name == 'musr' or benchmark_name.startswith('musr_'):
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


def load_chunk_file(chunk_file):
    """Load a single chunk file (for parallel I/O). Supports both old .npz and new .npy formats."""
    if str(chunk_file).endswith('.npz'):
        # Old format: npz with similarities and hash_ids
        data = np.load(chunk_file)
        return data['similarities']
    else:
        # New format: raw .npy with just similarities
        return np.load(chunk_file, mmap_mode='r')  # Memory-mapped for speed


def merge_test_point(test_data, output_dir, world_size):
    """
    Merge results for a single test point across all ranks.
    Returns results ready to be saved.
    """
    test_id = test_data['test_id']
    test_text = test_data['text']
    global_idx = test_data['global_idx']

    # Collect all chunk files from all ranks (both old .npz and new .npy formats)
    chunk_files = []
    for r in range(world_size):
        chunk_dir = output_dir / "temp_similarities" / f"rank_{r}" / f"test_{global_idx}"
        if chunk_dir.exists():
            chunk_files.extend(sorted(chunk_dir.glob("chunk_*_sims.npy")))  # New format
            chunk_files.extend(sorted(chunk_dir.glob("chunk_*.npz")))  # Old format (backward compat)

    if not chunk_files:
        return None

    # Load all chunks in parallel using thread pool
    with ThreadPoolExecutor(max_workers=16) as executor:
        chunks = list(executor.map(load_chunk_file, chunk_files))

    # Concatenate all similarities
    all_similarities = np.concatenate(chunks)

    # Compute top-100
    top_k = min(100, len(all_similarities))
    top_indices = np.argpartition(all_similarities, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(all_similarities[top_indices])[::-1]]
    top_scores = all_similarities[top_indices].tolist()  # Convert to list immediately
    top_indices = top_indices.tolist()

    # Free memory immediately - don't accumulate large arrays
    del all_similarities
    del chunks
    gc.collect()

    # Build result (only keep what's needed)
    result = {
        'test_id': test_id,
        'test_text': test_text,
        'top_100_indices': top_indices,
        'top_100_scores': top_scores,
    }

    return result


def worker_process(rank, world_size, all_test_data, output_dir, args):
    """
    Worker process to handle a subset of test points.
    """
    # Assign test points to this worker
    my_test_points = [tp for i, tp in enumerate(all_test_data) if i % world_size == rank]

    print(f"[Rank {rank}] Processing {len(my_test_points)} test points")

    # Process each test point
    results = []
    for test_data in tqdm(my_test_points, desc=f"[R{rank}]", position=rank):
        result = merge_test_point(test_data, output_dir, args.world_size)
        if result:
            results.append(result)

    # Save results for this worker
    output_file = output_dir / f"merged_rank_{rank}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f)

    print(f"[Rank {rank}] ✅ Complete! Saved {len(results)} results to {output_file}")
    return rank, len(results)


def combine_worker_outputs(output_dir, world_size, benchmark_groups, all_test_data):
    """
    Combine outputs from all workers and generate final CSV files + plots.
    """
    print("\n" + "="*80)
    print("Combining worker outputs and generating CSVs + plots...")
    print("="*80)

    # Load all worker results
    all_results = []
    for rank in range(world_size):
        output_file = output_dir / f"merged_rank_{rank}.json"
        if output_file.exists():
            with open(output_file, 'r') as f:
                all_results.extend(json.load(f))

    print(f"Loaded {len(all_results)} total results")

    # Create a mapping from test_id to result
    results_map = {r['test_id']: r for r in all_results}

    # Generate CSV and plots for each benchmark/mode
    for (benchmark, mode), test_points in benchmark_groups.items():
        print(f"\nProcessing {benchmark.upper()} - {mode.upper()}...")

        mode_dir = output_dir / f"{benchmark}_{mode}"
        mode_dir.mkdir(parents=True, exist_ok=True)

        # Track aggregate stats
        agg_stats = StreamingStats()
        all_top_scores = []

        # Build CSV data
        csv_data = []
        for test_data in test_points:
            test_id = test_data['test_id']
            if test_id not in results_map:
                continue

            result = results_map[test_id]

            # Collect top scores for aggregate plot
            top_scores = result['top_100_scores']
            if top_scores:
                all_top_scores.extend(top_scores[:10])  # Top 10 per test

            for i, (idx, score) in enumerate(zip(
                result['top_100_indices'],
                result['top_100_scores']
            )):
                csv_data.append({
                    'test_id': test_id,
                    'rank': i + 1,
                    'score': score,
                    'corpus_idx': idx
                })

        # Save CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_file = mode_dir / "top_100_contamination.csv"
            df.to_csv(csv_file, index=False)
            print(f"  ✅ Saved {len(df)} rows to {csv_file}")

            # Generate aggregate topk plot
            if all_top_scores:
                sorted_top = np.sort(all_top_scores)[::-1][:1000]
                plt.figure(figsize=(12, 8))
                plt.plot(range(1, len(sorted_top) + 1), sorted_top, marker='o', markersize=2, linewidth=1)
                plt.xlabel('Rank')
                plt.ylabel('Cosine Similarity')
                plt.title(f'{benchmark.upper()} {mode.upper()} - Top Scores Across All Test Points')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(mode_dir / "aggregate_topk.png", dpi=150)
                plt.close()
                print(f"  ✅ Saved aggregate_topk.png")

        else:
            print(f"  ⚠️  No data for {benchmark}_{mode}")

    print("\n" + "="*80)
    print("✅ All outputs generated!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Fast parallel merger")
    parser.add_argument('--output-dir', type=str, default='contamination_results')
    parser.add_argument('--world-size', type=int, default=8)
    parser.add_argument('--benchmarks', nargs='+', default=['musr', 'mbpp'])
    parser.add_argument('--modes', nargs='+', default=['input', 'output'])
    parser.add_argument('--rank', type=int, default=None, help='Worker rank (0-7), omit to run coordinator')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Load all test data (same as original script)
    print("Loading benchmark metadata...")
    all_test_data = []
    all_test_texts = []

    for benchmark in args.benchmarks:
        modes_to_process = ['input_output'] if (benchmark == 'musr' or benchmark.startswith('musr_')) else args.modes
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

    num_tests = len(all_test_texts)
    print(f"Found {num_tests} test points")

    # Group by benchmark/mode
    benchmark_groups = defaultdict(list)
    for test_data in all_test_data:
        key = (test_data['benchmark'], test_data['mode'])
        benchmark_groups[key].append(test_data)

    # If rank is specified, run as worker
    if args.rank is not None:
        worker_process(args.rank, args.world_size, all_test_data, output_dir, args)
    else:
        # Coordinator mode - wait for workers, then combine
        print("\n" + "="*80)
        print("Waiting for all merge workers to complete...")
        print("="*80)

        while True:
            complete = 0
            for r in range(args.world_size):
                if (output_dir / f"merged_rank_{r}.json").exists():
                    complete += 1

            if complete == args.world_size:
                break

            print(f"  Complete: {complete}/{args.world_size} workers", end='\r')
            time.sleep(2)

        print(f"\n✅ All {args.world_size} workers complete!")

        # Combine outputs
        combine_worker_outputs(output_dir, args.world_size, benchmark_groups, all_test_data)


if __name__ == "__main__":
    main()
