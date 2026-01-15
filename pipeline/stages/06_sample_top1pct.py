#!/usr/bin/env python3
"""
Sample Top 1% for Semantic Duplicate Analysis

For each MBPP test point, samples 100 random corpus items from the TOP 1% 
of similarity scores. Output is a CSV ready for semantic_duplicate_analysis.py.

Usage:
    python 06_sample_top1pct.py --config configs/dolci.yaml --output samples_dolci.csv
    
Or for all configs:
    python 06_sample_top1pct.py --all-configs --output-dir ./semantic_samples/
"""

import os
import sys
import gc
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from functools import partial


def load_config(config_path):
    """Load pipeline config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_corpus_text_mapping(corpus_jsonl_path):
    """Build mapping from hash_id (corpus ID) to text."""
    corpus_jsonl_path = Path(corpus_jsonl_path)
    if not corpus_jsonl_path.exists():
        print(f"⚠️  Corpus JSONL not found: {corpus_jsonl_path}")
        return {}

    print(f"Loading corpus texts from {corpus_jsonl_path}...")
    id_to_text = {}
    
    with open(corpus_jsonl_path) as f:
        for line in tqdm(f, desc="Indexing corpus"):
            try:
                data = json.loads(line)
                id_val = str(data.get('id', ''))
                if id_val:
                    id_to_text[id_val] = data.get('text', '')
            except json.JSONDecodeError:
                continue

    print(f"✓ Loaded {len(id_to_text):,} corpus texts")
    return id_to_text


def load_benchmark_test_texts(benchmark='mbpp'):
    """Load benchmark test texts."""
    from datasets import load_dataset

    if benchmark == 'mbpp':
        print("Loading MBPP benchmark...")
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

        test_data = {}
        for item in ds:
            test_id = str(item['task_id'])
            # Combine prompt + test_list for full context
            test_text = item['prompt']
            if item.get('test_list'):
                test_text += "\n\nTest cases:\n" + "\n".join(item['test_list'][:3])
            test_data[test_id] = test_text

        print(f"✓ Loaded {len(test_data)} MBPP test points")
        return test_data

    elif benchmark == 'codeforces':
        print("Loading Codeforces benchmark...")
        csv_path = Path(__file__).parent.parent / 'codeforces_uniform_recent.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"Codeforces CSV not found at {csv_path}")

        df = pd.read_csv(csv_path)
        test_data = {}
        for _, row in df.iterrows():
            test_id = str(row['id'])
            # Combine problem description with input/output format
            text_parts = [str(row.get('description', ''))]
            if pd.notna(row.get('input_format')):
                text_parts.append(f"Input: {row['input_format']}")
            if pd.notna(row.get('output_format')):
                text_parts.append(f"Output: {row['output_format']}")
            test_data[test_id] = "\n".join(text_parts)

        print(f"✓ Loaded {len(test_data)} Codeforces test points")
        return test_data

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def get_benchmark_index_ranges(config_path=None, benchmarks_config=None):
    """
    Calculate the global index ranges for each benchmark based on config order.
    Returns dict of benchmark_name -> (start_idx, end_idx, test_ids_list)
    """
    from datasets import load_dataset

    # Default benchmark order if not specified
    if benchmarks_config is None:
        benchmarks_config = [
            {'name': 'mbpp'},
            {'name': 'codeforces'},
        ]

    ranges = {}
    global_idx = 0

    for bench_cfg in benchmarks_config:
        bench_name = bench_cfg['name'].lower()
        start_idx = global_idx
        test_ids = []

        if bench_name == 'mbpp':
            ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
            for item in ds:
                test_ids.append(str(item['task_id']))
            global_idx += len(ds)

        elif bench_name == 'codeforces':
            csv_path = Path(__file__).parent.parent / 'codeforces_uniform_recent.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                test_ids = [str(row['id']) for _, row in df.iterrows()]
                global_idx += len(df)

        elif bench_name.startswith('musr'):
            try:
                subset = bench_name.replace('musr_', '') if '_' in bench_name else 'murder_mysteries'
                ds = load_dataset("TAUR-Lab/MuSR", split=subset)
                for i in range(len(ds)):
                    test_ids.append(f"musr_{subset}_{i}")
                global_idx += len(ds)
            except:
                pass

        elif bench_name == 'zebralogic':
            try:
                ds = load_dataset("WaltonFuworworworworworworworwortson/ZebraLogic", split="test")
                for i in range(len(ds)):
                    test_ids.append(f"zebra_{i}")
                global_idx += len(ds)
            except:
                pass

        if test_ids:
            ranges[bench_name] = (start_idx, global_idx, test_ids)

    return ranges


def get_hash_id_paths(output_dir):
    """
    Get paths to hash_id files without loading them into memory.
    Returns dict mapping chunk_num -> file path, or None if not available.
    """
    shared_hash_dir = output_dir / "temp_similarities" / "shared_hash_ids"
    if not shared_hash_dir.exists():
        return None

    hash_files = sorted(shared_hash_dir.glob("chunk_*_hash_ids.npy"))
    if not hash_files:
        return None

    hash_id_paths = {}
    for hf in hash_files:
        chunk_num = hf.stem.split('_')[1]  # "chunk_0000_hash_ids" -> "0000"
        hash_id_paths[chunk_num] = hf

    return hash_id_paths


def load_hash_ids_for_chunk(chunk_num, hash_id_paths):
    """Load hash_ids for a single chunk on-demand."""
    if hash_id_paths is None or chunk_num not in hash_id_paths:
        return None
    try:
        return np.load(hash_id_paths[chunk_num])
    except Exception:
        return None


def sample_top1pct_for_test_streaming(test_idx, output_dir, world_size, n_samples=100,
                                       hash_id_paths=None, parquet_files=None):
    """
    Memory-efficient streaming version: two-pass approach.

    Pass 1: Stream through all chunks to compute 99th percentile threshold
    Pass 2: Stream again to collect samples above threshold

    This never loads all data into memory at once.

    Args:
        hash_id_paths: Dict of chunk_num -> file path (from get_hash_id_paths)
        parquet_files: List of parquet file paths (fallback if hash_ids missing)

    Returns: list of {'similarity': float, 'hash_id': str} dicts
    """
    # First, collect all chunk file paths
    chunk_info = []  # List of (chunk_file, chunk_num, is_npz)

    for r in range(world_size):
        chunk_dir = output_dir / "temp_similarities" / f"rank_{r}" / f"test_{test_idx}"
        if not chunk_dir.exists():
            continue

        for chunk_file in sorted(chunk_dir.glob("chunk_*_sims.npy")):
            chunk_num = chunk_file.stem.split('_')[1]
            chunk_info.append((chunk_file, chunk_num, False))

        for chunk_file in sorted(chunk_dir.glob("chunk_*.npz")):
            chunk_info.append((chunk_file, None, True))

    if not chunk_info:
        return []

    # === PASS 1: Compute threshold using fast sampling ===
    # Sample ~100k values to estimate the 99th percentile
    SAMPLE_SIZE = 100_000
    samples_per_chunk = max(1, SAMPLE_SIZE // len(chunk_info))
    sampled_values = []

    for chunk_file, chunk_num, is_npz in chunk_info:
        try:
            if is_npz:
                data = np.load(chunk_file, allow_pickle=True)
                sims = data['similarities']
            else:
                sims = np.load(chunk_file, mmap_mode='r')

            # Fast random sampling using numpy
            chunk_len = len(sims)
            n_samples = min(samples_per_chunk, chunk_len)
            indices = np.random.choice(chunk_len, size=n_samples, replace=False)
            sampled_values.extend(sims[indices].tolist())
        except Exception:
            continue

    if not sampled_values:
        return []

    # Compute threshold from samples
    threshold = np.percentile(sampled_values, 99)
    del sampled_values  # Free memory

    # === PASS 2: Collect samples above threshold ===
    candidates = []  # List of (similarity, chunk_file, chunk_num, local_idx, is_npz)

    for chunk_file, chunk_num, is_npz in chunk_info:
        try:
            if is_npz:
                data = np.load(chunk_file, allow_pickle=True)
                sims = data['similarities']
            else:
                sims = np.load(chunk_file, mmap_mode='r')

            # Find indices above threshold
            above_threshold = np.where(sims >= threshold)[0]

            for local_idx in above_threshold:
                candidates.append((
                    float(sims[local_idx]),
                    chunk_file,
                    chunk_num,
                    int(local_idx),
                    is_npz
                ))
        except Exception:
            continue

    if not candidates:
        return []

    # Sample from candidates
    if len(candidates) > n_samples:
        sampled = [candidates[i] for i in np.random.choice(len(candidates), size=n_samples, replace=False)]
    else:
        sampled = candidates

    # === Resolve hash_ids for sampled items only ===
    samples = []

    # Group by chunk to minimize file reads
    from collections import defaultdict
    by_chunk = defaultdict(list)
    for sim, chunk_file, chunk_num, local_idx, is_npz in sampled:
        by_chunk[(chunk_file, chunk_num, is_npz)].append((sim, local_idx))

    for (chunk_file, chunk_num, is_npz), items in by_chunk.items():
        try:
            if is_npz:
                data = np.load(chunk_file, allow_pickle=True)
                hash_ids = data['hash_ids']
                for sim, local_idx in items:
                    samples.append({
                        'similarity': sim,
                        'hash_id': str(hash_ids[local_idx])
                    })
            else:
                # Load hash_ids for this chunk
                hash_ids = load_hash_ids_for_chunk(chunk_num, hash_id_paths)
                if hash_ids is not None:
                    for sim, local_idx in items:
                        samples.append({
                            'similarity': sim,
                            'hash_id': str(hash_ids[local_idx])
                        })
                else:
                    # Mark as missing - will need parquet fallback
                    for sim, local_idx in items:
                        samples.append({
                            'similarity': sim,
                            'hash_id': f"__MISSING__{chunk_num}_{local_idx}"
                        })
        except Exception:
            continue

    return samples


def sample_top1pct_for_test(test_idx, output_dir, world_size, n_samples=100,
                            hash_id_cache=None, parquet_files=None, chunk_size=5_000_000,
                            use_mmap=True):
    """
    Legacy function - redirects to streaming version for memory efficiency.
    Kept for API compatibility.
    """
    # Convert old hash_id_cache format to paths format if needed
    # In the new version, hash_id_cache should be hash_id_paths (just file paths)
    return sample_top1pct_for_test_streaming(
        test_idx, output_dir, world_size, n_samples,
        hash_id_paths=hash_id_cache,  # Now expects paths, not loaded arrays
        parquet_files=parquet_files
    )


def _worker_sample_test(args):
    """Worker function for parallel processing."""
    test_idx, test_id, output_dir, world_size, n_samples, hash_id_paths, test_texts = args

    samples = sample_top1pct_for_test_streaming(
        test_idx, output_dir, world_size, n_samples,
        hash_id_paths=hash_id_paths
    )

    if not samples:
        return []

    test_text = test_texts.get(test_id, '')
    results = []
    for sample in samples:
        results.append({
            'test_id': test_id,
            'test_text': test_text,
            'corpus_id': sample['hash_id'],
            'corpus_text': '',  # Will be filled later if needed
            'similarity': sample['similarity']
        })
    return results


def process_config(config_path, output_csv, n_samples=100):
    """Process a single config file and generate sample CSV."""
    
    config = load_config(config_path)
    
    # Determine output directory from config
    analysis_config = config.get('analysis', {})
    finalize_config = config.get('finalize', {})
    
    output_dir = Path(analysis_config.get('output_dir', './results/contamination'))
    corpus_jsonl = Path(finalize_config.get('corpus_file', ''))
    world_size = analysis_config.get('cluster', {}).get('world_size', 1)
    
    # Make paths absolute relative to pipeline root
    pipeline_root = Path(config_path).parent.parent
    if not output_dir.is_absolute():
        output_dir = pipeline_root / output_dir
    if not corpus_jsonl.is_absolute():
        corpus_jsonl = pipeline_root / corpus_jsonl
    
    print(f"\n{'='*60}")
    print(f"Processing config: {config_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Corpus JSONL: {corpus_jsonl}")
    print(f"  World size: {world_size}")
    print(f"{'='*60}")
    
    # Check if temp_similarities exists
    temp_sims_dir = output_dir / "temp_similarities"
    if not temp_sims_dir.exists():
        print(f"⚠️  No temp_similarities found at {temp_sims_dir}")
        print("   Run stage 04 first, or the data has been cleaned up.")
        return None
    
    # Load corpus text mapping
    id_to_text = load_corpus_text_mapping(corpus_jsonl)
    
    # Load MBPP test texts
    test_texts = load_benchmark_test_texts()
    
    # Find MBPP test directories
    # Test indices for MBPP are stored in the test_{idx} folders
    # We need to map back to test IDs
    
    # First, find which benchmarks were run
    mbpp_dirs = []
    for r in range(world_size):
        rank_dir = temp_sims_dir / f"rank_{r}"
        if rank_dir.exists():
            test_dirs = list(rank_dir.glob("test_*"))
            if test_dirs:
                mbpp_dirs.extend(test_dirs)
                break  # Just need to enumerate from one rank
    
    # Get unique test indices
    test_indices = set()
    for td in mbpp_dirs:
        try:
            idx = int(td.name.split('_')[1])
            test_indices.add(idx)
        except:
            continue
    
    # Load benchmark metadata to map indices to test IDs
    # This requires loading the benchmark the same way stage 04 does
    print(f"Found {len(test_indices)} test indices in temp_similarities")
    
    # We need to reload benchmarks to map global_idx -> test_id
    # For now, we'll process MBPP specifically
    benchmarks = analysis_config.get('benchmarks', [])
    mbpp_benchmarks = [b for b in benchmarks if 'mbpp' in b.get('name', '').lower()]
    
    if not mbpp_benchmarks:
        print("⚠️  No MBPP benchmark found in config")
        return None
    
    # Reload MBPP to get the mapping
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    
    # Build global_idx -> test_id mapping (assuming MBPP is loaded first or we know its offset)
    # For simplicity, we'll load all benchmarks in order and find MBPP's offset
    global_idx = 0
    mbpp_start_idx = None
    mbpp_test_ids = []
    
    for bench_cfg in benchmarks:
        bench_name = bench_cfg['name']
        if 'mbpp' in bench_name.lower():
            mbpp_start_idx = global_idx
            for item in ds:
                mbpp_test_ids.append(str(item['task_id']))
                global_idx += 1
        else:
            # Load other benchmarks to get their count
            # This is a simplification - in practice you'd load each benchmark
            # For now we'll just handle the common case where MBPP is one of the first
            if bench_name.startswith('musr'):
                # MuSR benchmarks have varying sizes
                try:
                    subset = bench_name.replace('musr_', '')
                    musr_ds = load_dataset("TAUR-Lab/MuSR", split=subset)
                    global_idx += len(musr_ds)
                except:
                    global_idx += 100  # Fallback estimate
            elif bench_name == 'zebralogic':
                global_idx += 1000  # Estimate
            elif bench_name == 'codeforces':
                global_idx += 500  # Estimate
    
    if mbpp_start_idx is None:
        print("⚠️  Could not determine MBPP offset in benchmark list")
        return None
    
    print(f"MBPP starts at global_idx {mbpp_start_idx}, has {len(mbpp_test_ids)} test points")

    # Get hash_id file paths (lazy loading - no memory used yet)
    hash_id_paths = get_hash_id_paths(output_dir)
    if hash_id_paths:
        print(f"Found {len(hash_id_paths)} hash_id chunk files (will load on-demand)")

    # Sample from each MBPP test point
    all_samples = []

    for i, test_id in enumerate(tqdm(mbpp_test_ids, desc="Sampling top 1%")):
        global_test_idx = mbpp_start_idx + i

        if global_test_idx not in test_indices:
            continue

        samples = sample_top1pct_for_test_streaming(global_test_idx, output_dir, world_size, n_samples,
                                                     hash_id_paths=hash_id_paths)
        
        if not samples:
            continue
        
        test_text = test_texts.get(test_id, '')
        
        for sample in samples:
            corpus_text = id_to_text.get(sample['hash_id'], '')
            all_samples.append({
                'test_id': test_id,
                'test_text': test_text,
                'corpus_id': sample['hash_id'],
                'corpus_text': corpus_text,
                'similarity': sample['similarity']
            })
    
    if not all_samples:
        print("⚠️  No samples collected")
        return None
    
    # Create DataFrame and save
    df = pd.DataFrame(all_samples)
    df = df.sort_values(['test_id', 'similarity'], ascending=[True, False])
    
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Saved {len(df)} samples to {output_csv}")
    print(f"   Test points with samples: {df['test_id'].nunique()}")
    print(f"   Top similarity: {df['similarity'].max():.4f}")
    print(f"   Threshold (99th pct): {df.groupby('test_id')['similarity'].min().mean():.4f}")
    
    return df


def process_direct(results_dir, corpus_jsonl, output_csv, data_dir=None, n_samples=100,
                   max_tests=None, num_workers=None, benchmark='mbpp', benchmarks_config=None):
    """Process with direct paths (no config file needed).

    Args:
        benchmark: Which benchmark to process ('mbpp', 'codeforces')
        benchmarks_config: List of benchmark configs to determine index offsets
        num_workers: Number of parallel workers (default: cpu_count() // 2)
    """
    results_dir = Path(results_dir)
    corpus_jsonl = Path(corpus_jsonl)
    output_csv = Path(output_csv)
    data_dir = Path(data_dir) if data_dir else None

    # Default to half of CPU cores (I/O bound task)
    if num_workers is None:
        num_workers = max(1, cpu_count() // 2)

    print(f"\n{'='*60}")
    print(f"Results dir: {results_dir}")
    print(f"Corpus JSONL: {corpus_jsonl}")
    print(f"Output: {output_csv}")
    print(f"Benchmark: {benchmark}")
    print(f"Workers: {num_workers}")
    print(f"{'='*60}")

    # Check if temp_similarities exists
    temp_sims_dir = results_dir / "temp_similarities"
    if not temp_sims_dir.exists():
        print(f"⚠️  No temp_similarities found at {temp_sims_dir}")
        return None

    # Load corpus text mapping (skip if /dev/null)
    if str(corpus_jsonl) != '/dev/null':
        id_to_text = load_corpus_text_mapping(corpus_jsonl)
    else:
        id_to_text = {}

    # Load benchmark test texts
    test_texts = load_benchmark_test_texts(benchmark)

    # Find test directories
    world_size = 1
    for r in range(8):
        if (temp_sims_dir / f"rank_{r}").exists():
            world_size = max(world_size, r + 1)

    # Get test indices
    test_indices = set()
    for r in range(world_size):
        rank_dir = temp_sims_dir / f"rank_{r}"
        if rank_dir.exists():
            for td in rank_dir.glob("test_*"):
                try:
                    idx = int(td.name.split('_')[1])
                    test_indices.add(idx)
                except:
                    continue
            break

    print(f"Found {len(test_indices)} test indices, world_size={world_size}")

    # Get benchmark index ranges
    bench_ranges = get_benchmark_index_ranges(benchmarks_config=benchmarks_config)

    if benchmark not in bench_ranges:
        print(f"⚠️  Benchmark '{benchmark}' not found in config")
        print(f"   Available: {list(bench_ranges.keys())}")
        return None

    start_idx, end_idx, bench_test_ids = bench_ranges[benchmark]
    print(f"{benchmark.upper()} index range: {start_idx} - {end_idx-1} ({len(bench_test_ids)} tests)")

    # Get hash_id file paths (lazy loading - no memory used yet)
    hash_id_paths = get_hash_id_paths(results_dir)
    if hash_id_paths:
        print(f"Found {len(hash_id_paths)} hash_id chunk files (will load on-demand)")

    # Build list of tasks to process
    tasks = []
    for i, test_id in enumerate(bench_test_ids):
        global_idx = start_idx + i
        if global_idx not in test_indices:
            continue
        tasks.append((global_idx, test_id, results_dir, world_size, n_samples, hash_id_paths, test_texts))
        if max_tests and len(tasks) >= max_tests:
            break

    print(f"Processing {len(tasks)} tests with {num_workers} workers (threaded)...")

    # Process in batches to control memory usage
    BATCH_SIZE = 50  # Process 50 tests at a time, then cleanup
    all_samples = []

    for batch_start in range(0, len(tasks), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(tasks))
        batch_tasks = tasks[batch_start:batch_end]

        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(tqdm(
                    executor.map(_worker_sample_test, batch_tasks),
                    total=len(batch_tasks),
                    desc=f"Batch {batch_start//BATCH_SIZE + 1}/{(len(tasks) + BATCH_SIZE - 1)//BATCH_SIZE}"
                ))
            for result in results:
                all_samples.extend(result)
        else:
            # Single-threaded fallback
            for task in tqdm(batch_tasks, desc=f"Batch {batch_start//BATCH_SIZE + 1}"):
                result = _worker_sample_test(task)
                all_samples.extend(result)

        # Force garbage collection between batches
        gc.collect()

    # Add corpus text if available
    if id_to_text:
        for sample in all_samples:
            sample['corpus_text'] = id_to_text.get(sample['corpus_id'], '')

    if not all_samples:
        print("⚠️  No samples collected")
        return None
    
    df = pd.DataFrame(all_samples)
    df = df.sort_values(['test_id', 'similarity'], ascending=[True, False])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Saved {len(df)} samples to {output_csv}")
    print(f"   Test points: {df['test_id'].nunique()}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Sample top 1% for semantic duplicate analysis")
    parser.add_argument('--config', '-c', help='Single config file to process')
    parser.add_argument('--all-configs', action='store_true', help='Process all configs in configs/')
    parser.add_argument('--results-dir', help='Direct path to results directory with temp_similarities')
    parser.add_argument('--corpus-jsonl', help='Direct path to corpus JSONL file')
    parser.add_argument('--data-dir', help='Direct path to embeddings parquet directory (for ID reconstruction)')
    parser.add_argument('--output', '-o', help='Output CSV')
    parser.add_argument('--output-dir', help='Output directory (for all configs)')
    parser.add_argument('--samples', '-n', type=int, default=100, help='Samples per test point')
    parser.add_argument('--max-tests', type=int, default=None, help='Limit number of tests to process (for quick testing)')
    parser.add_argument('--workers', '-w', type=int, default=None, help='Number of parallel workers (default: cpu_count/2)')
    parser.add_argument('--benchmark', '-b', default='mbpp', choices=['mbpp', 'codeforces'],
                        help='Benchmark to sample from (default: mbpp)')

    args = parser.parse_args()
    
    pipeline_root = Path(__file__).parent.parent
    configs_dir = pipeline_root / "configs"
    
    # Direct path mode
    if args.results_dir and args.corpus_jsonl:
        output_csv = args.output or f"./top1pct_{args.benchmark}_output.csv"
        process_direct(args.results_dir, args.corpus_jsonl, output_csv, args.data_dir,
                       args.samples, args.max_tests, args.workers, benchmark=args.benchmark)
    
    elif args.all_configs:
        config_files = list(configs_dir.glob("*.yaml"))
        output_dir = Path(args.output_dir or pipeline_root / "semantic_samples")
        
        for config_file in config_files:
            if config_file.name in ['example_custom.yaml']:
                continue
            
            output_csv = output_dir / f"top1pct_{config_file.stem}.csv"
            try:
                process_config(config_file, output_csv, args.samples)
            except Exception as e:
                print(f"❌ Error processing {config_file.name}: {e}")
                import traceback
                traceback.print_exc()
    
    elif args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = configs_dir / args.config
        
        output_csv = args.output or f"top1pct_{config_path.stem}.csv"
        process_config(config_path, output_csv, args.samples)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
