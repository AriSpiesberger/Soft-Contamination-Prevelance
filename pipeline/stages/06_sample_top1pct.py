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
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


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


def load_benchmark_test_texts():
    """Load MBPP test texts."""
    from datasets import load_dataset
    
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


def sample_top1pct_for_test(test_idx, output_dir, world_size, n_samples=100, parquet_files=None, chunk_size=5_000_000):
    """
    For a single test point, load all similarities across ranks,
    find the top 1% threshold, and sample 100 random items from that top 1%.
    
    If shared_hash_ids don't exist, reconstructs IDs from parquet files by matching
    chunk indices to parquet row order.
    
    Args:
        parquet_files: List of parquet file paths (needed if hash_ids missing)
        chunk_size: Rows per similarity chunk (default 5M, matches stage 04)
    
    Returns: list of {'similarity': float, 'hash_id': str} dicts
    """
    all_sims = []
    all_hash_ids = []
    
    # Check if hash_ids exist
    hash_dir = output_dir / "temp_similarities" / "rank_0" / "shared_hash_ids"
    has_hash_files = hash_dir.exists() and len(list(hash_dir.glob("*.npy"))) > 0
    
    # Collect all similarity data from all ranks
    for r in range(world_size):
        chunk_dir = output_dir / "temp_similarities" / f"rank_{r}" / f"test_{test_idx}"
        hash_dir_r = output_dir / "temp_similarities" / f"rank_{r}" / "shared_hash_ids"
        
        if not chunk_dir.exists():
            continue
            
        # Support both .npy and .npz formats
        chunk_files_npy = sorted(chunk_dir.glob("chunk_*_sims.npy"))
        chunk_files_npz = sorted(chunk_dir.glob("chunk_*.npz"))
        
        for chunk_file in chunk_files_npy:
            chunk_num = chunk_file.stem.split('_')[1]
            hash_file = hash_dir_r / f"chunk_{chunk_num}_hash_ids.npy"
            
            try:
                sims = np.load(chunk_file)
                if hash_file.exists():
                    hash_ids = np.load(hash_file)
                    all_sims.append(sims)
                    all_hash_ids.append(hash_ids)
                else:
                    # Store sims with placeholder - will reconstruct IDs later
                    all_sims.append(sims)
                    all_hash_ids.append(None)  # Mark as needing reconstruction
            except Exception as e:
                continue
        
        for chunk_file in chunk_files_npz:
            try:
                data = np.load(chunk_file, allow_pickle=True)
                all_sims.append(data['similarities'])
                all_hash_ids.append(data['hash_ids'])
            except Exception:
                continue
    
    if not all_sims:
        return []
    
    # Concatenate all similarities
    all_sims_flat = np.concatenate(all_sims)
    
    # Check if we need to reconstruct hash_ids from parquet
    if any(h is None for h in all_hash_ids):
        if parquet_files is None:
            print(f"  ⚠️  Test {test_idx}: No hash_ids and no parquet files provided")
            return []
        
        # Reconstruct hash_ids from parquet files
        # The order of similarities matches the order embeddings were processed
        # which is parquet file order, row by row
        all_ids = []
        for pf in parquet_files:
            try:
                import duckdb
                con = duckdb.connect(':memory:')
                ids = con.execute(f"SELECT id FROM read_parquet('{pf}')").fetchall()
                all_ids.extend([row[0] for row in ids])
                con.close()
            except Exception as e:
                continue
        
        if len(all_ids) != len(all_sims_flat):
            print(f"  ⚠️  Test {test_idx}: ID count mismatch ({len(all_ids)} IDs vs {len(all_sims_flat)} sims)")
            # Try to continue with what we have
            min_len = min(len(all_ids), len(all_sims_flat))
            all_ids = all_ids[:min_len]
            all_sims_flat = all_sims_flat[:min_len]
        
        all_hash_ids_flat = np.array(all_ids)
    else:
        all_hash_ids_flat = np.concatenate(all_hash_ids)
    
    # Find 99th percentile (top 1% threshold)
    threshold = np.percentile(all_sims_flat, 99)
    
    # Find indices above threshold
    top1_mask = all_sims_flat >= threshold
    top1_indices = np.where(top1_mask)[0]
    
    if len(top1_indices) == 0:
        return []
    
    # Sample n_samples from top 1%
    if len(top1_indices) > n_samples:
        sampled_indices = np.random.choice(top1_indices, size=n_samples, replace=False)
    else:
        sampled_indices = top1_indices
    
    # Build result
    samples = []
    for idx in sampled_indices:
        samples.append({
            'similarity': float(all_sims_flat[idx]),
            'hash_id': str(all_hash_ids_flat[idx])
        })
    
    return samples


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
    
    # Sample from each MBPP test point
    all_samples = []
    
    for i, test_id in enumerate(tqdm(mbpp_test_ids, desc="Sampling top 1%")):
        global_test_idx = mbpp_start_idx + i
        
        if global_test_idx not in test_indices:
            continue
        
        samples = sample_top1pct_for_test(global_test_idx, output_dir, world_size, n_samples)
        
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


def process_direct(results_dir, corpus_jsonl, output_csv, data_dir=None, n_samples=100):
    """Process with direct paths (no config file needed)."""
    results_dir = Path(results_dir)
    corpus_jsonl = Path(corpus_jsonl)
    output_csv = Path(output_csv)
    data_dir = Path(data_dir) if data_dir else None
    
    print(f"\n{'='*60}")
    print(f"Results dir: {results_dir}")
    print(f"Corpus JSONL: {corpus_jsonl}")
    print(f"Output: {output_csv}")
    print(f"{'='*60}")
    
    # Check if temp_similarities exists
    temp_sims_dir = results_dir / "temp_similarities"
    if not temp_sims_dir.exists():
        print(f"⚠️  No temp_similarities found at {temp_sims_dir}")
        return None
    
    # Load corpus text mapping
    id_to_text = load_corpus_text_mapping(corpus_jsonl)
    
    # Load MBPP test texts
    test_texts = load_benchmark_test_texts()
    
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
    
    # For MBPP, indices are 0-256 (257 tests)
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    mbpp_test_ids = [str(item['task_id']) for item in ds]
    
    # Load parquet files for hash_id reconstruction (if data_dir provided)
    parquet_files = None
    if data_dir and data_dir.exists():
        # Try flat first, then recursive
        parquet_files = sorted(data_dir.glob("*.parquet"))
        if not parquet_files:
            print(f"Searching recursively for parquet files...")
            parquet_files = sorted(data_dir.rglob("*.parquet"))
        print(f"Found {len(parquet_files)} parquet files for ID lookup")
    
    # Sample from each test
    all_samples = []
    for i, test_id in enumerate(tqdm(mbpp_test_ids, desc="Sampling top 1%")):
        if i not in test_indices:
            continue
        
        samples = sample_top1pct_for_test(i, results_dir, world_size, n_samples, parquet_files=parquet_files)
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
    
    args = parser.parse_args()
    
    pipeline_root = Path(__file__).parent.parent
    configs_dir = pipeline_root / "configs"
    
    # Direct path mode
    if args.results_dir and args.corpus_jsonl:
        output_csv = args.output or "./top1pct_output.csv"
        process_direct(args.results_dir, args.corpus_jsonl, output_csv, args.data_dir, args.samples)
    
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
