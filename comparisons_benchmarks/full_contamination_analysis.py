#!/usr/bin/env python3
"""
Full contamination analysis: Every test point against every embedding in all parquet files.
Processes MUSR and MBPP benchmarks in both input and output modes.
Saves full similarity vectors and top 100 matches for each test point.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import pyarrow.parquet as pq
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import pickle
import gzip
import time
from datetime import timedelta


def embed_texts(texts, model, tokenizer, device, batch_size=16):
    """Embed texts in batches."""
    embeddings = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                          max_length=8192, return_tensors='pt').to(device)
            out = model(**enc)
            mask = enc['attention_mask'].unsqueeze(-1).to(out[0].dtype)
            emb = (out[0] * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = F.normalize(emb, p=2, dim=1)
            embeddings.append(emb.float().cpu().numpy())
    return np.vstack(embeddings)


def load_all_parquet_data(data_dir: Path):
    """Load ALL embeddings, texts, and IDs from ALL parquet files."""
    start_time = time.time()
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")
    
    all_embeddings = []
    all_texts = []
    all_ids = []
    all_hash_ids = []
    file_boundaries = []  # Track which embeddings come from which file
    
    for pf_path in tqdm(parquet_files, desc="Loading parquet files"):
        try:
            pf = pq.ParquetFile(str(pf_path))
            cols = [f.name for f in pf.schema_arrow]
            
            if 'embeddings' not in cols:
                print(f"  ⚠️  Skipping {pf_path.name} (no embeddings column)")
                continue
            
            # Find columns
            text_col = None
            for c in ['text', 'content', 'paragraph', 'document']:
                if c in cols:
                    text_col = c
                    break
            
            hash_id_col = 'hash_id' if 'hash_id' in cols else None
            id_col = 'id' if 'id' in cols else None
            
            # Get embedding dimension
            emb_field = pf.schema_arrow.field('embeddings')
            emb_type = emb_field.type
            dim = emb_type.list_size if hasattr(emb_type, 'list_size') else 4096
            
            # Determine columns to read
            cols_to_read = ['embeddings']
            if text_col:
                cols_to_read.append(text_col)
            if id_col:
                cols_to_read.append(id_col)
            if hash_id_col:
                cols_to_read.append(hash_id_col)
            
            # Track start index for this file
            start_idx = len(all_embeddings)
            
            # Read all row groups
            for rg_idx in range(pf.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=cols_to_read)
                emb_col = table['embeddings']
                
                # Extract texts and IDs for this row group
                if text_col:
                    texts_rg = table[text_col].to_pylist()
                if id_col:
                    ids_rg = table[id_col].to_pylist()
                if hash_id_col:
                    hash_ids_rg = table[hash_id_col].to_pylist()
                
                row_offset = 0
                
                for chunk_idx in range(emb_col.num_chunks):
                    chunk = emb_col.chunk(chunk_idx)
                    n = len(chunk)
                    if n == 0:
                        continue
                    
                    # Extract texts and IDs for this chunk
                    if text_col:
                        all_texts.extend(texts_rg[row_offset:row_offset+n])
                    if id_col:
                        all_ids.extend(ids_rg[row_offset:row_offset+n])
                    if hash_id_col:
                        all_hash_ids.extend(hash_ids_rg[row_offset:row_offset+n])
                    row_offset += n
                    
                    # Extract embeddings
                    try:
                        values_chunk = chunk.flatten()
                        vals = values_chunk.to_numpy()
                        expected_size = n * dim
                        if len(vals) == expected_size:
                            mat = vals.reshape(n, dim).astype(np.float32)
                        else:
                            if len(vals) % n == 0:
                                inferred_dim = len(vals) // n
                                mat = vals.reshape(n, inferred_dim).astype(np.float32)
                                dim = inferred_dim
                            else:
                                py_list = chunk.to_pylist()
                                mat = np.array(py_list, dtype=np.float32)
                    except Exception as e:
                        py_list = chunk.to_pylist()
                        mat = np.array(py_list, dtype=np.float32)
                    
                    # Normalize
                    norms = np.linalg.norm(mat, axis=1, keepdims=True)
                    mat = mat / np.maximum(norms, 1e-9)
                    all_embeddings.append(mat)
            
            # Record file boundary
            end_idx = len(all_embeddings)
            file_boundaries.append({
                'file': str(pf_path),
                'start_idx': start_idx,
                'end_idx': end_idx,
                'count': end_idx - start_idx
            })
            
        except Exception as e:
            print(f"  ❌ Error loading {pf_path.name}: {e}")
            continue
    
    if not all_embeddings:
        raise ValueError("No embeddings loaded from any parquet files")
    
    # Stack all embeddings
    final_mat = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
    
    # Generate IDs if not available
    if not all_ids:
        all_ids = [f"emb_{i}" for i in range(len(final_mat))]
    if not all_hash_ids:
        all_hash_ids = [None] * len(final_mat)
    if not all_texts:
        all_texts = [None] * len(final_mat)
    
    elapsed = time.time() - start_time
    print(f"✅ Loaded {len(final_mat):,} total embeddings from {len(file_boundaries)} files")
    print(f"   Embedding dimension: {final_mat.shape[1]}")
    print(f"   ⏱️  Loading time: {timedelta(seconds=int(elapsed))} ({elapsed:.2f}s)")
    
    return final_mat, all_texts, all_ids, all_hash_ids, file_boundaries


def load_benchmark(benchmark_name: str, mode: str):
    """Load benchmark data and extract texts based on mode."""
    data = []
    
    if benchmark_name == 'musr':
        print(f"Loading MUSR dataset...")
        ds = load_dataset("TAUR-Lab/MuSR")
        split = list(ds.keys())[0]
        
        for item in ds[split]:
            task_id = str(item.get('task_id', f"musr_{len(data)}"))
            narrative = item.get('narrative', item.get('question', ''))
            answer = item.get('answer', '')
            
            data.append({
                'id': task_id,
                'input': narrative,
                'output': answer
            })
        
        print(f"  ✅ Loaded {len(data)} MUSR items")
    
    elif benchmark_name == 'mbpp':
        print(f"Loading MBPP dataset...")
        try:
            ds = load_dataset("evalplus/mbpp", "mbpp")
        except:
            ds = load_dataset("google-research-datasets/mbpp", "sanitized")
        
        for item in ds['test']:
            task_id = str(item.get('task_id', f"mbpp_{len(data)}"))
            prompt = item.get('prompt', item.get('text', ''))
            solution = item.get('canonical_solution', item.get('code', ''))
            
            data.append({
                'id': task_id,
                'input': prompt,
                'output': solution
            })
        
        print(f"  ✅ Loaded {len(data)} MBPP items")
    
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    # Extract texts based on mode
    texts = []
    ids = []
    for item in data:
        if mode == 'input':
            texts.append(item['input'])
        elif mode == 'output':
            texts.append(item['output'])
        else:
            texts.append(f"{item['input']}\n\n{item['output']}")
        ids.append(item['id'])
    
    return texts, ids


def process_benchmark_mode(benchmark_name: str, mode: str, corpus_embs, corpus_texts, 
                           corpus_ids, corpus_hash_ids, model, tokenizer, device, output_dir: Path,
                           batch_size=32):
    """Process all test points for a benchmark/mode combination."""
    mode_start_time = time.time()
    print(f"\n{'='*80}")
    print(f"Processing {benchmark_name.upper()} - {mode.upper()} mode")
    print(f"{'='*80}")
    
    # Load benchmark
    benchmark_start = time.time()
    test_texts, test_ids = load_benchmark(benchmark_name, mode)
    benchmark_time = time.time() - benchmark_start
    print(f"⏱️  Benchmark loading: {timedelta(seconds=int(benchmark_time))} ({benchmark_time:.2f}s)")
    
    # Embed all test texts
    embed_start = time.time()
    print(f"Embedding {len(test_texts)} test points...")
    test_embs = embed_texts(test_texts, model, tokenizer, device)
    embed_time = time.time() - embed_start
    print(f"⏱️  Embedding time: {timedelta(seconds=int(embed_time))} ({embed_time:.2f}s)")
    
    # Move corpus embeddings to GPU (keep there for all comparisons)
    gpu_transfer_start = time.time()
    print(f"Moving corpus embeddings to GPU...")
    corpus_embs_gpu = torch.from_numpy(corpus_embs).to(device).half()  # Use FP16 for speed
    gpu_transfer_time = time.time() - gpu_transfer_start
    print(f"⏱️  GPU transfer time: {timedelta(seconds=int(gpu_transfer_time))} ({gpu_transfer_time:.2f}s)")
    
    # Create output directory for this benchmark/mode
    mode_dir = output_dir / f"{benchmark_name}_{mode}"
    mode_dir.mkdir(parents=True, exist_ok=True)
    
    # Storage for aggregate histogram
    all_similarities = []
    
    # Process test points in batches for GPU efficiency
    print(f"\nComputing similarities for {len(test_texts)} test points (batch size: {batch_size})...")
    top_k = 100
    
    similarity_time_total = 0
    save_time_total = 0
    plot_time_total = 0
    
    for batch_start in tqdm(range(0, len(test_texts), batch_size), desc=f"{benchmark_name}_{mode}"):
        batch_end = min(batch_start + batch_size, len(test_texts))
        batch_indices = range(batch_start, batch_end)
        
        # Prepare batch of test embeddings on GPU
        batch_prep_start = time.time()
        test_embs_batch = torch.from_numpy(test_embs[batch_start:batch_end]).to(device).half()
        
        # Compute similarities on GPU (much faster!)
        # Shape: (batch_size, num_corpus_embeddings)
        similarities_batch_gpu = torch.matmul(test_embs_batch, corpus_embs_gpu.t())
        
        # Get top-k on GPU (faster than CPU argsort)
        top_scores_batch_gpu, top_indices_batch_gpu = torch.topk(
            similarities_batch_gpu, k=min(top_k, similarities_batch_gpu.size(1)), dim=1
        )
        
        # Move to CPU for processing
        similarities_batch = similarities_batch_gpu.cpu().numpy().astype(np.float32)
        top_scores_batch = top_scores_batch_gpu.cpu().numpy().astype(np.float32)
        top_indices_batch = top_indices_batch_gpu.cpu().numpy()
        similarity_time_total += time.time() - batch_prep_start
        
        # Process each test point in batch
        for i, test_idx in enumerate(batch_indices):
            test_id = test_ids[test_idx]
            test_text = test_texts[test_idx]
            
            # Get similarities and top-k for this test point
            similarities = similarities_batch[i]
            top_indices = top_indices_batch[i]
            top_scores = top_scores_batch[i]
            
            # Save top 100 matches
            top_matches = []
            for j in range(len(top_scores)):
                idx = top_indices[j]
                match = {
                    'rank': j + 1,
                    'score': float(top_scores[j]),
                    'index': int(idx),
                    'id': str(corpus_ids[idx]),
                    'hash_id': str(corpus_hash_ids[idx]) if corpus_hash_ids[idx] is not None else None,
                    'text': corpus_texts[idx] if corpus_texts and idx < len(corpus_texts) else None
                }
                top_matches.append(match)
            
            # Save individual results
            save_start = time.time()
            result = {
                'test_id': test_id,
                'test_text': test_text,
                'benchmark': benchmark_name,
                'mode': mode,
                'total_embeddings': len(corpus_embs),
                'top_100': top_matches,
                'stats': {
                    'max': float(similarities.max()),
                    'min': float(similarities.min()),
                    'mean': float(similarities.mean()),
                    'median': float(np.median(similarities)),
                    'std': float(similarities.std()),
                    'top_1': float(top_scores[0]),
                    'top_10': float(top_scores[9]) if len(top_scores) > 9 else None,
                    'top_50': float(top_scores[49]) if len(top_scores) > 49 else None,
                    'top_100': float(top_scores[99]) if len(top_scores) > 99 else None
                }
            }
            
            # Save top 100 JSON
            json_file = mode_dir / f"{test_id}_top100.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Save full similarity vector (compressed)
            vector_file = mode_dir / f"{test_id}_similarities.npy.gz"
            with gzip.open(vector_file, 'wb') as f:
                np.save(f, similarities.astype(np.float32))
            save_time_total += time.time() - save_start
            
            # Create individual plots (same as sanity_check.py)
            plot_start = time.time()
            # 1. Histogram of all similarities
            plt.figure(figsize=(10, 6))
            plt.hist(similarities, bins=100, alpha=0.7, edgecolor='black')
            plt.axvline(top_scores[0], color='r', linestyle='--', label=f'Top-1: {top_scores[0]:.4f}')
            plt.axvline(similarities.mean(), color='g', linestyle='--', label=f'Mean: {similarities.mean():.4f}')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Frequency')
            plt.title(f'{benchmark_name.upper()} {mode.upper()}: {test_id}\nSimilarity Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(mode_dir / f"{test_id}_histogram.png", dpi=150)
            plt.close()
            
            # 2. Top-k similarity scores line plot
            plt.figure(figsize=(12, 6))
            plt.plot(range(1, len(top_scores) + 1), top_scores, marker='o', markersize=3, linewidth=1.5)
            plt.xlabel('Rank')
            plt.ylabel('Cosine Similarity')
            plt.title(f'{benchmark_name.upper()} {mode.upper()}: {test_id}\nTop-{len(top_scores)} Similarity Scores')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(mode_dir / f"{test_id}_topk.png", dpi=150)
            plt.close()
            
            # 3. Cumulative distribution (sorted similarities)
            sorted_sims = np.sort(similarities)[::-1]
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(sorted_sims)), sorted_sims, linewidth=1)
            plt.xlabel('Rank (sorted)')
            plt.ylabel('Cosine Similarity')
            plt.title(f'{benchmark_name.upper()} {mode.upper()}: {test_id}\nCumulative Similarity Distribution')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(mode_dir / f"{test_id}_cumulative.png", dpi=150)
            plt.close()
            plot_time_total += time.time() - plot_start
            
            # Add to aggregate
            all_similarities.extend(similarities.tolist())
        
        # Clear GPU cache after each batch
        torch.cuda.empty_cache()
    
    print(f"\n⏱️  Timing breakdown:")
    print(f"   Similarity computation: {timedelta(seconds=int(similarity_time_total))} ({similarity_time_total:.2f}s)")
    print(f"   File saving: {timedelta(seconds=int(save_time_total))} ({save_time_total:.2f}s)")
    print(f"   Plot generation: {timedelta(seconds=int(plot_time_total))} ({plot_time_total:.2f}s)")
    
    # Create aggregate plots for this benchmark/mode (same as sanity_check.py)
    aggregate_start = time.time()
    print(f"\nCreating aggregate plots for {benchmark_name}_{mode}...")
    all_sims_array = np.array(all_similarities)
    
    # 1. Aggregate histogram
    plt.figure(figsize=(12, 8))
    plt.hist(all_sims_array, bins=200, alpha=0.7, edgecolor='black')
    plt.axvline(all_sims_array.max(), color='r', linestyle='--', 
                label=f'Max: {all_sims_array.max():.4f}')
    plt.axvline(all_sims_array.mean(), color='g', linestyle='--', 
                label=f'Mean: {all_sims_array.mean():.4f}')
    plt.axvline(np.percentile(all_sims_array, 99), color='orange', linestyle='--',
                label=f'99th percentile: {np.percentile(all_sims_array, 99):.4f}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title(f'{benchmark_name.upper()} {mode.upper()} - All Test Points vs All Embeddings\n'
              f'Total comparisons: {len(all_sims_array):,}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(mode_dir / f"aggregate_histogram.png", dpi=150)
    plt.close()
    
    # 2. Aggregate top-k line plot (top 1000 from all similarities)
    sorted_all = np.sort(all_sims_array)[::-1]
    top_k_agg = min(1000, len(sorted_all))
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, top_k_agg + 1), sorted_all[:top_k_agg], 
             marker='o', markersize=2, linewidth=1.5)
    plt.xlabel('Rank')
    plt.ylabel('Cosine Similarity')
    plt.title(f'{benchmark_name.upper()} {mode.upper()} - Top-{top_k_agg} Similarity Scores\n'
              f'Across all test points (total comparisons: {len(all_sims_array):,})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(mode_dir / f"aggregate_topk.png", dpi=150)
    plt.close()
    
    # 3. Aggregate cumulative distribution
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(sorted_all)), sorted_all, linewidth=1)
    plt.xlabel('Rank (sorted)')
    plt.ylabel('Cosine Similarity')
    plt.title(f'{benchmark_name.upper()} {mode.upper()} - Cumulative Similarity Distribution\n'
              f'All test points combined (total comparisons: {len(all_sims_array):,})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(mode_dir / f"aggregate_cumulative.png", dpi=150)
    plt.close()
    
    # Save aggregate stats
    agg_stats = {
        'benchmark': benchmark_name,
        'mode': mode,
        'num_test_points': len(test_texts),
        'total_embeddings': len(corpus_embs),
        'total_comparisons': len(all_sims_array),
        'stats': {
            'max': float(all_sims_array.max()),
            'min': float(all_sims_array.min()),
            'mean': float(all_sims_array.mean()),
            'median': float(np.median(all_sims_array)),
            'std': float(all_sims_array.std()),
            'p99': float(np.percentile(all_sims_array, 99)),
            'p95': float(np.percentile(all_sims_array, 95)),
            'p90': float(np.percentile(all_sims_array, 90))
        }
    }
    
    with open(mode_dir / "aggregate_stats.json", 'w') as f:
        json.dump(agg_stats, f, indent=2)
    
    aggregate_time = time.time() - aggregate_start
    mode_total_time = time.time() - mode_start_time
    
    print(f"\n⏱️  Aggregate plot generation: {timedelta(seconds=int(aggregate_time))} ({aggregate_time:.2f}s)")
    print(f"\n✅ Completed {benchmark_name}_{mode}")
    print(f"   ⏱️  Total time: {timedelta(seconds=int(mode_total_time))} ({mode_total_time:.2f}s)")
    print(f"   Saved to: {mode_dir}")


def main():
    parser = argparse.ArgumentParser(description="Full contamination analysis")
    parser.add_argument('--data-dir', required=True, help='Directory with parquet files (recursive)')
    parser.add_argument('--output-dir', default='sanity_check_plots', help='Output directory')
    parser.add_argument('--benchmarks', nargs='+', default=['musr', 'mbpp'],
                       choices=['musr', 'mbpp'], help='Benchmarks to process')
    parser.add_argument('--modes', nargs='+', default=['input', 'output'],
                       choices=['input', 'output'], help='Modes to process')
    parser.add_argument('--batch-size', type=int, default=64, 
                       help='Batch size for GPU similarity computation (default: 64, increase for A100)')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    total_start_time = time.time()
    
    # Load all corpus embeddings once
    print("="*80)
    print("Loading all corpus embeddings...")
    print("="*80)
    corpus_embs, corpus_texts, corpus_ids, corpus_hash_ids, file_boundaries = load_all_parquet_data(data_dir)
    
    # Save file boundaries info
    with open(output_dir / "file_boundaries.json", 'w') as f:
        json.dump(file_boundaries, f, indent=2)
    
    # Load model once
    model_start = time.time()
    print("\nLoading embedding model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "nvidia/llama-embed-nemotron-8b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device).eval()
    model_time = time.time() - model_start
    print(f"⏱️  Model loading time: {timedelta(seconds=int(model_time))} ({model_time:.2f}s)")
    
    # Process each benchmark/mode combination
    for benchmark in args.benchmarks:
        for mode in args.modes:
            process_benchmark_mode(
                benchmark, mode, corpus_embs, corpus_texts, corpus_ids, corpus_hash_ids,
                model, tokenizer, device, output_dir, batch_size=args.batch_size
            )
    
    total_time = time.time() - total_start_time
    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print(f"⏱️  Total execution time: {timedelta(seconds=int(total_time))} ({total_time:.2f}s)")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

