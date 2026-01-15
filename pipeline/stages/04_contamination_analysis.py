#!/usr/bin/env python3
"""
Production Contamination Analysis - Distributed across 8 A100 40GB GPUs.

Hardware: 8x A100 40GB VRAM, 200GB system RAM
Strategy: Each GPU processes 1/8 of parquet files against ALL test points.
          Results merged at end for correct top-100 across full corpus.

Usage (recommended):
    # Launch all 8 ranks in parallel
    for i in {0..7}; do
        CUDA_VISIBLE_DEVICES=$i python production_contamination_analysis.py \
            --data-dir /lambda/nfs/embeddings/embedding_folder \
            --rank $i --world-size 8 &
    done
    wait

Or with torchrun:
    torchrun --nproc_per_node=8 production_contamination_analysis.py \
        --data-dir /lambda/nfs/embeddings/embedding_folder
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import duckdb
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import gzip
import time
from datetime import timedelta
from collections import defaultdict
import gc
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import logging
from datetime import datetime


# ============================================================================
# Configuration for 8x A100 40GB + 200GB RAM - SPEED OPTIMIZED (FP16)
# ============================================================================
DEFAULT_CONFIG = {
    'gpu_batch_size': 8192,         # A100 can handle very large test batches
    'corpus_gpu_chunk': 12_000_000, # ~24GB in FP16 for 4096-dim - leaves headroom
    'max_rows_per_block': 5_000_000,   # Checkpoint every 5M rows (more frequent flushes to avoid huge metadata)
    'world_size': 8,
    'gpu_memory_threshold': 0.92,   # Push closer to limit for speed
    'ram_memory_threshold': 0.88,   # Push closer to limit for speed
    'prefetch_files': 3,            # Number of files to prefetch
    'flush_stagger_delay': 0.5,     # Seconds to stagger flushes between ranks (reduced since npz is fast)
}


def get_gpu_memory_usage(device_id):
    """Get GPU memory usage as fraction (0-1)."""
    try:
        allocated = torch.cuda.memory_allocated(device_id)
        total = torch.cuda.get_device_properties(device_id).total_memory
        return allocated / total
    except:
        return 0.0


def get_ram_usage():
    """Get RAM usage as fraction (0-1)."""
    try:
        import psutil
        return psutil.virtual_memory().percent / 100.0
    except:
        return 0.0


def check_memory_pressure(device_id, gpu_threshold=0.90, ram_threshold=0.85):
    """Check if memory pressure is too high."""
    gpu_usage = get_gpu_memory_usage(device_id)
    ram_usage = get_ram_usage()
    return gpu_usage > gpu_threshold or ram_usage > ram_threshold, gpu_usage, ram_usage


def save_checkpoint(rank, file_idx, test_results, num_tests, checkpoint_dir, rows_processed_in_file=0):
    """Save checkpoint for resuming after crash."""
    checkpoint = {
        'rank': rank,
        'last_file_idx': file_idx,
        'chunk_indices': [test_results[i]['chunk_idx'] for i in range(num_tests)],
        'rows_processed_in_file': rows_processed_in_file  # Track progress within large files
    }
    checkpoint_file = checkpoint_dir / f"checkpoint_rank_{rank}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f)


def load_checkpoint(rank, checkpoint_dir):
    """Load checkpoint if exists."""
    checkpoint_file = checkpoint_dir / f"checkpoint_rank_{rank}.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None


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

        # Reservoir sampling - add all values, let reservoir handle downsampling
        for x in values:
            if len(self.sample_reservoir) < self.sample_size:
                self.sample_reservoir.append(float(x))
            else:
                # Replace random element (proper reservoir sampling)
                j = np.random.randint(0, len(self.sample_reservoir))
                self.sample_reservoir[j] = float(x)

    def get_stats(self):
        variance = self.M2 / self.n if self.n > 1 else 0.0
        std = np.sqrt(variance)
        if self.sample_reservoir:
            sorted_samples = np.sort(self.sample_reservoir)
            p99 = float(np.percentile(sorted_samples, 99))
            p95 = float(np.percentile(sorted_samples, 95))
            p90 = float(np.percentile(sorted_samples, 90))
            median = float(np.percentile(sorted_samples, 50))
        else:
            p99 = p95 = p90 = median = 0.0
        return {
            'max': self.max_val, 'min': self.min_val, 'mean': self.mean,
            'median': median, 'std': std, 'p99': p99, 'p95': p95, 'p90': p90,
            'count': self.n
        }


def embed_texts(texts, model, tokenizer, device, batch_size=8):
    """Embed texts in batches using FP16 (matches old run)."""
    embeddings = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts", leave=False):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                          max_length=8192, return_tensors='pt').to(device)
            out = model(**enc)
            mask = enc['attention_mask'].unsqueeze(-1).to(out[0].dtype)
            emb = (out[0] * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = F.normalize(emb, p=2, dim=1)
            embeddings.append(emb.float().cpu().numpy())
            del enc, out, mask, emb
            torch.cuda.empty_cache()
    return np.vstack(embeddings)


class ParquetPrefetcher:
    """Prefetch parquet files in background thread to overlap I/O with compute."""

    def __init__(self, file_list, prefetch_count=3):
        self.file_list = file_list
        self.prefetch_count = prefetch_count
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=prefetch_count)
        self.pending_futures = {}
        # Thread-local storage for DuckDB connections
        self._local = threading.local()

    def _get_connection(self):
        """Get thread-local DuckDB connection."""
        if not hasattr(self._local, 'con'):
            self._local.con = duckdb.connect(':memory:')
            self._local.con.execute("SET arrow_large_buffer_size=true")
        return self._local.con

    def _load_file(self, idx):
        """Load a file in background with thread-local connection."""
        if idx >= len(self.file_list):
            return None
        pf_path = self.file_list[idx]
        con = self._get_connection()
        return load_single_parquet(pf_path, con)

    def prefetch(self, current_idx):
        """Start prefetching next files."""
        for i in range(current_idx + 1, min(current_idx + 1 + self.prefetch_count, len(self.file_list))):
            if i not in self.pending_futures and i not in self.cache:
                self.pending_futures[i] = self.executor.submit(self._load_file, i)

    def get(self, idx):
        """Get file data, waiting if needed."""
        # Check cache first
        with self.cache_lock:
            if idx in self.cache:
                data = self.cache.pop(idx)
                return data

        # Check pending futures
        if idx in self.pending_futures:
            future = self.pending_futures.pop(idx)
            return future.result()

        # Load synchronously
        return self._load_file(idx)

    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=False)


def load_single_parquet(pf_path, con, max_rows_per_load=1_500_000):
    """Load a single parquet file and return embeddings, texts, IDs.

    For large files (>max_rows_per_load), returns a generator that yields chunks.
    For small files, returns the data directly.
    """
    import pyarrow.parquet as pq

    try:
        # First, check file size using pyarrow metadata (fast, no loading)
        pf = pq.ParquetFile(pf_path)
        total_rows = pf.metadata.num_rows
        num_row_groups = pf.metadata.num_row_groups

        # For large files, use chunked loading via row groups
        if total_rows > max_rows_per_load:
            return _load_parquet_chunked(pf_path, pf, con)

        # Original loading logic for smaller files
        cols_query = f"DESCRIBE SELECT * FROM read_parquet('{pf_path}')"
        cols_result = con.execute(cols_query).fetchall()
        cols = [row[0] for row in cols_result]

        # Support both 'embedding' and 'embeddings' column names
        emb_col_name = 'embeddings' if 'embeddings' in cols else ('embedding' if 'embedding' in cols else None)
        if emb_col_name is None:
            return None, None, None, None, 0

        text_col = None
        for c in ['text', 'content', 'paragraph', 'document']:
            if c in cols:
                text_col = c
                break

        # Use hash_id if available, otherwise fall back to id
        hash_id_col = 'hash_id' if 'hash_id' in cols else ('id' if 'id' in cols else None)
        id_col = 'id' if 'id' in cols else None

        select_parts = [emb_col_name]
        if text_col:
            select_parts.append(text_col)
        if id_col:
            select_parts.append(id_col)
        if hash_id_col and hash_id_col != id_col:  # Avoid selecting same column twice
            select_parts.append(hash_id_col)

        query = f"SELECT {', '.join(select_parts)} FROM read_parquet('{pf_path}')"

        import pyarrow as pa
        try:
            arrow_reader = con.execute(query).arrow()
            arrow_table = arrow_reader.read_all()
        except Exception as arrow_err:
            print(f"\n⚠️  Arrow error reading {pf_path.name}: {arrow_err}")
            return None, None, None, None, 0

        if arrow_table.num_rows == 0:
            return None, None, None, None, 0

        emb_col = arrow_table[emb_col_name]
        batch_embs = []
        PYARROW_MAX = int((2**31 - 1) * 0.9)
        dim = 4096  # Default

        for chunk_idx in range(emb_col.num_chunks):
            chunk = emb_col.chunk(chunk_idx)
            n = len(chunk)

            if chunk_idx == 0:
                emb_type = chunk.type
                dim = emb_type.list_size if hasattr(emb_type, 'list_size') else 4096

            max_safe_rows = PYARROW_MAX // dim

            if n > max_safe_rows:
                for sub_start in range(0, n, max_safe_rows):
                    sub_end = min(sub_start + max_safe_rows, n)
                    sub_chunk = chunk.slice(sub_start, sub_end - sub_start)
                    values = sub_chunk.flatten().to_numpy()
                    mat = values.reshape(len(sub_chunk), dim).astype(np.float32)
                    batch_embs.append(mat)
            else:
                values = chunk.flatten().to_numpy()
                mat = values.reshape(n, dim).astype(np.float32)
                batch_embs.append(mat)

        if not batch_embs:
            return None, None, None, None, 0

        mat = np.vstack(batch_embs) if len(batch_embs) > 1 else batch_embs[0]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / np.maximum(norms, 1e-9)

        num_rows = len(mat)
        all_texts = arrow_table[text_col].to_pylist() if text_col else [None] * num_rows
        all_ids = arrow_table[id_col].to_pylist() if id_col else [f"emb_{i}" for i in range(num_rows)]
        all_hash_ids = arrow_table[hash_id_col].to_pylist() if hash_id_col else [None] * num_rows

        return mat, all_texts, all_ids, all_hash_ids, num_rows

    except Exception as e:
        print(f"\n❌ Error loading {pf_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, 0


def _load_parquet_chunked(pf_path, pf, con):
    """Generator that yields chunks from a large parquet file by row groups.

    Returns a special marker dict indicating this is a chunked file.
    The caller must iterate over the chunks.
    """
    import pyarrow.parquet as pq

    # Return a marker indicating chunked loading is needed
    return {
        '_chunked': True,
        'path': pf_path,
        'num_row_groups': pf.metadata.num_row_groups,
        'total_rows': pf.metadata.num_rows
    }


def iter_parquet_row_groups(pf_path, chunk_size=100_000, start_offset=0):
    """Iterate over a large parquet file in chunks using DuckDB LIMIT/OFFSET.

    Uses smaller chunks (100K rows) to avoid PyArrow's 2GB array limit.
    Args:
        start_offset: Row number to start from (for resuming from checkpoint)
    """
    import pyarrow.parquet as pq

    # Get total row count and columns
    pf = pq.ParquetFile(pf_path)
    total_rows = pf.metadata.num_rows

    schema = pf.schema_arrow
    col_names = schema.names

    # Support both 'embedding' and 'embeddings' column names
    emb_col_name = 'embeddings' if 'embeddings' in col_names else ('embedding' if 'embedding' in col_names else None)
    if emb_col_name is None:
        return

    text_col = None
    for c in ['text', 'content', 'paragraph', 'document']:
        if c in col_names:
            text_col = c
            break

    # Use hash_id if available, otherwise fall back to id
    hash_id_col = 'hash_id' if 'hash_id' in col_names else ('id' if 'id' in col_names else None)
    id_col = 'id' if 'id' in col_names else None

    select_parts = [emb_col_name]
    if text_col:
        select_parts.append(text_col)
    if id_col:
        select_parts.append(id_col)
    if hash_id_col and hash_id_col != id_col:  # Avoid selecting same column twice
        select_parts.append(hash_id_col)

    # Use thread-local DuckDB connection
    con = duckdb.connect(':memory:')
    con.execute("SET arrow_large_buffer_size=true")

    dim = 4096
    PYARROW_MAX = int((2**31 - 1) * 0.9)

    # Start from start_offset to support resuming from checkpoint
    for offset in range(start_offset, total_rows, chunk_size):
        try:
            query = f"SELECT {', '.join(select_parts)} FROM read_parquet('{pf_path}') LIMIT {chunk_size} OFFSET {offset}"
            arrow_table = con.execute(query).arrow().read_all()

            if arrow_table.num_rows == 0:
                continue

            emb_col = arrow_table[emb_col_name]
            batch_embs = []

            for chunk_idx in range(emb_col.num_chunks):
                chunk = emb_col.chunk(chunk_idx)
                n = len(chunk)

                if chunk_idx == 0 and offset == 0:
                    emb_type = chunk.type
                    dim = emb_type.list_size if hasattr(emb_type, 'list_size') else 4096

                max_safe_rows = PYARROW_MAX // dim

                if n > max_safe_rows:
                    for sub_start in range(0, n, max_safe_rows):
                        sub_end = min(sub_start + max_safe_rows, n)
                        sub_chunk = chunk.slice(sub_start, sub_end - sub_start)
                        values = sub_chunk.flatten().to_numpy()
                        mat = values.reshape(len(sub_chunk), dim).astype(np.float32)
                        batch_embs.append(mat)
                else:
                    values = chunk.flatten().to_numpy()
                    mat = values.reshape(n, dim).astype(np.float32)
                    batch_embs.append(mat)

            if not batch_embs:
                continue

            mat = np.vstack(batch_embs) if len(batch_embs) > 1 else batch_embs[0]
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            mat = mat / np.maximum(norms, 1e-9)

            num_rows = len(mat)
            all_texts = arrow_table[text_col].to_pylist() if text_col else [None] * num_rows
            all_ids = arrow_table[id_col].to_pylist() if id_col else [f"emb_{offset}_{i}" for i in range(num_rows)]
            all_hash_ids = arrow_table[hash_id_col].to_pylist() if hash_id_col else [None] * num_rows

            yield mat, all_texts, all_ids, all_hash_ids, num_rows

            # Free memory
            del arrow_table, mat, batch_embs
            gc.collect()

        except Exception as e:
            print(f"\n⚠️  Error reading chunk at offset {offset} from {pf_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    con.close()


def load_benchmark(benchmark_name: str, mode: str):
    """Load benchmark data."""
    data = []

    # Handle MuSR splits: musr_murder_mysteries, musr_object_placements, musr_team_allocation
    if benchmark_name.startswith('musr_'):
        split_name = benchmark_name.replace('musr_', '')  # e.g., 'murder_mysteries'
        ds = load_dataset("TAUR-Lab/MuSR")
        if split_name not in ds:
            raise ValueError(f"Unknown MuSR split: {split_name}. Available: {list(ds.keys())}")
        for idx, item in enumerate(ds[split_name]):
            task_id = f"{benchmark_name}_{idx}"  # e.g., musr_murder_mysteries_0
            narrative = item.get('narrative', item.get('question', ''))
            answer = item.get('answer', '')
            data.append({'id': task_id, 'input': narrative, 'output': answer})

    # 'musr' name - loads ALL MuSR splits combined
    elif benchmark_name == 'musr':
        ds = load_dataset("TAUR-Lab/MuSR")
        # Load all splits: murder_mysteries, object_placements, team_allocation
        for split_name in ds.keys():
            for idx, item in enumerate(ds[split_name]):
                task_id = f"musr_{split_name}_{idx}"
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

    elif benchmark_name == 'zebralogic':
        ds = load_dataset("WildEval/ZebraLogic", "grid_mode")
        for item in ds['test']:
            task_id = str(item.get('id', item.get('puzzle_id', f"zebralogic_{len(data)}")))
            puzzle = item.get('puzzle', '')
            solution = item.get('solution', '')
            data.append({'id': task_id, 'input': puzzle, 'output': solution})

    elif benchmark_name == 'codeforces':
        import pandas as pd
        # Load from codeforces_uniform_recent.csv in pipeline directory
        csv_path = Path(__file__).parent.parent / 'codeforces_uniform_recent.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"Codeforces CSV not found at {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"  Loading {len(df)} Codeforces problems...")
        
        for _, row in df.iterrows():
            problem_id = str(row['id'])
            
            # Concatenate problem text: description + input_format + output_format
            parts = []
            for col in ['description', 'input_format', 'output_format']:
                val = row.get(col, '')
                if pd.notna(val) and str(val).strip() and str(val).lower() != 'nan':
                    parts.append(str(val))
            text = '\n\n'.join(parts)
            
            data.append({
                'id': problem_id,
                'input': text,
                'output': '',
                'elo_bin': row.get('elo_bin'),
                'rating': row.get('rating')
            })

    # Build return values
    texts, ids, metadata = [], [], {}
    for item in data:
        if benchmark_name == 'musr' or benchmark_name.startswith('musr_'):
            texts.append(f"{item['input']}\n\n{item['output']}")
        elif benchmark_name == 'codeforces':
            texts.append(item['input'])  # Already concatenated
        elif benchmark_name == 'mbpp' or benchmark_name == 'zebralogic':
            if mode == 'input':
                texts.append(item['input'])
            elif mode == 'output':
                texts.append(item['output'])
            else:
                texts.append(f"{item['input']}\n\n{item['output']}")
        else:
            texts.append(f"{item['input']}\n\n{item['output']}")
        
        ids.append(item['id'])
        
        # Store metadata if present (for codeforces)
        if 'elo_bin' in item or 'rating' in item:
            metadata[item['id']] = {
                'elo_bin': item.get('elo_bin'),
                'rating': item.get('rating')
            }

    return texts, ids, metadata


def flush_buffers(test_results, num_tests, shared_hash_id_buffer, rank=None):
    """Flush all similarity buffers AND hash_ids to disk.

    Saves similarities per test, hash_ids ONCE (shared across all tests).
    Uses uncompressed .npy format for speed on network filesystems.
    """
    # Stagger flushes by rank to avoid I/O contention
    if rank is not None:
        stagger_delay = DEFAULT_CONFIG.get('flush_stagger_delay', 2.0)
        time.sleep(rank * stagger_delay)

    # Concatenate shared hash_ids once
    if not shared_hash_id_buffer:
        return  # Nothing to flush

    block_hash_ids = np.concatenate(shared_hash_id_buffer)

    # Get chunk index from first test (all tests have same chunk_idx)
    chunk_idx = test_results[0]['chunk_idx']

    # Save hash_ids ONCE for this chunk (shared by all tests)
    hash_ids_dir = test_results[0]['chunk_dir'].parent.parent / "shared_hash_ids"
    hash_ids_dir.mkdir(parents=True, exist_ok=True)
    hash_ids_file = hash_ids_dir / f"chunk_{chunk_idx:04d}_hash_ids.npy"
    np.save(hash_ids_file, block_hash_ids)  # Uncompressed for speed

    # Save similarities for each test (without duplicating hash_ids)
    for test_idx in range(num_tests):
        if not test_results[test_idx]['similarity_buffer']:
            continue

        block_sims = np.concatenate(test_results[test_idx]['similarity_buffer'])
        chunk_file = test_results[test_idx]['chunk_dir'] / f"chunk_{chunk_idx:04d}_sims.npy"

        # Save only similarities (hash_ids are in shared file)
        np.save(chunk_file, block_sims)  # Uncompressed for speed

        test_results[test_idx]['similarity_buffer'] = []
        test_results[test_idx]['chunk_idx'] += 1

    # Clear shared buffer after flush
    shared_hash_id_buffer.clear()


def setup_logging(rank, output_dir):
    """Setup logging for this rank."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(f"rank_{rank}")
    logger.setLevel(logging.DEBUG)

    # File handler - detailed logs
    fh = logging.FileHandler(log_dir / f"rank_{rank}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))

    # Console handler - important logs only
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(f'[R{rank}] %(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def run_worker(rank, world_size, args):
    """Run a single worker (one GPU) with OOM protection and checkpointing."""

    # --- Configuration Loading for Dynamic Naming ---
    import yaml
    PIPELINE_ROOT = Path(__file__).parent.parent
    
    # Only load config if PIPELINE_CONFIG is explicitly set and non-empty
    config_path_env = os.environ.get("PIPELINE_CONFIG", "").strip()
    
    if config_path_env:
        CONFIG_FILE = Path(config_path_env)
        if not CONFIG_FILE.is_absolute():
            CONFIG_FILE = PIPELINE_ROOT / config_path_env

        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                config = yaml.safe_load(f)
            
            DATASET_SHORT_NAME = config.get('pipeline', {}).get('dataset_short_name', config.get('pipeline', {}).get('name', 'dataset'))
            pct_val = int(config.get('chunking', {}).get('paragraph_sample_percentage', 0.01) * 100)
            pct_str = f"{pct_val}pct"
            
            # Construct Output Dir from config (only if not specified on CLI)
            if not hasattr(args, 'output_dir') or not args.output_dir:
                output_folder_name = f"contamination_{DATASET_SHORT_NAME}_{pct_str}"
                args.output_dir = str(PIPELINE_ROOT / "results" / output_folder_name)
                print(f"Auto-configured Output Dir (from config): {args.output_dir}")
            
            # Note: --data-dir is required on CLI, so we DON'T override it from config
            # Only set corpus_file from config if not already set
            corpus_file_config = config.get('analysis', {}).get('corpus_file', '')
            if corpus_file_config and not hasattr(args, 'corpus_file'):
                args.corpus_file = corpus_file_config
                print(f"Auto-configured Corpus File (from config): {args.corpus_file}")
        else:
            print(f"Warning: Config not found at {CONFIG_FILE}, using CLI args.")
    else:
        # No config - use CLI args as-is (already set in main())
        print(f"Using CLI output directory: {args.output_dir}")

    # Setup logging first
    log = setup_logging(rank, args.output_dir)

    # Set CUDA device - use cuda:0 since CUDA_VISIBLE_DEVICES restricts to one GPU
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # A100 optimizations
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for even faster matmul
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Auto-tune for best kernel

    log.info("=" * 60)
    log.info(f"WORKER STARTED - Rank {rank}/{world_size}")
    log.info(f"GPU: {torch.cuda.get_device_name(0)} (physical GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')})")
    log.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    log.info(f"TF32: {torch.backends.cuda.matmul.allow_tf32} | Precision: FP16")
    log.info("=" * 60)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint directory
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    worker_start = time.time()

    # Get parquet file(s) to process
    log.info("Scanning for parquet files...")

    # Use specific corpus_file if provided, otherwise scan directory
    if hasattr(args, 'corpus_file') and args.corpus_file:
        # Handle glob patterns (e.g., "*.parquet" or "embeddings_rank_*.parquet")
        import glob
        pattern = str(data_dir / args.corpus_file)
        matched_files = sorted(glob.glob(pattern))
        if matched_files:
            parquet_files = [Path(f) for f in matched_files]
            log.info(f"Using corpus file pattern: {args.corpus_file} (matched {len(parquet_files)} files)")
        else:
            log.error(f"No files matched corpus file pattern: {args.corpus_file}")
            return rank, 0
    else:
        all_parquets = sorted(data_dir.rglob("*.parquet"))
        # Filter for only paragraph parquets (exclude document parquets)
        parquet_files = [pf for pf in all_parquets if '/paragraphs/' in str(pf)]
        if len(parquet_files) < len(all_parquets):
            log.info(f"Scanning directory for parquet files (filtered {len(all_parquets) - len(parquet_files)} non-paragraph files)")
        else:
            log.info(f"Scanning directory for all parquet files")
    
    total_files = len(parquet_files)
    log.info(f"Found {total_files} total parquet files")

    # Each rank processes files where file_idx % world_size == rank
    # Apply global resume point first (--resume-from-file)
    global_resume = args.resume_from_file
    my_file_indices = [i for i in range(total_files) if i % world_size == rank and i >= global_resume]
    my_files = [parquet_files[i] for i in my_file_indices]

    if global_resume > 0:
        skipped_for_rank = len([i for i in range(total_files) if i % world_size == rank and i < global_resume])
        log.info(f"📂 Assigned {len(my_files)} files (skipping {skipped_for_rank} before resume point {global_resume})")
    else:
        log.info(f"📂 Assigned {len(my_files)}/{total_files} parquet files")

    # Check for existing checkpoint (local to this rank)
    checkpoint = load_checkpoint(rank, checkpoint_dir)
    resume_from_idx = 0
    resume_file_offset = 0
    if checkpoint:
        rows_in_file = checkpoint.get('rows_processed_in_file', 0)
        if rows_in_file > 0:
            # Resume from middle of the checkpointed file
            resume_from_idx = checkpoint['last_file_idx']
            resume_file_offset = rows_in_file
            log.info(f"🔄 RESUMING from file {resume_from_idx} at row {resume_file_offset:,}")
        else:
            # Checkpointed file was complete, start next file
            resume_from_idx = checkpoint['last_file_idx'] + 1
            resume_file_offset = 0
            log.info(f"🔄 RESUMING from local checkpoint index {resume_from_idx}")

    # Build global offset map (all ranks need this)
    log.info("Building global offset map...")
    con = duckdb.connect(':memory:')
    con.execute("SET arrow_large_buffer_size=true")

    file_offsets = {}
    file_row_counts = {}
    current_offset = 0

    for pf in tqdm(parquet_files, desc=f"[R{rank}] Indexing", disable=rank != 0):
        try:
            count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{pf}')").fetchone()[0]
        except:
            count = 0
        file_offsets[str(pf)] = current_offset
        file_row_counts[str(pf)] = count
        current_offset += count

    total_corpus_size = current_offset
    log.info(f"📊 Total corpus: {total_corpus_size:,} embeddings across {total_files} files")

    # Load embedding model in FP16 (matches old run)
    log.info("🤖 Loading embedding model (FP16)...")
    model_name = "nvidia/llama-embed-nemotron-8b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # FP16 to match old run
        trust_remote_code=True
    ).to(device).eval()
    log.info("✅ Model loaded")

    # Load and embed benchmarks
    log.info("📚 Loading benchmarks...")
    all_test_data = []
    all_test_texts = []

    for benchmark in args.benchmarks:
        modes_to_process = ['input_output'] if (benchmark == 'musr' or benchmark.startswith('musr_') or benchmark == 'mbpp' or benchmark == 'zebralogic' or benchmark == 'codeforces') else args.modes
        for mode in modes_to_process:
            test_texts, test_ids, test_metadata = load_benchmark(benchmark, mode)
            log.debug(f"  {benchmark.upper()}/{mode}: {len(test_texts)} test points")
            for text, test_id in zip(test_texts, test_ids):
                meta = test_metadata.get(test_id, {})
                all_test_data.append({
                    'benchmark': benchmark,
                    'mode': mode,
                    'test_id': test_id,
                    'text': text,
                    'global_idx': len(all_test_texts),
                    'elo_bin': meta.get('elo_bin'),
                    'rating': meta.get('rating'),
                })
                all_test_texts.append(text)

    num_tests = len(all_test_texts)
    log.info(f"🔢 Embedding {num_tests} test points...")
    all_test_embs = embed_texts(all_test_texts, model, tokenizer, device)
    log.info(f"✅ Test embeddings: {all_test_embs.shape}")

    # Free model memory
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    log.info("🧹 Model unloaded, GPU memory freed")

    # Initialize storage
    similarities_dir = output_dir / "temp_similarities" / f"rank_{rank}"
    similarities_dir.mkdir(parents=True, exist_ok=True)

    test_results = []
    for i in range(num_tests):
        chunk_dir = similarities_dir / f"test_{i}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        # Count existing chunks (both old .npz and new .npy formats)
        existing_new = list(chunk_dir.glob("chunk_*_sims.npy"))
        existing_old = list(chunk_dir.glob("chunk_*.npz"))
        existing_chunks = existing_new + existing_old
        test_results.append({
            'chunk_dir': chunk_dir,
            'similarity_buffer': [],
            'chunk_idx': len(existing_chunks)
        })

    # Shared hash_id buffer (same for all tests) - saves 1269x memory duplication!
    shared_hash_id_buffer = []

    # Process my parquet files with prefetching
    files_to_process = my_files[resume_from_idx:]
    log.info("=" * 60)
    log.info(f"🚀 STARTING PROCESSING: {len(files_to_process)} files")
    log.info("=" * 60)

    # Keep test embeddings on GPU for the entire run (they're small) - FP16 to match old run
    test_embs_gpu = torch.from_numpy(all_test_embs).to(device).half()
    log.info(f"📍 Test embeddings on GPU: {test_embs_gpu.shape} (FP16)")

    rows_in_buffer = 0
    files_processed = 0
    embeddings_processed = 0
    max_rows_per_block = args.max_rows_per_block  # Checkpoint by rows, not files
    process_start = time.time()
    last_log_time = process_start

    # Create prefetcher for async I/O (uses thread-local DuckDB connections)
    prefetcher = ParquetPrefetcher(files_to_process, prefetch_count=DEFAULT_CONFIG.get('prefetch_files', 3))

    pbar = tqdm(range(len(files_to_process)), desc=f"[R{rank}]", position=rank, ncols=100)

    def process_corpus_chunk(corpus_embs, corpus_hash_ids, corpus_gpu_chunk_size, test_embs_gpu, test_results, num_tests, device, rank, file_idx, checkpoint_dir):
        """Process a corpus embeddings array in GPU chunks. Returns (rows_processed, should_continue)."""
        nonlocal rows_in_buffer

        num_corpus = len(corpus_embs)
        rows_processed = 0

        for corpus_start in range(0, num_corpus, corpus_gpu_chunk_size):
            corpus_end = min(corpus_start + corpus_gpu_chunk_size, num_corpus)
            corpus_chunk = corpus_embs[corpus_start:corpus_end]
            hash_id_chunk = corpus_hash_ids[corpus_start:corpus_end]

            try:
                # FP16 to match old run - use non_blocking for async transfer
                corpus_gpu = torch.from_numpy(corpus_chunk).to(device, non_blocking=True).half()

                # Single massive matmul using pre-loaded test embeddings
                sims = torch.matmul(test_embs_gpu, corpus_gpu.t())

                # Non-blocking copy back to CPU
                sims_cpu = sims.float().cpu()
                torch.cuda.synchronize()
                sims_np = sims_cpu.numpy()

                # Store similarities per test, hash_ids once (shared)
                for test_idx in range(num_tests):
                    test_results[test_idx]['similarity_buffer'].append(sims_np[test_idx])
                shared_hash_id_buffer.append(np.array(hash_id_chunk))

                rows_processed += len(corpus_chunk)

                del sims, sims_cpu, sims_np
                del corpus_gpu

            except torch.cuda.OutOfMemoryError as oom_err:
                print(f"\n[Rank {rank}] ⚠️ GPU OOM! Flushing buffers...")
                torch.cuda.empty_cache()
                gc.collect()
                flush_buffers(test_results, num_tests, shared_hash_id_buffer, rank)
                save_checkpoint(rank, file_idx - 1, test_results, num_tests, checkpoint_dir)
                rows_in_buffer = 0
                continue

            del corpus_chunk

        return rows_processed

    for local_idx in pbar:
        file_idx = resume_from_idx + local_idx
        pf_path = files_to_process[local_idx]
        global_offset = file_offsets[str(pf_path)]

        # Start prefetching next files
        prefetcher.prefetch(local_idx)

        # Track rows processed in current file (for checkpoint resume)
        rows_processed_in_current_file = 0

        # Determine if we need to resume from middle of this file
        file_start_offset = resume_file_offset if local_idx == 0 else 0

        try:
            result = prefetcher.get(local_idx)
            if result is None:
                continue

            # Check if this is a large file that needs chunked loading
            if isinstance(result, dict) and result.get('_chunked'):
                # Large file - process by row groups
                if file_start_offset > 0:
                    log.info(f"📦 Large file detected: {pf_path.name} ({result['total_rows']:,} rows, {result['num_row_groups']} row groups) - resuming from row {file_start_offset:,}")
                else:
                    log.info(f"📦 Large file detected: {pf_path.name} ({result['total_rows']:,} rows, {result['num_row_groups']} row groups)")

                for corpus_embs, corpus_texts, corpus_ids, corpus_hash_ids, num_rows in iter_parquet_row_groups(pf_path, start_offset=file_start_offset):
                    rows_processed = process_corpus_chunk(
                        corpus_embs, corpus_hash_ids, args.corpus_gpu_chunk, test_embs_gpu,
                        test_results, num_tests, device, rank, file_idx, checkpoint_dir
                    )
                    embeddings_processed += rows_processed
                    rows_in_buffer += rows_processed
                    rows_processed_in_current_file += rows_processed

                    del corpus_embs, corpus_texts, corpus_ids, corpus_hash_ids
                    gc.collect()

                    # Check memory and flush if needed after each row group
                    mem_pressure, gpu_use, ram_use = check_memory_pressure(0)
                    if mem_pressure or rows_in_buffer >= max_rows_per_block:
                        pbar.set_postfix({'flushing': f'{rows_in_buffer/1e6:.1f}M (large file)'})
                        flush_buffers(test_results, num_tests, shared_hash_id_buffer, rank)
                        # Save checkpoint with current progress in this file
                        save_checkpoint(rank, file_idx, test_results, num_tests, checkpoint_dir,
                                      rows_processed_in_file=file_start_offset + rows_processed_in_current_file)
                        rows_in_buffer = 0
                        gc.collect()
                        torch.cuda.empty_cache()

                # File complete - save checkpoint with rows_processed_in_file=0 to mark completion
                save_checkpoint(rank, file_idx, test_results, num_tests, checkpoint_dir, rows_processed_in_file=0)
                files_processed += 1
                continue

            # Normal file processing
            corpus_embs, corpus_texts, corpus_ids, corpus_hash_ids, num_rows = result

            if corpus_embs is None:
                continue

            num_corpus = len(corpus_embs)
            embeddings_processed += num_corpus

            # Process in GPU chunks with OOM protection
            corpus_gpu_chunk_size = args.corpus_gpu_chunk

            for corpus_start in range(0, num_corpus, corpus_gpu_chunk_size):
                corpus_end = min(corpus_start + corpus_gpu_chunk_size, num_corpus)
                corpus_chunk = corpus_embs[corpus_start:corpus_end]
                hash_id_chunk = corpus_hash_ids[corpus_start:corpus_end]

                try:
                    # FP16 to match old run - use non_blocking for async transfer
                    corpus_gpu = torch.from_numpy(corpus_chunk).to(device, non_blocking=True).half()

                    # Single massive matmul using pre-loaded test embeddings
                    # test_embs_gpu is already on GPU from earlier
                    sims = torch.matmul(test_embs_gpu, corpus_gpu.t())

                    # Non-blocking copy back to CPU
                    sims_cpu = sims.float().cpu()
                    torch.cuda.synchronize()  # Ensure compute done before reusing GPU memory
                    sims_np = sims_cpu.numpy()

                    # Store similarities per test, hash_ids once (shared)
                    for test_idx in range(num_tests):
                        test_results[test_idx]['similarity_buffer'].append(sims_np[test_idx])
                    shared_hash_id_buffer.append(np.array(hash_id_chunk))

                    del sims, sims_cpu, sims_np
                    del corpus_gpu

                except torch.cuda.OutOfMemoryError as oom_err:
                    print(f"\n[Rank {rank}] ⚠️ GPU OOM! Flushing buffers and retrying...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    flush_buffers(test_results, num_tests, shared_hash_id_buffer, rank)
                    save_checkpoint(rank, file_idx - 1, test_results, num_tests, checkpoint_dir)
                    rows_in_buffer = 0
                    # Retry with smaller chunk
                    continue

                del corpus_chunk

            rows_in_buffer += num_corpus
            files_processed += 1

            del corpus_embs, corpus_texts, corpus_ids, corpus_hash_ids
            gc.collect()

            # Check memory pressure
            mem_pressure, gpu_use, ram_use = check_memory_pressure(0)  # Always device 0
            if mem_pressure:
                pbar.set_postfix({'mem_flush': f'GPU:{gpu_use:.0%} RAM:{ram_use:.0%}'})
                flush_buffers(test_results, num_tests, shared_hash_id_buffer, rank)
                save_checkpoint(rank, file_idx, test_results, num_tests, checkpoint_dir)
                rows_in_buffer = 0
                gc.collect()
                torch.cuda.empty_cache()

            # Flush to disk and checkpoint when we hit row limit (safer than file-based)
            elif rows_in_buffer >= max_rows_per_block:
                pbar.set_postfix({'flushing': f'{rows_in_buffer:,} rows'})
                flush_buffers(test_results, num_tests, shared_hash_id_buffer, rank)
                save_checkpoint(rank, file_idx, test_results, num_tests, checkpoint_dir)
                rows_in_buffer = 0
                gc.collect()

            # Update progress bar
            pbar.set_postfix({
                'embs': f'{embeddings_processed/1e6:.1f}M',
                'buf': f'{rows_in_buffer/1e6:.1f}M'
            })

            # Periodic detailed logging (every 30 seconds)
            now = time.time()
            if now - last_log_time > 30:
                elapsed = now - process_start
                rate = embeddings_processed / elapsed if elapsed > 0 else 0
                gpu_mem = get_gpu_memory_usage(0)  # Always device 0
                ram_mem = get_ram_usage()

                global_file_idx = my_file_indices[resume_from_idx + local_idx] if (resume_from_idx + local_idx) < len(my_file_indices) else 0
                log.info(f"📈 Progress: {files_processed}/{len(files_to_process)} files | "
                        f"{embeddings_processed/1e6:.1f}M embs | "
                        f"{rate/1e6:.2f}M/s | "
                        f"GPU:{gpu_mem:.0%} RAM:{ram_mem:.0%} | "
                        f"Global file ~{global_file_idx}")
                last_log_time = now

        except Exception as e:
            log.error(f"❌ Error processing {pf_path.name}: {e}")
            import traceback
            log.debug(traceback.format_exc())
            # Save checkpoint before continuing
            save_checkpoint(rank, file_idx - 1, test_results, num_tests, checkpoint_dir)
            continue

    # Final flush
    if rows_in_buffer > 0:
        log.info(f"💾 Final flush: {rows_in_buffer:,} rows")
        flush_buffers(test_results, num_tests, shared_hash_id_buffer, rank)
        save_checkpoint(rank, file_idx, test_results, num_tests, checkpoint_dir)

    # Cleanup
    prefetcher.cleanup()
    del test_embs_gpu
    torch.cuda.empty_cache()
    con.close()

    worker_time = time.time() - worker_start
    avg_rate = embeddings_processed / worker_time if worker_time > 0 else 0

    log.info("=" * 60)
    log.info(f"✅ WORKER {rank} COMPLETE!")
    log.info(f"   Files processed: {files_processed}")
    log.info(f"   Embeddings: {embeddings_processed:,}")
    log.info(f"   Time: {timedelta(seconds=int(worker_time))}")
    log.info(f"   Avg rate: {avg_rate/1e6:.2f}M embeddings/sec")
    log.info("=" * 60)

    # Write completion marker with stats
    completion_info = {
        'rank': rank,
        'files_processed': files_processed,
        'embeddings_processed': embeddings_processed,
        'time_seconds': worker_time,
        'avg_rate': avg_rate
    }
    with open(similarities_dir / "COMPLETE.json", 'w') as f:
        json.dump(completion_info, f)

    return rank, embeddings_processed


# Corpus ID mapping is NO LONGER USED
# Hash IDs are now saved in shared_hash_ids/*.npy files (one per chunk, shared across all tests)
# Similarities are saved per-test in test_*/chunk_*_sims.npy files


def process_single_test_top1pct(args_tuple):
    """
    Sample 100 random items from top 1% of similarity scores for a test point.
    Used for semantic duplicate analysis.
    """
    test_data, output_dir, world_size, id_to_text, n_samples = args_tuple
    
    test_id = test_data['test_id']
    test_text = test_data['text']
    global_idx = test_data['global_idx']
    benchmark = test_data['benchmark']
    mode = test_data['mode']
    
    # Collect chunk metadata from ALL ranks
    chunk_metadata = []
    for r in range(world_size):
        chunk_dir = output_dir / "temp_similarities" / f"rank_{r}" / f"test_{global_idx}"
        hash_ids_dir = output_dir / "temp_similarities" / f"rank_{r}" / "shared_hash_ids"
        
        chunk_files_npy = sorted(chunk_dir.glob("chunk_*_sims.npy")) if chunk_dir.exists() else []
        chunk_files_npz = sorted(chunk_dir.glob("chunk_*.npz")) if chunk_dir.exists() else []
        
        for chunk_file in chunk_files_npy:
            chunk_num = chunk_file.stem.split('_')[1]
            hash_ids_file = hash_ids_dir / f"chunk_{chunk_num}_hash_ids.npy"
            if hash_ids_file.exists():
                chunk_metadata.append(('npy', chunk_file, hash_ids_file))
        
        for chunk_file in chunk_files_npz:
            chunk_metadata.append(('npz', chunk_file, None))
    
    if not chunk_metadata:
        return None
    
    # Pass 1: Load ALL similarities to find 99th percentile
    all_sims = []
    all_hash_ids = []
    
    for fmt, sim_file, hash_file in chunk_metadata:
        try:
            if fmt == 'npy':
                sims = np.load(sim_file)  # Full load needed for percentile
                hash_ids = np.load(hash_file) if hash_file else None
            elif fmt == 'npz':
                data = np.load(sim_file, allow_pickle=True)
                sims = data['similarities']
                hash_ids = data['hash_ids']
            
            all_sims.append(sims)
            if hash_ids is not None:
                all_hash_ids.append(hash_ids)
        except Exception:
            continue
    
    if not all_sims:
        return None
    
    sims = np.concatenate(all_sims)
    hash_ids = np.concatenate(all_hash_ids) if all_hash_ids else None
    
    if len(sims) == 0 or hash_ids is None:
        return None
    
    # Find 99th percentile (top 1% threshold)
    threshold = np.percentile(sims, 99)
    
    # Get indices in top 1%
    top1pct_mask = sims >= threshold
    top1pct_indices = np.where(top1pct_mask)[0]
    
    if len(top1pct_indices) == 0:
        return None
    
    # Sample up to n_samples from top 1%
    sample_size = min(n_samples, len(top1pct_indices))
    sampled_indices = np.random.choice(top1pct_indices, size=sample_size, replace=False)
    
    # Build results
    samples = []
    for idx in sampled_indices:
        hash_id = str(hash_ids[idx])
        corpus_text = id_to_text.get(hash_id, '') if id_to_text else ''
        samples.append({
            'test_id': test_id,
            'test_text': test_text,
            'corpus_id': hash_id,
            'corpus_text': corpus_text,
            'similarity': float(sims[idx]),
            'benchmark': benchmark,
            'threshold_99pct': float(threshold),
            'total_embeddings': len(sims),
            'top1pct_count': len(top1pct_indices)
        })
    
    return samples


def process_single_test(args_tuple):
    """Process a single test point - designed for parallel execution."""
    test_data, output_dir, world_size = args_tuple

    import heapq
    test_id = test_data['test_id']
    test_text = test_data['text']
    global_idx = test_data['global_idx']
    benchmark = test_data['benchmark']
    mode = test_data['mode']

    # Collect chunk metadata from ALL ranks (don't load data yet!)
    chunk_metadata = []  # List of (sim_file, hash_file) tuples
    for r in range(world_size):
        chunk_dir = output_dir / "temp_similarities" / f"rank_{r}" / f"test_{global_idx}"

        # Support both new .npy and old .npz formats
        chunk_files_npy = sorted(chunk_dir.glob("chunk_*_sims.npy"))
        chunk_files_npz = sorted(chunk_dir.glob("chunk_*.npz"))

        # Collect new format (.npy files)
        for chunk_file in chunk_files_npy:
            chunk_num = chunk_file.stem.split('_')[1]
            hash_ids_dir = chunk_dir.parent.parent / "shared_hash_ids"
            hash_ids_file = hash_ids_dir / f"chunk_{chunk_num}_hash_ids.npy"
            if hash_ids_file.exists():
                chunk_metadata.append(('npy', chunk_file, hash_ids_file))

        # Collect old format (.npz files)
        for chunk_file in chunk_files_npz:
            chunk_metadata.append(('npz', chunk_file, None))

    if not chunk_metadata:
        return None  # Skip this test

    # STREAMING APPROACH: Find top-1000 WITHOUT concatenating everything
    top_k = 1000
    global_top_k = []  # Min-heap of (score, chunk_idx, local_idx)
    total_count = 0

    # For aggregate stats collection
    chunk_stats = []

    # Pass 1: Find top-K across all chunks (streaming, no concatenation)
    for chunk_idx, chunk_info in enumerate(chunk_metadata):
        fmt, sim_file, hash_file = chunk_info

        try:
            if fmt == 'npy':
                sims = np.load(sim_file, mmap_mode='r')  # Memory-mapped, not loaded
            elif fmt == 'npz':
                data = np.load(sim_file, allow_pickle=True)
                sims = data['similarities']
        except (ValueError, OSError, IOError) as e:
            # Skip corrupted or incomplete chunk files
            continue

        total_count += len(sims)

        # Find local top-K in this chunk
        chunk_top_k = min(top_k, len(sims))
        if chunk_top_k > 0:
            local_top_indices = np.argpartition(sims, -chunk_top_k)[-chunk_top_k:]

            # Add to global heap
            for local_idx in local_top_indices:
                score = float(sims[local_idx])
                if len(global_top_k) < top_k:
                    heapq.heappush(global_top_k, (score, chunk_idx, int(local_idx)))
                elif score > global_top_k[0][0]:
                    heapq.heapreplace(global_top_k, (score, chunk_idx, int(local_idx)))

        # Collect stats for aggregate (lightweight - just min/max/sum)
        chunk_stats.append({
            'min': float(np.min(sims)),
            'max': float(np.max(sims)),
            'sum': float(np.sum(sims)),
            'count': len(sims)
        })

        del sims
        if fmt == 'npz':
            del data

    # Sort heap to get final top-K in descending order
    global_top_k.sort(reverse=True)

    # Pass 2: Retrieve hash_ids for top-K results only
    topk_matches = []
    for rank, (score, chunk_idx, local_idx) in enumerate(global_top_k, 1):
        fmt, sim_file, hash_file = chunk_metadata[chunk_idx]

        try:
            if fmt == 'npy':
                hash_ids = np.load(hash_file, mmap_mode='r')
                hash_id = str(hash_ids[local_idx])
                del hash_ids
            elif fmt == 'npz':
                data = np.load(sim_file, allow_pickle=True)
                hash_id = str(data['hash_ids'][local_idx])
                del data

            topk_matches.append({
                'rank': rank,
                'score': score,
                'corpus_id': hash_id,
            })
        except (ValueError, OSError, IOError, IndexError):
            # Skip if we can't retrieve hash_id
            continue

    # Compute per-test stats from aggregate stats (already computed in Pass 1)
    # For per-test stats, do one more streaming pass (cheap - just scans, no allocation)
    per_test_stats = {'max': float('-inf'), 'min': float('inf'), 'sum': 0.0, 'count': 0}
    all_vals_for_median = []  # Only for median computation

    for chunk_info in chunk_metadata:
        fmt, sim_file, hash_file = chunk_info

        try:
            if fmt == 'npy':
                sims = np.load(sim_file, mmap_mode='r')
            elif fmt == 'npz':
                data = np.load(sim_file, allow_pickle=True)
                sims = data['similarities']
        except (ValueError, OSError, IOError):
            continue

        per_test_stats['max'] = max(per_test_stats['max'], float(np.max(sims)))
        per_test_stats['min'] = min(per_test_stats['min'], float(np.min(sims)))
        per_test_stats['sum'] += float(np.sum(sims))
        per_test_stats['count'] += len(sims)

        # Sample for aggregate histogram (100K per test for better accuracy)
        # With 1000 tests × 100K samples = 100M total (~400MB memory)
        sample_size = min(100000, len(sims))
        if len(sims) <= sample_size:
            all_vals_for_median.extend(sims.tolist())
        else:
            indices = np.random.choice(len(sims), sample_size, replace=False)
            all_vals_for_median.extend(sims[indices].tolist())

        del sims
        if fmt == 'npz':
            del data

    per_test_stats['mean'] = per_test_stats['sum'] / per_test_stats['count']
    per_test_stats['median'] = float(np.median(all_vals_for_median)) if all_vals_for_median else 0.0

    # For std, need another pass (or use two-pass algorithm)
    sum_sq_diff = 0.0
    for chunk_info in chunk_metadata:
        fmt, sim_file, hash_file = chunk_info

        try:
            if fmt == 'npy':
                sims = np.load(sim_file, mmap_mode='r')
            elif fmt == 'npz':
                data = np.load(sim_file, allow_pickle=True)
                sims = data['similarities']
        except (ValueError, OSError, IOError):
            continue

        sum_sq_diff += float(np.sum((sims - per_test_stats['mean']) ** 2))

        del sims
        if fmt == 'npz':
            del data

    per_test_stats['std'] = float(np.sqrt(sum_sq_diff / per_test_stats['count']))

    # Save JSON
    result = {
        'test_id': test_id,
        'test_text': test_text,
        'benchmark': benchmark,
        'mode': mode,
        'total_embeddings': total_count,
        'top_1000': topk_matches,
        'stats': per_test_stats
    }

    # Sanitize test_id to avoid filesystem issues (e.g., codeforces IDs have slashes)
    mode_dir = output_dir / f"{benchmark}_{mode}"
    mode_dir.mkdir(parents=True, exist_ok=True)
    safe_test_id = test_id.replace('/', '_')
    with open(mode_dir / f"{safe_test_id}_top1000.json", 'w') as f:
        json.dump(result, f, indent=2)

    # Return data for aggregate statistics
    top_100_scores = [score for score, _, _ in global_top_k[:100]]

    return {
        'test_id': test_id,
        'chunk_stats': chunk_stats,
        'top_100_scores': top_100_scores,
        'similarity_samples': all_vals_for_median  # Return sampled values for histogram
    }


def run_merger(args, world_size):
    """Merge results from all workers (run on rank 0 after all workers complete)."""

    # --- Configuration Loading for Dynamic Naming (Match run_worker) ---
    import yaml
    PIPELINE_ROOT = Path(__file__).parent.parent

    # Only load config if PIPELINE_CONFIG is explicitly set and non-empty
    config_path_env = os.environ.get("PIPELINE_CONFIG", "").strip()

    if config_path_env:
        CONFIG_FILE = Path(config_path_env)
        if not CONFIG_FILE.is_absolute():
            CONFIG_FILE = PIPELINE_ROOT / config_path_env

        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                config = yaml.safe_load(f)

            DATASET_SHORT_NAME = config.get('pipeline', {}).get('dataset_short_name', config.get('pipeline', {}).get('name', 'dataset'))
            pct_val = int(config.get('chunking', {}).get('paragraph_sample_percentage', 0.01) * 100)
            pct_str = f"{pct_val}pct"

            # Construct Output Dir from config (only if not specified on CLI)
            if not hasattr(args, 'output_dir') or not args.output_dir:
                output_folder_name = f"contamination_{DATASET_SHORT_NAME}_{pct_str}"
                args.output_dir = str(PIPELINE_ROOT / "results" / output_folder_name)
                print(f"Auto-configured Output Dir (from config): {args.output_dir}")

            # Note: --data-dir is required on CLI, so we DON'T override it from config

    output_dir = Path(args.output_dir)

    print("\n" + "="*80)
    print("MERGER: Waiting for all workers to complete...")
    print("="*80)

    # Wait for all ranks (with timeout)
    MERGER_TIMEOUT_SECONDS = 3600  # 1 hour timeout
    start_wait = time.time()
    while True:
        complete = []
        for r in range(world_size):
            marker = output_dir / "temp_similarities" / f"rank_{r}" / "COMPLETE.json"
            if marker.exists():
                complete.append(r)

        if len(complete) == world_size:
            break
        
        # Timeout check
        elapsed = time.time() - start_wait
        if elapsed > MERGER_TIMEOUT_SECONDS:
            print(f"\n⚠️ TIMEOUT after {elapsed/60:.0f} minutes waiting for workers.")
            print(f"   Complete: {complete}, Missing: {[r for r in range(world_size) if r not in complete]}")
            print("   Proceeding with available data...")
            break

        print(f"  Complete: {len(complete)}/{world_size} ranks (waited {elapsed:.0f}s)", end='\r')
        time.sleep(5)

    print(f"\n✅ All {world_size} ranks complete!")

    # Load completion stats
    total_embeddings = 0
    for r in range(world_size):
        marker = output_dir / "temp_similarities" / f"rank_{r}" / "COMPLETE.json"
        if marker.exists():
            with open(marker) as f:
                info = json.load(f)
                total_embeddings += info['embeddings_processed']
                print(f"  Rank {r}: {info['embeddings_processed']:,} embeddings in {timedelta(seconds=int(info['time_seconds']))}")
        else:
            print(f"  Rank {r}: <NO DATA found>")

    print(f"\nTotal embeddings processed: {total_embeddings:,}")

    # Load benchmark metadata
    print("\nLoading benchmark metadata...")
    all_test_data = []
    all_test_texts = []

    for benchmark in args.benchmarks:
        modes_to_process = ['input_output'] if (benchmark == 'musr' or benchmark.startswith('musr_') or benchmark == 'mbpp' or benchmark == 'zebralogic' or benchmark == 'codeforces') else args.modes
        for mode in modes_to_process:
            test_texts, test_ids, test_metadata = load_benchmark(benchmark, mode)
            for text, test_id in zip(test_texts, test_ids):
                meta = test_metadata.get(test_id, {})
                all_test_data.append({
                    'benchmark': benchmark,
                    'mode': mode,
                    'test_id': test_id,
                    'text': text,
                    'global_idx': len(all_test_texts),
                    'elo_bin': meta.get('elo_bin'),
                    'rating': meta.get('rating'),
                })
                all_test_texts.append(text)

    num_tests = len(all_test_texts)
    print(f"Found {num_tests} test points")

    # Group by benchmark/mode
    benchmark_groups = defaultdict(list)
    for test_data in all_test_data:
        key = (test_data['benchmark'], test_data['mode'])
        benchmark_groups[key].append(test_data)

    # Merge and save
    print("\n" + "="*80)
    print("Merging results and generating outputs...")
    print("="*80)

    # Determine number of parallel workers (leave 2 cores for system)
    import multiprocessing as mp
    num_workers = max(1, mp.cpu_count() - 2)
    print(f"Using {num_workers} parallel workers (of {mp.cpu_count()} CPUs)")

    for (benchmark, mode), test_points in benchmark_groups.items():
        print(f"\nProcessing {benchmark.upper()} - {mode.upper()} ({len(test_points)} test points)...")

        mode_dir = output_dir / f"{benchmark}_{mode}"
        mode_dir.mkdir(parents=True, exist_ok=True)

        # Prepare arguments for parallel processing
        args_list = [(test_data, output_dir, world_size) for test_data in test_points]

        # Process all tests in parallel
        print(f"  Processing {len(test_points)} tests in parallel...")
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_test, args_list),
                total=len(test_points),
                desc=f"  {benchmark}_{mode}"
            ))

        # Filter out None results (skipped tests)
        results = [r for r in results if r is not None]
        print(f"  ✅ Processed {len(results)} tests successfully")

        # Aggregate statistics from all test results
        # Sample 100K per test, but reservoir limit at 10M for plotting
        agg_stats = StreamingStats(sample_size=10_000_000)
        all_top_scores = []

        # Collect aggregate stats from parallel results
        for result in results:
            # Feed real sampled similarity values to StreamingStats
            # (result['similarity_samples'] contains values already sampled in process_single_test)
            if 'similarity_samples' in result and result['similarity_samples']:
                agg_stats.update_batch(np.array(result['similarity_samples']))

            # Collect top scores
            all_top_scores.extend(result['top_100_scores'])

        # Save aggregate stats
        final_stats = agg_stats.get_stats()
        with open(mode_dir / "aggregate_stats.json", 'w') as f:
            json.dump({
                'benchmark': benchmark,
                'mode': mode,
                'num_test_points': len(test_points),
                'total_embeddings': total_embeddings,
                'total_comparisons': final_stats['count'],
                'stats': final_stats
            }, f, indent=2)

        # Aggregate plots
        if agg_stats.sample_reservoir:
            sample_arr = np.array(agg_stats.sample_reservoir)

            # Ensure min/max are represented in sample (fixes disappearing outliers on log scale)
            if final_stats['max'] not in sample_arr:
                sample_arr = np.append(sample_arr, final_stats['max'])
            if final_stats['min'] not in sample_arr:
                sample_arr = np.append(sample_arr, final_stats['min'])

            # Calculate weights to scale sample to represent full dataset
            weight_per_sample = final_stats['count'] / len(sample_arr)
            weights = np.full(len(sample_arr), weight_per_sample)

            plt.figure(figsize=(12, 8))
            plt.hist(sample_arr, bins=200,
                     range=(final_stats['min'], final_stats['max']),
                     weights=weights,
                     log=True, alpha=0.7, edgecolor='black')
            plt.axvline(final_stats['max'], color='r', linestyle='--', label=f'Max: {final_stats["max"]:.4f}')
            plt.axvline(final_stats['mean'], color='g', linestyle='--', label=f'Mean: {final_stats["mean"]:.4f}')
            plt.axvline(final_stats['p99'], color='orange', linestyle='--', label=f'P99: {final_stats["p99"]:.4f}')
            plt.axvline(final_stats['p95'], color='purple', linestyle='--', label=f'P95: {final_stats["p95"]:.4f}')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Frequency (log scale)')
            plt.title(f'{benchmark.upper()} {mode.upper()} - Aggregate Distribution (Log Scale)\nTotal: {final_stats["count"]:,} comparisons (estimated from {len(sample_arr):,} samples)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(mode_dir / "aggregate_histogram_log.png", dpi=150)
            plt.close()
            
            # Linear histogram
            plt.figure(figsize=(12, 8))
            plt.hist(sample_arr, bins=200,
                     range=(final_stats['min'], final_stats['max']),
                     weights=weights,
                     alpha=0.7, edgecolor='black')
            plt.axvline(final_stats['max'], color='r', linestyle='--', label=f'Max: {final_stats["max"]:.4f}')
            plt.axvline(final_stats['mean'], color='g', linestyle='--', label=f'Mean: {final_stats["mean"]:.4f}')
            plt.axvline(final_stats['p99'], color='orange', linestyle='--', label=f'P99: {final_stats["p99"]:.4f}')
            plt.axvline(final_stats['p95'], color='purple', linestyle='--', label=f'P95: {final_stats["p95"]:.4f}')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Frequency')
            plt.title(f'{benchmark.upper()} {mode.upper()} - Aggregate Distribution (Linear Scale)\nTotal: {final_stats["count"]:,} comparisons (estimated from {len(sample_arr):,} samples)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(mode_dir / "aggregate_histogram_linear.png", dpi=150)
            plt.close()
            
            # CDF plot
            sorted_sample = np.sort(sample_arr)
            cdf = np.arange(1, len(sorted_sample) + 1) / len(sorted_sample)
            plt.figure(figsize=(12, 8))
            plt.plot(sorted_sample, cdf, linewidth=2, color='blue')
            plt.axvline(final_stats['max'], color='r', linestyle='--', label=f'Max: {final_stats["max"]:.4f}')
            plt.axvline(final_stats['mean'], color='g', linestyle='--', label=f'Mean: {final_stats["mean"]:.4f}')
            plt.axvline(final_stats['p99'], color='orange', linestyle='--', label=f'P99: {final_stats["p99"]:.4f}')
            plt.axvline(final_stats['p95'], color='purple', linestyle='--', label=f'P95: {final_stats["p95"]:.4f}')
            plt.axhline(0.99, color='orange', linestyle=':', alpha=0.5)
            plt.axhline(0.95, color='purple', linestyle=':', alpha=0.5)
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Cumulative Probability')
            plt.title(f'{benchmark.upper()} {mode.upper()} - Cumulative Distribution Function\nTotal: {final_stats["count"]:,} comparisons (estimated from {len(sample_arr):,} samples)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(mode_dir / "aggregate_cdf.png", dpi=150)
            plt.close()
            
            # Save sample for future plotting
            with gzip.open(mode_dir / "similarity_sample.npy.gz", 'wb') as f:
                np.save(f, sample_arr)

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

        print(f"  ✅ Saved to {mode_dir}")

    # Cleanup
    print("\nCleaning up temporary files...")
    import shutil
    try:
        # shutil.rmtree(output_dir / "temp_similarities")
        print("✅ Temporary files cleaned up (DISABLED for debugging)")
    except Exception as e:
        print(f"⚠️  Warning: Could not remove temp files: {e}")
        print("   (This is not critical - all results are saved)")

    print("\n" + "="*80)
    print("✅ MERGE COMPLETE!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Production Contamination Analysis (8x A100 40GB)")
    parser.add_argument('--data-dir', required=True, help='Directory with parquet files')
    parser.add_argument('--corpus-file', default=None, help='Specific corpus file pattern (e.g., "embeddings.parquet" or "*.parquet")')
    parser.add_argument('--output-dir', default='contamination_results', help='Base output directory')
    parser.add_argument('--dataset-name', default='dataset', help='Dataset name for output directory structure')
    parser.add_argument('--sample-size', default='unknown', help='Sample size/percentage for output directory structure')
    parser.add_argument('--benchmarks', nargs='+', default=['musr_murder_mysteries', 'musr_object_placements', 'musr_team_allocation', 'mbpp', 'zebralogic'],
                       help='Benchmarks to process. MuSR splits: musr_murder_mysteries, musr_object_placements, musr_team_allocation')
    parser.add_argument('--modes', nargs='+', default=['input', 'output'])
    parser.add_argument('--rank', type=int, default=None, help='Worker rank (0-7)')
    parser.add_argument('--world-size', type=int, default=8, help='Total workers')
    parser.add_argument('--gpu-batch-size', type=int, default=DEFAULT_CONFIG['gpu_batch_size'])
    parser.add_argument('--corpus-gpu-chunk', type=int, default=DEFAULT_CONFIG['corpus_gpu_chunk'])
    parser.add_argument('--max-rows-per-block', type=int, default=DEFAULT_CONFIG['max_rows_per_block'])
    parser.add_argument('--resume-from-file', type=int, default=0, help='Resume from this global parquet file index')
    parser.add_argument('--merge-only', action='store_true', help='Only run merger (skip workers)')
    parser.add_argument('--sample-top1pct', action='store_true', 
                        help='Sample 100 random items from top 1%% for semantic analysis (MBPP only)')
    parser.add_argument('--corpus-jsonl', default=None,
                        help='Corpus JSONL file for text lookup (required for --sample-top1pct)')
    parser.add_argument('--top1pct-samples', type=int, default=100,
                        help='Number of samples per test point in top 1%% mode')
    parser.add_argument('--top1pct-output', default=None,
                        help='Output CSV for top 1%% samples')
    args = parser.parse_args()

    # Construct full output directory with dataset_name and sample_size
    # Only append subdirectories if explicitly provided (not defaults)
    base_output_dir = Path(args.output_dir)

    # Check if using default values - if so, don't append subdirectories
    using_defaults = (args.dataset_name == 'dataset' and args.sample_size == 'unknown')

    if not using_defaults:
        # Format sample size (convert 0.01 to "1pct", etc.)
        try:
            sample_pct = float(args.sample_size)
            if sample_pct < 1:
                sample_str = f"{int(sample_pct * 100)}pct"
            else:
                sample_str = str(int(sample_pct))
        except (ValueError, TypeError):
            sample_str = str(args.sample_size)

        args.output_dir = str(base_output_dir / args.dataset_name / sample_str)
    else:
        # Use output_dir as-is when defaults are used
        args.output_dir = str(base_output_dir)

    # Get rank from environment or args
    if args.rank is not None:
        rank = args.rank
    elif 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
    elif 'LOCAL_RANK' in os.environ:
        rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0

    world_size = args.world_size

    if args.sample_top1pct:
        # Special mode: sample from top 1% for semantic analysis
        run_top1pct_sampler(args, world_size)
    elif args.merge_only:
        run_merger(args, world_size)
    else:
        run_worker(rank, world_size, args)

        # Rank 0 also runs the merger after its work is done
        if rank == 0:
            run_merger(args, world_size)


if __name__ == "__main__":
    main()
