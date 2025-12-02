#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Embedding Runner - H100 Cluster Edition
FIXED VERSION:
1. Solved Pickling Error by moving Collate function to global scope.
2. Fixed batch unpacking logic in collator.
3. Finds both 'data.jsonl' (v2) and 'part_*.jsonl' (v3) files.
"""

import os
import sys
import time
import gc
import signal
import shutil
import traceback
import queue as queue_module
import multiprocessing as mp
from collections import deque
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

import boto3
import orjson
import numpy as np

# Import S3 config
from s3_config import S3Config, default_config

# --- CONFIGURATION ---
MODEL_NAME = 'nvidia/llama-embed-nemotron-8b'
EMBEDDER_KEY = "embeddings"

# S3 Config - can be overridden by creating a custom S3Config instance
s3_config = default_config
S3_BUCKET = s3_config.bucket
INPUT_PREFIX = s3_config.input_prefix
OUTPUT_PREFIX = s3_config.output_prefix
REGION = s3_config.region

# Paths
LOCAL_CACHE_DIR = Path("/tmp/embedding_runner_cache")

# Tuning - Optimized for H100
TARGET_TOKENS_PER_BATCH = 400_000 
MAX_BATCH_SIZE = 512  # Increased for H100 80GB
MAX_SEQ_LENGTH = 512
NUM_LOADER_WORKERS = 24  # Increased for better I/O
PREFETCH_FACTOR = 4  # Increased for better prefetching

if not hasattr(np, 'int'):
    np.int = int

# =============================================================================
# S3 UTILITIES
# =============================================================================
def get_s3_client():
    return boto3.client('s3', region_name=s3_config.region, config=s3_config.get_boto_config())

def list_tasks(bucket, in_prefix, out_prefix):
    """Finds all jsonl files (data.jsonl OR part_*.jsonl)."""
    s3 = get_s3_client()
    
    print(f"Scanning inputs: s3://{bucket}/{in_prefix}")
    input_files = set()
    paginator = s3.get_paginator('list_objects_v2')
    
    # Scan for inputs
    pages = list(paginator.paginate(Bucket=bucket, Prefix=in_prefix))
    for page in tqdm(pages, desc="Scanning input files", unit="page"):
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.jsonl'):
                    input_files.add(obj['Key'])
    
    # Scan for already completed outputs to skip
    print(f"Scanning outputs: s3://{bucket}/{out_prefix}")
    done_files = set()
    pages = list(paginator.paginate(Bucket=bucket, Prefix=out_prefix))
    for page in tqdm(pages, desc="Scanning output files", unit="page"):
        if 'Contents' in page:
            for obj in page['Contents']:
                rel_path = obj['Key'].replace(out_prefix, "").lstrip("/")
                rel_jsonl = rel_path.replace('.parquet', '.jsonl')
                full_input_key = f"{in_prefix.rstrip('/')}/{rel_jsonl}"
                done_files.add(full_input_key)
    
    tasks = []
    for key in input_files:
        if key not in done_files:
            tasks.append(key)
            
    print(f"Found {len(input_files)} inputs. Skipping {len(done_files)} completed.")
    print(f"Tasks remaining: {len(tasks)}")
    return sorted(tasks)

def download_file(bucket, key, local_path):
    get_s3_client().download_file(bucket, key, str(local_path))

def upload_file(bucket, key, local_path):
    get_s3_client().upload_file(str(local_path), bucket, key)

# =============================================================================
# MODEL
# =============================================================================
def setup_model(gpu_id):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")  # H100 optimization
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")  # Right padding for efficiency
    
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    ).to(device).eval()
    return tokenizer, model, device

def mean_pooling(token_embeddings, attention_mask):
    """Fused mean pooling - stays in bf16 until final normalization."""
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

# =============================================================================
# PROFILING UTILITIES
# =============================================================================
class ProfileTimer:
    """Minimal CUDA-aware profiler for per-batch timing."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.times = {}

    def start(self, name):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        self.times[name] = time.perf_counter()

    def stop(self, name):
        if not self.enabled or name not in self.times:
            return 0
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - self.times[name]) * 1000
        return elapsed

def process_batch_iterative(initial_encoded_input, initial_indices, model, block_data, device, profiler=None):
    """
    Iterative queue with CUDA stream overlap for efficient processing.
    """
    queue = deque([(initial_encoded_input, initial_indices)])
    
    compute_stream = torch.cuda.Stream()
    transfer_stream = torch.cuda.Stream()
    
    while queue:
        current_encoded, current_indices = queue.popleft()
        
        # --- PREFETCH NEXT (on transfer stream) ---
        if queue:
            next_encoded, _ = queue[0]
            with torch.cuda.stream(transfer_stream):
                # Prefetch next batch to GPU
                _ = {
                    k: v.to(device, non_blocking=True)
                    for k, v in next_encoded.items()
                }
        
        # --- COMPUTE CURRENT (on compute stream) ---
        try:
            with torch.cuda.stream(compute_stream):
                # Wait for any prior transfer to complete
                compute_stream.wait_stream(transfer_stream)
                
                transfer_ms = 0
                inference_ms = 0
                d2h_ms = 0
                
                if profiler:
                    profiler.start('transfer')
                batch_gpu = {
                    k: v.to(device, non_blocking=True)
                    for k, v in current_encoded.items()
                }
                if profiler:
                    torch.cuda.synchronize()
                    transfer_ms = profiler.stop('transfer')
                
                if profiler:
                    profiler.start('inference')
                with torch.no_grad():
                    out = model(**batch_gpu)
                    embeddings = mean_pooling(out[0], batch_gpu['attention_mask'])
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                if profiler:
                    torch.cuda.synchronize()
                    inference_ms = profiler.stop('inference')
                
                if profiler:
                    profiler.start('d2h')
                # Stay in bf16 -> float32 on CPU to avoid GPU cast overhead
                cpu_embs = embeddings.cpu().float().numpy()
                if profiler:
                    d2h_ms = profiler.stop('d2h')
            
            # Store numpy arrays directly
            for local_idx, emb in zip(current_indices, cpu_embs):
                block_data[local_idx][EMBEDDER_KEY] = emb
            
            del batch_gpu, out, embeddings, cpu_embs
            
            # Return timing info if profiling
            if profiler:
                return {
                    'transfer_ms': transfer_ms,
                    'inference_ms': inference_ms,
                    'd2h_ms': d2h_ms,
                    'batch_size': len(current_indices)
                }
            return None
            
        except torch.cuda.OutOfMemoryError:
            # Cleanup GPU state
            try:
                del batch_gpu
            except NameError:
                pass
            try:
                del out
            except NameError:
                pass
            try:
                del embeddings
            except NameError:
                pass
            
            gc.collect()
            torch.cuda.empty_cache()
            
            batch_len = len(current_indices)
            if batch_len <= 1:
                seq_len = current_encoded['input_ids'].shape[1]
                print(f"  [OOM] Skipping single doc (seq_len={seq_len})")
                continue
            
            # Split and requeue
            mid = batch_len // 2
            
            batch_1 = {k: v[:mid].clone() for k, v in current_encoded.items()}
            indices_1 = current_indices[:mid]
            
            batch_2 = {k: v[mid:].clone() for k, v in current_encoded.items()}
            indices_2 = current_indices[mid:]
            
            # Insert at front for depth-first processing
            queue.appendleft((batch_2, indices_2))
            queue.appendleft((batch_1, indices_1))
            
            print(f"  [OOM Recovery] Split {batch_len} -> {mid} + {batch_len - mid}")

# =============================================================================
# DATA HELPERS (Global Scope for Pickling)
# =============================================================================
class BatchDataset(Dataset):
    def __init__(self, batches): self.batches = batches
    def __len__(self): return len(self.batches)
    def __getitem__(self, i): return self.batches[i]

class DataCollator:
    """Picklable collator class replaces the local function"""
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __call__(self, batch):
        # DataLoader(batch_size=1) returns a list with 1 item
        # The item is the tuple (texts, indices) from create_batches
        item = batch[0]
        txts, idxs = item[0], item[1]
        
        enc = self.tokenizer(
            txts, 
            padding=True, 
            truncation=True, 
            max_length=self.max_len, 
            return_tensors='pt'
        )
        return enc, idxs

def create_batches(items, target_tokens, max_bs):
    current_txt, current_idx = [], []
    max_len = 0
    for count, text, idx in items:
        max_len = max(max_len, count)
        next_bs = len(current_txt) + 1
        if (next_bs * max_len > target_tokens) or (next_bs > max_bs):
            if current_txt:
                yield (current_txt, current_idx)
                current_txt, current_idx, max_len = [], [], 0
        current_txt.append(text)
        current_idx.append(idx)
    if current_txt: yield (current_txt, current_idx)

# =============================================================================
# WORKER
# =============================================================================
def gpu_worker(rank, queue, errors, progress_queue):
    shutdown = False
    def h(*args): nonlocal shutdown; shutdown = True
    signal.signal(signal.SIGTERM, h)
    signal.signal(signal.SIGINT, h)

    try:
        tokenizer, model, device = setup_model(rank)
        print(f"[GPU {rank}] Model ready on {device}")
    except Exception as e:
        errors.put((rank, f"Init failed: {e}"))
        return
        
    # Create the collator once per worker
    collator = DataCollator(tokenizer, MAX_SEQ_LENGTH)
    profiler = ProfileTimer(enabled=True)
    files_processed = 0
    total_docs = 0
    total_batches = 0
    total_transfer_time = 0.0
    total_inference_time = 0.0
    total_d2h_time = 0.0

    while not shutdown:
        try:
            try:
                s3_key = queue.get(timeout=2)
            except queue_module.Empty:
                if queue.empty(): break
                continue

            file_start_time = time.time()
            fname = s3_key.split('/')[-1]
            local_in = LOCAL_CACHE_DIR / f"g{rank}" / "in" / fname
            local_out = LOCAL_CACHE_DIR / f"g{rank}" / "out" / fname.replace('.jsonl', '.parquet')
            
            # Download
            download_start = time.time()
            local_in.parent.mkdir(parents=True, exist_ok=True)
            download_file(S3_BUCKET, s3_key, local_in)
            download_time = time.time() - download_start
            file_size_mb = local_in.stat().st_size / (1024 * 1024) if local_in.exists() else 0

            # Load data
            load_start = time.time()
            data = []
            with open(local_in, 'rb') as f:
                for line in f:
                    if line.strip(): data.append(orjson.loads(line))
            load_time = time.time() - load_start
            
            if data:
                # Prepare batches
                prep_start = time.time()
                data.sort(key=lambda x: len(x.get('text', '')), reverse=False)
                tasks = []
                for i, d in enumerate(data):
                    tasks.append((int(len(d.get('text', ''))/3)+5, d['text'], i))
                
                batches = list(create_batches(tasks, TARGET_TOKENS_PER_BATCH, MAX_BATCH_SIZE))
                prep_time = time.time() - prep_start
                
                # Process batches
                loader = DataLoader(
                    BatchDataset(batches), 
                    batch_size=1, 
                    shuffle=False,
                    num_workers=NUM_LOADER_WORKERS, 
                    collate_fn=collator,
                    pin_memory=True, 
                    prefetch_factor=PREFETCH_FACTOR,
                    persistent_workers=True  # Keep workers alive between epochs
                )

                embed_start = time.time()
                batch_count = 0
                batch_timings = []
                
                with torch.no_grad():  # Wrap entire inference loop
                    for enc, idxs in loader:
                        timing = process_batch_iterative(enc, idxs, model, data, device, profiler)
                        batch_count += 1
                        if timing:
                            batch_timings.append(timing)
                            total_transfer_time += timing['transfer_ms'] / 1000.0
                            total_inference_time += timing['inference_ms'] / 1000.0
                            total_d2h_time += timing['d2h_ms'] / 1000.0
                
                embed_time = time.time() - embed_start
                total_batches += batch_count
                
                # Calculate per-batch averages for this file
                if batch_timings:
                    avg_transfer = sum(t['transfer_ms'] for t in batch_timings) / len(batch_timings)
                    avg_inference = sum(t['inference_ms'] for t in batch_timings) / len(batch_timings)
                    avg_d2h = sum(t['d2h_ms'] for t in batch_timings) / len(batch_timings)
                    avg_batch_size = sum(t['batch_size'] for t in batch_timings) / len(batch_timings)
                else:
                    avg_transfer = avg_inference = avg_d2h = avg_batch_size = 0

                # Save results
                save_start = time.time()
                import pyarrow as pa
                import pyarrow.parquet as pq
                
                valid = [d for d in data if EMBEDDER_KEY in d]
                if valid:
                    vectors = np.stack([d.pop(EMBEDDER_KEY) for d in valid])
                    table = pa.Table.from_pylist(valid)
                    vec_col = pa.FixedSizeListArray.from_arrays(
                        pa.array(vectors.ravel()), list_size=vectors.shape[1]
                    )
                    table = table.append_column(EMBEDDER_KEY, vec_col)
                    
                    local_out.parent.mkdir(parents=True, exist_ok=True)
                    pq.write_table(table, str(local_out), compression='zstd')
                    
                    rel_path = s3_key.replace(INPUT_PREFIX, "").lstrip("/")
                    out_key = f"{OUTPUT_PREFIX.rstrip('/')}/{rel_path}".replace('.jsonl', '.parquet')
                    
                    upload_start = time.time()
                    upload_file(S3_BUCKET, out_key, local_out)
                    upload_time = time.time() - upload_start
                    save_time = time.time() - save_start
                    
                    total_time = time.time() - file_start_time
                    files_processed += 1
                    total_docs += len(valid)
                    
                    # Calculate throughput
                    docs_per_sec = len(valid) / embed_time if embed_time > 0 else 0
                    embed_mb_per_sec = file_size_mb / embed_time if embed_time > 0 else 0
                    total_mb_per_sec = file_size_mb / total_time if total_time > 0 else 0
                    
                    # Report progress
                    progress_queue.put({
                        'rank': rank,
                        'file': fname,
                        'docs': len(valid),
                        'file_size_mb': file_size_mb,
                        'batches': batch_count,
                        'download_time': download_time,
                        'load_time': load_time,
                        'prep_time': prep_time,
                        'embed_time': embed_time,
                        'save_time': save_time,
                        'upload_time': upload_time,
                        'total_time': total_time,
                        'docs_per_sec': docs_per_sec,
                        'embed_mb_per_sec': embed_mb_per_sec,
                        'total_mb_per_sec': total_mb_per_sec,
                        'avg_transfer_ms': avg_transfer,
                        'avg_inference_ms': avg_inference,
                        'avg_d2h_ms': avg_d2h,
                        'avg_batch_size': avg_batch_size
                    })
                    
                    # Detailed breakdown
                    print(f"[GPU {rank}] ✓ {fname} | {file_size_mb:.1f}MB | {len(valid)} docs | "
                          f"{batch_count} batches")
                    print(f"  Time: {total_time:.1f}s total | "
                          f"Embed: {embed_time:.1f}s ({embed_mb_per_sec:.2f} MB/s) | "
                          f"IO: {download_time:.1f}s↓ + {upload_time:.1f}s↑")
                    print(f"  Speed: {docs_per_sec:.0f} docs/s | "
                          f"Transfer: {avg_transfer:.1f}ms | "
                          f"Inference: {avg_inference:.1f}ms | "
                          f"D2H: {avg_d2h:.1f}ms | "
                          f"Avg BS: {avg_batch_size:.0f}")
            
            # Cleanup
            if local_in.exists(): local_in.unlink()
            if local_out.exists(): local_out.unlink()
            gc.collect()

        except Exception as e:
            print(f"[GPU {rank}] ✗ Error processing {fname if 'fname' in locals() else 'unknown'}: {e}")
            traceback.print_exc()
            errors.put((rank, str(e)))
    
    # Final worker stats
    if total_batches > 0:
        avg_transfer = (total_transfer_time / total_batches) * 1000
        avg_inference = (total_inference_time / total_batches) * 1000
        avg_d2h = (total_d2h_time / total_batches) * 1000
        print(f"[GPU {rank}] Worker finished. Processed {files_processed} files, {total_docs:,} total docs, {total_batches} batches")
        print(f"  └─ Avg per batch: Transfer={avg_transfer:.1f}ms | "
              f"Inference={avg_inference:.1f}ms | D2H={avg_d2h:.1f}ms")
    else:
        print(f"[GPU {rank}] Worker finished. Processed {files_processed} files, {total_docs:,} total docs")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    if LOCAL_CACHE_DIR.exists(): shutil.rmtree(LOCAL_CACHE_DIR)
    
    print("="*60)
    print(f"CLUSTER EMBEDDING PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input: s3://{S3_BUCKET}/{INPUT_PREFIX}")
    print(f"Output: s3://{S3_BUCKET}/{OUTPUT_PREFIX}")
    print("="*60)
    
    tasks = list_tasks(S3_BUCKET, INPUT_PREFIX, OUTPUT_PREFIX)
    if not tasks:
        print("No input files found. Check S3_PREFIX.")
        sys.exit(0)

    m = mp.Manager()
    q = m.Queue()
    errs = m.Queue()
    progress_q = m.Queue()
    
    for t in tasks: q.put(t)
    total_tasks = len(tasks)
    
    procs = []
    n_gpus = torch.cuda.device_count()
    print(f"\nLaunching {n_gpus} GPU workers on {total_tasks} tasks...\n")
    
    # Start workers
    for i in range(n_gpus):
        p = mp.Process(target=gpu_worker, args=(i, q, errs, progress_q))
        p.start()
        procs.append(p)
    
    # Progress monitoring
    completed = 0
    total_docs = 0
    start_time = time.time()
    stats_by_gpu = {i: {'files': 0, 'docs': 0, 'time': 0.0} for i in range(n_gpus)}
    
    with tqdm(total=total_tasks, desc="Processing files", unit="file", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        # Monitor progress queue
        while completed < total_tasks:
            try:
                # Check for progress updates
                while not progress_q.empty():
                    update = progress_q.get_nowait()
                    rank = update['rank']
                    stats_by_gpu[rank]['files'] += 1
                    stats_by_gpu[rank]['docs'] += update['docs']
                    stats_by_gpu[rank]['time'] += update['total_time']
                    completed += 1
                    total_docs += update['docs']
                    pbar.update(1)
                    
                    # Show detailed per-step info
                    postfix = {
                        'docs': total_docs,
                        'embed': f"{update.get('embed_mb_per_sec', 0):.1f}MB/s",
                        'total': f"{update.get('total_mb_per_sec', 0):.1f}MB/s"
                    }
                    if 'avg_inference_ms' in update and update['avg_inference_ms'] > 0:
                        postfix['inf'] = f"{update['avg_inference_ms']:.0f}ms"
                    pbar.set_postfix(postfix)
                
                # Check if workers are still alive
                alive = sum(1 for p in procs if p.is_alive())
                if alive == 0 and completed < total_tasks:
                    # Workers died, check for errors
                    break
                
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Shutting down workers...")
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                break
    
    # Wait for all processes
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
            p.join()
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total files processed: {completed}/{total_tasks}")
    print(f"Total documents: {total_docs:,}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    if total_docs > 0:
        print(f"Overall throughput: {total_docs/total_time:.1f} docs/sec")
    print("\nPer-GPU Stats:")
    for rank, stats in stats_by_gpu.items():
        if stats['files'] > 0:
            throughput = stats['docs']/stats['time'] if stats['time'] > 0 else 0
            avg_time_per_file = stats['time'] / stats['files'] if stats['files'] > 0 else 0
            print(f"  GPU {rank}: {stats['files']} files, {stats['docs']:,} docs, "
                  f"{stats['time']/60:.1f} min, {throughput:.1f} docs/sec, "
                  f"{avg_time_per_file:.1f}s/file")
    
    # Check for errors
    if not errs.empty():
        print("\nErrors encountered:")
        while not errs.empty():
            rank, error = errs.get()
            print(f"  GPU {rank}: {error}")
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)