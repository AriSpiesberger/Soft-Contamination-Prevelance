#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-GPU Local Embedding Generation Script
Distributes work across GPUs by rank-based data sharding.
"""

import os
import sys
import json
import gc
import argparse
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Parse arguments first
parser = argparse.ArgumentParser()
parser.add_argument('--rank', type=int, default=0, help='GPU rank (default: 0)')
parser.add_argument('--world-size', type=int, default=1, help='Total number of GPUs (default: 1)')
args = parser.parse_args()

rank = args.rank
world_size = args.world_size

# --- CONFIGURATION LOADING ---
PIPELINE_ROOT = Path(__file__).parent.parent
CONFIG_FILE_RAW = os.environ.get("PIPELINE_CONFIG", str(PIPELINE_ROOT / "configs" / "default.yaml"))

# Resolve config path
CONFIG_FILE = Path(CONFIG_FILE_RAW)
if not CONFIG_FILE.is_absolute():
    CONFIG_FILE = PIPELINE_ROOT / CONFIG_FILE_RAW

def load_config():
    """Load pipeline configuration from YAML."""
    if not CONFIG_FILE.exists():
        print(f"[Rank {rank}] Error: Config file not found: {CONFIG_FILE}", file=sys.stderr)
        print(f"[Rank {rank}] Tried: {CONFIG_FILE_RAW}", file=sys.stderr)
        print(f"[Rank {rank}] Pipeline root: {PIPELINE_ROOT}", file=sys.stderr)
        sys.exit(1)

    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)

config = load_config()
embeddings_config = config.get('embeddings', {})

# Extract configuration values
local_config = embeddings_config.get('local', {})
MODEL_NAME = embeddings_config.get('model', 'nvidia/llama-embed-nemotron-8b')
INPUT_FILE = local_config.get('input_file', None)
OUTPUT_FILE = local_config.get('output_file', None)
MAX_SEQ_LENGTH = embeddings_config.get('max_seq_length', 512)
BATCH_SIZE = embeddings_config.get('max_batch_size', 32)
NUM_WORKERS = embeddings_config.get('num_loader_workers', 4)

# Resolve paths
def resolve_path(path_str):
    """Resolve path relative to pipeline root if not absolute."""
    path = Path(path_str)
    if not path.is_absolute():
        path = PIPELINE_ROOT / path
    return path

# STANDARD NAMING CONVENTION LOGIC
DATASET_SHORT_NAME = config.get('pipeline', {}).get('dataset_short_name', config.get('pipeline', {}).get('name', 'dataset'))
pct_val = int(config.get('chunking', {}).get('paragraph_sample_percentage', 0.01) * 100)
pct_str = f"{pct_val}pct"

# Auto-configure Input (from Stage 2)
if True:
    input_name = f"conversations_{DATASET_SHORT_NAME}_{pct_str}.jsonl"
    # Assuming data dir is ./data relative to pipeline root
    INPUT_FILE = PIPELINE_ROOT / "data" / input_name
    
# Auto-configure Output
if True:
    output_name = f"embeddings_{DATASET_SHORT_NAME}_{pct_str}.parquet"
    OUTPUT_FILE = PIPELINE_ROOT / "data" / output_name

print(f"[Rank {rank}] Auto-configured Input: {INPUT_FILE}")
print(f"[Rank {rank}] Auto-configured Output: {OUTPUT_FILE}")

if rank == 0:
    print(f"=" * 80)
    print(f"MULTI-GPU LOCAL EMBEDDING GENERATION")
    print(f"=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"World size: {world_size}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"=" * 80)
    print()

class JSONLDataset(Dataset):
    """Dataset for reading JSONL paragraphs with rank-based sharding and length sorting."""

    def __init__(self, jsonl_path, rank, world_size):
        self.paragraphs = []
        self.ids = []
        self.lengths = []
        self.rank = rank
        self.world_size = world_size

        print(f"[Rank {rank}] Loading paragraphs from {jsonl_path}...")
        with open(jsonl_path, 'r') as f:
            for idx, line in enumerate(tqdm(f, desc=f"[Rank {rank}] Reading JSONL", disable=(rank != 0))):
                # Shard data by rank
                if idx % world_size == rank:
                    data = json.loads(line)
                    text = data['text']
                    self.paragraphs.append(text)
                    self.ids.append(data['id'])
                    # Approximate token count (words * 1.3)
                    self.lengths.append(len(text.split()) * 1.3)

        # Sort by length descending (longest first - fail fast on OOM, fresh GPU memory)
        sorted_indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i], reverse=True)
        self.paragraphs = [self.paragraphs[i] for i in sorted_indices]
        self.ids = [self.ids[i] for i in sorted_indices]
        self.lengths = [self.lengths[i] for i in sorted_indices]

        print(f"[Rank {rank}] Loaded {len(self.paragraphs):,} paragraphs (sorted longest first)")

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx):
        return {
            'id': self.ids[idx],
            'text': self.paragraphs[idx],
            'approx_tokens': self.lengths[idx]
        }


def mean_pooling(model_output, attention_mask):
    """Mean pooling to get sentence embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def main():
    # Set GPU and memory settings
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    device = torch.device('cuda:0')

    print(f"[Rank {rank}] Using GPU {rank}")
    if torch.cuda.is_available():
        print(f"[Rank {rank}] GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"[Rank {rank}] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load model and tokenizer
    print(f"[Rank {rank}] Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Try to use Flash Attention 2 for faster, more memory-efficient computation
    try:
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2"
        )
        print(f"[Rank {rank}] ✓ Flash Attention 2 enabled")
    except Exception as e:
        print(f"[Rank {rank}] Flash Attention 2 not available ({e}), using default attention")
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    
    model = model.to(device)
    model.eval()

    # CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"[Rank {rank}] Model loaded (dtype: {model.dtype})")

    # Load dataset (sharded by rank)
    dataset = JSONLDataset(INPUT_FILE, rank=rank, world_size=world_size)

    # Dynamic batching: read from config (use conservative defaults)
    TARGET_TOKENS_PER_BATCH = embeddings_config.get('target_tokens_per_batch', 50000)
    MAX_BATCH_SIZE = embeddings_config.get('max_batch_size', 32)
    CHUNK_SIZE = 10000  # Save chunk every 10k embeddings

    # Output directory for chunks
    output_dir = OUTPUT_FILE.parent / f"{OUTPUT_FILE.stem}_rank_{rank}_chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find existing chunks to resume from
    existing_chunks = sorted(output_dir.glob("chunk_*.parquet"))
    start_idx = 0
    chunk_num = 0

    if existing_chunks:
        # Count items in existing chunks
        for chunk_file in existing_chunks:
            chunk_table = pq.read_table(chunk_file, columns=['id'])
            start_idx += len(chunk_table)
        chunk_num = len(existing_chunks)
        print(f"[Rank {rank}] Resuming: found {chunk_num} chunks, {start_idx:,} items done")

    # Generate embeddings with dynamic batching
    print(f"[Rank {rank}] Generating embeddings (saving every {CHUNK_SIZE:,} items)...")

    # Buffer for current chunk
    chunk_ids = []
    chunk_embeddings = []

    import time

    # Process in dynamic batches
    idx = start_idx
    total = len(dataset)
    pbar = tqdm(total=total, initial=start_idx, desc=f"[Rank {rank}] Embedding", disable=(rank != 0))

    # Initialize accumulators
    all_ids = []
    all_embeddings = []
    
    # Checkpoint settings
    CHECKPOINT_INTERVAL = 900  # 15 minutes
    checkpoint_path = output_dir / f"checkpoint_rank_{rank}.parquet"
    last_checkpoint_time = time.time()
    
    # Load existing checkpoint if resuming
    if checkpoint_path.exists():
        print(f"[Rank {rank}] Loading existing checkpoint...")
        try:
            checkpoint_table = pq.read_table(checkpoint_path)
            all_ids = checkpoint_table['id'].to_pylist()
            all_embeddings = [np.array(checkpoint_table['embedding'].to_pylist())]
            start_idx = len(all_ids)
            idx = start_idx
            pbar.n = start_idx
            pbar.refresh()
            print(f"[Rank {rank}] Resumed from checkpoint: {len(all_ids):,} items")
        except Exception as e:
            print(f"[Rank {rank}] Warning: Could not load checkpoint: {e}")

    # Graceful shutdown handler
    shutdown_requested = False
    def save_emergency_checkpoint():
        if all_ids:
            print(f"\n[Rank {rank}] Saving emergency checkpoint ({len(all_ids):,} items)...")
            try:
                emergency_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
                emergency_table = pa.table({
                    'id': all_ids,
                    'embedding': list(emergency_embeddings)
                })
                pq.write_table(emergency_table, checkpoint_path, compression='snappy')
                print(f"[Rank {rank}] Emergency checkpoint saved!")
            except Exception as e:
                print(f"[Rank {rank}] Error saving emergency checkpoint: {e}")
    
    import signal
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            print(f"\n[Rank {rank}] Interrupt received, saving checkpoint...")
            save_emergency_checkpoint()
            sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    with torch.no_grad():
        while idx < total:
            # Build batch dynamically based on sequence lengths
            batch_texts = []
            batch_ids = []
            max_seq_len = 0

            while idx < total and len(batch_texts) < MAX_BATCH_SIZE:
                item = dataset[idx]
                approx_tokens = min(item['approx_tokens'], MAX_SEQ_LENGTH)  # Cap at max seq length

                # Estimate total tokens if we add this sample
                new_max_len = max(max_seq_len, approx_tokens)
                new_batch_size = len(batch_texts) + 1
                estimated_tokens = new_max_len * new_batch_size

                # Check if adding this would exceed target (but always add at least one)
                if batch_texts and estimated_tokens > TARGET_TOKENS_PER_BATCH:
                    break

                batch_texts.append(item['text'])
                batch_ids.append(item['id'])
                max_seq_len = new_max_len
                idx += 1

            if not batch_texts:
                break

            # Process batch with OOM recovery (split in half on OOM)
            def process_batch(texts, ids):
                encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors='pt')
                input_ids = encoded['input_ids'].to(device, non_blocking=True)
                attention_mask = encoded['attention_mask'].to(device, non_blocking=True)
                
                try:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    emb = mean_pooling(outputs, attention_mask)
                    emb = F.normalize(emb, p=2, dim=1)
                    result = ids, emb.cpu().numpy()
                finally:
                    # Explicit tensor cleanup to prevent GPU memory fragmentation
                    del input_ids, attention_mask
                    if 'outputs' in dir():
                        del outputs
                
                return result

            def process_with_recovery(texts, ids, depth=0):
                try:
                    return process_batch(texts, ids)
                except torch.cuda.OutOfMemoryError:
                    if depth > 3 or len(texts) <= 1:
                        raise  # Give up after 3 splits or single item
                    torch.cuda.empty_cache()
                    gc.collect()  # Force garbage collection
                    print(f"\n[Rank {rank}] OOM on {len(texts)} items, splitting (depth={depth})...")
                    half = len(texts) // 2
                    ids1, emb1 = process_with_recovery(texts[:half], ids[:half], depth+1)
                    ids2, emb2 = process_with_recovery(texts[half:], ids[half:], depth+1)
                    return ids1 + ids2, np.vstack([emb1, emb2])

            result_ids, result_embeddings = process_with_recovery(batch_texts, batch_ids)
            all_ids.extend(result_ids)
            all_embeddings.append(result_embeddings)
            pbar.update(len(batch_texts))

            # Periodic garbage collection every 1000 batches
            if len(all_ids) % 10000 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            # Checkpoint every 15 minutes
            if time.time() - last_checkpoint_time > CHECKPOINT_INTERVAL:
                print(f"\n[Rank {rank}] Saving checkpoint ({len(all_ids):,} items)...")
                checkpoint_embeddings = np.vstack(all_embeddings)
                checkpoint_table = pa.table({
                    'id': all_ids,
                    'embedding': list(checkpoint_embeddings)
                })
                pq.write_table(checkpoint_table, checkpoint_path, compression='snappy')
                last_checkpoint_time = time.time()
                print(f"[Rank {rank}] Checkpoint saved")
                
                # Flush to disk and clear memory to prevent RAM growth
                del checkpoint_embeddings, checkpoint_table
                gc.collect()

    pbar.close()

    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)

    print(f"[Rank {rank}] Generated {len(all_ids):,} embeddings")
    print(f"[Rank {rank}] Embedding shape: {all_embeddings.shape}")

    # Save to parquet file
    # For single-GPU runs, save directly to final path; for multi-GPU, use rank suffix
    if world_size == 1:
        output_path = OUTPUT_FILE
    else:
        output_path = OUTPUT_FILE.parent / f"{OUTPUT_FILE.stem}_rank_{rank}.parquet"
    print(f"[Rank {rank}] Saving embeddings to {output_path}...")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create PyArrow table
    table = pa.table({
        'id': all_ids,
        'embedding': list(all_embeddings)
    })

    # Write to parquet
    pq.write_table(table, output_path, compression='snappy')

    print(f"[Rank {rank}] ✓ Saved successfully!")
    print(f"[Rank {rank}]   File: {output_path}")
    print(f"[Rank {rank}]   Size: {output_path.stat().st_size / (1024**2):.2f} MB")

    # Clean up checkpoint file
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[Rank {rank}] Checkpoint cleaned up")


if __name__ == "__main__":
    main()
