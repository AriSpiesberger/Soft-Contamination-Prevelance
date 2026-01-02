#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Local Embedding Generation Script
Features:
- Flash Attention 2 for 2-3x speedup
- FP16 precision for 2x memory savings
- Dynamic batching by sequence length (reduces padding waste)
- Streaming parquet writes to reduce memory
- TF32 enabled for faster matmul on Ampere+ GPUs
- Non-blocking GPU transfers
- Persistent workers and prefetching
"""

import os
import sys
import json
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import tiktoken

# --- CONFIGURATION LOADING ---
PIPELINE_ROOT = Path(__file__).parent.parent
CONFIG_FILE = os.environ.get("PIPELINE_CONFIG", PIPELINE_ROOT / "configs" / "default.yaml")

def load_config():
    """Load pipeline configuration from YAML."""
    config_path = Path(CONFIG_FILE)
    if not config_path.exists():
        print(f"Error: Config file not found: {CONFIG_FILE}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        return yaml.safe_load(f)

config = load_config()
embeddings_config = config.get('embeddings', {})

# Check mode
MODE = embeddings_config.get('mode', 's3')
if MODE != 'local':
    print(f"Error: This script requires mode='local' in config. Current mode: {MODE}")
    sys.exit(1)

# Extract configuration values
local_config = embeddings_config.get('local', {})
MODEL_NAME = embeddings_config.get('model', 'nvidia/llama-embed-nemotron-8b')
INPUT_FILE = local_config.get('input_file', './data/dolmino_random_paragraphs.jsonl')
OUTPUT_FILE = local_config.get('output_file', './data/dolmino_embeddings.parquet')
MAX_SEQ_LENGTH = embeddings_config.get('max_seq_length', 512)
BATCH_SIZE = embeddings_config.get('max_batch_size', 32)
NUM_WORKERS = embeddings_config.get('num_loader_workers', 4)
PREFETCH_FACTOR = embeddings_config.get('prefetch_factor', 4)

# Resolve paths
def resolve_path(path_str):
    """Resolve path relative to pipeline root if not absolute."""
    path = Path(path_str)
    if not path.is_absolute():
        path = PIPELINE_ROOT / path
    return path

INPUT_FILE = resolve_path(INPUT_FILE)
OUTPUT_FILE = resolve_path(OUTPUT_FILE)

print(f"=" * 80)
print(f"OPTIMIZED LOCAL EMBEDDING GENERATION")
print(f"=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max sequence length: {MAX_SEQ_LENGTH}")
print(f"=" * 80)


class JSONLDataset(Dataset):
    """Dataset for reading JSONL paragraphs with token-based filtering and length sorting."""

    def __init__(self, jsonl_path, max_tokens=None):
        self.data = []

        # Initialize tiktoken for fast token counting
        tokenizer_enc = tiktoken.get_encoding("cl100k_base")

        print(f"\nLoading paragraphs from {jsonl_path}...")
        total_loaded = 0
        filtered_count = 0

        with open(jsonl_path, 'r') as f:
            for line in tqdm(f, desc="Reading JSONL"):
                entry = json.loads(line)
                total_loaded += 1

                # Estimate token count
                if max_tokens:
                    token_count = len(tokenizer_enc.encode(entry['text'], disallowed_special=()))
                    if token_count > max_tokens:
                        filtered_count += 1
                        continue

                # Store all fields from JSONL
                item = {
                    'id': entry['id'],
                    'text': entry['text'],
                    'length': len(entry['text'])  # For sorting
                }

                # Add optional fields if they exist
                if 'source' in entry:
                    item['source'] = entry['source']
                if 'category' in entry:
                    item['category'] = entry['category']
                if 'token_size' in entry:
                    item['token_size'] = entry['token_size']

                self.data.append(item)

        # Sort by length for better batching efficiency
        print("Sorting by length for dynamic batching...")
        self.data.sort(key=lambda x: x['length'])

        print(f"Loaded {len(self.data):,} paragraphs")
        if max_tokens and filtered_count > 0:
            print(f"Filtered {filtered_count:,} paragraphs exceeding {max_tokens} tokens ({filtered_count/total_loaded*100:.1f}%)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TokenBatchSampler:
    """Batch sampler that groups items by total token count."""

    def __init__(self, dataset, max_batch_items, target_tokens, max_seq_length):
        self.dataset = dataset
        self.max_batch_items = max_batch_items
        self.target_tokens = target_tokens
        self.max_seq_length = max_seq_length
        self.tokenizer_enc = tiktoken.get_encoding("cl100k_base")

    def __iter__(self):
        batch = []
        batch_tokens = 0

        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            # Estimate tokens for this item
            item_tokens = min(
                len(self.tokenizer_enc.encode(item['text'], disallowed_special=())),
                self.max_seq_length
            )

            # Check if adding this item would exceed limits
            would_exceed_tokens = (batch_tokens + item_tokens) > self.target_tokens
            would_exceed_items = len(batch) >= self.max_batch_items

            if batch and (would_exceed_tokens or would_exceed_items):
                # Yield current batch and start new one
                yield batch
                batch = [idx]
                batch_tokens = item_tokens
            else:
                # Add to current batch
                batch.append(idx)
                batch_tokens += item_tokens

        # Yield final batch
        if batch:
            yield batch

    def __len__(self):
        # Estimate number of batches
        return (len(self.dataset) + self.max_batch_items - 1) // self.max_batch_items


def collate_fn(batch_indices, dataset):
    """Custom collate function that preserves all metadata."""
    batch_data = [dataset[idx] for idx in batch_indices]

    result = {
        'id': [item['id'] for item in batch_data],
        'text': [item['text'] for item in batch_data],
    }

    # Add optional fields if they exist in first item
    if batch_data and 'source' in batch_data[0]:
        result['source'] = [item.get('source', '') for item in batch_data]
    if batch_data and 'category' in batch_data[0]:
        result['category'] = [item.get('category', '') for item in batch_data]
    if batch_data and 'token_size' in batch_data[0]:
        result['token_size'] = [item.get('token_size', 0) for item in batch_data]

    return result


def mean_pooling(token_embeddings, attention_mask):
    """Optimized mean pooling."""
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def main():
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Enable TF32 for faster matmul on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ TF32 enabled for faster computation")

    # Load model and tokenizer
    print(f"\nLoading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Try to load with Flash Attention 2 for 2-3x speedup
    try:
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
            attn_implementation="flash_attention_2"
        )
        print("✓ Flash Attention 2 enabled (2-3x speedup)")
    except Exception as e:
        print(f"⚠ Flash Attention 2 not available, using default: {e}")
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
        )

    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully (dtype: {model.dtype})")

    # Load dataset with token filtering
    dataset = JSONLDataset(INPUT_FILE, max_tokens=MAX_SEQ_LENGTH)

    # Get target tokens from config
    TARGET_TOKENS = embeddings_config.get('target_tokens_per_batch', 400000)

    # Create dynamic batch sampler
    batch_sampler = TokenBatchSampler(
        dataset,
        max_batch_items=BATCH_SIZE,
        target_tokens=TARGET_TOKENS,
        max_seq_length=MAX_SEQ_LENGTH
    )

    print(f"Dynamic batching: target={TARGET_TOKENS:,} tokens/batch, max={BATCH_SIZE} items/batch")

    # Create custom collate function that receives dataset
    def collate_wrapper(batch_indices):
        return collate_fn(batch_indices, dataset)

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        collate_fn=collate_wrapper,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    # Prepare for streaming writes
    print(f"\nGenerating embeddings...")

    # Create output directory if needed
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Check for existing checkpoint
    CHECKPOINT_FILE = OUTPUT_FILE.parent / f"{OUTPUT_FILE.stem}.checkpoint"
    TEMP_OUTPUT_FILE = OUTPUT_FILE.parent / f"{OUTPUT_FILE.stem}.temp.parquet"

    start_idx = 0
    if TEMP_OUTPUT_FILE.exists() and CHECKPOINT_FILE.exists():
        # Resume from checkpoint
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint_data = json.load(f)
            start_idx = checkpoint_data.get('processed', 0)
        print(f"✓ Found checkpoint: resuming from {start_idx:,} / {len(dataset):,} paragraphs")
        print(f"  Progress: {start_idx / len(dataset) * 100:.1f}%")
    else:
        # Start fresh
        if TEMP_OUTPUT_FILE.exists():
            TEMP_OUTPUT_FILE.unlink()
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()

    # We'll write in chunks to avoid memory issues
    WRITE_CHUNK_SIZE = 10000
    CHECKPOINT_INTERVAL = 50000  # Save checkpoint every 50K paragraphs
    chunk_data = []  # Will store dicts with all fields
    total_processed = start_idx

    # Use parquet writer for streaming
    writer = None
    # Schema with all metadata fields (FP16 embeddings)
    schema = pa.schema([
        ('id', pa.string()),
        ('text', pa.string()),
        ('embedding', pa.list_(pa.float16())),
        ('source', pa.string()),
        ('category', pa.string()),
        ('token_size', pa.int32())
    ])

    # If resuming, open existing file for appending
    if start_idx > 0 and TEMP_OUTPUT_FILE.exists():
        writer = pq.ParquetWriter(TEMP_OUTPUT_FILE, schema, compression='snappy')
        print(f"✓ Opened checkpoint file for appending")

    with torch.no_grad():
        # Use autocast for mixed precision if on CUDA
        use_amp = device.type == 'cuda'

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Embedding", initial=start_idx // BATCH_SIZE, total=len(dataloader))):
            # Skip already processed batches
            batch_start_idx = batch_idx * BATCH_SIZE
            if batch_start_idx < start_idx:
                continue

            # Tokenize
            encoded = tokenizer(
                batch['text'],
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors='pt'
            )

            # Move to device (non-blocking for better performance)
            input_ids = encoded['input_ids'].to(device, non_blocking=True)
            attention_mask = encoded['attention_mask'].to(device, non_blocking=True)

            # Get embeddings with automatic mixed precision
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    embeddings = mean_pooling(outputs[0], attention_mask)
                    embeddings = F.normalize(embeddings, p=2, dim=1)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = mean_pooling(outputs[0], attention_mask)
                embeddings = F.normalize(embeddings, p=2, dim=1)

            # Move back to CPU and store with all metadata (convert to FP16)
            embeddings_np = embeddings.cpu().half().numpy()  # FP16 instead of FP32

            for i, embedding in enumerate(embeddings_np):
                row_data = {
                    'id': batch['id'][i],
                    'text': batch['text'][i],
                    'embedding': embedding,
                    'source': batch.get('source', [''])[i] if 'source' in batch else '',
                    'category': batch.get('category', [''])[i] if 'category' in batch else '',
                    'token_size': batch.get('token_size', [0])[i] if 'token_size' in batch else 0
                }
                chunk_data.append(row_data)

            total_processed += len(batch['id'])

            # Write chunk to parquet when we have enough
            if len(chunk_data) >= WRITE_CHUNK_SIZE:
                # Create PyArrow table for this chunk
                chunk_table = pa.table({
                    'id': [r['id'] for r in chunk_data],
                    'text': [r['text'] for r in chunk_data],
                    'embedding': [list(r['embedding']) for r in chunk_data],
                    'source': [r['source'] for r in chunk_data],
                    'category': [r['category'] for r in chunk_data],
                    'token_size': [r['token_size'] for r in chunk_data]
                })

                # Write to parquet (append mode)
                if writer is None:
                    writer = pq.ParquetWriter(TEMP_OUTPUT_FILE, schema, compression='snappy')

                writer.write_table(chunk_table)

                # Clear chunk buffer
                chunk_data = []

            # Save checkpoint periodically
            if total_processed % CHECKPOINT_INTERVAL == 0 and total_processed > start_idx:
                checkpoint_data = {
                    'processed': total_processed,
                    'total': len(dataset),
                    'progress': total_processed / len(dataset)
                }
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump(checkpoint_data, f)
                print(f"\n✓ Checkpoint saved: {total_processed:,} / {len(dataset):,} ({total_processed / len(dataset) * 100:.1f}%)")

    # Write any remaining data
    if chunk_data:
        chunk_table = pa.table({
            'id': [r['id'] for r in chunk_data],
            'text': [r['text'] for r in chunk_data],
            'embedding': [list(r['embedding']) for r in chunk_data],
            'source': [r['source'] for r in chunk_data],
            'category': [r['category'] for r in chunk_data],
            'token_size': [r['token_size'] for r in chunk_data]
        })

        if writer is None:
            writer = pq.ParquetWriter(TEMP_OUTPUT_FILE, schema, compression='snappy')

        writer.write_table(chunk_table)

    # Close writer
    if writer is not None:
        writer.close()

    # Rename temp file to final output
    if TEMP_OUTPUT_FILE.exists():
        if OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()
        TEMP_OUTPUT_FILE.rename(OUTPUT_FILE)
        print(f"\n✓ Moved temp file to final output")

    # Clean up checkpoint file
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    print(f"\n✓ Generated {total_processed:,} embeddings")
    print(f"✓ Embeddings saved successfully!")
    print(f"  File: {OUTPUT_FILE}")
    print(f"  Size: {OUTPUT_FILE.stat().st_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    main()
