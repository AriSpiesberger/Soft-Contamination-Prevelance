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
    """Dataset for reading JSONL paragraphs with length-based sorting."""

    def __init__(self, jsonl_path):
        self.data = []

        print(f"\nLoading paragraphs from {jsonl_path}...")
        with open(jsonl_path, 'r') as f:
            for line in tqdm(f, desc="Reading JSONL"):
                entry = json.loads(line)
                self.data.append({
                    'id': entry['id'],
                    'text': entry['text'],
                    'length': len(entry['text'])  # Approximate length for sorting
                })

        # Sort by length for better batching efficiency
        print("Sorting by length for dynamic batching...")
        self.data.sort(key=lambda x: x['length'])

        print(f"Loaded {len(self.data):,} paragraphs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'id': self.data[idx]['id'],
            'text': self.data[idx]['text']
        }


def collate_fn(batch):
    """Custom collate function for efficient batching."""
    return {
        'id': [item['id'] for item in batch],
        'text': [item['text'] for item in batch]
    }


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

    # Load dataset
    dataset = JSONLDataset(INPUT_FILE)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Already sorted by length
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        collate_fn=collate_fn,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    # Prepare for streaming writes
    print(f"\nGenerating embeddings...")

    # Create output directory if needed
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # We'll write in chunks to avoid memory issues
    WRITE_CHUNK_SIZE = 10000
    chunk_ids = []
    chunk_embeddings = []
    total_processed = 0

    # Use parquet writer for streaming
    writer = None
    schema = pa.schema([
        ('id', pa.string()),
        ('embedding', pa.list_(pa.float32()))
    ])

    with torch.no_grad():
        # Use autocast for mixed precision if on CUDA
        use_amp = device.type == 'cuda'

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Embedding")):
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

            # Move back to CPU and store
            chunk_ids.extend(batch['id'])
            chunk_embeddings.append(embeddings.cpu().float().numpy())
            total_processed += len(batch['id'])

            # Write chunk to parquet when we have enough
            if len(chunk_ids) >= WRITE_CHUNK_SIZE:
                embeddings_array = np.vstack(chunk_embeddings)

                # Create PyArrow table for this chunk
                chunk_table = pa.table({
                    'id': chunk_ids,
                    'embedding': list(embeddings_array)
                })

                # Write to parquet (append mode)
                if writer is None:
                    writer = pq.ParquetWriter(OUTPUT_FILE, schema, compression='snappy')

                writer.write_table(chunk_table)

                # Clear chunk buffers
                chunk_ids = []
                chunk_embeddings = []

    # Write any remaining data
    if chunk_ids:
        embeddings_array = np.vstack(chunk_embeddings)
        chunk_table = pa.table({
            'id': chunk_ids,
            'embedding': list(embeddings_array)
        })

        if writer is None:
            writer = pq.ParquetWriter(OUTPUT_FILE, schema, compression='snappy')

        writer.write_table(chunk_table)

    # Close writer
    if writer is not None:
        writer.close()

    print(f"\n✓ Generated {total_processed:,} embeddings")
    print(f"✓ Embeddings saved successfully!")
    print(f"  File: {OUTPUT_FILE}")
    print(f"  Size: {OUTPUT_FILE.stat().st_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    main()
