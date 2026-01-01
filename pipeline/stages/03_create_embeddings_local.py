#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Embedding Generation Script
Reads from local JSONL file and writes embeddings to local parquet file.
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
print(f"LOCAL EMBEDDING GENERATION")
print(f"=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max sequence length: {MAX_SEQ_LENGTH}")
print(f"=" * 80)


class JSONLDataset(Dataset):
    """Dataset for reading JSONL paragraphs."""

    def __init__(self, jsonl_path):
        self.paragraphs = []
        self.ids = []

        print(f"\nLoading paragraphs from {jsonl_path}...")
        with open(jsonl_path, 'r') as f:
            for line in tqdm(f, desc="Reading JSONL"):
                data = json.loads(line)
                self.paragraphs.append(data['text'])
                self.ids.append(data['id'])

        print(f"Loaded {len(self.paragraphs):,} paragraphs")

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx):
        return {
            'id': self.ids[idx],
            'text': self.paragraphs[idx]
        }


def mean_pooling(model_output, attention_mask):
    """Mean pooling to get sentence embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def main():
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load model and tokenizer
    print(f"\nLoading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Load dataset
    dataset = JSONLDataset(INPUT_FILE)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Generate embeddings
    print(f"\nGenerating embeddings...")
    all_ids = []
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Embedding"):
            # Tokenize
            encoded = tokenizer(
                batch['text'],
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors='pt'
            )

            # Move to device
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            # Get embeddings
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = mean_pooling(outputs, attention_mask)

            # Normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Move back to CPU and store
            all_ids.extend(batch['id'])
            all_embeddings.append(embeddings.cpu().numpy())

    # Concatenate all embeddings
    import numpy as np
    all_embeddings = np.vstack(all_embeddings)

    print(f"\nGenerated {len(all_ids):,} embeddings")
    print(f"Embedding shape: {all_embeddings.shape}")

    # Save to parquet
    print(f"\nSaving embeddings to {OUTPUT_FILE}...")

    # Create output directory if needed
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Create PyArrow table
    table = pa.table({
        'id': all_ids,
        'embedding': list(all_embeddings)
    })

    # Write to parquet
    pq.write_table(table, OUTPUT_FILE, compression='snappy')

    print(f"✓ Embeddings saved successfully!")
    print(f"  File: {OUTPUT_FILE}")
    print(f"  Size: {OUTPUT_FILE.stat().st_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    main()
