#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# Configuration values
local_config = embeddings_config.get('local', {})
MODEL_NAME = embeddings_config.get('model', 'nvidia/llama-embed-nemotron-8b')
INPUT_FILE = local_config.get('input_file', './data/dolmino_random_paragraphs.jsonl')
OUTPUT_FILE = local_config.get('output_file', './data/dolmino_embeddings.parquet')
MAX_SEQ_LENGTH = embeddings_config.get('max_seq_length', 512)
BATCH_SIZE = embeddings_config.get('max_batch_size', 32)
NUM_WORKERS = embeddings_config.get('num_loader_workers', 4)
PREFETCH_FACTOR = embeddings_config.get('prefetch_factor', 4)

def resolve_path(path_str):
    path = Path(path_str)
    return path if path.is_absolute() else PIPELINE_ROOT / path

INPUT_FILE = resolve_path(INPUT_FILE)
OUTPUT_FILE = resolve_path(OUTPUT_FILE)

print(f"=" * 80)
print(f"OPTIMIZED LOCAL EMBEDDING GENERATION")
print(f"=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"=" * 80)

class JSONLDataset(Dataset):
    def __init__(self, jsonl_path, max_tokens=None):
        self.data = []
        tokenizer_enc = tiktoken.get_encoding("cl100k_base")
        print(f"\nLoading paragraphs from {jsonl_path}...")
        with open(jsonl_path, 'r') as f:
            for line in tqdm(f, desc="Reading JSONL"):
                entry = json.loads(line)
                if max_tokens:
                    token_count = len(tokenizer_enc.encode(entry['text'], disallowed_special=()))
                    if token_count > max_tokens: continue
                
                item = {
                    'id': entry['id'],
                    'text': entry['text'],
                    'length': len(entry['text']),
                    'source': entry.get('source', ''),
                    'category': entry.get('category', ''),
                    'token_size': entry.get('token_size', 0)
                }
                self.data.append(item)

        print("Sorting by length for dynamic batching...")
        self.data.sort(key=lambda x: x['length'])
        print(f"Loaded {len(self.data):,} paragraphs")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

class TokenBatchSampler:
    def __init__(self, dataset, max_batch_items, target_tokens, max_seq_length):
        self.dataset = dataset
        self.max_batch_items = max_batch_items
        self.target_tokens = target_tokens
        self.max_seq_length = max_seq_length
        self.tokenizer_enc = tiktoken.get_encoding("cl100k_base")

    def __iter__(self):
        batch, batch_tokens = [], 0
        for idx in range(len(self.dataset)):
            item_tokens = min(len(self.tokenizer_enc.encode(self.dataset[idx]['text'], disallowed_special=())), self.max_seq_length)
            if batch and ((batch_tokens + item_tokens) > self.target_tokens or len(batch) >= self.max_batch_items):
                yield batch
                batch, batch_tokens = [idx], item_tokens
            else:
                batch.append(idx)
                batch_tokens += item_tokens
        if batch: yield batch

    def __len__(self): return (len(self.dataset) + self.max_batch_items - 1) // self.max_batch_items

def collate_fn(batch_data):
    return {
        'id': [item['id'] for item in batch_data],
        'text': [item['text'] for item in batch_data],
        'source': [item['source'] for item in batch_data],
        'category': [item['category'] for item in batch_data],
        'token_size': [item['token_size'] for item in batch_data]
    }

def mean_pooling(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    return torch.sum(token_embeddings * mask, dim=1) / mask.sum(dim=1).clamp(min=1e-9)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    try:
        model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, 
                                          torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
                                          attn_implementation="flash_attention_2").to(device)
        print("✓ Flash Attention 2 enabled")
    except Exception:
        model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, 
                                          torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32).to(device)
    
    model.eval()
    dataset = JSONLDataset(INPUT_FILE, max_tokens=MAX_SEQ_LENGTH)
    
    batch_sampler = TokenBatchSampler(dataset, BATCH_SIZE, 
                                      embeddings_config.get('target_tokens_per_batch', 400000), 
                                      MAX_SEQ_LENGTH)

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == 'cuda'),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        collate_fn=collate_fn,
        persistent_workers=(NUM_WORKERS > 0)
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    TEMP_OUTPUT_FILE = OUTPUT_FILE.parent / f"{OUTPUT_FILE.stem}.temp.parquet"
    
    # Explicit schema to ensure small file size and type consistency
    schema = pa.schema([
        ('id', pa.string()), 
        ('text', pa.string()), 
        ('embedding', pa.list_(pa.float16())),
        ('source', pa.string()), 
        ('category', pa.string()), 
        ('token_size', pa.int32())
    ])

    writer = None
    chunk_data = []
    
    def flush_chunk(data_list, parquet_writer, arrow_schema):
        """Helper to convert list of dicts to Table and write."""
        if not data_list:
            return parquet_writer
            
        # This is the critical fix: passing schema=arrow_schema
        table = pa.table({
            'id': [d['id'] for d in data_list],
            'text': [d['text'] for d in data_list],
            'embedding': [d['embedding'] for d in data_list],
            'source': [d['source'] for d in data_list],
            'category': [d['category'] for d in data_list],
            'token_size': [d['token_size'] for d in data_list],
        }, schema=arrow_schema)
        
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(TEMP_OUTPUT_FILE, arrow_schema, compression='snappy')
        
        parquet_writer.write_table(table)
        return parquet_writer

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Embedding"):
            encoded = tokenizer(batch['text'], padding=True, truncation=True, 
                                max_length=MAX_SEQ_LENGTH, return_tensors='pt').to(device)
            
            # Using updated non-deprecated autocast syntax
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=torch.float16):
                outputs = model(**encoded)
                embeddings = F.normalize(mean_pooling(outputs[0], encoded['attention_mask']), p=2, dim=1)
            
            embeddings_np = embeddings.cpu().half().numpy()

            for i in range(len(batch['id'])):
                chunk_data.append({
                    'id': batch['id'][i], 
                    'text': batch['text'][i], 
                    'embedding': embeddings_np[i],
                    'source': batch['source'][i], 
                    'category': batch['category'][i], 
                    'token_size': batch['token_size'][i]
                })

            if len(chunk_data) >= 10000:
                writer = flush_chunk(chunk_data, writer, schema)
                chunk_data = []

    # Final flush
    if chunk_data:
        writer = flush_chunk(chunk_data, writer, schema)
    
    if writer: 
        writer.close()
    
    TEMP_OUTPUT_FILE.rename(OUTPUT_FILE)
    print(f"✓ Embeddings saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()