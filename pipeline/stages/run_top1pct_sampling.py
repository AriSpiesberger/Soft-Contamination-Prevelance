#!/usr/bin/env python3
"""
Standalone Top 1% Sampler for Semantic Duplicate Analysis

This script runs a lightweight contamination analysis pass ONLY for MBPP,
sampling 100 random items from the top 1% of similarity scores per test point.

Does NOT modify any production code - completely standalone.

Usage:
    # Single config
    python run_top1pct_sampling.py --config configs/dolci.yaml --output samples_dolci.csv
    
    # All dolci configs
    python run_top1pct_sampling.py --all-dolci --output-dir ./semantic_samples/

Requirements:
    - Embedding parquet files must exist (from stage 03)
    - Corpus JSONL must exist (from stage 02)
"""

import os
import sys
import gc
import json
import yaml
import time
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
from collections import defaultdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))


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


def load_mbpp_benchmark():
    """Load MBPP test data."""
    from datasets import load_dataset
    
    print("Loading MBPP benchmark...")
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    
    test_texts = []
    test_ids = []
    
    for item in ds:
        test_id = str(item['task_id'])
        # Combine prompt + code for input_output mode
        test_text = item['prompt']
        if item.get('code'):
            test_text += "\n\n" + item['code']
        test_texts.append(test_text)
        test_ids.append(test_id)
    
    print(f"✓ Loaded {len(test_ids)} MBPP test points")
    return test_texts, test_ids


def embed_texts(texts, model, tokenizer, device, batch_size=8):
    """Embed texts in batches using FP16."""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                          max_length=512, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


def load_corpus_embeddings_streaming(parquet_path):
    """Generator that yields chunks of corpus embeddings."""
    import duckdb
    
    con = duckdb.connect()
    pf_path = str(parquet_path)
    
    # Get total rows
    total_rows = con.execute(f"SELECT COUNT(*) FROM '{pf_path}'").fetchone()[0]
    
    chunk_size = 100_000
    for offset in range(0, total_rows, chunk_size):
        result = con.execute(f"""
            SELECT id, text, embedding 
            FROM '{pf_path}' 
            LIMIT {chunk_size} OFFSET {offset}
        """).fetchall()
        
        if not result:
            break
            
        ids = [str(row[0]) for row in result]
        texts = [row[1] for row in result]
        embeddings = np.array([row[2] for row in result], dtype=np.float32)
        
        yield embeddings, ids, texts, len(result)
    
    con.close()


def run_sampling(config_path, output_csv, n_samples=100):
    """Run top 1% sampling for a single config."""
    
    config = load_config(config_path)
    pipeline_root = Path(config_path).parent.parent
    
    # Get paths from config
    analysis_cfg = config.get('analysis', {})
    finalize_cfg = config.get('finalize', {})
    embeddings_cfg = config.get('embeddings', {})
    
    # Corpus embeddings (parquet)
    corpus_dir = analysis_cfg.get('corpus_dir', '')
    corpus_file = analysis_cfg.get('corpus_file', 'embeddings.parquet')
    
    if embeddings_cfg.get('mode') == 'local':
        corpus_parquet = Path(embeddings_cfg.get('local', {}).get('output_file', ''))
    else:
        corpus_parquet = Path(corpus_dir) / corpus_file
    
    if not corpus_parquet.is_absolute():
        corpus_parquet = pipeline_root / corpus_parquet
    
    # Corpus text (JSONL)
    corpus_jsonl = Path(finalize_cfg.get('corpus_file', ''))
    if not corpus_jsonl.is_absolute():
        corpus_jsonl = pipeline_root / corpus_jsonl
    
    dataset_name = config.get('pipeline', {}).get('dataset_short_name', 
                   config.get('pipeline', {}).get('name', 'unknown'))
    
    print(f"\n{'='*70}")
    print(f"Config: {config_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Corpus Parquet: {corpus_parquet}")
    print(f"Corpus JSONL: {corpus_jsonl}")
    print(f"{'='*70}")
    
    # Validate paths
    if not corpus_parquet.exists():
        print(f"❌ Corpus parquet not found: {corpus_parquet}")
        return None
    
    # Load corpus text mapping
    id_to_text = load_corpus_text_mapping(corpus_jsonl)
    
    # Load MBPP test data
    test_texts, test_ids = load_mbpp_benchmark()
    
    # Setup GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load embedding model
    print("Loading embedding model...")
    from transformers import AutoModel, AutoTokenizer
    
    model_name = embeddings_cfg.get('model', 'nvidia/llama-embed-nemotron-8b')
    # Use a smaller model for test embedding if the main one is too large
    # For now, we'll compute similarities directly without re-embedding tests
    
    # Actually, we need to embed the test texts with the same model used for corpus
    # But that model is large. Let's check if test embeddings are cached somewhere...
    
    # Alternative approach: Load corpus embeddings and compute sims in GPU chunks
    # This avoids needing to load the embedding model
    
    print("Loading corpus embeddings and computing similarities...")
    
    # First, embed test texts using a lightweight approach
    # We'll use the same model that was used for corpus embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, 
                                       torch_dtype=torch.float16).to(device)
    model.eval()
    
    print("Embedding MBPP test texts...")
    test_embeddings = embed_texts(test_texts, model, tokenizer, device, batch_size=4)
    test_embeddings = torch.from_numpy(test_embeddings).to(device).half()
    print(f"✓ Test embeddings shape: {test_embeddings.shape}")
    
    # Free model memory
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Process corpus in streaming fashion
    # For each test point, we need to track:
    # 1. All similarity scores (or a reservoir sample for percentile estimation)
    # 2. Top 1% candidates with their hash_ids
    
    # Two-pass approach:
    # Pass 1: Compute all similarities, estimate 99th percentile per test
    # Pass 2: Sample from top 1%
    
    # For memory efficiency, use reservoir sampling to estimate percentiles
    RESERVOIR_SIZE = 100_000  # Per test point
    
    print("\nPass 1: Computing similarities and estimating 99th percentile...")
    
    # Initialize reservoirs for each test
    reservoirs = [[] for _ in range(len(test_ids))]
    counts = [0 for _ in range(len(test_ids))]
    
    # Also keep track of top-1% candidates (we'll over-collect and filter later)
    # Use a min-heap to keep top N candidates per test
    import heapq
    TOP_CANDIDATES = 10000  # Keep this many candidates, then filter to top 1%
    candidate_heaps = [[] for _ in range(len(test_ids))]  # (score, hash_id)
    
    total_corpus = 0
    
    for corpus_embs, corpus_ids, corpus_texts, chunk_size in tqdm(
        load_corpus_embeddings_streaming(corpus_parquet), 
        desc="Processing corpus"
    ):
        total_corpus += chunk_size
        
        # Move to GPU
        corpus_gpu = torch.from_numpy(corpus_embs).to(device).half()
        
        # Compute similarities: (num_tests, chunk_size)
        with torch.no_grad():
            sims = torch.matmul(test_embeddings, corpus_gpu.t())
            sims_np = sims.float().cpu().numpy()
        
        del corpus_gpu
        
        # Update reservoirs and candidate heaps for each test
        for test_idx in range(len(test_ids)):
            test_sims = sims_np[test_idx]
            counts[test_idx] += len(test_sims)
            
            # Reservoir sampling for percentile estimation
            for i, sim in enumerate(test_sims):
                if len(reservoirs[test_idx]) < RESERVOIR_SIZE:
                    reservoirs[test_idx].append(sim)
                else:
                    j = np.random.randint(0, counts[test_idx])
                    if j < RESERVOIR_SIZE:
                        reservoirs[test_idx][j] = sim
            
            # Keep top candidates (score, hash_id)
            for i, sim in enumerate(test_sims):
                hash_id = corpus_ids[i]
                if len(candidate_heaps[test_idx]) < TOP_CANDIDATES:
                    heapq.heappush(candidate_heaps[test_idx], (sim, hash_id))
                elif sim > candidate_heaps[test_idx][0][0]:
                    heapq.heapreplace(candidate_heaps[test_idx], (sim, hash_id))
        
        del sims_np
        gc.collect()
    
    print(f"\n✓ Processed {total_corpus:,} corpus embeddings")
    
    # Pass 2: For each test, compute 99th percentile and sample from top 1%
    print("\nPass 2: Sampling from top 1% for each test point...")
    
    all_samples = []
    
    for test_idx, test_id in enumerate(tqdm(test_ids, desc="Sampling")):
        reservoir = np.array(reservoirs[test_idx])
        if len(reservoir) == 0:
            continue
        
        # Estimate 99th percentile
        threshold = np.percentile(reservoir, 99)
        
        # Get candidates above threshold
        candidates = [(score, hash_id) for score, hash_id in candidate_heaps[test_idx] 
                      if score >= threshold]
        
        if not candidates:
            # Fallback: use all candidates we have
            candidates = candidate_heaps[test_idx]
        
        if not candidates:
            continue
        
        # Sample up to n_samples
        sample_size = min(n_samples, len(candidates))
        sampled = np.random.choice(len(candidates), size=sample_size, replace=False)
        
        test_text = test_texts[test_idx]
        
        for idx in sampled:
            score, hash_id = candidates[idx]
            corpus_text = id_to_text.get(hash_id, '')
            
            all_samples.append({
                'dataset': dataset_name,
                'test_id': test_id,
                'test_text': test_text,
                'corpus_id': hash_id,
                'corpus_text': corpus_text,
                'similarity': float(score),
                'threshold_99pct': float(threshold),
                'total_corpus': total_corpus
            })
    
    if not all_samples:
        print("⚠️  No samples collected")
        return None
    
    # Save results
    df = pd.DataFrame(all_samples)
    df = df.sort_values(['test_id', 'similarity'], ascending=[True, False])
    
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Saved {len(df):,} samples to {output_csv}")
    print(f"   Test points: {df['test_id'].nunique()}")
    print(f"   Top similarity: {df['similarity'].max():.4f}")
    print(f"   Mean 99th pct threshold: {df['threshold_99pct'].mean():.4f}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Sample top 1% for semantic duplicate analysis")
    parser.add_argument('--config', '-c', help='Single config file to process')
    parser.add_argument('--all-dolci', action='store_true', help='Process all dolci configs')
    parser.add_argument('--output', '-o', help='Output CSV (for single config)')
    parser.add_argument('--output-dir', help='Output directory (for multiple configs)')
    parser.add_argument('--samples', '-n', type=int, default=100, help='Samples per test point')
    
    args = parser.parse_args()
    
    pipeline_root = Path(__file__).parent.parent
    configs_dir = pipeline_root / "configs"
    
    if args.all_dolci:
        # Process dolci configs
        dolci_configs = [
            configs_dir / "dolci.yaml",
            configs_dir / "dolci_dpo.yaml", 
            configs_dir / "dolci_rl.yaml",
        ]
        
        output_dir = Path(args.output_dir or pipeline_root / "semantic_samples")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for config_file in dolci_configs:
            if not config_file.exists():
                print(f"⚠️  Config not found: {config_file}")
                continue
            
            output_csv = output_dir / f"top1pct_{config_file.stem}.csv"
            try:
                run_sampling(config_file, output_csv, args.samples)
            except Exception as e:
                print(f"❌ Error processing {config_file.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    elif args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = configs_dir / args.config
        
        output_csv = args.output or f"top1pct_{config_path.stem}.csv"
        run_sampling(config_path, output_csv, args.samples)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
