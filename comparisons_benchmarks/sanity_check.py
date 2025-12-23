#!/usr/bin/env python3
"""
Quick sanity check: embed test data, compare against ONE parquet file (all contents).
Simple and fast - just read one file and compare.
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
import os


def embed_texts(texts, model, tokenizer, device, batch_size=16):
    """Embed texts in batches."""
    embeddings = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                          max_length=512, return_tensors='pt').to(device)
            out = model(**enc)
            mask = enc['attention_mask'].unsqueeze(-1).to(out[0].dtype)
            emb = (out[0] * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = F.normalize(emb, p=2, dim=1)
            embeddings.append(emb.float().cpu().numpy())
    return np.vstack(embeddings)


def load_parquet_data(parquet_path: Path):
    """Load ALL embeddings, texts, and IDs from a single parquet file."""
    print(f"Loading {parquet_path.name}...")
    
    pf = pq.ParquetFile(str(parquet_path))
    cols = [f.name for f in pf.schema_arrow]
    
    if 'embeddings' not in cols:
        raise ValueError(f"No 'embeddings' column in {parquet_path.name}")
    
    # Find text and ID columns
    text_col = None
    for c in ['text', 'content', 'paragraph', 'document']:
        if c in cols:
            text_col = c
            break
    
    id_col = None
    for c in ['id', 'hash_id', 'doc_hash', 'doc_id']:
        if c in cols:
            id_col = c
            break
    
    print(f"  Found columns: embeddings, text={text_col}, id={id_col}")
    
    # Get embedding dimension from schema
    emb_field = pf.schema_arrow.field('embeddings')
    emb_type = emb_field.type
    if hasattr(emb_type, 'list_size'):
        dim = emb_type.list_size
    else:
        dim = 4096
    
    print(f"  Embedding dimension: {dim}")
    
    # Determine columns to read
    cols_to_read = ['embeddings']
    if text_col:
        cols_to_read.append(text_col)
    if id_col:
        cols_to_read.append(id_col)
    
    # Read all row groups, process chunks without combining
    all_mats = []
    all_texts = []
    all_ids = []
    
    for rg_idx in range(pf.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=cols_to_read)
        emb_col = table['embeddings']
        
        # Extract texts and IDs for this row group (aligned with embeddings)
        if text_col:
            texts_rg = table[text_col].to_pylist()
        if id_col:
            ids_rg = table[id_col].to_pylist()
        
        # Track offset for this row group
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
            row_offset += n
            
            # Extract values from fixed_size_list chunk
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
            all_mats.append(mat)
    
    if not all_mats:
        raise ValueError("No embeddings found in file")
    
    final_mat = np.vstack(all_mats) if len(all_mats) > 1 else all_mats[0]
    print(f"  Loaded {len(final_mat):,} embeddings (dim={final_mat.shape[1]})")
    
    # Generate IDs if not available
    if not all_ids:
        all_ids = [f"{parquet_path.stem}_{i}" for i in range(len(final_mat))]
    
    return final_mat, all_texts, all_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet-file', required=True, help='Single parquet file to check')
    parser.add_argument('--benchmark', default='musr', choices=['musr', 'mbpp', 'humaneval'],
                       help='Benchmark dataset to compare against')
    parser.add_argument('--test-idx', type=int, default=0, help='Which test example to use')
    args = parser.parse_args()

    parquet_path = Path(args.parquet_file)
    if not parquet_path.exists():
        raise FileNotFoundError(f"File not found: {parquet_path}")

    # Load benchmark test data
    print(f"Loading {args.benchmark} dataset...")
    if args.benchmark == 'musr':
        ds = load_dataset("TAUR-Lab/MuSR")
        split = list(ds.keys())[0]
        item = ds[split][args.test_idx]
        test_text = f"{item.get('narrative', item.get('question', ''))}\n\n{item.get('answer', '')}"
    elif args.benchmark == 'mbpp':
        try:
            ds = load_dataset("evalplus/mbpp", "mbpp")
        except:
            ds = load_dataset("google-research-datasets/mbpp", "sanitized")
        item = ds['test'][args.test_idx]
        test_text = f"{item.get('prompt', item.get('text', ''))}\n\n{item.get('canonical_solution', item.get('code', ''))}"
    elif args.benchmark == 'humaneval':
        ds = load_dataset("openai/openai_humaneval")
        item = ds['test'][args.test_idx]
        test_text = f"{item.get('prompt', '')}\n\n{item.get('canonical_solution', '')}"
    
    print(f"\n=== Test text (first 500 chars) ===")
    print(test_text[:500])
    print("...")

    # Load model
    print("\nLoading embedding model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "nvidia/llama-embed-nemotron-8b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device).eval()

    # Embed test text
    print("Embedding test text...")
    test_emb = embed_texts([test_text], model, tokenizer, device)[0]
    print(f"Test embedding shape: {test_emb.shape}")

    # Free model memory
    del model, tokenizer
    torch.cuda.empty_cache()

    # Load ALL embeddings, texts, and IDs from the parquet file
    corpus_embs, corpus_texts, corpus_ids = load_parquet_data(parquet_path)

    # Compute similarities (both are already normalized)
    print("\nComputing cosine similarities...")
    similarities = corpus_embs @ test_emb

    # Get top-100
    top_k = 100
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_scores = similarities[top_indices]

    print(f"\n=== Top-{top_k} Similarity Distribution ===")
    print(f"Top-1:   {top_scores[0]:.4f}")
    print(f"Top-10:  {top_scores[9]:.4f}")
    print(f"Top-50:  {top_scores[49]:.4f}")
    print(f"Top-100: {top_scores[99]:.4f}")
    print(f"Mean:    {top_scores.mean():.4f}")
    print(f"Std:     {top_scores.std():.4f}")

    print(f"\n=== Overall Distribution ===")
    print(f"Max:     {similarities.max():.4f}")
    print(f"Min:     {similarities.min():.4f}")
    print(f"Mean:    {similarities.mean():.4f}")
    print(f"Median:  {np.median(similarities):.4f}")
    print(f"Std:     {similarities.std():.4f}")

    print(f"\n=== Top 10 matches ===")
    for i in range(min(10, len(top_scores))):
        idx = top_indices[i]
        print(f"  {i+1}. score={top_scores[i]:.4f}, id={corpus_ids[idx]}, index={idx}")
    
    # Create output directory
    output_dir = Path("sanity_check_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Generate base filename from parquet file and benchmark
    base_name = f"{parquet_path.stem}_{args.benchmark}_{args.test_idx}"
    
    # Create plots
    print(f"\nGenerating plots in {output_dir}...")
    
    # 1. Histogram of all similarities
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=100, alpha=0.7, edgecolor='black')
    plt.axvline(top_scores[0], color='r', linestyle='--', label=f'Top-1: {top_scores[0]:.4f}')
    plt.axvline(similarities.mean(), color='g', linestyle='--', label=f'Mean: {similarities.mean():.4f}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title(f'Similarity Distribution: {parquet_path.name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_histogram.png", dpi=150)
    plt.close()
    
    # 2. Top-k similarity scores
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, top_k + 1), top_scores, marker='o', markersize=3, linewidth=1.5)
    plt.xlabel('Rank')
    plt.ylabel('Cosine Similarity')
    plt.title(f'Top-{top_k} Similarity Scores')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_topk.png", dpi=150)
    plt.close()
    
    # 3. Cumulative distribution
    sorted_sims = np.sort(similarities)[::-1]
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(sorted_sims)), sorted_sims, linewidth=1)
    plt.xlabel('Rank (sorted)')
    plt.ylabel('Cosine Similarity')
    plt.title('Cumulative Similarity Distribution (sorted)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_cumulative.png", dpi=150)
    plt.close()
    
    # Save top 100 comparisons with text and IDs
    print(f"Saving top {top_k} comparisons...")
    top_comparisons = []
    for i in range(top_k):
        idx = top_indices[i]
        comparison = {
            'rank': i + 1,
            'score': float(top_scores[i]),
            'index': int(idx),
            'id': str(corpus_ids[idx]),
            'text': corpus_texts[idx] if corpus_texts and idx < len(corpus_texts) else None
        }
        top_comparisons.append(comparison)
    
    # Save as JSON
    output_file = output_dir / f"{base_name}_top100.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'test_text': test_text,
            'benchmark': args.benchmark,
            'test_idx': args.test_idx,
            'parquet_file': str(parquet_path),
            'total_embeddings': len(corpus_embs),
            'top_100': top_comparisons
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved plots and top {top_k} comparisons to {output_dir}/")
    print(f"   - {base_name}_histogram.png")
    print(f"   - {base_name}_topk.png")
    print(f"   - {base_name}_cumulative.png")
    print(f"   - {base_name}_top100.json")


if __name__ == "__main__":
    main()
