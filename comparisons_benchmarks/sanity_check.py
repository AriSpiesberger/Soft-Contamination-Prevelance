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


def load_parquet_embeddings(parquet_path: Path):
    """Load ALL embeddings from a single parquet file."""
    print(f"Loading {parquet_path.name}...")
    
    pf = pq.ParquetFile(str(parquet_path))
    cols = [f.name for f in pf.schema_arrow]
    
    if 'embeddings' not in cols:
        raise ValueError(f"No 'embeddings' column in {parquet_path.name}")
    
    # Get embedding dimension from schema
    emb_field = pf.schema_arrow.field('embeddings')
    emb_type = emb_field.type
    if hasattr(emb_type, 'list_size'):
        dim = emb_type.list_size
    else:
        # Fallback: assume 4096 (standard for your embeddings)
        dim = 4096
    
    print(f"  Embedding dimension: {dim}")
    
    # Read all row groups, process chunks without combining
    all_mats = []
    
    for rg_idx in range(pf.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=['embeddings'])
        emb_col = table['embeddings']
        
        for chunk_idx in range(emb_col.num_chunks):
            chunk = emb_col.chunk(chunk_idx)
            n = len(chunk)
            if n == 0:
                continue
            
            # Extract values from fixed_size_list chunk
            # Use flatten() to get underlying values array, then reshape
            try:
                # Flatten the list array to access underlying values
                values_chunk = chunk.flatten()
                vals = values_chunk.to_numpy()
                
                # Reshape: n rows, dim columns
                expected_size = n * dim
                if len(vals) == expected_size:
                    mat = vals.reshape(n, dim).astype(np.float32)
                else:
                    # Try to infer dimension
                    if len(vals) % n == 0:
                        inferred_dim = len(vals) // n
                        print(f"  Using inferred dim={inferred_dim} (schema said {dim}, vals={len(vals)}, n={n})")
                        mat = vals.reshape(n, inferred_dim).astype(np.float32)
                        dim = inferred_dim  # Update for consistency
                    else:
                        # Fallback: convert via Python (slower but works)
                        print(f"  Cannot reshape cleanly, using Python conversion")
                        py_list = chunk.to_pylist()
                        mat = np.array(py_list, dtype=np.float32)
            except Exception as e:
                # Fallback: convert via Python (slower but works)
                print(f"  Fallback to Python conversion for chunk {chunk_idx}: {e}")
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
    
    return final_mat


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

    # Load ALL embeddings from the parquet file
    corpus_embs = load_parquet_embeddings(parquet_path)

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
        print(f"  {i+1}. score={top_scores[i]:.4f}, index={top_indices[i]}")


if __name__ == "__main__":
    main()
