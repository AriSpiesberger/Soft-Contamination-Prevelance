#!/usr/bin/env python3
"""
Quick sanity check: embed 1 MUSR example with bf16, compare against 1M corpus paragraphs.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
import pyarrow.parquet as pq
from pathlib import Path
import argparse


def embed_single_text(text: str, model, tokenizer, device) -> np.ndarray:
    """Embed a single text using bfloat16."""
    with torch.inference_mode():
        enc = tokenizer([text], padding=True, truncation=True,
                       max_length=8192, return_tensors='pt').to(device)
        out = model(**enc)
        mask = enc['attention_mask'].unsqueeze(-1).to(out[0].dtype)
        emb = (out[0] * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = F.normalize(emb, p=2, dim=1)
        return emb.float().cpu().numpy()[0]


def load_corpus_embeddings(data_dir: Path, max_docs: int = 1_000_000) -> tuple:
    """Load up to max_docs embeddings from parquet files."""
    parquet_files = sorted(data_dir.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    all_embeddings = []
    all_ids = []
    total = 0

    for pf_path in tqdm(parquet_files, desc="Loading corpus"):
        if total >= max_docs:
            break

        pf = pq.ParquetFile(str(pf_path))
        table = pf.read()

        # Get embeddings column
        if 'embeddings' in table.column_names:
            emb_col = 'embeddings'
        elif 'embedding' in table.column_names:
            emb_col = 'embedding'
        else:
            print(f"No embeddings column in {pf_path.name}, cols: {table.column_names}")
            continue

        # Get ID column
        if 'id' in table.column_names:
            id_col = 'id'
        elif 'hash_id' in table.column_names:
            id_col = 'hash_id'
        else:
            id_col = None

        embeddings = table[emb_col].to_pylist()
        ids = table[id_col].to_pylist() if id_col else list(range(len(embeddings)))

        # Take only what we need
        remaining = max_docs - total
        embeddings = embeddings[:remaining]
        ids = ids[:remaining]

        all_embeddings.extend(embeddings)
        all_ids.extend(ids)
        total += len(embeddings)

        if total >= max_docs:
            break

    print(f"Loaded {len(all_embeddings)} embeddings")
    return np.array(all_embeddings, dtype=np.float32), all_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Directory with corpus parquet files')
    parser.add_argument('--max-docs', type=int, default=1_000_000, help='Max corpus docs to compare')
    parser.add_argument('--test-idx', type=int, default=0, help='Which MUSR test example to use')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load MUSR
    print("Loading MUSR dataset...")
    ds = load_dataset("TAUR-Lab/MuSR")

    # Get first test example
    split = list(ds.keys())[0]
    item = ds[split][args.test_idx]
    test_text = f"{item.get('narrative', item.get('question', ''))}\n\n{item.get('answer', '')}"
    print(f"\n=== Test text (first 500 chars) ===")
    print(test_text[:500])
    print("...")

    # Load model with bf16
    print("\nLoading embedding model with bfloat16...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "nvidia/llama-embed-nemotron-8b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device).eval()

    # Embed test text
    print("Embedding test text...")
    test_emb = embed_single_text(test_text, model, tokenizer, device)
    print(f"Test embedding shape: {test_emb.shape}, dtype: {test_emb.dtype}")

    # Free model memory
    del model, tokenizer
    torch.cuda.empty_cache()

    # Load corpus embeddings
    corpus_embs, corpus_ids = load_corpus_embeddings(data_dir, args.max_docs)

    # Compute similarities
    print("\nComputing cosine similarities...")
    # test_emb is already L2 normalized, corpus should be too
    # But let's normalize corpus just in case
    corpus_norms = np.linalg.norm(corpus_embs, axis=1, keepdims=True)
    corpus_embs_normed = corpus_embs / np.clip(corpus_norms, 1e-9, None)

    similarities = corpus_embs_normed @ test_emb

    # Get top-100
    top_k = 100
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_scores = similarities[top_indices]
    top_ids = [corpus_ids[i] for i in top_indices]

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
    for i in range(10):
        print(f"  {i+1}. score={top_scores[i]:.4f}, id={top_ids[i]}")


if __name__ == "__main__":
    main()
