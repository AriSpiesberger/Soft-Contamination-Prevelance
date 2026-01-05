#!/usr/bin/env python3
"""
Verify CSV scores by re-embedding and comparing cosine similarities.
Uses the same Nemotron model to ensure consistency.
"""

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def load_nemotron_model(device='cuda'):
    """Load the same Nemotron model used in the pipeline."""
    model_name = "nvidia/llama-embed-nemotron-8b"
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        dtype=torch.float16,
        trust_remote_code=True
    ).to(device)
    model.eval()

    return model, tokenizer


def embed_text(text, model, tokenizer, device='cuda'):
    """Embed a single text using Nemotron (same as pipeline)."""
    with torch.inference_mode():
        # Tokenize
        encoded = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors='pt'
        ).to(device)

        # Get embeddings
        outputs = model(**encoded)

        # Mean pooling with attention mask
        attention_mask = encoded['attention_mask'].unsqueeze(-1).to(outputs[0].dtype)
        embeddings = (outputs[0] * attention_mask).sum(1) / attention_mask.sum(1).clamp(min=1e-9)

        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.float().cpu().numpy()[0]


def cosine_similarity(emb1, emb2):
    """Calculate cosine similarity between two embeddings."""
    return float(np.dot(emb1, emb2))


def verify_csv_scores(csv_path, model, tokenizer, num_samples=10, device='cuda'):
    """
    Verify scores in CSV by re-embedding and comparing.

    Args:
        csv_path: Path to CSV file
        model: Loaded Nemotron model
        tokenizer: Loaded tokenizer
        num_samples: Number of samples to verify
        device: CUDA device

    Returns:
        dict with verification results
    """
    print(f"\n{'='*80}")
    print(f"Verifying: {csv_path.name}")
    print(f"{'='*80}")

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Total entries: {len(df)}")

    # Sample entries to verify (include top matches and random ones)
    if len(df) > num_samples:
        # Take top 5 and random 5
        top_indices = df.nlargest(5, 'cosine_similarity').index.tolist()
        remaining = df.drop(top_indices).index.tolist()
        random_indices = np.random.choice(remaining, min(5, len(remaining)), replace=False).tolist()
        sample_indices = top_indices + random_indices
    else:
        sample_indices = df.index.tolist()

    results = {
        'csv_path': str(csv_path),
        'total_entries': len(df),
        'samples_verified': len(sample_indices),
        'matches': [],
        'discrepancies': []
    }

    print(f"\nVerifying {len(sample_indices)} samples...")

    for idx in tqdm(sample_indices, desc="Verifying"):
        row = df.iloc[idx]

        test_text = row['test_text']
        corpus_text = row['corpus_text']
        stored_score = row['cosine_similarity']

        # Skip if text is missing
        if pd.isna(test_text) or pd.isna(corpus_text) or not test_text or not corpus_text:
            print(f"Skipping row {idx}: missing text")
            continue

        # Re-embed
        test_emb = embed_text(test_text, model, tokenizer, device)
        corpus_emb = embed_text(corpus_text, model, tokenizer, device)

        # Calculate similarity
        computed_score = cosine_similarity(test_emb, corpus_emb)

        # Compare (allow small floating point differences)
        diff = abs(computed_score - stored_score)

        result = {
            'index': int(idx),
            'test_id': row['test_id'],
            'rank': int(row['rank']),
            'stored_score': float(stored_score),
            'computed_score': float(computed_score),
            'difference': float(diff),
            'test_text_preview': str(test_text[:100]) if test_text else '',
            'corpus_text_preview': str(corpus_text[:100]) if corpus_text else ''
        }

        if diff > 0.01:  # More than 1% difference
            results['discrepancies'].append(result)
            print(f"\n⚠️  DISCREPANCY at row {idx}:")
            print(f"   Test ID: {row['test_id']}")
            print(f"   Stored:   {stored_score:.6f}")
            print(f"   Computed: {computed_score:.6f}")
            print(f"   Diff:     {diff:.6f}")
        else:
            results['matches'].append(result)

    return results


def print_summary(results):
    """Print verification summary."""
    print(f"\n{'='*80}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*80}")
    print(f"CSV: {Path(results['csv_path']).name}")
    print(f"Total entries: {results['total_entries']:,}")
    print(f"Samples verified: {results['samples_verified']}")
    print(f"Matches: {len(results['matches'])} ✅")
    print(f"Discrepancies: {len(results['discrepancies'])} {'⚠️' if results['discrepancies'] else '✅'}")

    if results['matches']:
        scores_stored = [m['stored_score'] for m in results['matches']]
        scores_computed = [m['computed_score'] for m in results['matches']]
        diffs = [m['difference'] for m in results['matches']]

        print(f"\nMatch Statistics:")
        print(f"  Mean difference: {np.mean(diffs):.8f}")
        print(f"  Max difference:  {np.max(diffs):.8f}")
        print(f"  Score range (stored):   [{min(scores_stored):.4f}, {max(scores_stored):.4f}]")
        print(f"  Score range (computed): [{min(scores_computed):.4f}, {max(scores_computed):.4f}]")

    if results['discrepancies']:
        print(f"\n⚠️  DISCREPANCIES FOUND:")
        for disc in results['discrepancies']:
            print(f"\n  Row {disc['index']} (test_id={disc['test_id']}):")
            print(f"    Stored:   {disc['stored_score']:.6f}")
            print(f"    Computed: {disc['computed_score']:.6f}")
            print(f"    Diff:     {disc['difference']:.6f}")
            print(f"    Test:     {disc['test_text_preview'][:80]}...")
            print(f"    Corpus:   {disc['corpus_text_preview'][:80]}...")


def main():
    parser = argparse.ArgumentParser(description='Verify CSV scores by re-embedding')
    parser.add_argument('--csv', required=True, help='Path to CSV file to verify')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to verify')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        return 1

    print("="*80)
    print("CSV SCORE VERIFICATION")
    print("="*80)
    print(f"CSV file: {csv_path}")
    print(f"Samples to verify: {args.num_samples}")
    print(f"Device: {args.device}")

    # Load model
    model, tokenizer = load_nemotron_model(args.device)

    # Verify scores
    results = verify_csv_scores(csv_path, model, tokenizer, args.num_samples, args.device)

    # Print summary
    print_summary(results)

    # Return exit code
    if results['discrepancies']:
        print("\n❌ Verification FAILED - discrepancies found")
        return 1
    else:
        print("\n✅ Verification PASSED - all scores match!")
        return 0


if __name__ == "__main__":
    exit(main())
