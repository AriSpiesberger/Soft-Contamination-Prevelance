#!/usr/bin/env python3
"""Check for overlaps between annotation files to verify mutual exclusivity."""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from itertools import combinations
import hashlib
import re


def ngrams(text: str, n: int = 5) -> set:
    """Extract character n-grams from text."""
    text = re.sub(r'\s+', ' ', str(text).lower().strip())
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def text_hash(text: str) -> str:
    """Hash normalized text for exact matching."""
    normalized = re.sub(r'\s+', ' ', str(text).lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


def load_file(path: Path) -> pd.DataFrame:
    """Load CSV and add normalized columns."""
    df = pd.read_csv(path)
    # Keep only rows with both text fields
    if 'test_text' in df.columns and 'corpus_text' in df.columns:
        df = df.dropna(subset=['test_text', 'corpus_text'])
    return df


def main():
    data_dir = Path(__file__).parent / "training_data"

    # All labeled annotation files
    files = {
        'full(1)': 'mbpp_annotations_full(1).csv',
        'old_with_text': 'mbpp_annotations_old_with_text.csv',
        'existing': 'mbpp_annotations_existing.csv',
        'current': 'mbpp_annotations_current.csv',
        'codeforces': 'codeforces_annotations.csv',
        'cf_review': 'codeforces_annotations_review.csv',
    }

    dfs = {}
    for name, filename in files.items():
        path = data_dir / filename
        if path.exists():
            dfs[name] = load_file(path)
            print(f"Loaded {name}: {len(dfs[name])} rows")
        else:
            print(f"MISSING: {filename}")

    # ========================================================
    # 1. EXACT OVERLAP: test_id x corpus_id pairs
    # ========================================================
    print("\n" + "="*70)
    print("1. EXACT OVERLAP BY test_id x corpus_id PAIRS")
    print("="*70)

    pair_sets = {}
    for name, df in dfs.items():
        if 'test_id' in df.columns and 'corpus_id' in df.columns:
            pairs = set(zip(df['test_id'].astype(str), df['corpus_id'].astype(str)))
            pair_sets[name] = pairs
            print(f"  {name}: {len(pairs)} unique pairs")
        else:
            print(f"  {name}: no test_id/corpus_id columns")

    # Pairwise overlap
    print("\nPairwise overlap (test_id x corpus_id):")
    for (a, pairs_a), (b, pairs_b) in combinations(pair_sets.items(), 2):
        overlap = pairs_a & pairs_b
        if overlap:
            print(f"  {a} & {b}: {len(overlap)} shared pairs")
            # Show a few examples
            for pair in list(overlap)[:3]:
                print(f"    Example: test_id={pair[0]}, corpus_id={pair[1]}")
        else:
            print(f"  {a} & {b}: 0 (mutually exclusive OK)")

    # ========================================================
    # 2. EXACT TEXT OVERLAP (hash-based)
    # ========================================================
    print("\n" + "="*70)
    print("2. EXACT TEXT OVERLAP (normalized text hash)")
    print("="*70)

    text_hash_sets = {}
    for name, df in dfs.items():
        if 'test_text' in df.columns and 'corpus_text' in df.columns:
            hashes = set()
            for _, row in df.iterrows():
                h = text_hash(str(row['test_text'])[:500] + "|||" + str(row['corpus_text'])[:500])
                hashes.add(h)
            text_hash_sets[name] = hashes
            print(f"  {name}: {len(hashes)} unique text pairs")

    print("\nPairwise text overlap:")
    for (a, hashes_a), (b, hashes_b) in combinations(text_hash_sets.items(), 2):
        overlap = hashes_a & hashes_b
        if overlap:
            pct_a = len(overlap) / len(hashes_a) * 100
            pct_b = len(overlap) / len(hashes_b) * 100
            print(f"  {a} & {b}: {len(overlap)} shared text pairs ({pct_a:.1f}% of {a}, {pct_b:.1f}% of {b})")
        else:
            print(f"  {a} & {b}: 0 (mutually exclusive OK)")

    # ========================================================
    # 3. N-GRAM SIMILARITY (test_text only, sample-based)
    # ========================================================
    print("\n" + "="*70)
    print("3. N-GRAM SIMILARITY ANALYSIS (5-gram Jaccard on test_text)")
    print("="*70)

    # For efficiency, sample test_texts from each file
    SAMPLE_SIZE = 200
    NGRAM_N = 5
    SIMILARITY_THRESHOLD = 0.5

    test_text_samples = {}
    for name, df in dfs.items():
        if 'test_text' in df.columns:
            # Get unique test_texts
            unique_texts = df['test_text'].dropna().unique()
            sample = unique_texts[:min(SAMPLE_SIZE, len(unique_texts))]
            test_text_samples[name] = [(t, ngrams(t, NGRAM_N)) for t in sample]
            print(f"  {name}: sampled {len(sample)} unique test_texts")

    print(f"\nCross-file near-duplicates (Jaccard >= {SIMILARITY_THRESHOLD}):")
    for (a, samples_a), (b, samples_b) in combinations(test_text_samples.items(), 2):
        near_dupes = []
        for text_a, ngrams_a in samples_a:
            for text_b, ngrams_b in samples_b:
                sim = jaccard(ngrams_a, ngrams_b)
                if sim >= SIMILARITY_THRESHOLD:
                    near_dupes.append((sim, text_a[:80], text_b[:80]))

        near_dupes.sort(reverse=True)
        if near_dupes:
            print(f"\n  {a} <-> {b}: {len(near_dupes)} near-duplicate test_texts")
            for sim, ta, tb in near_dupes[:5]:
                print(f"    Jaccard={sim:.3f}")
                print(f"      A: {ta}...")
                print(f"      B: {tb}...")
        else:
            print(f"\n  {a} <-> {b}: 0 near-duplicates OK")

    # ========================================================
    # 4. test_id OVERLAP (are the same benchmark items being annotated?)
    # ========================================================
    print("\n" + "="*70)
    print("4. test_id OVERLAP (same benchmark items across files)")
    print("="*70)

    testid_sets = {}
    for name, df in dfs.items():
        if 'test_id' in df.columns:
            tids = set(df['test_id'].astype(str).unique())
            testid_sets[name] = tids
            print(f"  {name}: {len(tids)} unique test_ids")

    print("\nPairwise test_id overlap:")
    for (a, tids_a), (b, tids_b) in combinations(testid_sets.items(), 2):
        overlap = tids_a & tids_b
        only_a = tids_a - tids_b
        only_b = tids_b - tids_a
        pct_a = len(overlap) / len(tids_a) * 100 if tids_a else 0
        pct_b = len(overlap) / len(tids_b) * 100 if tids_b else 0
        print(f"  {a} & {b}: {len(overlap)} shared test_ids ({pct_a:.1f}% of {a}, {pct_b:.1f}% of {b})")
        print(f"    Only in {a}: {len(only_a)}, Only in {b}: {len(only_b)}")

    # ========================================================
    # 5. LABEL AGREEMENT on overlapping pairs
    # ========================================================
    print("\n" + "="*70)
    print("5. LABEL AGREEMENT ON OVERLAPPING PAIRS")
    print("="*70)

    for (a, df_a), (b, df_b) in combinations(dfs.items(), 2):
        if 'test_id' not in df_a.columns or 'test_id' not in df_b.columns:
            continue
        if 'corpus_id' not in df_a.columns or 'corpus_id' not in df_b.columns:
            continue
        if 'is_sd' not in df_a.columns or 'is_sd' not in df_b.columns:
            continue

        # Merge on test_id x corpus_id
        df_a_keyed = df_a[['test_id', 'corpus_id', 'is_sd']].copy()
        df_b_keyed = df_b[['test_id', 'corpus_id', 'is_sd']].copy()
        df_a_keyed['test_id'] = df_a_keyed['test_id'].astype(str)
        df_a_keyed['corpus_id'] = df_a_keyed['corpus_id'].astype(str)
        df_b_keyed['test_id'] = df_b_keyed['test_id'].astype(str)
        df_b_keyed['corpus_id'] = df_b_keyed['corpus_id'].astype(str)

        merged = df_a_keyed.merge(df_b_keyed, on=['test_id', 'corpus_id'], suffixes=('_a', '_b'))

        if len(merged) == 0:
            continue

        # Normalize is_sd to bool
        merged['is_sd_a'] = merged['is_sd_a'].astype(str).str.lower() == 'true'
        merged['is_sd_b'] = merged['is_sd_b'].astype(str).str.lower() == 'true'

        agree = (merged['is_sd_a'] == merged['is_sd_b']).sum()
        disagree = len(merged) - agree
        print(f"\n  {a} <-> {b}: {len(merged)} overlapping pairs")
        print(f"    Label agreement: {agree}/{len(merged)} ({agree/len(merged)*100:.1f}%)")
        if disagree > 0:
            conflicts = merged[merged['is_sd_a'] != merged['is_sd_b']]
            print(f"    CONFLICTS: {disagree}")
            for _, row in conflicts.head(3).iterrows():
                print(f"      test_id={row['test_id']}, corpus_id={row['corpus_id']}: {a}={row['is_sd_a']} vs {b}={row['is_sd_b']}")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
