"""Self-consistency analysis for Gemini codeforces SD classifier.

Reads gemini_consistency_runs.csv (one row per Gemini call), computes:
  - per-pair: mode label, # of unique labels, agreement rate (mode count / runs)
  - per-original-category: mean agreement, mean # unique labels
  - overall: mean pairwise agreement, Fleiss' kappa
  - confusion matrix: original Gemini label vs Gemini's mode-of-runs label
"""

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DEFAULT_IN = ROOT / "human_annotation" / "gemini_consistency_runs.csv"
DEFAULT_OUT = ROOT / "human_annotation" / "gemini_consistency_summary.csv"

CATEGORIES = ["exact", "equivalent", "subset", "superset", "related", "unrelated"]


def fleiss_kappa(label_matrix, categories):
    """Fleiss' kappa for n_subjects x n_raters categorical labels."""
    n_subj = label_matrix.shape[0]
    n_raters = label_matrix.shape[1]
    counts = np.zeros((n_subj, len(categories)), dtype=int)
    cat_idx = {c: i for i, c in enumerate(categories)}
    for i in range(n_subj):
        for r in range(n_raters):
            lbl = label_matrix[i, r]
            if lbl in cat_idx:
                counts[i, cat_idx[lbl]] += 1
    if n_raters < 2:
        return float("nan")
    p_i = ((counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_e = ((counts.sum(axis=0) / (n_subj * n_raters)) ** 2).sum()
    if p_e >= 1.0:
        return float("nan")
    return (p_bar - p_e) / (1 - p_e)


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_IN)
    out = sys.argv[2] if len(sys.argv) > 2 else str(DEFAULT_OUT)

    df = pd.read_csv(inp)
    df = df[df["error"].isna() | (df["error"].astype(str).str.strip() == "")]
    df["match_type"] = df["match_type"].astype(str).str.lower().str.strip()
    df["original_match_type"] = df["original_match_type"].astype(str).str.lower().str.strip()

    # Per-pair stats
    rows = []
    for pair_id, g in df.groupby("pair_id"):
        labels = g["match_type"].tolist()
        c = Counter(labels)
        mode_label, mode_count = c.most_common(1)[0]
        rows.append({
            "pair_id": int(pair_id),
            "test_id": g["test_id"].iloc[0],
            "corpus_id": g["corpus_id"].iloc[0],
            "original": g["original_match_type"].iloc[0],
            "n_runs": len(labels),
            "n_unique_labels": len(c),
            "mode": mode_label,
            "mode_count": mode_count,
            "agreement_rate": mode_count / len(labels),
            "labels_distribution": ";".join(f"{k}:{v}" for k, v in c.most_common()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out, index=False)
    print(f"wrote per-pair summary: {out}\n")

    # Headline numbers
    n_pairs = len(summary)
    n_runs = int(summary["n_runs"].mode().iloc[0])
    print(f"n pairs: {n_pairs}  |  runs/pair: {n_runs}")
    print(f"mean per-pair agreement (mode_count / n_runs): {summary['agreement_rate'].mean():.3f}")
    print(f"frac pairs unanimous (all {n_runs} agree):     {(summary['agreement_rate'] == 1.0).mean():.3f}")
    print(f"mean # unique labels per pair:                 {summary['n_unique_labels'].mean():.2f}")

    # Pairwise agreement: for each pair, % of run-pairs that agree
    pairwise = []
    for _, g in df.groupby("pair_id"):
        labs = g["match_type"].tolist()
        n = len(labs)
        if n < 2:
            continue
        agree = sum(labs[i] == labs[j] for i in range(n) for j in range(i + 1, n))
        total = n * (n - 1) // 2
        pairwise.append(agree / total)
    print(f"mean pairwise agreement:                       {np.mean(pairwise):.3f}")

    # Fleiss' kappa
    label_matrix = []
    for _, g in df.groupby("pair_id"):
        labs = g["match_type"].tolist()[:n_runs]
        if len(labs) == n_runs:
            label_matrix.append(labs)
    label_matrix = np.array(label_matrix)
    kappa = fleiss_kappa(label_matrix, CATEGORIES)
    print(f"Fleiss' kappa (n={len(label_matrix)} pairs):     {kappa:.3f}")

    # Per-original-category breakdown
    print("\nper original category (mean agreement, mean unique labels, n):")
    by_cat = summary.groupby("original").agg(
        mean_agreement=("agreement_rate", "mean"),
        mean_unique=("n_unique_labels", "mean"),
        n=("pair_id", "count"),
    ).reindex([c for c in CATEGORIES if c in summary["original"].unique()])
    print(by_cat.to_string(float_format=lambda x: f"{x:.3f}"))

    # Confusion matrix: original vs mode
    print("\nconfusion: rows = original Gemini label, cols = mode-of-runs label")
    cats_present = sorted(set(summary["original"].unique()) | set(summary["mode"].unique()),
                          key=lambda c: (CATEGORIES.index(c) if c in CATEGORIES else 99))
    cm = pd.crosstab(summary["original"], summary["mode"]).reindex(
        index=cats_present, columns=cats_present, fill_value=0)
    print(cm.to_string())

    # Self-consistency check: of pairs where original was X, how often is mode also X?
    summary["mode_matches_original"] = summary["original"] == summary["mode"]
    print("\nfraction whose mode label = original Gemini label, by category:")
    consistency_by_cat = summary.groupby("original")["mode_matches_original"].agg(["mean", "count"])
    consistency_by_cat = consistency_by_cat.reindex([c for c in CATEGORIES if c in summary["original"].unique()])
    print(consistency_by_cat.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
