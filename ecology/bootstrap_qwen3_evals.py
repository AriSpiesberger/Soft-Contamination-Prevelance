"""
Clustered bootstrap significance tests for Qwen3 paired eval (n=20 trials).

Pairing: same sample_id is evaluated by BOTH the contaminated and clean
models. The contam_split (125 samples) and clean_split (125 samples) are
disjoint sets of samples.

For each epoch:
  - Paired bootstrap of (contam_model - clean_model) within contam_split
  - Paired bootstrap of (contam_model - clean_model) within clean_split
  - DiD bootstrap: independently resample each split's paired diffs
    (since the splits contain different samples).

Reports observed value, 95% CI, and two-sided bootstrap p-value.
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

EVAL_DIR = Path(__file__).parent / "outcomes" / "outputs_qwen3" / "evals"
CONTAM_DIR = EVAL_DIR / "contaminated_model_20"
CLEAN_DIR = EVAL_DIR / "clean_model_20"

N_BOOT = 10000
ALPHA = 0.05
SEED = 42


def load_pass_rates(path):
    """Return {epoch: {sample_id: pass_rate}} (skipping 'final')."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            ep = row["epoch"]
            if ep == "final":
                continue
            ep = int(ep)
            sid = int(row["sample_id"])
            out.setdefault(ep, {})[sid] = float(row["pass_rate"])
    return out


def paired_bootstrap_diff(a, b, n_boot=N_BOOT, alpha=ALPHA, seed=SEED):
    """Paired bootstrap: resample sample-indices, compute mean(a-b) diff."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    d = a - b
    obs = d.mean()
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    ci_lo = np.percentile(boot, 100 * alpha / 2)
    ci_hi = np.percentile(boot, 100 * (1 - alpha / 2))
    centered = boot - obs
    p_val = float(np.mean(np.abs(centered) >= np.abs(obs)))
    return obs, ci_lo, ci_hi, p_val, boot


def did_bootstrap(d_contam_split, d_clean_split,
                  n_boot=N_BOOT, alpha=ALPHA, seed=SEED):
    """DiD = mean(d_contam_split) - mean(d_clean_split).
    Splits are disjoint → resample each independently."""
    rng = np.random.default_rng(seed)
    a = np.asarray(d_contam_split)
    b = np.asarray(d_clean_split)
    obs = a.mean() - b.mean()
    idx_a = rng.integers(0, len(a), size=(n_boot, len(a)))
    idx_b = rng.integers(0, len(b), size=(n_boot, len(b)))
    boot = a[idx_a].mean(axis=1) - b[idx_b].mean(axis=1)
    ci_lo = np.percentile(boot, 100 * alpha / 2)
    ci_hi = np.percentile(boot, 100 * (1 - alpha / 2))
    centered = boot - obs
    p_val = float(np.mean(np.abs(centered) >= np.abs(obs)))
    return obs, ci_lo, ci_hi, p_val


def sig_star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def aligned(contam_dict, clean_dict):
    shared = sorted(set(contam_dict) & set(clean_dict))
    a = np.array([contam_dict[s] for s in shared])
    b = np.array([clean_dict[s] for s in shared])
    return a, b


def main():
    cm_contam = load_pass_rates(CONTAM_DIR / "eval_contam_split.csv")
    cm_clean = load_pass_rates(CONTAM_DIR / "eval_clean_split.csv")
    cl_contam = load_pass_rates(CLEAN_DIR / "eval_contam_split.csv")
    cl_clean = load_pass_rates(CLEAN_DIR / "eval_clean_split.csv")

    epochs = sorted(set(cm_contam) & set(cm_clean)
                    & set(cl_contam) & set(cl_clean))
    print(f"Epochs evaluated: {epochs}")
    print(f"Bootstrap resamples: {N_BOOT}, alpha={ALPHA}, seed={SEED}\n")

    # --- Per-split paired diff (contam_model - clean_model) ---
    rows_split = {}
    for label, split_name, contam_d, clean_d in [
        ("CONTAM SPLIT: contam_model - clean_model (paired by sample_id)",
         "contam", cm_contam, cl_contam),
        ("CLEAN SPLIT:  contam_model - clean_model (paired by sample_id)",
         "clean", cm_clean, cl_clean),
    ]:
        print("=" * 96)
        print(label)
        print("=" * 96)
        print(f"{'Ep':>3}  {'n':>4}  {'Contam_M':>9}  {'Clean_M':>9}  "
              f"{'Diff':>8}  {'95% CI Lo':>10}  {'95% CI Hi':>10}  "
              f"{'p-value':>9}  {'Sig':>4}")
        print("-" * 96)
        for e in epochs:
            a, b = aligned(contam_d[e], clean_d[e])
            diff, lo, hi, p, _ = paired_bootstrap_diff(a, b)
            rows_split.setdefault(e, {})[split_name] = a - b
            print(f"{e:>3}  {len(a):>4}  {a.mean():>9.4f}  {b.mean():>9.4f}  "
                  f"{diff:>+8.4f}  {lo:>+10.4f}  {hi:>+10.4f}  "
                  f"{p:>9.4f}  {sig_star(p):>4}")
        print()

    # --- DiD ---
    print("=" * 96)
    print("DIFFERENCE-IN-DIFFERENCES (contam_split paired-diff - clean_split paired-diff)")
    print("Independent bootstrap across disjoint splits.")
    print("=" * 96)
    print(f"{'Ep':>3}  {'DiD':>8}  {'95% CI Lo':>10}  {'95% CI Hi':>10}  "
          f"{'p-value':>9}  {'Sig':>4}")
    print("-" * 96)
    for e in epochs:
        d_contam = rows_split[e]["contam"]
        d_clean = rows_split[e]["clean"]
        did, lo, hi, p = did_bootstrap(d_contam, d_clean)
        print(f"{e:>3}  {did:>+8.4f}  {lo:>+10.4f}  {hi:>+10.4f}  "
              f"{p:>9.4f}  {sig_star(p):>4}")


if __name__ == "__main__":
    main()
