"""
Clustered bootstrap test for difference in pass rates.

Resamples at the sample level (n=125 clusters), preserving all 10 trials
per cluster. This respects the within-sample correlation structure.
"""

import os
import csv
import sys
import numpy as np

# Flush output immediately
sys.stdout.reconfigure(line_buffering=True)

EVALS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "outcomes", "outputs_olmo", "evals",
)
BASE = os.path.join(EVALS_DIR, "checkpoint_evals")
BASE_CSV = os.path.join(EVALS_DIR, "base_model_eval_results_20260413_034651.csv")
STATS_DIR = os.path.join(EVALS_DIR, "stats")
os.makedirs(STATS_DIR, exist_ok=True)
OUT_CSV = os.path.join(STATS_DIR, "clustered_bootstrap.csv")

CHECKPOINTS = [
    "checkpoint-167", "checkpoint-334", "checkpoint-501", "checkpoint-668",
    "checkpoint-835", "checkpoint-1002", "checkpoint-1169", "checkpoint-1336",
    "checkpoint-1503", "checkpoint-1670", "final",
]
STEP_INCREMENT = 167
N_BOOT = 10000
ALPHA = 0.05


def parse_checkpoint_num(name):
    if name == "final":
        return 1837
    return int(name.split("-")[1])


def get_binary_by_sample(csv_path, test_split_filter):
    """Return dict {sample_id: [list of 0/1 for each trial]}."""
    results = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["test_split"] != test_split_filter:
                continue
            sid = int(row["sample_id"])
            if sid not in results:
                results[sid] = []
            results[sid].append(1 if row["correct"].strip() == "1" else 0)
    return results


def clustered_bootstrap(binary_a, binary_b, n_boot=N_BOOT, alpha=ALPHA):
    """
    Clustered bootstrap: resample 125 samples (clusters) with replacement.
    For each resampled cluster, grab ALL raw binary trials (10 per model).
    Compute overall pass rate from the full raw 0/1 pool, not from averages.

    Returns: observed_diff, ci_lo, ci_hi, p_value
    """
    rng = np.random.default_rng(42)
    sids = sorted(set(binary_a) & set(binary_b))
    n = len(sids)
    k = len(binary_a[sids[0]])  # trials per sample (10)

    # Build raw trial matrices: shape (n_samples, k_trials)
    raw_a = np.array([binary_a[s] for s in sids])  # (125, 10)
    raw_b = np.array([binary_b[s] for s in sids])  # (125, 10)

    # Observed: overall pass rate across all 1250 raw trials
    obs_rate_a = raw_a.sum() / raw_a.size
    obs_rate_b = raw_b.sum() / raw_b.size
    obs_diff = obs_rate_a - obs_rate_b

    # Vectorized bootstrap: resample cluster indices (n_boot x n)
    idx = rng.integers(0, n, size=(n_boot, n))

    # For each bootstrap, grab all raw trials for resampled clusters
    # raw_a[idx] shape: (n_boot, 125, 10) -> sum over last two axes / total trials
    boot_a = raw_a[idx]  # (n_boot, 125, 10)
    boot_b = raw_b[idx]  # (n_boot, 125, 10)
    total_trials = n * k
    boot_rate_a = boot_a.reshape(n_boot, -1).sum(axis=1) / total_trials
    boot_rate_b = boot_b.reshape(n_boot, -1).sum(axis=1) / total_trials
    boot_diffs = boot_rate_a - boot_rate_b

    ci_lo = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_hi = np.percentile(boot_diffs, 100 * (1 - alpha / 2))

    # Two-sided p-value: centered bootstrap
    centered = boot_diffs - obs_diff
    p_val = np.mean(np.abs(centered) >= np.abs(obs_diff))

    return obs_diff, ci_lo, ci_hi, p_val


# --- Load data ---
# Finetuned models: binary_data[model][split][epoch] = {sid: [0/1, ...]}
binary_data = {}
for model in ["contaminated", "clean"]:
    binary_data[model] = {"contaminated": {}, "clean": {}}
    for ckpt in CHECKPOINTS:
        csv_path = os.path.join(BASE, model, ckpt, "results.csv")
        if not os.path.exists(csv_path):
            continue
        epoch = round(parse_checkpoint_num(ckpt) / STEP_INCREMENT)
        for split in ["contaminated", "clean"]:
            binary_data[model][split][epoch] = get_binary_by_sample(csv_path, split)

# Base model
base_binary = {}
for split in ["contaminated", "clean"]:
    base_binary[split] = get_binary_by_sample(BASE_CSV, split)

epochs = sorted(binary_data["contaminated"]["contaminated"].keys())


OUT_ROWS = []


def run_bootstrap_table(comparison, label_a, label_b, get_a, get_b):
    for split in ["contaminated", "clean"]:
        print(f"\n{'='*90}")
        print(f"{comparison} | TEST SPLIT: {split.upper()}")
        print(f"Clustered bootstrap (n=125 clusters, 10 trials each, {N_BOOT} resamples)")
        print(f"{'='*90}")
        print(f"{'Ep':>3}  {label_a:>10}  {label_b:>10}  {'Diff':>8}  {'95% CI Lo':>10}  {'95% CI Hi':>10}  {'p-value':>9}  {'Sig':>4}")
        print("-" * 90)
        for e in epochs:
            ba = get_a(split, e)
            bb = get_b(split, e)
            rate_a = float(np.mean([np.mean(ba[s]) for s in sorted(ba)]))
            rate_b = float(np.mean([np.mean(bb[s]) for s in sorted(bb)]))
            diff, ci_lo, ci_hi, p = clustered_bootstrap(ba, bb)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{e:>3}  {rate_a:>10.4f}  {rate_b:>10.4f}  {diff:>+8.4f}  {ci_lo:>+10.4f}  {ci_hi:>+10.4f}  {p:>9.4f}  {sig:>4}")
            OUT_ROWS.append({
                "comparison": comparison, "split": split, "epoch": e,
                "rate_a": rate_a, "rate_b": rate_b,
                "diff": float(diff), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
                "p_value": float(p), "sig": sig,
                "n_boot": N_BOOT, "alpha": ALPHA,
            })


# 1) Contaminated vs Clean
run_bootstrap_table(
    "contaminated_vs_clean",
    "Contam", "Clean",
    lambda split, e: binary_data["contaminated"][split][e],
    lambda split, e: binary_data["clean"][split][e],
)

# 2) Contaminated vs Base
run_bootstrap_table(
    "contaminated_vs_base",
    "Contam", "Base",
    lambda split, e: binary_data["contaminated"][split][e],
    lambda split, e: base_binary[split],
)

# 3) Clean vs Base
run_bootstrap_table(
    "clean_vs_base",
    "Clean", "Base",
    lambda split, e: binary_data["clean"][split][e],
    lambda split, e: base_binary[split],
)

# Write results CSV
fieldnames = ["comparison", "split", "epoch", "rate_a", "rate_b",
              "diff", "ci_lo", "ci_hi", "p_value", "sig", "n_boot", "alpha"]
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(OUT_ROWS)
print(f"\nWrote {len(OUT_ROWS)} rows to {OUT_CSV}")
