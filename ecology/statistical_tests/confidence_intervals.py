"""
Confidence intervals for model accuracy and paired differences.

Unit of analysis: per-sample pass rate (k/10), n=125 independent samples.
The 10 evals per sample are NOT independent, so we collapse first.
"""

import os
import csv
import numpy as np
from scipy import stats

BASE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "outcomes", "outputs_olmo", "evals", "checkpoint_evals",
)
BASE_CSV = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "outcomes", "outputs_olmo", "evals", "base_model_eval_results_20260413_034651.csv",
)

MODELS = ["contaminated", "clean"]
CHECKPOINTS = [
    "checkpoint-167", "checkpoint-334", "checkpoint-501", "checkpoint-668",
    "checkpoint-835", "checkpoint-1002", "checkpoint-1169", "checkpoint-1336",
    "checkpoint-1503", "checkpoint-1670", "final",
]
STEP_INCREMENT = 167
N_BOOTSTRAP = 10000
ALPHA = 0.05


def parse_checkpoint_num(name):
    if name == "final":
        return 1837
    return int(name.split("-")[1])


def get_per_sample_accuracy(csv_path, test_split_filter):
    """Return dict {sample_id: pass_rate} where pass_rate = n_correct / 10."""
    counts = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["test_split"] != test_split_filter:
                continue
            sid = int(row["sample_id"])
            if sid not in counts:
                counts[sid] = [0, 0]
            counts[sid][0] += 1
            if row["correct"].strip() == "1":
                counts[sid][1] += 1
    return {sid: c / t for sid, (t, c) in counts.items()}


def to_array(sample_dict):
    """Convert {sid: acc} to sorted array of accuracies."""
    return np.array([sample_dict[s] for s in sorted(sample_dict)])


def ci_t(x, alpha=ALPHA):
    """T-based confidence interval for the mean."""
    n = len(x)
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return mean, mean - t_crit * se, mean + t_crit * se


def ci_bootstrap(x, n_boot=N_BOOTSTRAP, alpha=ALPHA):
    """Bootstrap percentile confidence interval for the mean."""
    rng = np.random.default_rng(42)
    boot_means = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return x.mean(), lo, hi


# --- Load data ---
data = {}
for model in MODELS:
    data[model] = {"contaminated": {}, "clean": {}}
    for ckpt in CHECKPOINTS:
        csv_path = os.path.join(BASE, model, ckpt, "results.csv")
        if not os.path.exists(csv_path):
            continue
        epoch = round(parse_checkpoint_num(ckpt) / STEP_INCREMENT)
        for split in ["contaminated", "clean"]:
            data[model][split][epoch] = to_array(
                get_per_sample_accuracy(csv_path, split)
            )

base_data = {}
for split in ["contaminated", "clean"]:
    base_data[split] = to_array(get_per_sample_accuracy(BASE_CSV, split))

epochs = sorted(data["contaminated"]["contaminated"].keys())

# =====================================================
# PART 1: Per-model accuracy CIs at each epoch
# =====================================================
print("=" * 95)
print("ACCURACY CONFIDENCE INTERVALS (n=125 independent samples, pass@10)")
print("=" * 95)

for split in ["contaminated", "clean"]:
    print(f"\n--- Test split: {split.upper()} ---")
    print(f"{'':>8}  {'':>3}  {'--- T-based 95% CI ---':>30}  {'--- Bootstrap 95% CI ---':>30}")
    print(f"{'Model':>8}  {'Ep':>3}  {'Mean':>7}  {'Lo':>7}  {'Hi':>7}  {'Width':>7}  {'Mean':>7}  {'Lo':>7}  {'Hi':>7}  {'Width':>7}")
    print("-" * 95)

    # Base model (no epoch)
    x = base_data[split]
    tm, tlo, thi = ci_t(x)
    bm, blo, bhi = ci_bootstrap(x)
    print(f"{'Base':>8}  {'--':>3}  {tm:>7.3f}  {tlo:>7.3f}  {thi:>7.3f}  {thi-tlo:>7.3f}  {bm:>7.3f}  {blo:>7.3f}  {bhi:>7.3f}  {bhi-blo:>7.3f}")

    for model in MODELS:
        for e in epochs:
            x = data[model][split][e]
            tm, tlo, thi = ci_t(x)
            bm, blo, bhi = ci_bootstrap(x)
            label = "Contam" if model == "contaminated" else "Clean"
            print(f"{label:>8}  {e:>3}  {tm:>7.3f}  {tlo:>7.3f}  {thi:>7.3f}  {thi-tlo:>7.3f}  {bm:>7.3f}  {blo:>7.3f}  {bhi:>7.3f}  {bhi-blo:>7.3f}")
        print()

# =====================================================
# PART 2: Paired difference CIs (reject null = no diff)
# =====================================================
print("\n\n" + "=" * 100)
print("PAIRED DIFFERENCE CIs (d_i = model_i - baseline_i, n=125)")
print("If CI excludes 0 => reject null of no difference at 95% level")
print("=" * 100)


def print_paired_ci(title, label, get_model, get_ref):
    for split in ["contaminated", "clean"]:
        print(f"\n--- {title} | Test split: {split.upper()} ---")
        print(f"{'Ep':>3}  {label+' Mean':>12}  {'Ref Mean':>9}  {'Mean Diff':>10}  {'T-Lo':>7}  {'T-Hi':>7}  {'Boot-Lo':>7}  {'Boot-Hi':>7}  {'t-stat':>7}  {'p-val':>9}  {'Sig':>4}")
        print("-" * 100)
        for e in epochs:
            a = get_model(split, e)
            b = get_ref(split, e)
            d = a - b
            tm, tlo, thi = ci_t(d)
            bm, blo, bhi = ci_bootstrap(d)
            t_stat, p_val = stats.ttest_rel(a, b)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{e:>3}  {a.mean():>12.4f}  {b.mean():>9.4f}  {tm:>+10.4f}  {tlo:>+7.4f}  {thi:>+7.4f}  {blo:>+7.4f}  {bhi:>+7.4f}  {t_stat:>7.3f}  {p_val:>9.6f}  {sig:>4}")


# Contaminated model vs Base
print_paired_ci(
    "CONTAMINATED MODEL vs BASE",
    "Contam",
    lambda split, e: data["contaminated"][split][e],
    lambda split, e: base_data[split],
)

# Clean model vs Base
print_paired_ci(
    "CLEAN MODEL vs BASE",
    "Clean",
    lambda split, e: data["clean"][split][e],
    lambda split, e: base_data[split],
)

# Contaminated vs Clean (direct)
print_paired_ci(
    "CONTAMINATED MODEL vs CLEAN MODEL",
    "Contam",
    lambda split, e: data["contaminated"][split][e],
    lambda split, e: data["clean"][split][e],
)
