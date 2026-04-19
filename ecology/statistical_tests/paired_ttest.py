"""Paired t-test and McNemar's test: contaminated vs clean model, and both vs base model, per epoch."""

import os
import csv
import numpy as np
from scipy import stats

EVALS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "outcomes", "outputs_olmo", "evals",
)
BASE = os.path.join(EVALS_DIR, "checkpoint_evals")
STATS_DIR = os.path.join(EVALS_DIR, "stats")
os.makedirs(STATS_DIR, exist_ok=True)
OUT_TTEST_CSV = os.path.join(STATS_DIR, "paired_ttest.csv")
OUT_MCNEMAR_CSV = os.path.join(STATS_DIR, "mcnemar.csv")

MODELS = ["contaminated", "clean"]
CHECKPOINTS = [
    "checkpoint-167", "checkpoint-334", "checkpoint-501", "checkpoint-668",
    "checkpoint-835", "checkpoint-1002", "checkpoint-1169", "checkpoint-1336",
    "checkpoint-1503", "checkpoint-1670", "final",
]
STEP_INCREMENT = 167


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


def get_binary_results(csv_path, test_split_filter):
    """Return dict {(sample_id, sample_num): 0 or 1} for McNemar's test."""
    results = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["test_split"] != test_split_filter:
                continue
            key = (int(row["sample_id"]), int(row["sample_num"]))
            results[key] = 1 if row["correct"].strip() == "1" else 0
    return results


# --- Load base model results ---
BASE_CSV = os.path.join(EVALS_DIR, "base_model_eval_results_20260413_034651.csv")


def get_base_per_sample_accuracy(csv_path, test_split_filter):
    """Return dict {sample_id: pass_rate} from base model CSV (has 'model' column)."""
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


# Collect finetuned model data (per-sample accuracy for t-test, binary for McNemar)
data = {}
binary_data = {}
for model in MODELS:
    data[model] = {"contaminated": {}, "clean": {}}
    binary_data[model] = {"contaminated": {}, "clean": {}}
    for ckpt in CHECKPOINTS:
        csv_path = os.path.join(BASE, model, ckpt, "results.csv")
        if not os.path.exists(csv_path):
            continue
        epoch = round(parse_checkpoint_num(ckpt) / STEP_INCREMENT)
        for split in ["contaminated", "clean"]:
            sample_accs = get_per_sample_accuracy(csv_path, split)
            sids = sorted(sample_accs.keys())
            data[model][split][epoch] = (sids, np.array([sample_accs[s] for s in sids]))
            binary_data[model][split][epoch] = get_binary_results(csv_path, split)

# Collect base model data (single set, no epochs)
base_data = {}
base_binary = {}
for split in ["contaminated", "clean"]:
    sample_accs = get_base_per_sample_accuracy(BASE_CSV, split)
    sids = sorted(sample_accs.keys())
    base_data[split] = (sids, np.array([sample_accs[s] for s in sids]))
    base_binary[split] = get_binary_results(BASE_CSV, split)

epochs = sorted(data["contaminated"]["contaminated"].keys())


TTEST_ROWS = []


def run_comparison(comparison, label_a, label_b, get_a, get_b):
    """Paired t-test; print and accumulate rows."""
    for split in ["contaminated", "clean"]:
        print(f"\n{'='*78}")
        print(f"{comparison} | TEST SPLIT: {split.upper()} (n=125 paired samples)")
        print(f"{'='*78}")
        print(f"{'Epoch':>6}  {label_a:>12}  {label_b:>12}  {'Diff':>8}  {'t-stat':>8}  {'p-value':>10}  {'Sig':>5}")
        print("-" * 78)
        for e in epochs:
            sids_a, accs_a = get_a(split, e)
            sids_b, accs_b = get_b(split, e)
            assert sids_a == sids_b, f"Sample IDs don't match at epoch {e}"
            t_stat, p_val = stats.ttest_rel(accs_a, accs_b)
            diff = float(accs_a.mean() - accs_b.mean())
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{e:>6}  {accs_a.mean():>12.4f}  {accs_b.mean():>12.4f}  {diff:>+8.4f}  {t_stat:>8.3f}  {p_val:>10.6f}  {sig:>5}")
            TTEST_ROWS.append({
                "comparison": comparison, "split": split, "epoch": e,
                "mean_a": float(accs_a.mean()), "mean_b": float(accs_b.mean()),
                "diff": diff, "t_stat": float(t_stat), "p_value": float(p_val),
                "sig": sig, "n_paired": len(accs_a),
            })


# 1) Contaminated model vs Clean model
run_comparison(
    "contaminated_vs_clean",
    "Contam Mean", "Clean Mean",
    lambda split, e: data["contaminated"][split][e],
    lambda split, e: data["clean"][split][e],
)

# 2) Contaminated model vs Base model
run_comparison(
    "contaminated_vs_base",
    "Contam Mean", "Base Mean",
    lambda split, e: data["contaminated"][split][e],
    lambda split, e: base_data[split],
)

# 3) Clean model vs Base model
run_comparison(
    "clean_vs_base",
    "Clean Mean", "Base Mean",
    lambda split, e: data["clean"][split][e],
    lambda split, e: base_data[split],
)

ttest_fields = ["comparison", "split", "epoch", "mean_a", "mean_b",
                "diff", "t_stat", "p_value", "sig", "n_paired"]
with open(OUT_TTEST_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=ttest_fields); w.writeheader(); w.writerows(TTEST_ROWS)
print(f"\nWrote {len(TTEST_ROWS)} rows to {OUT_TTEST_CSV}")


# ============================================================
# McNEMAR'S TEST (binary per-trial: 1250 paired observations)
# ============================================================

def mcnemar_test(binary_a, binary_b):
    """Run McNemar's test on two dicts of {(sid, snum): 0/1}.
    Returns (b, c, chi2, p) where b = A right & B wrong, c = A wrong & B right."""
    keys = sorted(set(binary_a) & set(binary_b))
    b = sum(1 for k in keys if binary_a[k] == 1 and binary_b[k] == 0)  # A correct, B wrong
    c = sum(1 for k in keys if binary_a[k] == 0 and binary_b[k] == 1)  # A wrong, B correct
    # McNemar's with continuity correction
    if b + c == 0:
        return b, c, 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = stats.chi2.sf(chi2, df=1)
    return b, c, chi2, p


MCNEMAR_ROWS = []


def run_mcnemar(comparison, label_a, label_b, get_a, get_b):
    for split in ["contaminated", "clean"]:
        print(f"\n{'='*90}")
        print(f"McNEMAR: {comparison} | TEST SPLIT: {split.upper()} (n=1250 paired trials)")
        print(f"{'='*90}")
        print(f"{'Epoch':>6}  {label_a+' only':>14}  {label_b+' only':>14}  {'chi2':>8}  {'p-value':>10}  {'Sig':>5}")
        print("-" * 90)
        for e in epochs:
            ba = get_a(split, e)
            bb = get_b(split, e)
            b, c, chi2, p = mcnemar_test(ba, bb)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{e:>6}  {b:>14}  {c:>14}  {chi2:>8.3f}  {p:>10.6f}  {sig:>5}")
            MCNEMAR_ROWS.append({
                "comparison": comparison, "split": split, "epoch": e,
                "a_only": b, "b_only": c, "chi2": float(chi2),
                "p_value": float(p), "sig": sig,
            })


print("\n\n" + "#" * 90)
print("# McNEMAR'S TEST (per-trial binary correctness, 125 samples x 10 evals = 1250 pairs)")
print("#" * 90)

run_mcnemar("contaminated_vs_clean", "Contam", "Clean",
            lambda split, e: binary_data["contaminated"][split][e],
            lambda split, e: binary_data["clean"][split][e])

run_mcnemar("contaminated_vs_base", "Contam", "Base",
            lambda split, e: binary_data["contaminated"][split][e],
            lambda split, e: base_binary[split])

run_mcnemar("clean_vs_base", "Clean", "Base",
            lambda split, e: binary_data["clean"][split][e],
            lambda split, e: base_binary[split])

mcnemar_fields = ["comparison", "split", "epoch", "a_only", "b_only",
                  "chi2", "p_value", "sig"]
with open(OUT_MCNEMAR_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=mcnemar_fields); w.writeheader(); w.writerows(MCNEMAR_ROWS)
print(f"\nWrote {len(MCNEMAR_ROWS)} rows to {OUT_MCNEMAR_CSV}")
