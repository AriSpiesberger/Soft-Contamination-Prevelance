"""Paired t-test and McNemar's test: contaminated vs clean model, and both vs base model, per epoch."""

import os
import csv
import numpy as np
from scipy import stats

BASE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "outcomes", "ecology_evals",
    "Soft-Contamination-Prevelance", "ecology", "outputs", "checkpoint_evals",
)

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
BASE_CSV = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "outcomes", "base_model_eval_results_20260413_034651.csv",
)


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


def print_comparison(title, model_a_label, model_b_label, get_a, get_b):
    """Print paired t-test table."""
    for split in ["contaminated", "clean"]:
        print(f"\n{'='*78}")
        print(f"{title} | TEST SPLIT: {split.upper()} (n=125 paired samples)")
        print(f"{'='*78}")
        print(f"{'Epoch':>6}  {model_a_label:>12}  {model_b_label:>12}  {'Diff':>8}  {'t-stat':>8}  {'p-value':>10}  {'Sig':>5}")
        print("-" * 78)
        for e in epochs:
            sids_a, accs_a = get_a(split, e)
            sids_b, accs_b = get_b(split, e)
            assert sids_a == sids_b, f"Sample IDs don't match at epoch {e}"
            t_stat, p_val = stats.ttest_rel(accs_a, accs_b)
            diff = accs_a.mean() - accs_b.mean()
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{e:>6}  {accs_a.mean():>12.4f}  {accs_b.mean():>12.4f}  {diff:>+8.4f}  {t_stat:>8.3f}  {p_val:>10.6f}  {sig:>5}")


# 1) Contaminated model vs Clean model
print_comparison(
    "CONTAMINATED MODEL vs CLEAN MODEL",
    "Contam Mean", "Clean Mean",
    lambda split, e: data["contaminated"][split][e],
    lambda split, e: data["clean"][split][e],
)

# 2) Contaminated model vs Base model
print_comparison(
    "CONTAMINATED MODEL vs BASE MODEL",
    "Contam Mean", "Base Mean",
    lambda split, e: data["contaminated"][split][e],
    lambda split, e: base_data[split],  # same base for all epochs
)

# 3) Clean model vs Base model
print_comparison(
    "CLEAN MODEL vs BASE MODEL",
    "Clean Mean", "Base Mean",
    lambda split, e: data["clean"][split][e],
    lambda split, e: base_data[split],  # same base for all epochs
)


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


def print_mcnemar(title, label_a, label_b, get_a, get_b):
    for split in ["contaminated", "clean"]:
        print(f"\n{'='*90}")
        print(f"McNEMAR: {title} | TEST SPLIT: {split.upper()} (n=1250 paired trials)")
        print(f"{'='*90}")
        print(f"{'Epoch':>6}  {label_a+' only':>14}  {label_b+' only':>14}  {'chi2':>8}  {'p-value':>10}  {'Sig':>5}")
        print("-" * 90)
        for e in epochs:
            ba = get_a(split, e)
            bb = get_b(split, e)
            b, c, chi2, p = mcnemar_test(ba, bb)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{e:>6}  {b:>14}  {c:>14}  {chi2:>8.3f}  {p:>10.6f}  {sig:>5}")


print("\n\n" + "#" * 90)
print("# McNEMAR'S TEST (per-trial binary correctness, 125 samples x 10 evals = 1250 pairs)")
print("#" * 90)

# 1) Contaminated vs Clean
print_mcnemar(
    "CONTAMINATED MODEL vs CLEAN MODEL",
    "Contam", "Clean",
    lambda split, e: binary_data["contaminated"][split][e],
    lambda split, e: binary_data["clean"][split][e],
)

# 2) Contaminated vs Base
print_mcnemar(
    "CONTAMINATED MODEL vs BASE MODEL",
    "Contam", "Base",
    lambda split, e: binary_data["contaminated"][split][e],
    lambda split, e: base_binary[split],
)

# 3) Clean vs Base
print_mcnemar(
    "CLEAN MODEL vs BASE MODEL",
    "Clean", "Base",
    lambda split, e: binary_data["clean"][split][e],
    lambda split, e: base_binary[split],
)
