"""Two-sided paired t-test on eval_checkpoints_vllm / eval_qwen3_checkpoints output.

Expects each input dir to contain:
    eval_contam_split.csv, eval_clean_split.csv
with per-sample rows: epoch, checkpoint, sample_id, pass_rate, ...

Pairs contaminated-model vs clean-model pass_rates per sample, per epoch,
per test split.

Usage:
    python paired_ttest.py \\
        --contam-dir ecology/outcomes/outputs_qwen3/evals/exp_contaminated_e1-e10 \\
        --clean-dir  ecology/outcomes/outputs_qwen3/evals/exp_clean_e1-e10 \\
        --out-csv    ecology/outcomes/outputs_qwen3/evals/stats/paired_ttest.csv
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from scipy import stats

SPLIT_FILE = {"contaminated": "eval_contam_split.csv", "clean": "eval_clean_split.csv"}


def load_pass_rates(model_dir, split):
    """Return dict epoch -> {sample_id: pass_rate}."""
    out = defaultdict(dict)
    path = os.path.join(model_dir, SPLIT_FILE[split])
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            epoch = int(row["epoch"])
            out[epoch][int(row["sample_id"])] = float(row["pass_rate"])
    return out


def run(contam_dir, clean_dir, out_csv):
    contam = {s: load_pass_rates(contam_dir, s) for s in ("contaminated", "clean")}
    clean = {s: load_pass_rates(clean_dir, s) for s in ("contaminated", "clean")}

    rows = []
    for split in ("contaminated", "clean"):
        epochs = sorted(set(contam[split]) & set(clean[split]))
        print(f"\n{'='*95}")
        print(f"Paired t-test (two-sided): CONTAM vs CLEAN model | test split = {split.upper()}")
        print(f"{'='*95}")
        print(f"{'Epoch':>6}  {'Contam':>9}  {'Clean':>9}  {'Diff':>9}  {'t':>8}  {'p':>10}  {'95% CI':>22}  Sig")
        print("-" * 95)
        for e in epochs:
            sids = sorted(set(contam[split][e]) & set(clean[split][e]))
            a = np.array([contam[split][e][s] for s in sids])
            b = np.array([clean[split][e][s] for s in sids])
            diff = a - b
            t_stat, p_val = stats.ttest_rel(a, b)
            n = len(diff)
            se = diff.std(ddof=1) / np.sqrt(n)
            tcrit = stats.t.ppf(0.975, df=n - 1)
            ci_lo = diff.mean() - tcrit * se
            ci_hi = diff.mean() + tcrit * se
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            ci = f"[{ci_lo:+.4f},{ci_hi:+.4f}]"
            print(f"{e:>6}  {a.mean():>9.4f}  {b.mean():>9.4f}  {diff.mean():>+9.4f}  "
                  f"{t_stat:>8.3f}  {p_val:>10.6f}  {ci:>22}  {sig}")
            rows.append({
                "split": split, "epoch": e, "n_paired": n,
                "contam_mean": float(a.mean()), "clean_mean": float(b.mean()),
                "diff_mean": float(diff.mean()),
                "t_stat": float(t_stat), "p_value": float(p_val),
                "ci95_low": float(ci_lo), "ci95_high": float(ci_hi), "sig": sig,
            })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fields = ["split", "epoch", "n_paired", "contam_mean", "clean_mean",
              "diff_mean", "t_stat", "p_value", "ci95_low", "ci95_high", "sig"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contam-dir", required=True, help="Dir with contaminated-model eval_*_split.csv")
    ap.add_argument("--clean-dir", required=True, help="Dir with clean-model eval_*_split.csv")
    ap.add_argument("--out-csv", required=True, help="Output CSV path for test statistics")
    args = ap.parse_args()
    run(args.contam_dir, args.clean_dir, args.out_csv)


if __name__ == "__main__":
    main()
