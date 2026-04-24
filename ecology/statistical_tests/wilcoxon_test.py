"""Wilcoxon signed-rank tests + bootstrap CIs across 3 model variants.

For each (test_split, epoch) prints mean pass_rate (with 95% bootstrap CI) for
clean / contaminated / exact, plus paired Wilcoxon p-values for:
    - clean   vs contaminated
    - exact   vs contaminated

Layout per run dir is auto-detected: vLLM eval_*_split.csv or OLMo per-trial.

Usage:
    python wilcoxon_test.py \\
        --clean-dir  ecology/outcomes/outputs_qwen3/evals/exp_clean_e1-e10 \\
        --contam-dir ecology/outcomes/outputs_qwen3/evals/exp_contaminated_e1-e10 \\
        --exact-dir  ecology/outcomes/outputs_qwen3/evals/exp_exact_e1-e10 \\
        --label "Qwen3-8B" \\
        --out-csv    ecology/outcomes/outputs_qwen3/evals/stats/wilcoxon.csv
"""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats

SPLIT_FILE = {"contaminated": "eval_contam_split.csv", "clean": "eval_clean_split.csv"}
CKPT_STEP_DEFAULT = 167


def detect_layout(d):
    if all(os.path.exists(os.path.join(d, f)) for f in SPLIT_FILE.values()):
        return "vllm"
    if glob.glob(os.path.join(d, "checkpoint-*", "results.csv")):
        return "olmo"
    raise SystemExit(f"Unrecognized layout: {d}")


def epoch_of(ckpt, step, max_seen):
    if ckpt == "final":
        return max_seen + 1
    m = re.match(r"checkpoint-(\d+)$", ckpt)
    return round(int(m.group(1)) / step) if m else None


def load_vllm(model_dir, split):
    out = defaultdict(dict)
    with open(os.path.join(model_dir, SPLIT_FILE[split]), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ep = row["epoch"]
            if ep == "final":
                continue
            out[int(ep)][int(row["sample_id"])] = float(row["pass_rate"])
    return out


def load_olmo(model_dir, split, step):
    ckpts = sorted(
        [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))],
        key=lambda n: (0, int(n.split("-")[1])) if n.startswith("checkpoint-") else (1, 0),
    )
    numbered = [c for c in ckpts if c.startswith("checkpoint-")]
    max_epoch = max((epoch_of(c, step, 0) for c in numbered), default=0)
    out = defaultdict(dict)
    for c in ckpts:
        path = os.path.join(model_dir, c, "results.csv")
        if not os.path.exists(path):
            continue
        epoch = epoch_of(c, step, max_epoch)
        if epoch is None:
            continue
        agg = defaultdict(lambda: [0, 0])
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["test_split"] != split:
                    continue
                sid = int(row["sample_id"])
                agg[sid][0] += 1
                if row["correct"].strip() == "1":
                    agg[sid][1] += 1
        for sid, (n, k) in agg.items():
            out[epoch][sid] = k / n if n else 0.0
    return out


def load(model_dir, split, step):
    return load_vllm(model_dir, split) if detect_layout(model_dir) == "vllm" \
        else load_olmo(model_dir, split, step)


def boot_mean_ci(values, n_boot=10000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    idx = rng.randint(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    a = (1 - ci) / 2
    return float(v.mean()), float(np.percentile(means, 100 * a)), float(np.percentile(means, 100 * (1 - a)))


def wilcoxon_p(a, b):
    """Two-sided Wilcoxon signed-rank on paired arrays. Drops zero diffs (default)."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 1 or np.all(a == b):
        return float("nan"), float("nan")
    try:
        r = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="auto")
    except ValueError:
        return float("nan"), float("nan")
    return float(r.statistic), float(r.pvalue)


def fmt_ci(m, lo, hi):
    return f"{m*100:5.2f} [{lo*100:5.2f},{hi*100:5.2f}]"


def fmt_p(p):
    if not np.isfinite(p):
        return "    n/a"
    if p < 0.001:
        return f"{p:7.1e}"
    return f"{p:7.4f}"


def sig_marker(p):
    return ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "") if np.isfinite(p) else ""


def run(clean_dir, contam_dir, exact_dir, label, out_csv, ckpt_step):
    runs = {
        "clean": {s: load(clean_dir, s, ckpt_step) for s in ("contaminated", "clean")},
        "contaminated": {s: load(contam_dir, s, ckpt_step) for s in ("contaminated", "clean")},
    }
    if exact_dir:
        runs["exact"] = {s: load(exact_dir, s, ckpt_step) for s in ("contaminated", "clean")}

    rows = []
    for split in ("contaminated", "clean"):
        epoch_sets = [set(runs[m][split].keys()) for m in runs]
        epochs = sorted(set.intersection(*epoch_sets))
        print(f"\n{'='*108}")
        print(f"  {label}  |  test split = {split.upper()}  |  Wilcoxon signed-rank (two-sided), bootstrap 95% CIs (n_boot=10000)")
        print(f"{'='*108}")
        header = f"{'Ep':>3}  {'Clean acc % [95% CI]':>22}  {'Contam acc % [95% CI]':>22}"
        if "exact" in runs:
            header += f"  {'Exact acc % [95% CI]':>22}"
        header += f"  {'p (clean vs contam)':>20}  {'p (exact vs contam)':>20}"
        print(header)
        print("-" * len(header))
        for e in epochs:
            sids = sorted(set.intersection(*[set(runs[m][split][e].keys()) for m in runs]))
            arrs = {m: np.array([runs[m][split][e][s] for s in sids]) for m in runs}
            ci_clean = boot_mean_ci(arrs["clean"])
            ci_contam = boot_mean_ci(arrs["contaminated"])
            ci_exact = boot_mean_ci(arrs["exact"]) if "exact" in arrs else (np.nan, np.nan, np.nan)

            _, p_cc = wilcoxon_p(arrs["clean"], arrs["contaminated"])
            if "exact" in arrs:
                _, p_ec = wilcoxon_p(arrs["exact"], arrs["contaminated"])
            else:
                p_ec = float("nan")

            line = (
                f"{e:>3}  {fmt_ci(*ci_clean):>22}  {fmt_ci(*ci_contam):>22}"
            )
            if "exact" in runs:
                line += f"  {fmt_ci(*ci_exact):>22}"
            line += f"  {fmt_p(p_cc):>14} {sig_marker(p_cc):>3}  {fmt_p(p_ec):>14} {sig_marker(p_ec):>3}"
            print(line)

            row = {
                "model": label, "split": split, "epoch": e, "n_paired": len(sids),
                "clean_mean": ci_clean[0], "clean_ci_lo": ci_clean[1], "clean_ci_hi": ci_clean[2],
                "contam_mean": ci_contam[0], "contam_ci_lo": ci_contam[1], "contam_ci_hi": ci_contam[2],
                "exact_mean": ci_exact[0], "exact_ci_lo": ci_exact[1], "exact_ci_hi": ci_exact[2],
                "wilcoxon_p_clean_vs_contam": p_cc, "sig_clean_vs_contam": sig_marker(p_cc),
                "wilcoxon_p_exact_vs_contam": p_ec, "sig_exact_vs_contam": sig_marker(p_ec),
            }
            rows.append(row)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fields = ["model", "split", "epoch", "n_paired",
              "clean_mean", "clean_ci_lo", "clean_ci_hi",
              "contam_mean", "contam_ci_lo", "contam_ci_hi",
              "exact_mean", "exact_ci_lo", "exact_ci_hi",
              "wilcoxon_p_clean_vs_contam", "sig_clean_vs_contam",
              "wilcoxon_p_exact_vs_contam", "sig_exact_vs_contam"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {out_csv}")
    print("legend: *** p<0.001  ** p<0.01  * p<0.05")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean-dir", required=True)
    ap.add_argument("--contam-dir", required=True)
    ap.add_argument("--exact-dir", default=None)
    ap.add_argument("--label", required=True, help="Model label for header / CSV")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--ckpt-step", type=int, default=CKPT_STEP_DEFAULT)
    args = ap.parse_args()
    run(args.clean_dir, args.contam_dir, args.exact_dir, args.label, args.out_csv, args.ckpt_step)


if __name__ == "__main__":
    main()
