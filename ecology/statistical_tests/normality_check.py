"""Test normality of paired differences (contam - clean) per sample, per epoch, per split.

Result drives the choice between paired t-test (normal) and Wilcoxon signed-rank
(non-normal). Uses Shapiro-Wilk + reports skewness/kurtosis. Writes a CSV summary
and optional Q-Q + histogram diagnostic figure per model.

Usage:
    python normality_check.py \\
        --contam-dir ecology/outcomes/outputs_qwen3/evals/exp_contaminated_e1-e10 \\
        --clean-dir  ecology/outcomes/outputs_qwen3/evals/exp_clean_e1-e10 \\
        --out-csv    ecology/outcomes/outputs_qwen3/evals/stats/normality.csv \\
        --plot-dir   ecology/outcomes/outputs_qwen3/evals/plots/normality
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


def run(contam_dir, clean_dir, out_csv, plot_dir, ckpt_step):
    contam = {s: load(contam_dir, s, ckpt_step) for s in ("contaminated", "clean")}
    clean = {s: load(clean_dir, s, ckpt_step) for s in ("contaminated", "clean")}

    rows = []
    diffs_by_split = {s: [] for s in ("contaminated", "clean")}
    for split in ("contaminated", "clean"):
        epochs = sorted(set(contam[split]) & set(clean[split]))
        print(f"\n{'='*92}")
        print(f"Normality of paired diffs (contam - clean) | split = {split.upper()}")
        print(f"{'='*92}")
        print(f"{'Epoch':>6}  {'n':>4}  {'mean':>8}  {'std':>7}  {'skew':>7}  {'kurt':>7}  "
              f"{'W':>7}  {'p (SW)':>10}  Decision")
        print("-" * 92)
        for e in epochs:
            sids = sorted(set(contam[split][e]) & set(clean[split][e]))
            d = np.array([contam[split][e][s] - clean[split][e][s] for s in sids])
            if len(d) < 3:
                continue
            W, p = stats.shapiro(d)
            sk = float(stats.skew(d))
            ku = float(stats.kurtosis(d))
            decision = "t-test ok" if p >= 0.05 else "use Wilcoxon"
            print(f"{e:>6}  {len(d):>4}  {d.mean():>+8.4f}  {d.std(ddof=1):>7.4f}  "
                  f"{sk:>+7.3f}  {ku:>+7.3f}  {W:>7.4f}  {p:>10.6f}  {decision}")
            rows.append({
                "split": split, "epoch": e, "n": len(d),
                "mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)),
                "skew": sk, "kurtosis": ku,
                "shapiro_W": float(W), "shapiro_p": float(p),
                "normal_at_0.05": bool(p >= 0.05),
            })
            diffs_by_split[split].append((e, d))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fields = ["split", "epoch", "n", "mean_diff", "std_diff", "skew", "kurtosis",
              "shapiro_W", "shapiro_p", "normal_at_0.05"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n_total = len(rows)
    n_normal = sum(1 for r in rows if r["normal_at_0.05"])
    print(f"\n{n_normal}/{n_total} epoch x split cells pass Shapiro-Wilk at alpha=0.05")
    print(f"recommendation: {'paired t-test' if n_normal == n_total else 'Wilcoxon signed-rank (or report both)'}")
    print(f"wrote {out_csv}")

    if plot_dir:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        os.makedirs(plot_dir, exist_ok=True)
        for split in ("contaminated", "clean"):
            cells = diffs_by_split[split]
            if not cells:
                continue
            n = len(cells)
            fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.5))
            if n == 1:
                axes = axes.reshape(2, 1)
            for j, (e, d) in enumerate(cells):
                axes[0, j].hist(d, bins=20, color="C0", edgecolor="black", alpha=0.75)
                axes[0, j].axvline(0, color="red", linestyle="--", alpha=0.6)
                axes[0, j].set_title(f"E{e}", fontsize=10)
                if j == 0:
                    axes[0, j].set_ylabel("Histogram\n(diff)", fontsize=10)
                stats.probplot(d, dist="norm", plot=axes[1, j])
                axes[1, j].set_title("")
                axes[1, j].get_lines()[0].set_markersize(3)
                if j == 0:
                    axes[1, j].set_ylabel("Q-Q\n(diff vs normal)", fontsize=10)
                else:
                    axes[1, j].set_ylabel("")
            fig.suptitle(f"Paired diffs (contam-clean) — {split} test split",
                         fontsize=12, fontweight="bold")
            fig.tight_layout()
            p = os.path.join(plot_dir, f"normality_{split}.png")
            fig.savefig(p, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"saved: {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contam-dir", required=True)
    ap.add_argument("--clean-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--plot-dir", default=None)
    ap.add_argument("--ckpt-step", type=int, default=CKPT_STEP_DEFAULT)
    args = ap.parse_args()
    run(args.contam_dir, args.clean_dir, args.out_csv, args.plot_dir, args.ckpt_step)


if __name__ == "__main__":
    main()
