"""Plot accuracy-over-epoch curves for any number of model runs (clean / contaminated / exact / ...).

Auto-detects each run's layout:
    - vLLM-style:  {dir}/eval_{contam,clean}_split.csv with per-sample pass_rate
    - OLMo-style:  {dir}/checkpoint-*/results.csv with per-trial correct (0/1)

Usage:
    python plot_evals.py \\
        --run clean=ecology/outcomes/outputs_qwen35/evals/exp_clean \\
        --run contaminated=ecology/outcomes/outputs_qwen35/evals/exp_contaminated \\
        --run exact=ecology/outcomes/outputs_qwen35/evals/exp_exact \\
        --title "Qwen3.5-9B" \\
        --out-dir ecology/outcomes/outputs_qwen35/evals/plots
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

SPLIT_FILE = {"contaminated": "eval_contam_split.csv", "clean": "eval_clean_split.csv"}
COLORS = {"clean": "C0", "contaminated": "C3", "exact": "C2"}
MARKERS = {"clean": "s", "contaminated": "o", "exact": "D"}


def detect_layout(d):
    if all(os.path.exists(os.path.join(d, f)) for f in SPLIT_FILE.values()):
        return "vllm"
    if glob.glob(os.path.join(d, "checkpoint-*", "results.csv")):
        return "olmo"
    raise SystemExit(f"Unrecognized layout: {d}")


def epoch_of(ckpt_name, step, max_seen):
    if ckpt_name == "final":
        return max_seen + 1
    m = re.match(r"checkpoint-(\d+)$", ckpt_name)
    return round(int(m.group(1)) / step) if m else None


def load_per_sample_vllm(run_dir):
    """Return dict split -> epoch -> list[pass_rate] across samples."""
    out = {s: defaultdict(list) for s in SPLIT_FILE}
    final_rows = {s: [] for s in SPLIT_FILE}
    max_epoch = 0
    for split, fname in SPLIT_FILE.items():
        with open(os.path.join(run_dir, fname), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ep = row["epoch"]
                if ep == "final":
                    final_rows[split].append(float(row["pass_rate"]))
                    continue
                ep_i = int(ep)
                max_epoch = max(max_epoch, ep_i)
                out[split][ep_i].append(float(row["pass_rate"]))
    for split, vals in final_rows.items():
        if vals:
            out[split][max_epoch + 1].extend(vals)
    return out


def load_per_sample_olmo(run_dir, step):
    """Aggregate per-trial -> per-sample pass_rate, dict split -> epoch -> list."""
    ckpts = sorted(
        [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))],
        key=lambda n: (0, int(n.split("-")[1])) if n.startswith("checkpoint-") else (1, 0),
    )
    numbered = [c for c in ckpts if c.startswith("checkpoint-")]
    max_epoch = max((epoch_of(c, step, 0) for c in numbered), default=0)

    out = {"contaminated": defaultdict(list), "clean": defaultdict(list)}
    for ckpt in ckpts:
        path = os.path.join(run_dir, ckpt, "results.csv")
        if not os.path.exists(path):
            continue
        epoch = epoch_of(ckpt, step, max_epoch)
        if epoch is None:
            continue
        agg = {"contaminated": defaultdict(lambda: [0, 0]),
               "clean": defaultdict(lambda: [0, 0])}
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                split = row["test_split"]
                if split not in agg:
                    continue
                sid = int(row["sample_id"])
                agg[split][sid][0] += 1
                if row["correct"].strip() == "1":
                    agg[split][sid][1] += 1
        for split, samples in agg.items():
            for sid, (n, k) in samples.items():
                out[split][epoch].append(k / n if n else 0.0)
    return out


def load_per_sample(run_dir, ckpt_step):
    layout = detect_layout(run_dir)
    if layout == "vllm":
        return load_per_sample_vllm(run_dir)
    return load_per_sample_olmo(run_dir, ckpt_step)


def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, 100 * alpha), np.percentile(boot_means, 100 * (1 - alpha))


def plot_runs(runs_data, title, out_dir):
    """runs_data: dict run_label -> {split -> epoch -> [pass_rates]}"""
    os.makedirs(out_dir, exist_ok=True)

    MAX_EPOCH = 10
    runs_data = {
        lbl: {split: {e: v for e, v in by_ep.items() if e <= MAX_EPOCH}
              for split, by_ep in data.items()}
        for lbl, data in runs_data.items()
    }

    LABEL_FS = 26
    TITLE_FS = 28
    SUPTITLE_FS = 30
    TICK_FS = 22
    LEGEND_FS = 22
    LW = 3.5
    MS = 12

    # --- Plot 1: accuracy on each test split, one panel per split ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    for ax, split in zip(axes, ["contaminated", "clean"]):
        for label, data in runs_data.items():
            d = data.get(split, {})
            if not d:
                continue
            epochs = sorted(d.keys())
            means = np.array([np.mean(d[e]) for e in epochs])
            ci_lo = np.array([bootstrap_ci(d[e])[0] for e in epochs])
            ci_hi = np.array([bootstrap_ci(d[e])[1] for e in epochs])
            ax.plot(epochs, means * 100, marker=MARKERS.get(label, "x"),
                    color=COLORS.get(label, None), label=label.capitalize(),
                    linewidth=LW, markersize=MS)
            ax.fill_between(epochs, ci_lo * 100, ci_hi * 100, alpha=0.15,
                            color=COLORS.get(label, None))
        ax.set_xlabel("Epoch", fontsize=LABEL_FS)
        ax.set_title(f"{split.capitalize()} test split", fontsize=TITLE_FS)
        ax.legend(fontsize=LEGEND_FS)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=TICK_FS)
    axes[0].set_ylabel("Accuracy (%)", fontsize=LABEL_FS)
    fig.suptitle(f"{title}: model accuracy over training epochs",
                 fontsize=SUPTITLE_FS, fontweight="bold")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "accuracy_over_epochs.pdf")
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {p1}")

    # --- Plot 2: contam-vs-clean test gap per model ---
    fig, ax = plt.subplots(figsize=(11, 7))
    for label, data in runs_data.items():
        epochs = sorted(set(data.get("contaminated", {})) & set(data.get("clean", {})))
        if not epochs:
            continue
        gap = np.array([np.mean(data["contaminated"][e]) - np.mean(data["clean"][e])
                        for e in epochs])
        ax.plot(epochs, gap * 100, marker=MARKERS.get(label, "x"),
                color=COLORS.get(label, None), label=label.capitalize(),
                linewidth=LW, markersize=MS)
    ax.axhline(0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel("Epoch", fontsize=LABEL_FS)
    ax.set_ylabel("Contam - Clean (%)", fontsize=LABEL_FS)
    ax.set_title(f"{title}: contamination gap", fontsize=TITLE_FS, fontweight="bold")
    ax.legend(fontsize=LEGEND_FS)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    fig.tight_layout()
    p2 = os.path.join(out_dir, "accuracy_gap.pdf")
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {p2}")


def parse_run(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--run expects label=path, got: {s}")
    label, path = s.split("=", 1)
    return label.strip(), path.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, type=parse_run,
                    help="Run as label=path/to/eval_dir (repeatable)")
    ap.add_argument("--title", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ckpt-step", type=int, default=167,
                    help="Steps per epoch for OLMo-style layouts (default 167)")
    args = ap.parse_args()

    runs_data = {}
    for label, path in args.run:
        print(f"loading {label}: {path}")
        runs_data[label] = load_per_sample(path, args.ckpt_step)

    plot_runs(runs_data, args.title, args.out_dir)


if __name__ == "__main__":
    main()
