"""Plot per-epoch distributions of per-sample accuracy (pass@10) for contaminated vs clean models."""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    counts = {}  # sample_id -> [total, correct]
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


# Collect: data[model][split][epoch] = array of 125 per-sample accuracies
data = {}
for model in MODELS:
    data[model] = {"contaminated": {}, "clean": {}}
    for ckpt in CHECKPOINTS:
        csv_path = os.path.join(BASE, model, ckpt, "results.csv")
        if not os.path.exists(csv_path):
            continue
        epoch = round(parse_checkpoint_num(ckpt) / STEP_INCREMENT)
        for split in ["contaminated", "clean"]:
            sample_accs = get_per_sample_accuracy(csv_path, split)
            data[model][split][epoch] = np.array(
                [sample_accs[sid] for sid in sorted(sample_accs)]
            )

epochs = sorted(data["contaminated"]["contaminated"].keys())

# --- Plot: violin/box plots per epoch, one row per test split ---
fig, axes = plt.subplots(2, 1, figsize=(18, 11), sharex=True)

model_colors = {"contaminated": "#d62728", "clean": "#2ca02c"}
model_labels = {"contaminated": "Contaminated Model", "clean": "Clean Model"}

for ax, test_split in zip(axes, ["contaminated", "clean"]):
    positions_contam = np.arange(len(epochs)) * 3
    positions_clean = positions_contam + 1

    bp_contam = ax.boxplot(
        [data["contaminated"][test_split][e] for e in epochs],
        positions=positions_contam, widths=0.8, patch_artist=True,
        showmeans=True, meanline=True,
        boxprops=dict(facecolor="#d62728", edgecolor="#8b0000", linewidth=1.5, alpha=0.7),
        medianprops=dict(color="black", linewidth=2),
        meanprops=dict(color="black", linewidth=2, linestyle="--"),
        whiskerprops=dict(color="#8b0000", linewidth=1.5),
        capprops=dict(color="#8b0000", linewidth=1.5),
        flierprops=dict(markeredgecolor="#8b0000", marker="o", markersize=4),
    )
    bp_clean = ax.boxplot(
        [data["clean"][test_split][e] for e in epochs],
        positions=positions_clean, widths=0.8, patch_artist=True,
        showmeans=True, meanline=True,
        boxprops=dict(facecolor="#2ca02c", edgecolor="#006400", linewidth=1.5, alpha=0.7),
        medianprops=dict(color="black", linewidth=2),
        meanprops=dict(color="black", linewidth=2, linestyle="--"),
        whiskerprops=dict(color="#006400", linewidth=1.5),
        capprops=dict(color="#006400", linewidth=1.5),
        flierprops=dict(markeredgecolor="#006400", marker="o", markersize=4),
    )

    # X-axis
    ax.set_xticks(positions_contam + 0.5)
    ax.set_xticklabels([f"Epoch {e}" for e in epochs], fontsize=10, rotation=45, ha="right")

    # Y-axis
    ax.set_ylabel("Per-Sample Accuracy (pass@10)", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylim(-0.05, 1.05)

    split_label = "Contaminated" if test_split == "contaminated" else "Clean"
    ax.set_title(f"Evaluated on {split_label} Test Split (n=125 samples)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    # Legend
    ax.legend(
        [bp_contam["boxes"][0], bp_clean["boxes"][0]],
        ["Contaminated Model", "Clean Model"],
        fontsize=11, loc="upper left", framealpha=0.9,
    )

fig.suptitle("Per-Sample Accuracy Distributions Across Training Epochs\n(OLMo-3-7B, MuSR, 10 evals per sample)",
             fontsize=15, fontweight="bold")
plt.tight_layout()
outdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outcomes")
plt.savefig(os.path.join(outdir, "accuracy_distributions.png"), dpi=150, bbox_inches="tight")
plt.savefig(os.path.join(outdir, "accuracy_distributions.pdf"), bbox_inches="tight")
print("Saved to ecology/outcomes/accuracy_distributions.png and .pdf")
