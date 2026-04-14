"""Plot contaminated vs clean model accuracy over training checkpoints."""

import os
import csv
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

def parse_checkpoint_num(name):
    if name == "final":
        return 1837  # next step after 1670 (increment of 167)
    return int(name.split("-")[1])

def compute_accuracy(csv_path, test_split_filter=None):
    """Compute accuracy from a results.csv, optionally filtering by test_split."""
    correct, total = 0, 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if test_split_filter and row["test_split"] != test_split_filter:
                continue
            total += 1
            if row["correct"].strip() == "1":
                correct += 1
    return correct / total if total > 0 else None

# Collect results: results[model_type][test_split] = list of (step, accuracy)
results = {}
for model in MODELS:
    results[model] = {"contaminated": [], "clean": []}
    for ckpt in CHECKPOINTS:
        csv_path = os.path.join(BASE, model, ckpt, "results.csv")
        if not os.path.exists(csv_path):
            print(f"Missing: {csv_path}")
            continue
        step = parse_checkpoint_num(ckpt)
        for split in ["contaminated", "clean"]:
            acc = compute_accuracy(csv_path, split)
            if acc is not None:
                results[model][split].append((step, acc))

# --- Map steps to epochs ---
STEP_INCREMENT = 167
STEPS_PER_EPOCH = STEP_INCREMENT  # each checkpoint = 1 epoch
def step_to_epoch(s):
    return round(s / STEPS_PER_EPOCH)

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

colors = {"contaminated": "#d62728", "clean": "#2ca02c"}
markers = {"contaminated": "o", "clean": "s"}
linestyles = {"contaminated": "-", "clean": "--"}

for ax, test_split in zip(axes, ["contaminated", "clean"]):
    for model in MODELS:
        data = results[model][test_split]
        if not data:
            continue
        steps, accs = zip(*sorted(data))
        label = f"{'Contaminated' if model == 'contaminated' else 'Clean'} Model"
        ax.plot(steps, accs, marker=markers[model], label=label,
                color=colors[model], linewidth=2.5, markersize=8,
                linestyle=linestyles[model], markeredgecolor="white",
                markeredgewidth=0.8)

        # Annotate epoch number on each dot
        for s, a in zip(steps, accs):
            epoch = step_to_epoch(s)
            ax.annotate(f"E{epoch}", (s, a), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=7.5,
                        color=colors[model], fontweight="bold")

    ax.set_xlabel("Training Step", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=13, fontweight="bold")
    split_label = "Contaminated" if test_split == "contaminated" else "Clean"
    ax.set_title(f"Evaluated on {split_label} Test Split", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(0.35, 0.78)

    # X-axis: show all checkpoint steps as ticks
    all_steps = sorted(set(s for m in MODELS for s, _ in results[m][test_split]))
    ax.set_xticks(all_steps)
    ax.set_xticklabels([str(s) for s in all_steps], rotation=45, ha="right", fontsize=9)
    ax.tick_params(axis="y", labelsize=10)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

fig.suptitle("Model Accuracy Over Training (OLMo-3-7B on MuSR)",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "outcomes", "training_curves.png"),
    dpi=150, bbox_inches="tight",
)
plt.savefig(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "outcomes", "training_curves.pdf"),
    bbox_inches="tight",
)
print("Saved to ecology/outcomes/training_curves.png and .pdf")
plt.show()
