"""Plot Qwen3 checkpoint evaluation results: contaminated vs clean accuracy over epochs."""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

EVAL_DIR = Path(__file__).parent / "outcomes" / "qwen3_evals"
OUT_DIR = EVAL_DIR / "plots"


def load_summary(path):
    epochs, contam_acc, clean_acc = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = row["epoch"]
            if ep == "final":
                continue  # skip final for clean time-series plotting
            epochs.append(int(ep))
            contam_acc.append(float(row["contaminated_accuracy"]))
            clean_acc.append(float(row["clean_accuracy"]))
    return np.array(epochs), np.array(contam_acc), np.array(clean_acc)


def load_per_sample(path):
    """Load per-sample results grouped by epoch for bootstrap CIs."""
    by_epoch = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = row["epoch"]
            if ep == "final":
                continue
            ep = int(ep)
            if ep not in by_epoch:
                by_epoch[ep] = []
            by_epoch[ep].append(float(row["pass_rate"]))
    return by_epoch


def bootstrap_ci(values, n_boot=10000, ci=0.95):
    rng = np.random.RandomState(42)
    values = np.asarray(values)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, 100 * alpha), np.percentile(boot_means, 100 * (1 - alpha))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load summaries
    contam_epochs, contam_contam, contam_clean = load_summary(
        EVAL_DIR / "contaminated_model_20" / "eval_summary.csv")
    clean_epochs, clean_contam, clean_clean = load_summary(
        EVAL_DIR / "clean_model_20" / "eval_summary.csv")

    # Load per-sample for CIs
    contam_per_contam = load_per_sample(EVAL_DIR / "contaminated_model_20" / "eval_contam_split.csv")
    contam_per_clean = load_per_sample(EVAL_DIR / "contaminated_model_20" / "eval_clean_split.csv")
    clean_per_contam = load_per_sample(EVAL_DIR / "clean_model_20" / "eval_contam_split.csv")
    clean_per_clean = load_per_sample(EVAL_DIR / "clean_model_20" / "eval_clean_split.csv")

    # --- Plot 1: Contaminated test split (both models) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, split_label, contam_data, clean_data, contam_per, clean_per in [
        (ax1, "Contaminated Test Split",
         (contam_epochs, contam_contam), (clean_epochs, clean_contam),
         contam_per_contam, clean_per_contam),
        (ax2, "Clean Test Split",
         (contam_epochs, contam_clean), (clean_epochs, clean_clean),
         contam_per_clean, clean_per_clean),
    ]:
        # Contaminated model
        ep, acc = contam_data
        ci_lo = np.array([bootstrap_ci(contam_per[e])[0] for e in ep])
        ci_hi = np.array([bootstrap_ci(contam_per[e])[1] for e in ep])
        ax.plot(ep, acc * 100, marker="o", color="C3", label="Contaminated model", linewidth=2)
        ax.fill_between(ep, ci_lo * 100, ci_hi * 100, alpha=0.15, color="C3")

        # Clean model
        ep, acc = clean_data
        ci_lo = np.array([bootstrap_ci(clean_per[e])[0] for e in ep])
        ci_hi = np.array([bootstrap_ci(clean_per[e])[1] for e in ep])
        ax.plot(ep, acc * 100, marker="s", color="C0", label="Clean model", linewidth=2)
        ax.fill_between(ep, ci_lo * 100, ci_hi * 100, alpha=0.15, color="C0")

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_title(split_label, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 11))

    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    fig.suptitle("Qwen3-8B: Contaminated vs Clean Model Accuracy Over Training",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_over_epochs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'accuracy_over_epochs.png'}")

    # --- Plot 2: Difference (contam_acc - clean_acc) for each model ---
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(contam_epochs, (contam_contam - contam_clean) * 100,
            marker="o", color="C3", label="Contaminated model", linewidth=2)
    ax.plot(clean_epochs, (clean_contam - clean_clean) * 100,
            marker="s", color="C0", label="Clean model", linewidth=2)
    ax.axhline(0, color="black", linestyle="--", alpha=0.4)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy Gap: Contam Test - Clean Test (%)", fontsize=12)
    ax.set_title("Qwen3-8B: Accuracy Gap (Contaminated - Clean Test Split)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 11))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'accuracy_gap.png'}")

    # --- Plot 3: DiD over epochs ---
    paired_path = EVAL_DIR / "paired_20" / "paired_analysis.csv"
    if paired_path.exists():
        epochs, dids = [], []
        with open(paired_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                dids.append(float(row["did"]) * 100)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, dids, marker="o", color="C3", linewidth=2)
        ax.axhline(0, color="black", linestyle="--", alpha=0.4)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("DiD (%)", fontsize=12)
        ax.set_title("Qwen3-8B: Difference-in-Differences Over Training",
                      fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 11))
        fig.tight_layout()
        fig.savefig(OUT_DIR / "did_over_epochs.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {OUT_DIR / 'did_over_epochs.png'}")

    print("Done!")


if __name__ == "__main__":
    main()
