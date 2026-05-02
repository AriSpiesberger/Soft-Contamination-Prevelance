"""
Bucket the per-sample lm-eval logs by the canonical 210/211 contam/clean split
(mbpp_test_train_half.csv / mbpp_test_eval_half.csv) and produce a per-checkpoint
table of pass@1 on the contaminated subset vs the truly-held-out clean subset.

Looks at every condition under finetuning/outputs/harness_results/ that has a
summary.json, finds its run_*_seedXXXX/.../samples_mbpp_*.jsonl files, and
aggregates pass@1 over the 10 seeds for each subset separately.

Usage:
    uv run python finetuning/mbpp/analyze_split.py
    uv run python finetuning/mbpp/analyze_split.py --csv outputs/split_analysis.csv

The shallow-generalization claim says: pass@1 lifts on BOTH subsets after
training on contam-half paraphrases. The clean lift is the headline.
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev

PWD = Path(__file__).parent.parent
RESULTS_DIR = PWD / "outputs" / "harness_results"
TRAIN_HALF_CSV = PWD / "mbpp_data" / "mbpp_test_train_half.csv"
EVAL_HALF_CSV = PWD / "mbpp_data" / "mbpp_test_eval_half.csv"


def load_task_ids(csv_path: Path) -> set[int]:
    with open(csv_path) as f:
        return {int(row["task_id"]) for row in csv.DictReader(f)}


def score_one_run(samples_path: Path, train_half: set[int], eval_half: set[int]):
    """Return (contam_pass, clean_pass, n_contam, n_clean) for one seed's jsonl."""
    contam_hits = contam_total = 0
    clean_hits = clean_total = 0
    other_total = 0
    with open(samples_path) as f:
        for line in f:
            row = json.loads(line)
            task_id = int(row["doc"]["task_id"])
            passed = float(row.get("pass_at_1", 0.0))
            if task_id in train_half:
                contam_total += 1
                contam_hits += passed
            elif task_id in eval_half:
                clean_total += 1
                clean_hits += passed
            else:
                other_total += 1
    return (
        contam_hits / contam_total if contam_total else float("nan"),
        clean_hits / clean_total if clean_total else float("nan"),
        contam_total, clean_total, other_total,
    )


def aggregate_condition(cond_dir: Path, train_half: set[int], eval_half: set[int]):
    contam_scores, clean_scores = [], []
    sizes = None
    for run_dir in sorted(cond_dir.glob("run_*_seed*/")):
        for samples in run_dir.rglob("samples_mbpp_*.jsonl"):
            c, k, nc, nk, no = score_one_run(samples, train_half, eval_half)
            contam_scores.append(c)
            clean_scores.append(k)
            sizes = (nc, nk, no)
    return contam_scores, clean_scores, sizes


def binomial_se(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, p))
    return (p * (1 - p) / n) ** 0.5


def paired_diff_se(p1: float, p2: float, n: int) -> float:
    """SE of p1-p2 when both are over the SAME n items, treated as independent
    (upper bound; true paired SE requires per-item correlation). For two
    correlated proportions over the same items, this overstates the SE."""
    return (binomial_se(p1, n) ** 2 + binomial_se(p2, n) ** 2) ** 0.5


def fmt(scores):
    if not scores:
        return "—"
    m = mean(scores)
    s = stdev(scores) if len(scores) > 1 else 0.0
    return f"{m:.4f} ± {s:.4f}"


def epoch_key(label: str) -> int:
    m = re.match(r"checkpoint-(\d+)", label)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None,
                    help="Optional path to write a CSV of the table.")
    args = ap.parse_args()

    train_half = load_task_ids(TRAIN_HALF_CSV)
    eval_half = load_task_ids(EVAL_HALF_CSV)
    print(f"contam (train_half): {len(train_half)}  |  clean (eval_half): {len(eval_half)}  |  disjoint: {not (train_half & eval_half)}")
    print()

    # Pick the most recent dir for each label.
    by_label: dict[str, Path] = {}
    for cond_dir in sorted(RESULTS_DIR.glob("*_*"), key=lambda p: p.name):
        m = re.match(r"(.+)_(\d{8}_\d{6})$", cond_dir.name)
        if not m or not (cond_dir / "summary.json").exists():
            continue
        label = m.group(1)
        # Latest timestamp wins.
        if label not in by_label or cond_dir.name > by_label[label].name:
            by_label[label] = cond_dir

    # Order: baseline, then checkpoint-N ascending.
    def sort_key(label):
        if label == "baseline":
            return (0, 0)
        return (1, epoch_key(label))

    rows = []
    baseline_contam = baseline_clean = None
    for label in sorted(by_label, key=sort_key):
        cond_dir = by_label[label]
        contam_scores, clean_scores, sizes = aggregate_condition(cond_dir, train_half, eval_half)
        if not contam_scores:
            continue
        contam_mean = mean(contam_scores)
        clean_mean = mean(clean_scores)
        n_contam = sizes[0] if sizes else 0
        n_clean = sizes[1] if sizes else 0
        # Across-seeds std (only meaningful if >1 seed); otherwise binomial SE
        # over the items in the subset (greedy n=1 case).
        if len(contam_scores) > 1:
            contam_se = stdev(contam_scores) / (len(contam_scores) ** 0.5)
            clean_se = stdev(clean_scores) / (len(clean_scores) ** 0.5)
        else:
            contam_se = binomial_se(contam_mean, n_contam)
            clean_se = binomial_se(clean_mean, n_clean)

        if label == "baseline":
            baseline_contam, baseline_clean = contam_mean, clean_mean
            d_contam_se = d_clean_se = None
        d_contam = contam_mean - baseline_contam if baseline_contam is not None and label != "baseline" else None
        d_clean = clean_mean - baseline_clean if baseline_clean is not None and label != "baseline" else None
        d_contam_se = paired_diff_se(contam_mean, baseline_contam, n_contam) if d_contam is not None else None
        d_clean_se = paired_diff_se(clean_mean, baseline_clean, n_clean) if d_clean is not None else None

        rows.append({
            "label": label,
            "n_seeds": len(contam_scores),
            "contam_mean": contam_mean, "contam_se": contam_se,
            "clean_mean": clean_mean, "clean_se": clean_se,
            "delta_contam": d_contam, "delta_contam_se": d_contam_se,
            "delta_clean": d_clean, "delta_clean_se": d_clean_se,
            "n_contam": n_contam, "n_clean": n_clean,
            "n_other": sizes[2] if sizes else None,
        })

    # Pretty print.
    header = (
        f"{'condition':<14}  {'n':>3}  "
        f"{'contam pass@1':>18}  {'Δ contam':>15}  "
        f"{'clean pass@1':>18}  {'Δ clean':>15}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        contam_str = f"{r['contam_mean']:.4f} ± {r['contam_se']:.4f}"
        clean_str = f"{r['clean_mean']:.4f} ± {r['clean_se']:.4f}"
        if r["delta_contam"] is not None:
            d_contam_str = f"{r['delta_contam']*100:+.2f} ± {r['delta_contam_se']*100:.2f}pp"
            d_clean_str = f"{r['delta_clean']*100:+.2f} ± {r['delta_clean_se']*100:.2f}pp"
        else:
            d_contam_str = "—"
            d_clean_str = "—"
        print(f"{r['label']:<14}  {r['n_seeds']:>3}  {contam_str:>18}  {d_contam_str:>15}  {clean_str:>18}  {d_clean_str:>15}")
    if rows and rows[0]["n_other"]:
        print(f"\n(per run: {rows[0]['n_contam']} contam items, {rows[0]['n_clean']} clean items, "
              f"{rows[0]['n_other']} excluded MBPP-test items not in either half)")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV -> {out}")


if __name__ == "__main__":
    main()
