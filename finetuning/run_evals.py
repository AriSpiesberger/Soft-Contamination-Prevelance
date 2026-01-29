#!/usr/bin/env python3
"""
Run evaluations using the existing p3_eval_mbpp.py script.
Evaluates baseline + checkpoints at epochs 3, 6, 10.

Usage:
    python run_evals.py
"""

import subprocess
import sys
import json
from pathlib import Path

PWD = Path(__file__).parent
CHECKPOINT_DIR = PWD / "outputs" / "checkpoints"
RESULTS_FILE = PWD / "outputs" / "eval_results.json"

EXPERIMENTS = ["sem_dupes", "exact_dupes", "cosine_sim"]
EPOCHS = [3, 6, 10]


def find_epoch_checkpoint(exp_dir: Path, epoch: int) -> str:
    if epoch == 10:
        final_dir = exp_dir / "final"
        if final_dir.exists():
            return str(final_dir)

    checkpoints = sorted(exp_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not checkpoints:
        return None

    steps_per_epoch = int(checkpoints[0].name.split("-")[1])
    target_step = epoch * steps_per_epoch

    for ckpt in checkpoints:
        step = int(ckpt.name.split("-")[1])
        if step >= target_step - 2:
            return str(ckpt)
    return None


def run_eval(finetuned_path: str = None, split: str = "eval") -> float:
    """Run p3_eval_mbpp.py and return pass@1 percentage."""
    cmd = [sys.executable, "p3_eval_mbpp.py", "--test-split", split, "--no-wandb"]

    if finetuned_path:
        cmd.extend(["--finetuned-path", finetuned_path])

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    if result.returncode != 0:
        print(f"FAILED: {result.stderr[-500:]}")
        return None

    # Parse pass@1 from output
    for line in result.stdout.split("\n"):
        if "pass@1:" in line:
            # Format: "  pass@1: 45.00% (93/207)"
            try:
                pct = float(line.split("pass@1:")[1].split("%")[0].strip())
                return pct
            except:
                pass

    print(f"Could not parse result from: {result.stdout[-500:]}")
    return None


def main():
    all_results = {}

    # Fix CSV path in p3_eval_mbpp.py to use correct location
    print("Updating p3_eval_mbpp.py paths...")
    p3_path = PWD / "p3_eval_mbpp.py"
    content = p3_path.read_text()

    # Update paths if needed
    old_train = 'TEST_TRAIN_HALF = Path(__file__).parent.parent / "mbpp_test_train_half.csv"'
    new_train = 'TEST_TRAIN_HALF = Path(__file__).parent / "mbpp_data" / "mbpp_test_train_half.csv"'
    old_eval = 'TEST_EVAL_HALF = Path(__file__).parent.parent / "mbpp_test_eval_half.csv"'
    new_eval = 'TEST_EVAL_HALF = Path(__file__).parent / "mbpp_data" / "mbpp_test_eval_half.csv"'

    if old_train in content:
        content = content.replace(old_train, new_train)
        content = content.replace(old_eval, new_eval)
        p3_path.write_text(content)
        print("  Updated CSV paths")

    # Baseline
    print("\n" + "="*60)
    print("BASELINE EVALUATION")
    print("="*60)

    baseline_eval = run_eval(None, "eval")
    baseline_train = run_eval(None, "train")

    all_results["baseline"] = {
        "mbpp_eval": baseline_eval,
        "mbpp_train": baseline_train,
    }
    print(f"\nBaseline: eval={baseline_eval}%, train={baseline_train}%")

    # Save intermediate results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Each experiment
    for exp_name in EXPERIMENTS:
        exp_dir = CHECKPOINT_DIR / exp_name
        if not exp_dir.exists():
            print(f"\nSkipping {exp_name} - not found")
            continue

        all_results[exp_name] = {}

        for epoch in EPOCHS:
            ckpt_path = find_epoch_checkpoint(exp_dir, epoch)
            if ckpt_path is None:
                print(f"\nSkipping {exp_name} epoch {epoch} - checkpoint not found")
                continue

            print("\n" + "="*60)
            print(f"{exp_name} EPOCH {epoch}")
            print("="*60)

            eval_acc = run_eval(ckpt_path, "eval")
            train_acc = run_eval(ckpt_path, "train")

            all_results[exp_name][f"epoch_{epoch}"] = {
                "mbpp_eval": eval_acc,
                "mbpp_train": train_acc,
            }
            print(f"\n{exp_name} ep{epoch}: eval={eval_acc}%, train={train_acc}%")

            # Save after each eval
            with open(RESULTS_FILE, 'w') as f:
                json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("ALL EVALUATIONS COMPLETE!")
    print(f"Results saved to: {RESULTS_FILE}")
    print("="*60)
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
