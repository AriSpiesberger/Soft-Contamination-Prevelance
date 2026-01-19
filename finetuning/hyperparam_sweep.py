"""
Bayesian hyperparameter optimization for MBPP KL-regularized finetuning.

Uses Optuna for efficient exploration of:
- LoRA rank (r): 4-64
- KL beta: 0.01-1.0 (log scale)
- Epochs: 1-5
- Learning rate: 1e-5 to 5e-4 (log scale)

Optimizes for combined train + eval accuracy gain over baseline.

Run with: python hyperparam_sweep.py --n-trials 20
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

try:
    import optuna
except ImportError:
    print("Optuna not installed. Run: pip install optuna")
    sys.exit(1)

pwd = Path(__file__).parent
RESULTS_FILE = pwd / "outputs" / "hyperparam_sweep_results.jsonl"
STUDY_DB = f"sqlite:///{pwd / 'outputs' / 'optuna_study.db'}"

# Baseline pass@1 scores (from unfinetuned model)
# Update these after running baseline evaluation
BASELINE_TRAIN_PASS1 = 45.0  # baseline on train split
BASELINE_EVAL_PASS1 = 45.0   # baseline on eval split


def run_eval(run_id: str, split: str) -> Optional[float]:
    """Run evaluation on a specific split. Returns pass@1 or None on failure."""
    eval_cmd = [
        sys.executable, "p3_eval_mbpp.py",
        "--finetuned",
        "--wandb-id", run_id,
        "--test-split", split,
        "--no-wandb",
    ]

    try:
        eval_result = subprocess.run(
            eval_cmd,
            cwd=pwd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )

        if eval_result.returncode != 0:
            print(f"Eval ({split}) failed: {eval_result.stderr[-500:]}")
            return None

        # Extract pass@1 from output
        for line in eval_result.stdout.split("\n"):
            if "pass@1:" in line and "%" in line:
                try:
                    return float(line.split("%")[0].split(":")[-1].strip())
                except:
                    pass

        print(f"Could not parse {split} pass@1")
        return None

    except subprocess.TimeoutExpired:
        print(f"Eval ({split}) timed out")
        return None


def run_experiment(lora_r: int, kl_beta: float, epochs: int, learning_rate: float) -> Tuple[float, float, float]:
    """Run a single training + eval experiment.
    Returns (train_pass1, eval_pass1, combined_gain) or (-1, -1, -1) on failure."""

    print(f"\n{'='*60}")
    print(f"Trial: lora_r={lora_r}, kl_beta={kl_beta:.4f}, epochs={epochs}, lr={learning_rate:.2e}")
    print(f"{'='*60}\n")

    # Training command
    train_cmd = [
        sys.executable, "p2_train_mbpp_kl.py",
        "-e", str(epochs),
        "-k", str(kl_beta),
        "-r", str(lora_r),
        "--no-wandb",
    ]

    try:
        train_result = subprocess.run(
            train_cmd,
            cwd=pwd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )

        if train_result.returncode != 0:
            print(f"Training failed: {train_result.stderr[-500:]}")
            return -1.0, -1.0, -1.0

        # Extract run_id from output
        run_id = None
        for line in train_result.stdout.split("\n"):
            if "Wandb run id:" in line:
                run_id = line.split(":")[-1].strip()
                break

        if not run_id:
            print("Could not find run_id in output")
            return -1.0, -1.0, -1.0

        print(f"Training complete. Run ID: {run_id}")

    except subprocess.TimeoutExpired:
        print("Training timed out")
        return -1.0, -1.0, -1.0

    # Evaluate on both train and eval splits
    train_pass1 = run_eval(run_id, "train")
    eval_pass1 = run_eval(run_id, "eval")

    if train_pass1 is None or eval_pass1 is None:
        return -1.0, -1.0, -1.0

    # Calculate gains over baseline
    train_gain = train_pass1 - BASELINE_TRAIN_PASS1
    eval_gain = eval_pass1 - BASELINE_EVAL_PASS1
    combined_gain = train_gain + eval_gain

    print(f"\nResults:")
    print(f"  Train pass@1: {train_pass1:.2f}% (gain: {train_gain:+.2f}%)")
    print(f"  Eval pass@1:  {eval_pass1:.2f}% (gain: {eval_gain:+.2f}%)")
    print(f"  Combined gain: {combined_gain:+.2f}%")

    # Log result
    result = {
        "lora_r": lora_r,
        "kl_beta": kl_beta,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "run_id": run_id,
        "train_pass1": train_pass1,
        "eval_pass1": eval_pass1,
        "train_gain": train_gain,
        "eval_gain": eval_gain,
        "combined_gain": combined_gain,
        "timestamp": datetime.now().isoformat(),
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")

    return train_pass1, eval_pass1, combined_gain


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function. Maximizes combined train + eval gain."""

    # Hyperparameter search space
    lora_r = trial.suggest_int("lora_r", 4, 64, log=True)  # 4, 8, 16, 32, 64
    kl_beta = trial.suggest_float("kl_beta", 0.01, 1.0, log=True)
    epochs = trial.suggest_int("epochs", 1, 4)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)

    train_pass1, eval_pass1, combined_gain = run_experiment(lora_r, kl_beta, epochs, learning_rate)

    # Store individual metrics for analysis
    trial.set_user_attr("train_pass1", train_pass1)
    trial.set_user_attr("eval_pass1", eval_pass1)

    if combined_gain < -100:  # Failed trial
        return -100.0

    return combined_gain


def main():
    parser = argparse.ArgumentParser(description="Bayesian hyperparameter optimization for MBPP finetuning")
    parser.add_argument("-n", "--n-trials", type=int, default=20, help="Number of trials to run")
    parser.add_argument("--resume", action="store_true", help="Resume previous study")
    parser.add_argument("--study-name", type=str, default="mbpp-kl-finetune-gain", help="Optuna study name")
    args = parser.parse_args()

    # Create output directory
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=STUDY_DB,
        direction="maximize",  # Maximize combined train + eval gain
        load_if_exists=args.resume,
        sampler=optuna.samplers.TPESampler(seed=42),  # Tree-structured Parzen Estimator
    )

    print(f"Starting Bayesian optimization with {args.n_trials} trials...")
    print(f"Optimizing: combined train + eval accuracy gain")
    print(f"Baselines: train={BASELINE_TRAIN_PASS1}%, eval={BASELINE_EVAL_PASS1}%")
    print(f"Study DB: {STUDY_DB}")
    print(f"Results file: {RESULTS_FILE}")

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)

    best = study.best_trial
    print(f"\nBest trial:")
    print(f"  Combined gain: {best.value:+.2f}%")
    print(f"  Train pass@1: {best.user_attrs.get('train_pass1', 'N/A')}")
    print(f"  Eval pass@1: {best.user_attrs.get('eval_pass1', 'N/A')}")
    print(f"  Params: {best.params}")

    print(f"\nTop 5 trials:")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        trials_df = trials_df.sort_values("value", ascending=False).head(5)
        cols = ["number", "value", "params_lora_r", "params_kl_beta", "params_epochs", "params_learning_rate"]
        if "user_attrs_train_pass1" in trials_df.columns:
            cols.extend(["user_attrs_train_pass1", "user_attrs_eval_pass1"])
        print(trials_df[[c for c in cols if c in trials_df.columns]].to_string())

    print(f"\nFull results saved to: {RESULTS_FILE}")
    print(f"Study database: {STUDY_DB}")
    print(f"\nTo visualize: optuna-dashboard {STUDY_DB}")


if __name__ == "__main__":
    main()
