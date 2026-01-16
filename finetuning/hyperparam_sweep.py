"""
Bayesian hyperparameter optimization for MBPP KL-regularized finetuning.

Uses Optuna for efficient exploration of:
- LoRA rank (r): 4-64
- KL beta: 0.01-1.0 (log scale)
- Epochs: 1-5
- Learning rate: 1e-5 to 5e-4 (log scale)

Run with: python hyperparam_sweep.py --n-trials 20
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import optuna
except ImportError:
    print("Optuna not installed. Run: pip install optuna")
    sys.exit(1)

pwd = Path(__file__).parent
RESULTS_FILE = pwd / "outputs" / "hyperparam_sweep_results.jsonl"
STUDY_DB = f"sqlite:///{pwd / 'outputs' / 'optuna_study.db'}"


def run_experiment(lora_r: int, kl_beta: float, epochs: int, learning_rate: float) -> float:
    """Run a single training + eval experiment. Returns eval pass@1 or -1 on failure."""
    
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
            return -1.0
        
        # Extract run_id from output
        run_id = None
        for line in train_result.stdout.split("\n"):
            if "Wandb run id:" in line:
                run_id = line.split(":")[-1].strip()
                break
        
        if not run_id:
            print("Could not find run_id in output")
            return -1.0
        
        print(f"Training complete. Run ID: {run_id}")
        
    except subprocess.TimeoutExpired:
        print("Training timed out")
        return -1.0
    
    # Evaluation on eval split
    eval_cmd = [
        sys.executable, "p3_eval_mbpp.py",
        "--finetuned",
        "--wandb-id", run_id,
        "--test-split", "eval",
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
            print(f"Eval failed: {eval_result.stderr[-500:]}")
            return -1.0
        
        # Extract pass@1 from output
        eval_pass1 = None
        for line in eval_result.stdout.split("\n"):
            if "pass@1:" in line and "%" in line:
                try:
                    pct = float(line.split("%")[0].split(":")[-1].strip())
                    eval_pass1 = pct
                except:
                    pass
        
        if eval_pass1 is None:
            print("Could not parse eval pass@1")
            return -1.0
        
        print(f"Eval complete. pass@1: {eval_pass1}%")
        
        # Log result
        result = {
            "lora_r": lora_r,
            "kl_beta": kl_beta,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "run_id": run_id,
            "eval_pass1": eval_pass1,
            "timestamp": datetime.now().isoformat(),
        }
        
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(result) + "\n")
        
        return eval_pass1
        
    except subprocess.TimeoutExpired:
        print("Eval timed out")
        return -1.0


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function."""
    
    # Hyperparameter search space
    lora_r = trial.suggest_int("lora_r", 4, 64, log=True)  # 4, 8, 16, 32, 64
    kl_beta = trial.suggest_float("kl_beta", 0.01, 1.0, log=True)
    epochs = trial.suggest_int("epochs", 1, 4)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    
    eval_pass1 = run_experiment(lora_r, kl_beta, epochs, learning_rate)
    
    if eval_pass1 < 0:
        # Return a bad value for failed trials (but don't crash)
        return 0.0
    
    return eval_pass1


def main():
    parser = argparse.ArgumentParser(description="Bayesian hyperparameter optimization for MBPP finetuning")
    parser.add_argument("-n", "--n-trials", type=int, default=20, help="Number of trials to run")
    parser.add_argument("--resume", action="store_true", help="Resume previous study")
    parser.add_argument("--study-name", type=str, default="mbpp-kl-finetune", help="Optuna study name")
    args = parser.parse_args()
    
    # Create output directory
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=STUDY_DB,
        direction="maximize",  # We want to maximize eval pass@1
        load_if_exists=args.resume,
        sampler=optuna.samplers.TPESampler(seed=42),  # Tree-structured Parzen Estimator
    )
    
    print(f"Starting Bayesian optimization with {args.n_trials} trials...")
    print(f"Study DB: {STUDY_DB}")
    print(f"Results file: {RESULTS_FILE}")
    
    # Add baseline result if we have it (helps guide the search)
    # Baseline: lora_r=16, kl_beta=0.1, epochs=1 -> 44.08%
    
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    
    print(f"\nBest trial:")
    print(f"  Value (eval pass@1): {study.best_trial.value:.2f}%")
    print(f"  Params: {study.best_trial.params}")
    
    print(f"\nTop 5 trials:")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        trials_df = trials_df.sort_values("value", ascending=False).head(5)
        print(trials_df[["number", "value", "params_lora_r", "params_kl_beta", "params_epochs", "params_learning_rate"]].to_string())
    
    print(f"\nFull results saved to: {RESULTS_FILE}")
    print(f"Study database: {STUDY_DB}")
    print(f"\nTo visualize: optuna-dashboard {STUDY_DB}")


if __name__ == "__main__":
    main()
