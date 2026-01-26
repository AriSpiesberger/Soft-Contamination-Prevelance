"""
Run semantic vs exact duplicate experiments.

Experiments:
1. Semantic duplicates (~5 paraphrases per task) - train to 10 epochs, eval at 3, 6, 10
2. Exact duplicates (5 identical copies per task) - train to 10 epochs, eval at 3, 6, 10

Standard SFT (no KL regularization).
Trains ONCE per experiment, evaluates checkpoints at specified epochs.
"""
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

pwd = Path(__file__).parent

# Default hyperparams
DEFAULT_PARAMS = {
    "lora_r": 32,
    "learning_rate": 2e-4,
}

# Aligned dataset paths
SEMANTIC_ALIGNED_CSV = pwd / "mbpp_data" / "mbpp_train_semantic_fixed.csv"
EXACT_ALIGNED_CSV = pwd / "mbpp_data" / "mbpp_train_exact_5x_aligned.csv"

# Output
RESULTS_FILE = pwd / "outputs" / "duplicate_experiment_results.jsonl"

# Train to max epochs, eval at these checkpoints
MAX_EPOCHS = 10
EVAL_AT_EPOCHS = [3, 6, 10]


def run_train(csv_path: str, max_epochs: int, experiment_name: str, params: dict) -> str:
    """Run training to max_epochs and return the run_id."""
    print(f"\n{'='*60}")
    print(f"Training: {experiment_name} for {max_epochs} epochs")
    print(f"Data: {csv_path}")
    print(f"Params: lora_r={params['lora_r']}, lr={params['learning_rate']:.2e}")
    print(f"Will save checkpoints every epoch")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "-u", "p2_train_mbpp_sft.py",  # -u for unbuffered output
        "-i", str(csv_path),
        "-e", str(max_epochs),
        "-r", str(params["lora_r"]),
        "-l", str(params["learning_rate"]),
        "--no-wandb",
    ]

    # Stream output so user can see training progress
    process = subprocess.Popen(
        cmd, cwd=pwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )

    run_id = None
    output_lines = []
    for line in process.stdout:
        print(line, end='')  # Stream to console
        output_lines.append(line)
        if "Run id:" in line:
            run_id = line.split(":")[-1].strip()

    process.wait()

    if process.returncode != 0:
        print(f"Training failed with return code {process.returncode}")
        return None

    if not run_id:
        print("Could not find run_id in output")
        # Try to find it in last lines
        for line in output_lines[-20:]:
            if "Run id:" in line:
                run_id = line.split(":")[-1].strip()
                break

    return run_id


def find_checkpoint_path(run_id: str, epoch: int, max_epochs: int) -> Path:
    """Find the checkpoint directory for a specific epoch."""
    base_dir = pwd / f"outputs/checkpoints/olmo3-mbpp-sft-{run_id}"

    if not base_dir.exists():
        return None

    # List all checkpoints sorted by step number
    checkpoints = sorted(base_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))

    if not checkpoints:
        # Final model might be saved directly in base_dir
        if (base_dir / "adapter_config.json").exists():
            return base_dir
        return None

    # With epoch-based saving, checkpoint count should match epoch
    if epoch <= len(checkpoints):
        return checkpoints[epoch - 1]

    # For the final epoch, also check base_dir
    if epoch == max_epochs and (base_dir / "adapter_config.json").exists():
        return base_dir

    return None


def run_eval(checkpoint_path: Path, split: str, is_baseline: bool = False) -> float:
    """Run evaluation on a checkpoint and return pass@1 percentage."""
    if is_baseline:
        cmd = [
            sys.executable, "p3_eval_mbpp.py",
            "--test-split", split,
            "--no-wandb",
        ]
    else:
        cmd = [
            sys.executable, "p3_eval_mbpp.py",
            "--finetuned",
            "--finetuned-path", str(checkpoint_path),
            "--test-split", split,
            "--no-wandb",
        ]

    result = subprocess.run(cmd, cwd=pwd, capture_output=True, text=True, timeout=7200)

    if result.returncode != 0:
        print(f"    Eval failed: {result.stderr[-300:]}")
        return None

    for line in result.stdout.split("\n"):
        if "pass@1:" in line and "%" in line:
            try:
                return float(line.split("%")[0].split(":")[-1].strip())
            except:
                pass

    return None


def run_baseline_eval() -> dict:
    """Evaluate the unfinetuned base model."""
    print(f"\n{'#'*60}")
    print(f"# Evaluating BASELINE (unfinetuned model)")
    print(f"{'#'*60}")

    baseline = {}
    for split in tqdm(["train", "eval"], desc="Baseline eval"):
        print(f"  Evaluating baseline on {split} split...")
        result = run_eval(None, split, is_baseline=True)
        baseline[split] = result
        if result is not None:
            print(f"    Baseline {split}: {result:.2f}%")
        else:
            print(f"    Baseline {split}: FAILED")

    return baseline


def run_experiment(name: str, csv_path: str, params: dict, eval_epochs: list, max_epochs: int) -> list:
    """Run a single experiment: train once, eval at multiple epochs."""
    print(f"\n{'#'*60}")
    print(f"# Experiment: {name}")
    print(f"# Train to {max_epochs} epochs, eval at epochs {eval_epochs}")
    print(f"{'#'*60}")

    # Train once
    run_id = run_train(csv_path, max_epochs, name, params)
    if not run_id:
        return [{"experiment": name, "error": "Training failed"}]

    results = []

    # Evaluate at each checkpoint
    print(f"\nEvaluating checkpoints...")
    for epoch in tqdm(eval_epochs, desc=f"Eval {name}"):
        checkpoint_path = find_checkpoint_path(run_id, epoch, max_epochs)

        if checkpoint_path is None:
            print(f"  Epoch {epoch}: checkpoint not found")
            results.append({
                "experiment": name,
                "epochs": epoch,
                "error": "Checkpoint not found",
                "run_id": run_id,
            })
            continue

        print(f"  Epoch {epoch}: {checkpoint_path.name}")
        train_pass1 = run_eval(checkpoint_path, "train")
        eval_pass1 = run_eval(checkpoint_path, "eval")

        result = {
            "experiment": name,
            "epochs": epoch,
            "csv_path": str(csv_path),
            "run_id": run_id,
            "checkpoint_path": str(checkpoint_path),
            "train_pass1": train_pass1,
            "eval_pass1": eval_pass1,
            "params": params,
            "timestamp": datetime.now().isoformat(),
        }
        results.append(result)

        # Log incrementally
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(result) + "\n")

        if train_pass1 and eval_pass1:
            print(f"    Train: {train_pass1:.2f}%, Eval: {eval_pass1:.2f}%")
        else:
            print(f"    ERROR")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-only", action="store_true")
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument("--eval-epochs", type=int, nargs="+", default=EVAL_AT_EPOCHS)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_PARAMS["lora_r"])
    parser.add_argument("--lr", type=float, default=DEFAULT_PARAMS["learning_rate"])
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline evaluation")
    args = parser.parse_args()

    max_epochs = args.max_epochs

    params = {
        "lora_r": args.lora_r,
        "learning_rate": args.lr,
    }

    # Prepare data if needed
    if args.prepare_data or not SEMANTIC_ALIGNED_CSV.exists() or not EXACT_ALIGNED_CSV.exists():
        print("Preparing aligned training data...")
        subprocess.run([sys.executable, "prepare_aligned_data.py"], cwd=pwd, check=True)

    # Run baseline evaluation first
    if args.skip_baseline:
        baseline = {"train": None, "eval": None}
        print("Skipping baseline evaluation")
    else:
        baseline = run_baseline_eval()

    all_results = []

    # Semantic duplicates
    if not args.exact_only:
        results = run_experiment("semantic_duplicates", SEMANTIC_ALIGNED_CSV, params, args.eval_epochs, max_epochs)
        all_results.extend(results)

    # Exact duplicates
    if not args.semantic_only:
        results = run_experiment("exact_duplicates_5x", EXACT_ALIGNED_CSV, params, args.eval_epochs, max_epochs)
        all_results.extend(results)

    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"{'Experiment':<25} {'Epoch':>8} {'Train%':>12} {'Eval%':>12}")
    print("-"*70)

    # Print baseline
    base_train = f"{baseline['train']:.2f}" if baseline.get('train') else "ERR"
    base_eval = f"{baseline['eval']:.2f}" if baseline.get('eval') else "ERR"
    print(f"{'BASELINE (no finetune)':<25} {'-':>8} {base_train:>12} {base_eval:>12}")
    print("-"*70)

    for r in all_results:
        if "error" not in r:
            train_str = f"{r['train_pass1']:.2f}" if r.get('train_pass1') else "ERR"
            eval_str = f"{r['eval_pass1']:.2f}" if r.get('eval_pass1') else "ERR"
            print(f"{r['experiment']:<25} {r['epochs']:>8} {train_str:>12} {eval_str:>12}")
        else:
            print(f"{r.get('experiment', '?'):<25} {r.get('epochs', '?'):>8} {'ERROR':>12} {'ERROR':>12}")
    print("="*70)
    print(f"Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
