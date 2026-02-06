"""
Run benchmark evaluations on all checkpoints of trained models.
Evaluates: ARC-Challenge, HellaSwag, GSM8K on all checkpoints
           True Detective (degradation test) on final models only
Uses multiple GPUs in parallel.

Usage:
    python run_benchmark_evals.py                    # Run all benchmarks
    python run_benchmark_evals.py --skip-degradation # Skip True Detective
    python run_benchmark_evals.py --only-degradation # Only run True Detective on finals
"""

import json
import os
import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

OUTPUT_DIR = Path(__file__).parent / "outputs"
EVAL_DIR = OUTPUT_DIR / "benchmark_evals"
SCRIPT_DIR = Path(__file__).parent

# Model configurations
# TODO: Update checkpoint timestamps below after retraining
MODEL_CONFIGS = {
    "olmo_contaminated": {
        "base_model": "allenai/Olmo-3-1025-7B",
        "model_dir": OUTPUT_DIR / "exp_contaminated_20260123_061624",  # update timestamp after retraining
    },
    "olmo_clean": {
        "base_model": "allenai/Olmo-3-1025-7B",
        "model_dir": OUTPUT_DIR / "exp_clean_20260123_061624",  # update timestamp after retraining
    },
    "qwen_contaminated": {
        "base_model": "Qwen/Qwen3-8B-Base",
        "model_dir": OUTPUT_DIR / "qwen_contaminated_20260127_194900",  # update timestamp after retraining
    },
    "qwen_clean": {
        "base_model": "Qwen/Qwen3-8B-Base",
        "model_dir": OUTPUT_DIR / "qwen_clean_20260127_194900",  # update timestamp after retraining
    },
}

BENCHMARKS = ["arc_challenge", "hellaswag", "gsm8k"]

# Degradation benchmarks only run on final models
DEGRADATION_BENCHMARKS = ["true_detective"]


def run_true_detective_eval(model_name, gpu_id, output_dir):
    """Run True Detective degradation test on a single model's final checkpoint."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Determine which model family
    if "olmo" in model_name:
        model_arg = "olmo"
    elif "qwen" in model_name:
        model_arg = "qwen"
    else:
        model_arg = "all"

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "eval_true_detective.py"),
        "--model", model_arg,
    ]

    print(f"[GPU {gpu_id}] True Detective: {model_name}")

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=14400)
        if result.returncode == 0:
            return model_name, "final", "true_detective", "success"
        else:
            print(f"[GPU {gpu_id}] True Detective Error: {result.stderr[:500]}")
            return model_name, "final", "true_detective", f"error: {result.stderr[:200]}"
    except Exception as e:
        return model_name, "final", "true_detective", f"exception: {str(e)}"


def get_checkpoints(model_dir):
    """Get all checkpoints including final."""
    checkpoints = []
    if model_dir.exists():
        for ckpt in sorted(model_dir.glob("checkpoint-*")):
            checkpoints.append(ckpt)
        final = model_dir / "final"
        if final.exists():
            checkpoints.append(final)
    return checkpoints


def run_single_eval(args):
    """Run a single evaluation (called in parallel)."""
    model_name, base_model, adapter_path, benchmark, gpu_id, output_dir = args

    ckpt_name = adapter_path.name
    result_file = output_dir / f"{model_name}_{ckpt_name}_{benchmark}.json"

    if result_file.exists():
        print(f"Skipping (exists): {result_file.name}")
        return model_name, ckpt_name, benchmark, str(result_file)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["HF_ALLOW_CODE_EVAL"] = "1"  # Enable code execution for humaneval/mbpp

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={base_model},peft={adapter_path},trust_remote_code=True,dtype=bfloat16",
        "--tasks", benchmark,
        "--batch_size", "4",
        "--output_path", str(output_dir),
        "--num_fewshot", "0",
    ]

    print(f"[GPU {gpu_id}] {model_name}/{ckpt_name}/{benchmark}")

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=14400)
        if result.returncode == 0:
            return model_name, ckpt_name, benchmark, "success"
        else:
            print(f"[GPU {gpu_id}] Error: {result.stderr[:500]}")
            return model_name, ckpt_name, benchmark, f"error: {result.stderr[:200]}"
    except Exception as e:
        return model_name, ckpt_name, benchmark, f"exception: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluations")
    parser.add_argument("--skip-degradation", action="store_true",
                        help="Skip True Detective degradation test")
    parser.add_argument("--only-degradation", action="store_true",
                        help="Only run True Detective on final models")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = EVAL_DIR / timestamp
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {eval_dir}")

    num_gpus = 8
    results = []

    # Run checkpoint benchmarks (unless only-degradation)
    if not args.only_degradation:
        # Build list of all evaluation jobs
        jobs = []
        for model_name, config in MODEL_CONFIGS.items():
            base_model = config["base_model"]
            model_dir = config["model_dir"]

            checkpoints = get_checkpoints(model_dir)
            if not checkpoints:
                print(f"No checkpoints found for {model_name} in {model_dir}")
                continue

            print(f"{model_name}: {len(checkpoints)} checkpoints")

            for ckpt in checkpoints:
                for benchmark in BENCHMARKS:
                    jobs.append((model_name, base_model, ckpt, benchmark, None, eval_dir))

        print(f"\nTotal checkpoint jobs: {len(jobs)}")
        print(f"Using {num_gpus} GPUs in parallel\n")

        # Assign GPUs round-robin and run in parallel
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = {}
            for i, job in enumerate(jobs):
                gpu_id = i % num_gpus
                job_with_gpu = (*job[:4], gpu_id, job[5])
                future = executor.submit(run_single_eval, job_with_gpu)
                futures[future] = job_with_gpu

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"Completed: {result[0]}/{result[1]}/{result[2]} -> {result[3]}")

    # Run True Detective degradation test on final models only
    if not args.skip_degradation:
        print(f"\n{'='*60}")
        print("RUNNING TRUE DETECTIVE DEGRADATION TEST (final models only)")
        print(f"{'='*60}\n")

        # Run once per model family (olmo and qwen) - the script handles both variants
        degradation_jobs = []
        seen_families = set()
        for model_name in MODEL_CONFIGS.keys():
            family = "olmo" if "olmo" in model_name else "qwen"
            if family not in seen_families:
                seen_families.add(family)
                degradation_jobs.append((model_name, 0, eval_dir))

        for job in degradation_jobs:
            model_name, gpu_id, out_dir = job
            result = run_true_detective_eval(model_name, gpu_id, out_dir)
            results.append(result)
            print(f"Completed: {result[0]}/{result[1]}/{result[2]} -> {result[3]}")

    # Aggregate results from lm_eval output files
    print(f"\n{'='*60}")
    print("AGGREGATING RESULTS")
    print(f"{'='*60}\n")

    aggregated = {}
    for result_file in eval_dir.glob("**/*results.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)

            # Extract model/checkpoint info from path
            if "results" in data:
                for task, metrics in data["results"].items():
                    key = result_file.parent.name if result_file.parent != eval_dir else result_file.stem
                    if key not in aggregated:
                        aggregated[key] = {}
                    aggregated[key][task] = metrics
        except Exception as e:
            print(f"Error reading {result_file}: {e}")

    # Save aggregated summary
    summary = {
        "timestamp": timestamp,
        "job_results": results,
        "metrics": aggregated,
    }
    with open(eval_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")

    for model_key, tasks in sorted(aggregated.items()):
        print(f"\n{model_key}:")
        for task, metrics in tasks.items():
            # Get the main metric (usually acc or acc_norm)
            acc = metrics.get("acc,none", metrics.get("acc_norm,none", metrics.get("exact_match,none", "N/A")))
            if isinstance(acc, float):
                acc = f"{acc*100:.1f}%"
            print(f"  {task}: {acc}")

    print(f"\n{'='*60}")
    print(f"All evaluations complete!")
    print(f"Full results saved to: {eval_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
