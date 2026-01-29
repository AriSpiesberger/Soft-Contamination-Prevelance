#!/usr/bin/env python3
"""
Fast parallel evaluation on 8 GPUs.
Runs multiple checkpoints in parallel, one per GPU.
"""

import os
import subprocess
import sys
import json
import concurrent.futures
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


def run_single_eval(args):
    """Run a single evaluation on a specific GPU."""
    name, ckpt_path, split, gpu_id = args

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [sys.executable, "p3_eval_mbpp_fast.py", "--test-split", split, "--batch-size", "8"]
    if ckpt_path:
        cmd.extend(["--finetuned-path", ckpt_path])

    print(f"[GPU {gpu_id}] Starting {name} {split}...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env, cwd=str(PWD))

        if result.returncode != 0:
            print(f"[GPU {gpu_id}] {name} {split} FAILED: {result.stderr[-300:]}")
            return (name, split, None)

        # Parse pass@1
        for line in result.stdout.split("\n"):
            if "pass@1:" in line:
                try:
                    pct = float(line.split("pass@1:")[1].split("%")[0].strip())
                    print(f"[GPU {gpu_id}] {name} {split}: {pct}%")
                    return (name, split, pct)
                except:
                    pass

        print(f"[GPU {gpu_id}] {name} {split}: parse error\n{result.stdout[-300:]}")
        return (name, split, None)

    except subprocess.TimeoutExpired:
        print(f"[GPU {gpu_id}] {name} {split} TIMEOUT")
        return (name, split, None)


def main():
    # Build list of all evals to run
    eval_jobs = []

    # Baseline on GPU 0 and 1
    eval_jobs.append(("baseline", None, "eval", 0))
    eval_jobs.append(("baseline", None, "train", 1))

    # All checkpoints distributed across GPUs
    gpu_idx = 2
    for exp_name in EXPERIMENTS:
        exp_dir = CHECKPOINT_DIR / exp_name
        if not exp_dir.exists():
            print(f"Skipping {exp_name} - not found")
            continue

        for epoch in EPOCHS:
            ckpt_path = find_epoch_checkpoint(exp_dir, epoch)
            if ckpt_path:
                eval_jobs.append((f"{exp_name}_ep{epoch}", ckpt_path, "eval", gpu_idx % 8))
                gpu_idx += 1
                eval_jobs.append((f"{exp_name}_ep{epoch}", ckpt_path, "train", gpu_idx % 8))
                gpu_idx += 1

    print(f"\nRunning {len(eval_jobs)} evals across 8 GPUs in parallel...")

    # Run in parallel (8 at a time)
    all_results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_single_eval, job) for job in eval_jobs]

        for future in concurrent.futures.as_completed(futures):
            name, split, pct = future.result()

            # Parse name to store results
            if name == "baseline":
                if "baseline" not in all_results:
                    all_results["baseline"] = {}
                all_results["baseline"][f"mbpp_{split}"] = pct
            else:
                # e.g. "sem_dupes_ep3"
                parts = name.rsplit("_ep", 1)
                exp_name = parts[0]
                epoch = int(parts[1])

                if exp_name not in all_results:
                    all_results[exp_name] = {}
                if f"epoch_{epoch}" not in all_results[exp_name]:
                    all_results[exp_name][f"epoch_{epoch}"] = {}
                all_results[exp_name][f"epoch_{epoch}"][f"mbpp_{split}"] = pct

            # Save intermediate results
            RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(RESULTS_FILE, 'w') as f:
                json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
