#!/usr/bin/env python3
"""
Distributed evaluation runner for wandb runs.

Queries wandb project for runs that need evaluation (missing eval/finetuned/arc_easy/acc,none)
and distributes evaluation jobs across multiple GPUs.

Usage:
    python distributed_eval_runner.py                    # Run with defaults (GPUs 1-7)
    python distributed_eval_runner.py --gpus 1,2,3,4     # Specify GPUs
    python distributed_eval_runner.py --dry-run          # Just print what would run
    python distributed_eval_runner.py --max-runs 5       # Limit number of runs to process
"""

import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import wandb


@dataclass
class RunInfo:
    """Information about a wandb run."""
    id: str
    name: str
    state: str
    created_at: str
    has_eval_metric: bool


def get_runs_needing_eval(
    project: str,
    entity: Optional[str] = None,
    eval_metric: str = "eval/finetuned/arc_easy/acc,none",
) -> list[RunInfo]:
    """
    Query wandb for runs that need evaluation.
    
    Returns runs sorted newest to oldest that:
    - Have state == "finished"
    - Do NOT have the eval_metric logged
    """
    api = wandb.Api()
    
    # Get all runs from project, sorted by created_at descending (newest first)
    project_path = f"{entity}/{project}" if entity else project
    runs = api.runs(
        project_path,
        order="-created_at",  # Newest first
    )
    
    runs_needing_eval = []
    runs_skipped_not_finished = []
    runs_skipped_has_eval = []
    
    for run in runs:
        # Check run state
        if run.state != "finished":
            runs_skipped_not_finished.append(run.name or run.id)
            continue
        
        # Check if eval metric exists in summary
        has_eval = eval_metric in run.summary
        
        if has_eval:
            runs_skipped_has_eval.append(run.name or run.id)
            continue
        
        runs_needing_eval.append(RunInfo(
            id=run.id,
            name=run.name or run.id,
            state=run.state,
            created_at=str(run.created_at),
            has_eval_metric=has_eval,
        ))
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Wandb Project: {project_path}")
    print(f"{'='*60}")
    print(f"Total runs found: {len(runs_needing_eval) + len(runs_skipped_not_finished) + len(runs_skipped_has_eval)}")
    print(f"  - Skipped (not finished): {len(runs_skipped_not_finished)}")
    print(f"  - Skipped (already has eval): {len(runs_skipped_has_eval)}")
    print(f"  - Need evaluation: {len(runs_needing_eval)}")
    print(f"{'='*60}\n")
    
    if runs_skipped_not_finished:
        print(f"Runs skipped (not finished): {runs_skipped_not_finished[:10]}{'...' if len(runs_skipped_not_finished) > 10 else ''}")
    if runs_skipped_has_eval:
        print(f"Runs skipped (has eval): {runs_skipped_has_eval[:10]}{'...' if len(runs_skipped_has_eval) > 10 else ''}")
    
    return runs_needing_eval


def run_eval_job(
    wandb_id: str,
    wandb_project: str,
    gpu_id: int,
    script_path: str = "p3_1_eval_baseline.py",
    dry_run: bool = False,
) -> tuple[str, int, str]:
    """
    Run evaluation job for a single wandb run on specified GPU.
    
    Returns: (wandb_id, return_code, output)
    """
    cmd = [
        "python", script_path,
        "--finetuned-only",
        "--wandb-project", wandb_project,
        "--wandb-id", wandb_id,
    ]
    
    env_cmd = f"CUDA_VISIBLE_DEVICES={gpu_id}"
    full_cmd = f"{env_cmd} {' '.join(cmd)}"
    
    if dry_run:
        print(f"[DRY RUN] Would run: {full_cmd}")
        return (wandb_id, 0, "dry run")
    
    print(f"[GPU {gpu_id}] Starting evaluation for run: {wandb_id}")
    
    try:
        # Run the command with the GPU environment variable
        import os
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,  # Run from script directory
        )
        
        if result.returncode == 0:
            print(f"[GPU {gpu_id}] ✓ Completed: {wandb_id}")
        else:
            print(f"[GPU {gpu_id}] ✗ Failed: {wandb_id}")
            print(f"  stderr: {result.stderr[:500] if result.stderr else 'None'}")
        
        return (wandb_id, result.returncode, result.stdout + result.stderr)
    
    except Exception as e:
        print(f"[GPU {gpu_id}] ✗ Error running {wandb_id}: {e}")
        return (wandb_id, -1, str(e))


def distribute_jobs(
    runs: list[RunInfo],
    gpus: list[int],
    wandb_project: str,
    dry_run: bool = False,
    max_runs: Optional[int] = None,
) -> dict[str, tuple[int, str]]:
    """
    Distribute evaluation jobs across GPUs.
    
    Uses ThreadPoolExecutor to run one job per GPU in parallel.
    """
    if max_runs:
        runs = runs[:max_runs]
    
    if not runs:
        print("No runs to process!")
        return {}
    
    print(f"\nDistributing {len(runs)} runs across GPUs: {gpus}")
    print("-" * 60)
    
    results = {}
    
    # Use ThreadPoolExecutor with one worker per GPU
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        # Track which GPU is assigned to which future
        gpu_queue = list(gpus)
        futures = {}
        run_idx = 0
        
        while run_idx < len(runs) or futures:
            # Submit new jobs if we have available GPUs and remaining runs
            while gpu_queue and run_idx < len(runs):
                gpu_id = gpu_queue.pop(0)
                run = runs[run_idx]
                
                future = executor.submit(
                    run_eval_job,
                    wandb_id=run.id,
                    wandb_project=wandb_project,
                    gpu_id=gpu_id,
                    dry_run=dry_run,
                )
                futures[future] = (run.id, gpu_id)
                run_idx += 1
            
            if not futures:
                break
            
            # Wait for at least one job to complete
            done_futures = []
            for future in as_completed(futures):
                wandb_id, gpu_id = futures[future]
                try:
                    result = future.result()
                    results[result[0]] = (result[1], result[2])
                except Exception as e:
                    print(f"Job for {wandb_id} raised exception: {e}")
                    results[wandb_id] = (-1, str(e))
                
                # Return GPU to queue
                gpu_queue.append(gpu_id)
                done_futures.append(future)
            
            # Remove completed futures
            for f in done_futures:
                del futures[f]
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Distribute evaluation jobs across GPUs for wandb runs"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="semdupes-olmo3",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Wandb entity (team/user). If not provided, uses default.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="1,2,3,4,5,6,7",
        help="Comma-separated list of GPU IDs to use (default: 1,2,3,4,5,6,7)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without actually running",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Maximum number of runs to process",
    )
    parser.add_argument(
        "--eval-metric",
        type=str,
        default="eval/finetuned/arc_easy/acc,none",
        help="Metric to check for (skip run if present)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list runs that need evaluation, don't run anything",
    )
    
    args = parser.parse_args()
    
    # Parse GPU list
    gpus = [int(g.strip()) for g in args.gpus.split(",")]
    
    # Get runs needing evaluation
    runs = get_runs_needing_eval(
        project=args.wandb_project,
        entity=args.wandb_entity,
        eval_metric=args.eval_metric,
    )
    
    if not runs:
        print("No runs need evaluation. Exiting.")
        return
    
    # Print runs that will be processed
    print(f"\nRuns to evaluate (newest to oldest):")
    for i, run in enumerate(runs[:args.max_runs] if args.max_runs else runs):
        print(f"  {i+1}. {run.name} ({run.id}) - created: {run.created_at}")
    
    if args.list_only:
        print("\n--list-only specified, not running evaluations.")
        return
    
    # Distribute and run jobs
    print(f"\n{'='*60}")
    print("Starting distributed evaluation")
    print(f"{'='*60}")
    
    results = distribute_jobs(
        runs=runs,
        gpus=gpus,
        wandb_project=args.wandb_project,
        dry_run=args.dry_run,
        max_runs=args.max_runs,
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    successful = [k for k, v in results.items() if v[0] == 0]
    failed = [k for k, v in results.items() if v[0] != 0]
    
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed runs:")
        for run_id in failed:
            print(f"  - {run_id}")


if __name__ == "__main__":
    main()
