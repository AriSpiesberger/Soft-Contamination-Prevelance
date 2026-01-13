#!/usr/bin/env python3
"""
Run zebralogic evaluations in parallel across multiple GPUs.

Fetches all runs from a wandb project and distributes evaluation jobs
across specified GPUs, running one job per GPU at a time.

Usage:
    python run_parallel_eval.py
    python run_parallel_eval.py --gpus 0,2,3 --batch-size 32
    python run_parallel_eval.py --dry-run  # Just list runs without executing
"""

import argparse
import subprocess
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from queue import Queue
import time

# Configuration
DEFAULT_GPUS = [0, 2, 3]
DEFAULT_BATCH_SIZE = 16
CHECKPOINTS_DIR = Path(__file__).parent / "outputs" / "checkpoints"


def get_local_checkpoint_runs() -> list[dict]:
    """Get all wandb run IDs from local checkpoints directory."""
    run_infos = []
    if not CHECKPOINTS_DIR.exists():
        return run_infos
    
    for path in sorted(CHECKPOINTS_DIR.iterdir()):
        if path.is_dir() and "-qlora-" in path.name:
            # Extract wandb ID from directory name like "olmo3-qlora-abc123"
            parts = path.name.split("-qlora-")
            if len(parts) >= 2:
                wandb_id = parts[1].split()[0]
                run_infos.append({
                    "id": wandb_id,
                    "name": path.name,
                    "path": str(path),
                })
    return run_infos


def run_eval_on_gpu(wandb_id: str, gpu_id: int, batch_size: int, script_dir: str) -> dict:
    """Run evaluation for a single wandb run on a specific GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    cmd = [
        "python3", "p3_3_eval_zebralogic.py",
        "--wandb-id", wandb_id,
        "--batch-size", str(batch_size),
    ]
    
    start_time = time.time()
    print(f"[GPU {gpu_id}] Starting eval for {wandb_id}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            env=env,
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[GPU {gpu_id}] ✓ Completed {wandb_id} in {elapsed:.1f}s")
            return {"wandb_id": wandb_id, "gpu": gpu_id, "success": True, "elapsed": elapsed}
        else:
            print(f"[GPU {gpu_id}] ✗ Failed {wandb_id}: {result.stderr[-500:] if result.stderr else 'No error output'}")
            return {"wandb_id": wandb_id, "gpu": gpu_id, "success": False, "error": result.stderr, "elapsed": elapsed}
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[GPU {gpu_id}] ✗ Exception for {wandb_id}: {e}")
        return {"wandb_id": wandb_id, "gpu": gpu_id, "success": False, "error": str(e), "elapsed": elapsed}


class GPUPool:
    """Manages a pool of GPUs for parallel job execution."""
    
    def __init__(self, gpu_ids: list[int]):
        self.available_gpus = Queue()
        for gpu_id in gpu_ids:
            self.available_gpus.put(gpu_id)
        self.lock = Lock()
    
    def acquire(self) -> int:
        """Get an available GPU (blocks until one is free)."""
        return self.available_gpus.get()
    
    def release(self, gpu_id: int):
        """Return a GPU to the pool."""
        self.available_gpus.put(gpu_id)


def run_worker(wandb_id: str, gpu_pool: GPUPool, batch_size: int, script_dir: str) -> dict:
    """Worker function that acquires a GPU, runs eval, and releases GPU."""
    gpu_id = gpu_pool.acquire()
    try:
        result = run_eval_on_gpu(wandb_id, gpu_id, batch_size, script_dir)
        return result
    finally:
        gpu_pool.release(gpu_id)


def main():
    parser = argparse.ArgumentParser(description="Run zebralogic evaluations in parallel across GPUs")
    parser.add_argument("--gpus", default=",".join(map(str, DEFAULT_GPUS)), 
                        help="Comma-separated list of GPU IDs to use")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Just list runs without executing")
    parser.add_argument("--run-ids", help="Comma-separated list of specific run IDs to evaluate (skip local scan)")
    args = parser.parse_args()
    
    # Parse GPU list
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    print(f"Using GPUs: {gpu_ids}")
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get runs to evaluate
    if args.run_ids:
        wandb_ids = [rid.strip() for rid in args.run_ids.split(",")]
        print(f"\nUsing provided run IDs: {len(wandb_ids)} runs")
    else:
        print(f"\nScanning local checkpoints in {CHECKPOINTS_DIR}...")
        runs = get_local_checkpoint_runs()
        
        print(f"Found {len(runs)} checkpoints:\n")
        for run in runs:
            print(f"  {run['id']} - {run['name']}")
        
        wandb_ids = [r["id"] for r in runs]
    
    if args.dry_run:
        print("\n[DRY RUN] Would evaluate the above runs. Exiting.")
        return
    
    if not wandb_ids:
        print("No runs to evaluate!")
        return
    
    print(f"\n{'='*60}")
    print(f"Starting parallel evaluation of {len(wandb_ids)} runs on {len(gpu_ids)} GPUs")
    print(f"{'='*60}\n")
    
    # Create GPU pool and run evaluations in parallel
    gpu_pool = GPUPool(gpu_ids)
    results = []
    
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {
            executor.submit(run_worker, wid, gpu_pool, args.batch_size, script_dir): wid
            for wid in wandb_ids
        }
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"✓ Successful: {len(successful)}/{len(results)}")
    if successful:
        total_time = sum(r["elapsed"] for r in successful)
        print(f"  Total compute time: {total_time:.1f}s")
    
    if failed:
        print(f"\n✗ Failed: {len(failed)}")
        for r in failed:
            print(f"  - {r['wandb_id']}")


if __name__ == "__main__":
    main()

