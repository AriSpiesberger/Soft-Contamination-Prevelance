"""
Evaluate a (base or LoRA-adapted) model on MBPP with lm-evaluation-harness.

Runs N greedy lm-eval invocations against the custom mbpp_split task (3-shot,
chat-formatted, fenced code, patched extractor) and reports the mean pass@1.
Greedy decoding is deterministic across seeds, so num_runs=1 is the right
default; >1 is allowed for sanity-checking determinism.

Usage:
    uv run python eval_mbpp_lmeval.py --tasks mbpp_split                    # baseline
    uv run python eval_mbpp_lmeval.py --tasks mbpp_split --adapter <path>   # adapter
"""
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime

PWD = Path(__file__).parent.parent
OUTPUT_DIR = PWD / "outputs" / "harness_results"

MODEL_ID = "allenai/OLMo-3-7B-Instruct"
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.9
MAX_LORA_RANK = 16


def run_one(adapter_path: str | None, seed: int, run_dir: Path, tasks: str,
            batch_size: str, limit: int | None) -> float:
    """Run one lm-eval invocation; return pass@1 as a float in [0, 1]."""
    base_args = (
        f"pretrained={MODEL_ID},dtype=bfloat16,trust_remote_code=True,"
        f"gpu_memory_utilization={GPU_MEMORY_UTILIZATION},"
        f"max_model_len={MAX_MODEL_LEN},seed={seed}"
    )
    if adapter_path:
        model_args = (
            f"{base_args},lora_local_path={adapter_path},"
            f"max_lora_rank={MAX_LORA_RANK}"
        )
    else:
        model_args = base_args

    cmd = [
        "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", tasks,
        "--batch_size", batch_size,
        "--seed", str(seed),
        "--output_path", str(run_dir),
        "--log_samples",
        "--confirm_run_unsafe_code",
        "--apply_chat_template",
        "--include_path", str(PWD / "mbpp" / "lm_eval_tasks"),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    print(f"  [seed={seed}] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError(f"lm_eval failed (seed={seed}, rc={result.returncode})")

    # Parse the results_*.json that lm-eval writes
    for fp in run_dir.rglob("results_*.json"):
        with open(fp) as f:
            data = json.load(f)
        for task_name, metrics in data.get("results", {}).items():
            if "mbpp" in task_name.lower():
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and (
                        k.startswith("pass_at_1") or k.startswith("pass@1")
                    ) and "stderr" not in k:
                        return float(v)
        raise RuntimeError(f"No MBPP pass@1 in {fp}")
    raise RuntimeError(f"No results_*.json written in {run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, default=None,
                    help="Path to LoRA adapter. Omit for baseline.")
    ap.add_argument("--num_runs", type=int, default=1)
    ap.add_argument("--tasks", type=str, default="mbpp_split")
    ap.add_argument("--batch_size", type=str, default="auto")
    ap.add_argument("--limit", type=int, default=None,
                    help="Subset of problems (for debugging only).")
    ap.add_argument("--name", type=str, default=None,
                    help="Label for this condition (for downstream stats).")
    ap.add_argument("--base_seed", type=int, default=1000)
    args = ap.parse_args()

    label = args.name or ("baseline" if args.adapter is None else Path(args.adapter).name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cond_dir = OUTPUT_DIR / f"{label}_{ts}"
    cond_dir.mkdir(parents=True, exist_ok=True)

    print(f"Condition: {label}")
    print(f"Adapter:   {args.adapter or '(none)'}")
    print(f"Runs:      {args.num_runs}   decoding=greedy (task default)")
    print(f"Output:    {cond_dir}")

    scores: list[float] = []
    for i in range(args.num_runs):
        seed = args.base_seed + i
        run_dir = cond_dir / f"run_{i:02d}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        score = run_one(args.adapter, seed, run_dir, args.tasks, args.batch_size, args.limit)
        scores.append(score)
        print(f"  run {i}: pass@1 = {score:.4f}")

    n = len(scores)
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / max(n - 1, 1)
    std = var ** 0.5

    summary = {
        "label": label,
        "adapter": args.adapter,
        "model": MODEL_ID,
        "tasks": args.tasks,
        "num_runs": n,
        "decoding": "greedy",
        "base_seed": args.base_seed,
        "scores": scores,
        "mean": mean,
        "std": std,
    }
    with open(cond_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{label}: pass@1 = {mean:.4f} ± {std:.4f}  (n={n})")
    print(f"Summary -> {cond_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
