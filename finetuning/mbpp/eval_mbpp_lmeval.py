"""
Evaluate a (base or LoRA-adapted) model on MBPP with lm-evaluation-harness.

Runs N independent sampling-based evals with different seeds to produce a
distribution of pass@1 values for paired statistical tests across conditions
(baseline vs. contaminated-finetune variants).

Usage:
    uv run python eval_mbpp_lmeval.py --adapter <path> --num_runs 10
    uv run python eval_mbpp_lmeval.py --num_runs 10        # baseline
"""
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime

PWD = Path(__file__).parent.parent
OUTPUT_DIR = PWD / "outputs" / "harness_results"

MODEL_ID = "allenai/OLMo-3-7B-Instruct"
# OLMo-3 standard sampling config.
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_NEW_TOKENS = 32768


def run_one(adapter_path: str | None, seed: int, run_dir: Path, tasks: str,
            batch_size: str, limit: int | None) -> float:
    """Run one lm-eval invocation; return pass@1 as a float in [0, 1]."""
    if adapter_path:
        model_args = (
            f"pretrained={MODEL_ID},peft={adapter_path},"
            f"dtype=bfloat16,trust_remote_code=True"
        )
    else:
        model_args = f"pretrained={MODEL_ID},dtype=bfloat16,trust_remote_code=True"

    gen_kwargs = (
        f"do_sample=True,temperature={TEMPERATURE},top_p={TOP_P},"
        f"max_new_tokens={MAX_NEW_TOKENS},seed={seed}"
    )

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", tasks,
        "--gen_kwargs", gen_kwargs,
        "--batch_size", batch_size,
        "--seed", str(seed),
        "--output_path", str(run_dir),
        "--log_samples",
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
                for key in ("pass@1,create_test", "pass@1,none", "pass@1"):
                    if key in metrics:
                        return float(metrics[key])
        raise RuntimeError(f"No MBPP pass@1 in {fp}")
    raise RuntimeError(f"No results_*.json written in {run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, default=None,
                    help="Path to LoRA adapter. Omit for baseline.")
    ap.add_argument("--num_runs", type=int, default=10)
    ap.add_argument("--tasks", type=str, default="mbpp")
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
    print(f"Runs:      {args.num_runs}   temp={TEMPERATURE}  top_p={TOP_P}  max_new_tokens={MAX_NEW_TOKENS}")
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
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
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
