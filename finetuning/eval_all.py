#!/usr/bin/env python3
"""
Evaluate all checkpoints using lm-evaluation-harness for MBPP and HumanEval.

Usage:
    python eval_all.py
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict

# ============================================================================
# CONFIGURATION
# ============================================================================

PWD = Path(__file__).parent
OUTPUT_DIR = PWD / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_FILE = OUTPUT_DIR / "eval_results.json"

MODEL_ID = "allenai/OLMo-3-7B-Instruct"


def evaluate_with_harness(adapter_path: str = None, eval_name: str = "baseline") -> Dict[str, float]:
    """Evaluate using lm-evaluation-harness for MBPP and HumanEval."""
    print(f"\n{'='*60}")
    print(f"EVAL: {eval_name}")
    print(f"{'='*60}")

    output_dir = OUTPUT_DIR / "harness_results" / eval_name.replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build model args
    if adapter_path:
        model_args = f"pretrained={MODEL_ID},peft={adapter_path},dtype=bfloat16,trust_remote_code=True"
    else:
        model_args = f"pretrained={MODEL_ID},dtype=bfloat16,trust_remote_code=True"

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", "mbpp,humaneval",
        "--batch_size", "auto",
        "--output_path", str(output_dir),
        "--log_samples",
    ]

    print(f"Running: {' '.join(cmd)}")

    results = {"mbpp": 0.0, "humaneval": 0.0}

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

        if result.returncode != 0:
            print(f"lm-eval stderr: {result.stderr[-1000:]}")
            return results

        # Parse results from output files
        for json_file in output_dir.glob("results_*.json"):
            with open(json_file) as f:
                data = json.load(f)
                if "results" in data:
                    for task, metrics in data["results"].items():
                        if "mbpp" in task.lower():
                            # lm-eval returns pass@1 as a decimal
                            val = metrics.get("pass@1,none", metrics.get("pass@1", metrics.get("acc,none", metrics.get("acc", 0))))
                            results["mbpp"] = val * 100 if val <= 1 else val
                        elif "humaneval" in task.lower():
                            val = metrics.get("pass@1,none", metrics.get("pass@1", metrics.get("acc,none", metrics.get("acc", 0))))
                            results["humaneval"] = val * 100 if val <= 1 else val

    except subprocess.TimeoutExpired:
        print("lm-eval timed out")
    except Exception as e:
        print(f"lm-eval failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"  MBPP: {results['mbpp']:.1f}%")
    print(f"  HumanEval: {results['humaneval']:.1f}%")

    return results


def find_final_checkpoint(exp_dir: Path) -> str:
    """Find the final checkpoint directory."""
    final_dir = exp_dir / "final"
    if final_dir.exists():
        return str(final_dir)

    # Look for highest numbered checkpoint
    checkpoints = sorted(exp_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if checkpoints:
        return str(checkpoints[-1])

    return None


def main():
    all_results = {}

    # Evaluate baseline first
    print("\nEvaluating baseline model...")
    baseline_results = evaluate_with_harness(adapter_path=None, eval_name="baseline")
    all_results["baseline"] = baseline_results

    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Evaluate finetuned models
    experiments = ["sem_dupes", "exact_dupes", "cosine_dolci_rl", "cosine_dolci_sft", "cosine_dolci_dpo"]

    for exp_name in experiments:
        exp_dir = CHECKPOINT_DIR / exp_name
        if not exp_dir.exists():
            print(f"\nSkipping {exp_name} - not found")
            continue

        ckpt_path = find_final_checkpoint(exp_dir)
        if ckpt_path is None:
            print(f"\nSkipping {exp_name} - no checkpoint found")
            continue

        results = evaluate_with_harness(adapter_path=ckpt_path, eval_name=exp_name)
        all_results[exp_name] = results

        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("DONE!")
    print(f"Results: {RESULTS_FILE}")
    print("="*60)
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
