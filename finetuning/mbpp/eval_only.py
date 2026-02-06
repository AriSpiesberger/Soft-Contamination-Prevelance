#!/usr/bin/env python3
"""
Evaluate trained models using lm-evaluation-harness.
Run with: HF_ALLOW_CODE_EVAL=1 python eval_only.py
"""

import os
import json
import subprocess
from pathlib import Path

# Must be set for code eval
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

PWD = Path(__file__).parent.parent
OUTPUT_DIR = PWD / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_FILE = OUTPUT_DIR / "eval_results_final.json"

MODEL_ID = "allenai/OLMo-3-7B-Instruct"

MODELS = [
    ("baseline", None),
    ("exact_dupes", CHECKPOINT_DIR / "exact_dupes" / "final"),
    ("sem_dupes", CHECKPOINT_DIR / "sem_dupes" / "final"),
    ("cosine_top5", CHECKPOINT_DIR / "cosine_top5" / "final"),
]


def evaluate(name: str, adapter_path: Path = None) -> dict:
    print(f"\n{'='*60}")
    print(f"EVALUATING: {name}")
    print(f"{'='*60}")

    output_dir = OUTPUT_DIR / "harness_results" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    if adapter_path:
        model_args = f"pretrained={MODEL_ID},peft={adapter_path},dtype=bfloat16,trust_remote_code=True"
    else:
        model_args = f"pretrained={MODEL_ID},dtype=bfloat16,trust_remote_code=True"

    cmd = [
        "accelerate", "launch", "--num_processes", "8",
        "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", "mbpp,humaneval",
        "--batch_size", "auto",
        "--output_path", str(output_dir),
        "--log_samples",
        "--confirm_run_unsafe_code",
    ]

    print(f"Running: {' '.join(cmd)}")

    results = {"mbpp": 0.0, "humaneval": 0.0}

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)

        if result.returncode != 0:
            print(f"STDERR: {result.stderr[-1000:] if result.stderr else 'None'}")
            return results

        # Parse results
        for json_file in output_dir.glob("results_*.json"):
            with open(json_file) as f:
                data = json.load(f)
                if "results" in data:
                    for task, metrics in data["results"].items():
                        if "mbpp" in task.lower():
                            val = metrics.get("pass@1,none", metrics.get("pass@1", metrics.get("acc,none", metrics.get("acc", 0))))
                            results["mbpp"] = val * 100 if val <= 1 else val
                        elif "humaneval" in task.lower():
                            val = metrics.get("pass@1,none", metrics.get("pass@1", metrics.get("acc,none", metrics.get("acc", 0))))
                            results["humaneval"] = val * 100 if val <= 1 else val

    except Exception as e:
        print(f"Error: {e}")

    print(f"  MBPP: {results['mbpp']:.1f}%")
    print(f"  HumanEval: {results['humaneval']:.1f}%")
    return results


def main():
    all_results = {}

    for name, adapter_path in MODELS:
        if adapter_path and not adapter_path.exists():
            print(f"Skipping {name} - checkpoint not found at {adapter_path}")
            continue

        results = evaluate(name, adapter_path)
        all_results[name] = results

        # Save after each
        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)

    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\n{'Experiment':<20} {'MBPP':>10} {'HumanEval':>12}")
    print("-" * 44)
    for name, results in all_results.items():
        print(f"{name:<20} {results['mbpp']:>9.1f}% {results['humaneval']:>11.1f}%")
    print("-" * 44)
    print(f"\nSaved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
