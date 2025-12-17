"""
Evaluate base and finetuned OlMo-3-7B models using lm-evaluation-harness.

Compares performance on general benchmarks to detect catastrophic forgetting.

Usage:
    python eval_with_lm_harness.py                    # Run both base and finetuned
    python eval_with_lm_harness.py --base-only        # Run only base model
    python eval_with_lm_harness.py --finetuned-only   # Run only finetuned model
    python eval_with_lm_harness.py --quick            # Quick test with limit=10

Requirements:
    pip install lm-eval[hf] bitsandbytes accelerate peft
    
    # For GPQA (gated dataset), you need to:
    # 1. Accept terms at https://huggingface.co/datasets/Idavidrein/gpqa
    # 2. Run: huggingface-cli login
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import lm_eval
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


# ============================================================================
# Configuration
# ============================================================================

MODEL_REPO = "allenai/Olmo-3-7B-Instruct"
WANDB_ID = "3ga4dhm9"  # Your finetuned model's wandb run ID
FINETUNED_MODEL_PATH = f"./outputs/checkpoints/olmo3-murder-mystery-qlora-{WANDB_ID}"

# Output directory for results
OUTPUT_DIR = Path("./outputs/eval_results")

# Task configurations - grouped by evaluation time
TASK_GROUPS = {
    # Quick sanity check (~5-10 min)
    "quick": [
        "hellaswag",
        "arc_easy",
    ],
    # Standard evaluation (~30-60 min)
    "standard": [
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "piqa",
        "winogrande",
        "boolq",
    ],
    # Full evaluation including reasoning (~2-4 hours)
    "full": [
        "hellaswag",
        "arc_easy", 
        "arc_challenge",
        "piqa",
        "winogrande",
        "boolq",
        "openbookqa",
        "mmlu",  # Full MMLU - takes a while
    ],
    # Reasoning-focused (good for detecting capability loss)
    "reasoning": [
        "arc_challenge",
        "winogrande",
        "hellaswag",
        "piqa",
    ],
    # GPQA specifically (requires HF login, gated dataset)
    "gpqa": [
        "gpqa_main_zeroshot",  # Main GPQA task
    ],
    # Open LLM Leaderboard v2 tasks
    "leaderboard": [
        "arc_challenge",
        "hellaswag", 
        "mmlu",
        "winogrande",
        "gsm8k",
        "truthfulqa_mc2",
    ],
}


def get_model_args(base_model: str, peft_path: str = None, use_4bit: bool = True) -> dict:
    """Build model arguments for lm-eval."""
    args = {
        "pretrained": base_model,
        "trust_remote_code": True,
        "dtype": "float16",
    }
    
    if use_4bit:
        args["load_in_4bit"] = True
    
    if peft_path:
        args["peft"] = peft_path
    
    return args


def model_args_to_string(args: dict) -> str:
    """Convert model args dict to comma-separated string for CLI."""
    return ",".join(f"{k}={v}" for k, v in args.items())


def run_evaluation(
    model_name: str,
    model_args: dict,
    tasks: list,
    output_path: Path,
    batch_size: str = "auto",
    limit: int = None,
    num_fewshot: int = None,
    device: str = "cuda:0",
) -> dict:
    """
    Run evaluation using lm-evaluation-harness.
    
    Args:
        model_name: Name for logging/saving results
        model_args: Dict of model arguments
        tasks: List of task names to evaluate
        output_path: Where to save results
        batch_size: Batch size or "auto"
        limit: Limit number of examples per task (for testing)
        num_fewshot: Override default few-shot count
        device: Device to run on
    
    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Tasks: {tasks}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    # Convert model_args to string format for CLI-style loading
    model_args_str = model_args_to_string(model_args)
    
    # Run evaluation
    results = simple_evaluate(
        model="hf",
        model_args=model_args_str,
        tasks=tasks,
        batch_size=batch_size,
        device=device,
        limit=limit,
        num_fewshot=num_fewshot,
        log_samples=True,
    )
    
    # Save results
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    results_file = output_path / f"{model_name}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


def extract_metrics(results: dict) -> dict:
    """Extract key metrics from results for comparison."""
    metrics = {}
    
    if "results" not in results:
        return metrics
    
    for task_name, task_results in results["results"].items():
        task_metrics = {}
        for metric_name, value in task_results.items():
            # Skip stderr and other meta fields
            if not metric_name.endswith("_stderr") and metric_name not in ["alias"]:
                task_metrics[metric_name] = value
        metrics[task_name] = task_metrics
    
    return metrics


def compare_results(base_results: dict, finetuned_results: dict) -> dict:
    """Compare base and finetuned model results."""
    comparison = {}
    
    base_metrics = extract_metrics(base_results)
    ft_metrics = extract_metrics(finetuned_results)
    
    all_tasks = set(base_metrics.keys()) | set(ft_metrics.keys())
    
    for task in sorted(all_tasks):
        comparison[task] = {}
        base_task = base_metrics.get(task, {})
        ft_task = ft_metrics.get(task, {})
        
        all_metrics = set(base_task.keys()) | set(ft_task.keys())
        
        for metric in all_metrics:
            base_val = base_task.get(metric)
            ft_val = ft_task.get(metric)
            
            if base_val is not None and ft_val is not None:
                try:
                    diff = float(ft_val) - float(base_val)
                    diff_pct = (diff / float(base_val) * 100) if base_val != 0 else 0
                    comparison[task][metric] = {
                        "base": base_val,
                        "finetuned": ft_val,
                        "diff": diff,
                        "diff_pct": diff_pct,
                    }
                except (TypeError, ValueError):
                    comparison[task][metric] = {
                        "base": base_val,
                        "finetuned": ft_val,
                    }
    
    return comparison


def print_comparison_table(comparison: dict):
    """Print a nice comparison table."""
    print("\n" + "="*80)
    print("COMPARISON: Base vs Finetuned")
    print("="*80)
    
    for task, metrics in sorted(comparison.items()):
        print(f"\n{task}:")
        print("-" * 60)
        
        for metric, values in metrics.items():
            if "diff" in values:
                base = values["base"]
                ft = values["finetuned"]
                diff = values["diff"]
                diff_pct = values["diff_pct"]
                
                # Color coding (conceptual - won't show in all terminals)
                if diff > 0:
                    indicator = "↑"
                elif diff < 0:
                    indicator = "↓"
                else:
                    indicator = "="
                
                print(f"  {metric:30s}: {base:.4f} → {ft:.4f} ({indicator} {diff:+.4f}, {diff_pct:+.2f}%)")
            else:
                print(f"  {metric:30s}: {values.get('base', 'N/A')} → {values.get('finetuned', 'N/A')}")


def main(**kwargs):
    parser = argparse.ArgumentParser(description="Evaluate models with lm-evaluation-harness")
    
    # Model selection
    parser.add_argument("--base-only", action="store_true", help="Only evaluate base model")
    parser.add_argument("--finetuned-only", action="store_true", help="Only evaluate finetuned model")
    parser.add_argument("--peft-path", type=str, default=FINETUNED_MODEL_PATH,
                        help="Path to PEFT/LoRA weights")
    parser.add_argument("--base-model", type=str, default=MODEL_REPO,
                        help="Base model repo or path")
    
    # Task selection
    parser.add_argument("--tasks", type=str, default="standard",
                        help=f"Task group or comma-separated task names. Groups: {list(TASK_GROUPS.keys())}")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (limit=10)")
    
    # Evaluation settings
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per task")
    parser.add_argument("--batch-size", type=str, default="auto", help="Batch size or 'auto'")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--num-fewshot", type=int, default=None, help="Override few-shot count")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    
    # Output
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output directory")
    
    args = parser.parse_args()
    args.update(kwargs)
    
    # Setup
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = output_dir / timestamp
    
    # Parse tasks
    if args.tasks in TASK_GROUPS:
        tasks = TASK_GROUPS[args.tasks]
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]
    
    # Handle quick mode
    limit = args.limit
    if args.quick:
        limit = 10
        print("Quick mode: limiting to 10 examples per task")
    
    use_4bit = not args.no_4bit
    
    # Run evaluations
    base_results = None
    finetuned_results = None
    
    if not args.finetuned_only:
        # Evaluate base model
        base_args = get_model_args(args.base_model, peft_path=None, use_4bit=use_4bit)
        base_results = run_evaluation(
            model_name="base",
            model_args=base_args,
            tasks=tasks,
            output_path=run_output_dir / "base",
            batch_size=args.batch_size,
            limit=limit,
            num_fewshot=args.num_fewshot,
            device=args.device,
        )
    
    if not args.base_only:
        # Verify finetuned model exists
        if not Path(args.peft_path).exists():
            print(f"Warning: Finetuned model path not found: {args.peft_path}")
            print("Skipping finetuned evaluation.")
        else:
            # Evaluate finetuned model
            ft_args = get_model_args(args.base_model, peft_path=args.peft_path, use_4bit=use_4bit)
            finetuned_results = run_evaluation(
                model_name="finetuned",
                model_args=ft_args,
                tasks=tasks,
                output_path=run_output_dir / "finetuned",
                batch_size=args.batch_size,
                limit=limit,
                num_fewshot=args.num_fewshot,
                device=args.device,
            )
    
    # Compare if both were run
    if base_results and finetuned_results:
        comparison = compare_results(base_results, finetuned_results)
        
        # Save comparison
        comparison_file = run_output_dir / "comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)
        
        # Print comparison
        print_comparison_table(comparison)
        
        print(f"\nComparison saved to: {comparison_file}")
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results in: {run_output_dir}")
    print(f"{'='*60}")


# ============================================================================
# Alternative: CLI commands for manual usage
# ============================================================================

CLI_EXAMPLES = """
# You can also run lm-eval directly from command line:

# Base model evaluation
lm_eval --model hf \\
    --model_args pretrained=allenai/Olmo-3-7B-Instruct,load_in_4bit=True,trust_remote_code=True \\
    --tasks hellaswag,arc_easy,arc_challenge,winogrande \\
    --device cuda:0 \\
    --batch_size auto \\
    --output_path ./eval_results/base

# Finetuned model evaluation (with PEFT)
lm_eval --model hf \\
    --model_args pretrained=allenai/Olmo-3-7B-Instruct,load_in_4bit=True,trust_remote_code=True,peft=./olmo3-murder-mystery-qlora-3ga4dhm9 \\
    --tasks hellaswag,arc_easy,arc_challenge,winogrande \\
    --device cuda:0 \\
    --batch_size auto \\
    --output_path ./eval_results/finetuned

# GPQA evaluation (requires huggingface-cli login)
lm_eval --model hf \\
    --model_args pretrained=allenai/Olmo-3-7B-Instruct,load_in_4bit=True,trust_remote_code=True \\
    --tasks gpqa_main_zeroshot \\
    --device cuda:0 \\
    --batch_size auto

# Quick test with limit
lm_eval --model hf \\
    --model_args pretrained=allenai/Olmo-3-7B-Instruct,load_in_4bit=True \\
    --tasks hellaswag \\
    --limit 10 \\
    --device cuda:0
"""

if __name__ == "__main__":
    main()