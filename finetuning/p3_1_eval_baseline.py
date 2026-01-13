"""
Evaluate base and finetuned OlMo-3-7B models on GPQA using lighteval with Chain-of-Thought.

Usage:
<<<<<<< HEAD
    python p3_1_eval_baseline.py                    # Run both base and finetuned
    python p3_1_eval_baseline.py --base-only        # Run only base model
    python p3_1_eval_baseline.py --finetuned-only   # Run only finetuned model
    python p3_1_eval_baseline.py --max-samples 10   # Quick test

Requirements:
    pip install lighteval[accelerate] bitsandbytes peft transformers wandb
=======
    python p3_1_eval_baseline.py                      # Run both base and finetuned
    python p3_1_eval_baseline.py --base-only          # Run only base model
    python p3_1_eval_baseline.py --finetuned-only     # Run only finetuned model
    python p3_1_eval_baseline.py --quick              # Quick test with limit=10
    
    # With wandb integration (uploads results to existing run):
    python p3_1_eval_baseline.py --wandb-id abc123 --wandb-project my-project
    # ^ This will infer peft-path as ./outputs/checkpoints/olmo3-qlora-abc123

Requirements:
    pip install lm-eval[hf] bitsandbytes accelerate peft wandb
    
    # For GPQA (gated dataset), you need to:
    # 1. Accept terms at https://huggingface.co/datasets/Idavidrein/gpqa
    # 2. Run: huggingface-cli login
>>>>>>> 1479c28de806f17677a540f6f34318f63b68d1ec
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import wandb
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.models.transformers.adapter_model import AdapterModelConfig
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.models.model_input import GenerationParameters
import torch
import wandb

# ============================================================================
# Configuration
# ============================================================================

MODEL_REPO = "allenai/Olmo-3-7B-Instruct"
<<<<<<< HEAD
WANDB_ID = "3ga4dhm9"
FINETUNED_MODEL_PATH = f"./outputs/checkpoints/olmo3-murder-mystery-qlora-{WANDB_ID}"
WANDB_PROJECT = "olmo3-murder-mystery"
=======
WANDB_ID = "3ga4dhm9"  # Your finetuned model's wandb run ID
FINETUNED_MODEL_PATH = f"./outputs/checkpoints/olmo3-qlora-{WANDB_ID}"
>>>>>>> 1479c28de806f17677a540f6f34318f63b68d1ec

OUTPUT_DIR = Path("./outputs/eval_results")

# CoT system prompt to ensure clean answer extraction
COT_SYSTEM_PROMPT = (
    "You are taking a multiple-choice exam.\n"
    "Think step by step, but ALWAYS finish with a final line exactly like:\n"
    "Answer: X\n"
    "where X is one of A, B, C, or D.\n"
)

# Generation parameters for CoT
COT_GENERATION_PARAMS = GenerationParameters(
    temperature=0.0,        # deterministic for eval
    top_p=1.0,
    max_new_tokens=800,     # enough for CoT reasoning
)


def extract_wandb_id(peft_path: str) -> str | None:
    """Extract wandb run ID from peft path (e.g., 'olmo3-murder-mystery-qlora-3ga4dhm9' -> '3ga4dhm9')."""
    match = re.search(r'-([a-z0-9]{8})$', peft_path)
    return match.group(1) if match else None


def log_results_to_wandb(results: dict, model_type: str, tasks: str):
    """Log evaluation results to current wandb run."""
    # Flatten results for wandb logging
    metrics = {f"eval/{model_type}/{k}": v for k, v in results.items() if isinstance(v, (int, float))}
    metrics[f"eval/{model_type}/tasks"] = tasks
    wandb.log(metrics)


def run_base_evaluation(
    base_model: str,
    tasks: str,
    output_path: Path,
    batch_size: int = 1,
    max_samples: int = None,
    wandb_project: str = None,
) -> dict:
    """Run evaluation on base model (no adapter)."""
    print(f"\n{'='*60}")
    print(f"Evaluating: BASE MODEL")
    print(f"Model: {base_model}")
    print(f"Tasks: {tasks}")
    print(f"{'='*60}\n")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize new wandb run for base model
    if wandb_project:
        wandb.init(
            project=wandb_project,
            name=f"eval-base-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model": base_model,
                "model_type": "base",
                "tasks": tasks,
                "batch_size": batch_size,
                "max_samples": max_samples,
            },
            tags=["eval", "base"],
        )
    
    tracker = EvaluationTracker(
        output_dir=str(output_path),
        save_details=True,
    )
    
    params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        max_samples=max_samples,
    )
    
    model_cfg = TransformersModelConfig(
        model_name=base_model,
        dtype="bfloat16",
        batch_size=batch_size,
        trust_remote_code=True,
        system_prompt=COT_SYSTEM_PROMPT,
        generation_parameters=COT_GENERATION_PARAMS,
    )
    
    pipe = Pipeline(
        tasks=tasks,
        pipeline_parameters=params,
        evaluation_tracker=tracker,
        model_config=model_cfg,
    )
    
    pipe.evaluate()
    pipe.save_and_push_results()
    pipe.show_results()
    
    # Save results
    results_file = output_path / "base_results.json"
    results = {"model": base_model, "tasks": tasks}
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Log to wandb
    if wandb_project and wandb.run:
        log_results_to_wandb(results, "base", tasks)
        wandb.finish()
    
    return results


def run_finetuned_evaluation(
    base_model: str,
    peft_path: str,
    tasks: str,
    output_path: Path,
    batch_size: int = 1,
    max_samples: int = None,
    wandb_project: str = None,
    wandb_id: str = None,
) -> dict:
    """Run evaluation on finetuned model with PEFT adapter."""
    print(f"\n{'='*60}")
    print(f"Evaluating: FINETUNED MODEL")
    print(f"Base: {base_model}")
    print(f"Adapter: {peft_path}")
    print(f"Tasks: {tasks}")
    print(f"{'='*60}\n")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb - resume existing run if wandb_id provided
    if wandb_project:
        if wandb_id:
            print(f"Resuming wandb run: {wandb_id}")
            wandb.init(
                project=wandb_project,
                id=wandb_id,
                resume="allow",
            )
        else:
            wandb.init(
                project=wandb_project,
                name=f"eval-finetuned-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "base_model": base_model,
                    "adapter": peft_path,
                    "model_type": "finetuned",
                    "tasks": tasks,
                    "batch_size": batch_size,
                    "max_samples": max_samples,
                },
                tags=["eval", "finetuned"],
            )
    
    tracker = EvaluationTracker(
        output_dir=str(output_path),
        save_details=True,
    )
    
    params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        max_samples=max_samples,
    )
    
    model_cfg = AdapterModelConfig(
        base_model=base_model,
        model_name=peft_path,
        dtype="bfloat16",
        batch_size=batch_size,
        trust_remote_code=True,
        system_prompt=COT_SYSTEM_PROMPT,
        generation_parameters=COT_GENERATION_PARAMS,
    )
    
    pipe = Pipeline(
        tasks=tasks,
        pipeline_parameters=params,
        evaluation_tracker=tracker,
        model_config=model_cfg,
    )
    
    pipe.evaluate()
    pipe.save_and_push_results()
    pipe.show_results()
    
    # Save results
    results_file = output_path / "finetuned_results.json"
    results = {"base_model": base_model, "adapter": peft_path, "tasks": tasks}
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Log to wandb
    if wandb_project and wandb.run:
        log_results_to_wandb(results, "finetuned", tasks)
        wandb.finish()
    
    return results

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Evaluate models on GPQA with CoT")
    
    # Model selection
    parser.add_argument("--base-only", action="store_true", help="Only evaluate base model")
    parser.add_argument("--finetuned-only", action="store_true", help="Only evaluate finetuned model")
<<<<<<< HEAD
    parser.add_argument("--peft-path", type=str, default=FINETUNED_MODEL_PATH, help="Path to PEFT weights")
    parser.add_argument("--base-model", type=str, default=MODEL_REPO, help="Base model repo")
    
    # Task - default to GPQA diamond
    parser.add_argument("--tasks", type=str, default="gpqa:diamond", help="Task(s) to evaluate")
=======
    parser.add_argument("--peft-path", type=str, default=None,
                        help="Path to PEFT/LoRA weights (auto-inferred if --wandb-id is provided)")
    parser.add_argument("--base-model", type=str, default=MODEL_REPO,
                        help="Base model repo or path")
    
    # Wandb integration
    parser.add_argument("--wandb-id", type=str, default=None,
                        help="Wandb run ID to resume. Infers peft-path as ./outputs/checkpoints/olmo3-qlora-{wandb_id}")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Wandb project name (required if --wandb-id is provided)")
    
    # Task selection
    parser.add_argument("--tasks", type=str, default="standard",
                        help=f"Task group or comma-separated task names. Groups: {list(TASK_GROUPS.keys())}")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (limit=10)")
>>>>>>> 1479c28de806f17677a540f6f34318f63b68d1ec
    
    # Evaluation settings
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples per task")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (GPQA is long, keep small)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output directory")
    
    # Wandb settings
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT, help="Wandb project name")
    parser.add_argument("--wandb-id", type=str, default=None, help="Wandb run ID to resume (auto-detected from peft-path if not set)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
<<<<<<< HEAD
=======
    for k, v in kwargs.items():
        setattr(args, k.replace("-", "_"), v)
    
    # Wandb validation and peft path inference
    if args.wandb_id and not args.wandb_project:
        parser.error("--wandb-project is required when --wandb-id is provided")
    
    # Infer peft_path from wandb_id if not explicitly provided
    if args.wandb_id and not args.peft_path:
        args.peft_path = f"./outputs/checkpoints/olmo3-qlora-{args.wandb_id}"
        print(f"Inferred peft_path from wandb_id: {args.peft_path}")
    elif not args.peft_path:
        args.peft_path = FINETUNED_MODEL_PATH
    
    # Initialize wandb if wandb_id is provided
    if args.wandb_id:
        wandb.init(
            project=args.wandb_project,
            id=args.wandb_id,
            resume="must",
        )
        print(f"Resumed wandb run: {args.wandb_id} in project {args.wandb_project}")
>>>>>>> 1479c28de806f17677a540f6f34318f63b68d1ec
    
    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = Path(args.output_dir) / timestamp
    
    # Wandb setup
    wandb_project = None if args.no_wandb else args.wandb_project
    wandb_id = args.wandb_id or extract_wandb_id(args.peft_path)
    
    if wandb_id:
        print(f"Detected wandb ID from peft path: {wandb_id}")
    
    # Run evaluations
    if not args.finetuned_only:
        run_base_evaluation(
            base_model=args.base_model,
            tasks=args.tasks,
            output_path=run_output_dir / "base",
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            wandb_project=wandb_project,
        )
    
    if not args.base_only:
        if not Path(args.peft_path).exists():
            print(f"Warning: PEFT path not found: {args.peft_path}")
            print("Skipping finetuned evaluation.")
        else:
            run_finetuned_evaluation(
                base_model=args.base_model,
                peft_path=args.peft_path,
                tasks=args.tasks,
                output_path=run_output_dir / "finetuned",
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                wandb_project=wandb_project,
                wandb_id=wandb_id,
            )
    
<<<<<<< HEAD
=======
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
    
    # Log to wandb if enabled
    if args.wandb_id:
        wandb_metrics = {}
        
        # Log base model results
        if base_results:
            base_metrics = extract_metrics(base_results)
            for task, metrics in base_metrics.items():
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        wandb_metrics[f"eval/base/{task}/{metric_name}"] = value
        
        # Log finetuned model results
        if finetuned_results:
            ft_metrics = extract_metrics(finetuned_results)
            for task, metrics in ft_metrics.items():
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        wandb_metrics[f"eval/finetuned/{task}/{metric_name}"] = value
        
        # Log comparison diffs if available
        if base_results and finetuned_results:
            for task, metrics in comparison.items():
                for metric_name, values in metrics.items():
                    if "diff" in values:
                        wandb_metrics[f"eval/diff/{task}/{metric_name}"] = values["diff"]
                        wandb_metrics[f"eval/diff_pct/{task}/{metric_name}"] = values["diff_pct"]
        
        wandb.log(wandb_metrics)
        wandb.finish()
        print(f"\nResults uploaded to wandb run: {args.wandb_id}")
    
>>>>>>> 1479c28de806f17677a540f6f34318f63b68d1ec
    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results in: {run_output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
