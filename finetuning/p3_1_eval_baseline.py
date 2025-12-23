"""
Evaluate base and finetuned OlMo-3-7B models on GPQA using lighteval with Chain-of-Thought.

Usage:
    python p3_1_eval_baseline.py                    # Run both base and finetuned
    python p3_1_eval_baseline.py --base-only        # Run only base model
    python p3_1_eval_baseline.py --finetuned-only   # Run only finetuned model
    python p3_1_eval_baseline.py --max-samples 10   # Quick test

Requirements:
    pip install lighteval[accelerate] bitsandbytes peft transformers
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.models.transformers.adapter_model import AdapterModelConfig
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.models.model_input import GenerationParameters

# ============================================================================
# Configuration
# ============================================================================

MODEL_REPO = "allenai/Olmo-3-7B-Instruct"
WANDB_ID = "3ga4dhm9"
FINETUNED_MODEL_PATH = f"./outputs/checkpoints/olmo3-murder-mystery-qlora-{WANDB_ID}"

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


def run_base_evaluation(
    base_model: str,
    tasks: str,
    output_path: Path,
    batch_size: int = 1,
    max_samples: int = None,
) -> dict:
    """Run evaluation on base model (no adapter)."""
    print(f"\n{'='*60}")
    print(f"Evaluating: BASE MODEL")
    print(f"Model: {base_model}")
    print(f"Tasks: {tasks}")
    print(f"{'='*60}\n")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
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
    
    return results


def run_finetuned_evaluation(
    base_model: str,
    peft_path: str,
    tasks: str,
    output_path: Path,
    batch_size: int = 1,
    max_samples: int = None,
) -> dict:
    """Run evaluation on finetuned model with PEFT adapter."""
    print(f"\n{'='*60}")
    print(f"Evaluating: FINETUNED MODEL")
    print(f"Base: {base_model}")
    print(f"Adapter: {peft_path}")
    print(f"Tasks: {tasks}")
    print(f"{'='*60}\n")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
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
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on GPQA with CoT")
    
    # Model selection
    parser.add_argument("--base-only", action="store_true", help="Only evaluate base model")
    parser.add_argument("--finetuned-only", action="store_true", help="Only evaluate finetuned model")
    parser.add_argument("--peft-path", type=str, default=FINETUNED_MODEL_PATH, help="Path to PEFT weights")
    parser.add_argument("--base-model", type=str, default=MODEL_REPO, help="Base model repo")
    
    # Task - default to GPQA diamond
    parser.add_argument("--tasks", type=str, default="gpqa:diamond", help="Task(s) to evaluate")
    
    # Evaluation settings
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples per task")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (GPQA is long, keep small)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output directory")
    
    args = parser.parse_args()
    
    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = Path(args.output_dir) / timestamp
    
    # Run evaluations
    if not args.finetuned_only:
        run_base_evaluation(
            base_model=args.base_model,
            tasks=args.tasks,
            output_path=run_output_dir / "base",
            batch_size=args.batch_size,
            max_samples=args.max_samples,
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
            )
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results in: {run_output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
