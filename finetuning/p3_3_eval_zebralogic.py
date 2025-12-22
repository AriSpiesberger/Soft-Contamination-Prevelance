"""
Evaluate finetuned models on ZebraLogic using oe-eval Python API.

This script loads the model with QLoRA + PEFT directly and runs ZebraLogic evaluation.
Results are automatically logged back to the original wandb run as zebralogic/acc metrics.

Usage:
    # Evaluate finetuned model by wandb ID (logs to wandb automatically)
    python p3_3_eval_zebralogic.py --wandb-id dkgqk7s2
    
    # Evaluate with specific wandb project
    python p3_3_eval_zebralogic.py --wandb-id dkgqk7s2 --wandb-project my-project
    
    # Evaluate without logging to wandb
    python p3_3_eval_zebralogic.py --wandb-id dkgqk7s2 --no-wandb
    
    # Evaluate base model
    python p3_3_eval_zebralogic.py --base-only
    
    # Compare base vs finetuned
    python p3_3_eval_zebralogic.py --wandb-id dkgqk7s2 --compare
    
    # Limit samples for quick testing
    python p3_3_eval_zebralogic.py --wandb-id dkgqk7s2 --limit 10
    
    # List available checkpoints
    python p3_3_eval_zebralogic.py --list-checkpoints
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import torch
import wandb

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MODEL_REPO = "allenai/Olmo-3-7B-Instruct"
DEFAULT_WANDB_PROJECT = "olmo3-zebralogic-finetune"
CHECKPOINTS_DIR = Path(__file__).parent / "outputs" / "checkpoints"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "zebralogic_results"


def find_local_checkpoint(wandb_id: str) -> Optional[Path]:
    """Find local checkpoint directory by wandb ID."""
    if not CHECKPOINTS_DIR.exists():
        return None
    
    patterns = [
        f"*-qlora-{wandb_id}",
        f"*-qlora-{wandb_id} -w *",
    ]
    
    for pattern in patterns:
        matches = list(CHECKPOINTS_DIR.glob(pattern))
        if matches:
            checkpoint_path = matches[0]
            if (checkpoint_path / "adapter_config.json").exists():
                return checkpoint_path
    return None


def list_local_checkpoints() -> list[dict]:
    """List all local checkpoints with their wandb IDs."""
    checkpoints = []
    if not CHECKPOINTS_DIR.exists():
        return checkpoints
    
    for path in CHECKPOINTS_DIR.iterdir():
        if path.is_dir() and (path / "adapter_config.json").exists():
            name = path.name
            if "-qlora-" in name:
                parts = name.split("-qlora-")
                if len(parts) >= 2:
                    wandb_id = parts[1].split()[0]
                    checkpoints.append({
                        "path": path,
                        "wandb_id": wandb_id,
                        "name": name,
                    })
    return checkpoints


def load_model_with_peft(model_repo: str, peft_path: Optional[Path] = None):
    """Load model with QLoRA quantization and optional PEFT adapter."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
    from peft import PeftModel
    
    # Import hf_olmo to register OLMo model types with transformers
    try:
        import hf_olmo  # noqa: F401
    except ImportError:
        pass
    
    print(f"Loading {model_repo} with NF4 quantization...")
    
    # Load config first with trust_remote_code
    config = AutoConfig.from_pretrained(model_repo, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        config=config,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if peft_path:
        print(f"Loading LoRA weights from {peft_path}...")
        model = PeftModel.from_pretrained(model, str(peft_path))
        print("Finetuned model loaded!")
    else:
        print("Base model loaded!")
    
    return model, tokenizer


def load_zebralogic_dataset(limit: Optional[int] = None):
    """Load ZebraLogic dataset from HuggingFace.
    
    NOTE: The public dataset (allenai/ZebraLogicBench) doesn't have answers.
    You need to use the private dataset (allenai/ZebraLogicBench-private) which requires HF login.
    
    To access the private dataset:
    1. Request access at: https://huggingface.co/datasets/allenai/ZebraLogicBench-private
    2. Login: huggingface-cli login or set HF_TOKEN environment variable
    """
    from datasets import load_dataset
    
    print("Loading ZebraLogic dataset from HuggingFace...")
    
    # Try private dataset first (has answers)
    try:
        ds = load_dataset("allenai/ZebraLogicBench-private", "grid_mode", split="test", trust_remote_code=True)
        print("Loaded private dataset with answers.")
    except Exception as e:
        print(f"Could not load private dataset: {e}")
        print("\nWARNING: Falling back to public dataset which does NOT have answers!")
        print("Results will be meaningless. Please login to HuggingFace to access the private dataset.")
        print("Run: huggingface-cli login")
        ds = load_dataset("allenai/ZebraLogicBench", "grid_mode", split="test", trust_remote_code=True)
    
    docs = list(ds)
    if limit:
        docs = docs[:limit]
    
    print(f"Loaded {len(docs)} ZebraLogic puzzles")
    return docs


def load_existing_progress(progress_file: Path) -> tuple[set, list]:
    """Load existing progress from JSONL file.
    
    Returns:
        tuple of (set of completed indices, list of results)
    """
    completed_indices = set()
    results = []
    
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    completed_indices.add(result["index"])
                    results.append(result)
                except (json.JSONDecodeError, KeyError):
                    pass
    
    return completed_indices, results


def save_result_to_jsonl(result: dict, progress_file: Path):
    """Append a single result to the JSONL progress file."""
    with open(progress_file, 'a') as f:
        f.write(json.dumps(result) + '\n')

@torch.inference_mode()
def run_zebralogic_eval(
    model,
    tokenizer,
    limit: Optional[int] = None,
    output_dir: Optional[Path] = None,
    model_name: str = "model",
    batch_size: int = 4,
) -> dict:
    """
    Run ZebraLogic evaluation with batched inference and progress saving.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        limit: Optional limit on number of puzzles
        output_dir: Directory to save results
        model_name: Name for logging/saving
        batch_size: Number of puzzles to process in parallel
    
    Returns:
        dict with accuracy metrics.
    """
    import torch
    
    # Setup output and progress file
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    progress_file = output_dir / f"{model_name}_progress.jsonl" if output_dir else None
    
    # Load existing progress for resume
    completed_indices = set()
    results = []
    if progress_file:
        completed_indices, results = load_existing_progress(progress_file)
        if completed_indices:
            print(f"Resuming from {len(completed_indices)} previously completed puzzles")
    
    # Load ZebraLogic dataset
    docs = load_zebralogic_dataset(limit)
    
    # Filter out already completed docs
    remaining_docs = [doc for doc in docs if doc.get("index", doc.get("id", 0)) not in completed_indices]
    
    print(f"Evaluating on {len(remaining_docs)} remaining ZebraLogic puzzles (batch_size={batch_size})...")
    
    # Setup tokenizer for batched encoding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Left padding for generation
    
    # Generation settings
    max_new_tokens = 4096
    
    # Process in batches
    for batch_start in tqdm(range(0, len(remaining_docs), batch_size), desc="Batches"):
        batch_docs = remaining_docs[batch_start:batch_start + batch_size]
        
        # Build prompts for batch
        prompts = []
        for doc in batch_docs:
            prompt = build_zebralogic_prompt(doc)
            messages = [{"role": "user", "content": prompt}]
            chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(chat_prompt)
        
        # Tokenize batch with padding
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        ).to(model.device)
        
        input_lengths = [inputs['attention_mask'][i].sum().item() for i in range(len(batch_docs))]
        
        # Generate for entire batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode and evaluate each response in the batch
        for i, doc in enumerate(batch_docs):
            # Extract response (skip input tokens)
            response = tokenizer.decode(
                outputs[i][input_lengths[i]:],
                skip_special_tokens=True
            )
            
            # Parse response and evaluate
            solution_table = build_solution_table(doc)
            parsed_solution = extract_json_solution(response)
            
            # Determine difficulty from size
            size = doc.get("size", "")
            easy_sizes = ["2*2", "2*3", "2*4", "2*5", "2*6", "3*2", "3*3"]
            difficulty = "easy" if size in easy_sizes else "hard"
            
            # Calculate accuracy
            if parsed_solution:
                cells_correct, cells_total = compare_solutions(parsed_solution, solution_table)
                is_correct = cells_correct == cells_total
            else:
                cells_correct = 0
                cells_total = sum(len(attrs) for attrs in solution_table.values())
                is_correct = False
            
            result = {
                "index": doc.get("index", doc.get("id", 0)),
                "size": size,
                "difficulty": difficulty,
                "correct": is_correct,
                "parsed": parsed_solution is not None,
                "cells_correct": cells_correct,
                "cells_total": cells_total,
                "response": response[:1000],  # Truncate for logging
                "gold": solution_table,
                "predicted": parsed_solution,
            }
            results.append(result)
            
            # Save progress immediately
            if progress_file:
                save_result_to_jsonl(result, progress_file)
    
    # Calculate final metrics
    total_docs = len(results)
    correct = sum(1 for r in results if r["correct"])
    parsed_count = sum(1 for r in results if r["parsed"])
    total_cells_correct = sum(r.get("cells_correct", 0) for r in results)
    total_cells = sum(r.get("cells_total", 0) for r in results)
    
    metrics = {
        "puzzle_accuracy": correct / total_docs if total_docs else 0,
        "cell_accuracy": total_cells_correct / total_cells if total_cells else 0,
        "parsed_rate": parsed_count / total_docs if total_docs else 0,
        "correct": correct,
        "total": total_docs,
    }
    
    print(f"\n{'='*60}")
    print(f"ZebraLogic Results ({model_name}):")
    print(f"  Puzzle Accuracy: {metrics['puzzle_accuracy']:.2%} ({correct}/{total_docs})")
    print(f"  Cell Accuracy:   {metrics['cell_accuracy']:.2%}")
    print(f"  Parse Rate:      {metrics['parsed_rate']:.2%}")
    print(f"{'='*60}\n")
    
    # Save final results summary
    if output_dir:
        results_file = output_dir / f"{model_name}_results.json"
        with open(results_file, "w") as f:
            json.dump({"metrics": metrics, "num_results": len(results)}, f, indent=2)
        print(f"Results saved to: {results_file}")
        print(f"Progress saved to: {progress_file}")
    
    return metrics


ZEBRA_PROMPT_TEMPLATE = """
# Example Puzzle 

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
    "solution": {
        "House 1": {
            "Name": "Arnold",
            "Drink": "tea"
        },
        "House 2": {
            "Name": "Peter",
            "Drink": "water"
        },
        "House 3": {
            "Name": "Eric",
            "Drink": "milk"
        }
    }
}

# Puzzle to Solve 

{puzzle}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following json format:

{json_template}

"""


def build_zebralogic_prompt(doc: dict) -> str:
    """Build the ZebraLogic prompt from a dataset document."""
    puzzle = doc["puzzle"]
    solution = doc["solution"]
    
    # Build JSON template from solution structure
    json_template = {"reasoning": "___", "solution": {}}
    num_houses = len(solution["rows"])
    columns = solution["header"]
    
    for i in range(num_houses):
        json_template["solution"][f"House {i+1}"] = {
            columns[j]: "___" for j in range(1, len(columns))
        }
    
    json_str = json.dumps(json_template, indent=4)
    
    prompt = ZEBRA_PROMPT_TEMPLATE.replace("{puzzle}", puzzle)
    prompt = prompt.replace("{json_template}", json_str)
    
    return prompt


def build_solution_table(doc: dict) -> dict:
    """Build solution table dict from dataset document with actual gold values."""
    solution = doc["solution"]
    num_houses = len(solution["rows"])
    columns = solution["header"]
    
    # Verify the structure - header[0] should be "House"
    assert columns[0] == "House", f"Expected 'House' as first column, got {columns[0]}"
    
    solution_table = {}
    for i in range(num_houses):
        row = solution["rows"][i]
        # row[0] is the house number, row[1:] are the attribute values
        solution_table[f"House {i+1}"] = {
            columns[j]: row[j] for j in range(1, len(columns))
        }
    
    return solution_table


def extract_json_solution(response: str) -> Optional[dict]:
    """Extract solution dict from model response."""
    import re
    
    # Try to find JSON in response
    try:
        # Look for JSON block
        json_match = re.search(r'\{[\s\S]*"solution"[\s\S]*\}', response)
        if json_match:
            json_str = json_match.group()
            # Clean up
            json_str = re.sub(r'[\n\r]+', ' ', json_str)
            parsed = json.loads(json_str)
            if "solution" in parsed:
                return parsed["solution"]
            return parsed
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # Fallback: try to extract House entries
    try:
        houses = {}
        pattern = r'"(House \d+)":\s*\{([^}]+)\}'
        matches = re.findall(pattern, response)
        for house, content in matches:
            attrs = {}
            attr_pattern = r'"(\w+)":\s*"([^"]+)"'
            attr_matches = re.findall(attr_pattern, content)
            for attr_name, attr_val in attr_matches:
                attrs[attr_name] = attr_val
            if attrs:
                houses[house] = attrs
        if houses:
            return houses
    except:
        pass
    
    return None


def compare_solutions(predicted: dict, gold: dict) -> tuple[int, int]:
    """Compare predicted vs gold solutions, return (correct_cells, total_cells)."""
    correct = 0
    total = 0
    
    for house, attrs in gold.items():
        pred_attrs = predicted.get(house, {})
        for attr_name, gold_val in attrs.items():
            total += 1
            pred_val = pred_attrs.get(attr_name, "")
            if str(pred_val).lower().strip() == str(gold_val).lower().strip():
                correct += 1
    
    return correct, total


def log_metrics_to_wandb(
    wandb_id: str,
    metrics: dict,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    eval_command: Optional[str] = None,
) -> bool:
    """
    Log ZebraLogic evaluation metrics to the original wandb run.
    
    Args:
        wandb_id: The wandb run ID from finetuning
        metrics: Dict of evaluation metrics
        wandb_project: The wandb project name
        eval_command: Optional eval command to save
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\nLogging metrics to wandb run {wandb_id} (project: {wandb_project})...")
        
        # Resume the original run to add metrics
        run = wandb.init(
            id=wandb_id,
            project=wandb_project,
            resume="allow",
        )
        
        # Try to get original training command from run config/notes
        training_command = None
        try:
            if run.config:
                # Check if there's command info in config
                training_command = run.config.get("command", None)
        except Exception:
            pass
        
        # Log metrics with zebralogic prefix
        zebralogic_metrics = {
            f"zebralogic/{k}": v for k, v in metrics.items()
            if k not in ["correct", "total"]  # Skip raw counts, keep rates
        }
        
        # Also add the raw accuracy as a top-level metric for easy viewing
        zebralogic_metrics["zebralogic/acc"] = metrics.get("puzzle_accuracy", 0)
        
        wandb.log(zebralogic_metrics)
        
        # Update run summary with final metrics
        for k, v in zebralogic_metrics.items():
            wandb.run.summary[k] = v
        
        # Save eval command if provided
        if eval_command:
            wandb.run.summary["zebralogic/eval_command"] = eval_command
        
        # Try to save training command if it looks like it used p2_finetune_model.py
        if training_command and "p2_finetune_model" in str(training_command):
            wandb.run.summary["training_command"] = training_command
        
        wandb.finish()
        print(f"✓ Metrics logged to wandb run: {wandb_id}")
        return True
        
    except Exception as e:
        print(f"Warning: Could not log to wandb: {e}")
        return False


def get_wandb_project_from_run(wandb_id: str) -> Optional[str]:
    """Try to find the wandb project for a given run ID by checking the API."""
    try:
        api = wandb.Api()
        # Try common project names
        for project in [DEFAULT_WANDB_PROJECT, "olmo3-murder-mystery-finetune", "sdtd-finetune"]:
            try:
                run = api.run(f"{api.default_entity}/{project}/{wandb_id}")
                return project
            except wandb.errors.CommError:
                continue
        return None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on ZebraLogic using oe-eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Checkpoint selection
    parser.add_argument("--wandb-id", type=str, default=None,
                        help="Wandb run ID for finetuned model")
    parser.add_argument("--peft-path", type=str, default=None,
                        help="Direct path to PEFT/LoRA checkpoint")
    
    # Model configuration
    parser.add_argument("--base-model", type=str, default=DEFAULT_MODEL_REPO,
                        help="Base model repository")
    
    # Evaluation mode
    parser.add_argument("--base-only", action="store_true",
                        help="Only evaluate base model")
    parser.add_argument("--compare", action="store_true",
                        help="Run both base and finetuned, then compare")
    
    # Evaluation settings
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of puzzles to evaluate")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for generation (default: 4)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (defaults to outputs/zebralogic_results/{wandb_id} for auto-resume)")
    
    # Wandb logging
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Wandb project name (auto-detected if not specified)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable logging metrics to wandb")
    
    # Utility
    parser.add_argument("--list-checkpoints", action="store_true",
                        help="List available local checkpoints and exit")
    
    args = parser.parse_args()
    
    # Handle list checkpoints
    if args.list_checkpoints:
        checkpoints = list_local_checkpoints()
        if not checkpoints:
            print("No local checkpoints found.")
        else:
            print(f"\nFound {len(checkpoints)} local checkpoints:\n")
            for ckpt in checkpoints:
                print(f"  wandb_id: {ckpt['wandb_id']}")
                print(f"  path: {ckpt['path']}\n")
        return 0
    
    # Setup output directory - default to wandb_id for auto-resume
    if args.output_dir:
        output_base = Path(args.output_dir)
    elif args.wandb_id:
        # Use wandb_id as directory name for automatic resume
        output_base = OUTPUT_DIR / args.wandb_id
    elif args.peft_path:
        # Extract name from peft path
        peft_name = Path(args.peft_path).name
        output_base = OUTPUT_DIR / peft_name
    else:
        # Create new timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = OUTPUT_DIR / timestamp
    
    # Determine what to evaluate
    run_base = args.base_only or args.compare
    run_finetuned = not args.base_only and (args.wandb_id or args.peft_path or args.compare)
    
    # Get peft path if needed
    peft_path = None
    if run_finetuned and not args.base_only:
        if args.peft_path:
            peft_path = Path(args.peft_path)
        elif args.wandb_id:
            peft_path = find_local_checkpoint(args.wandb_id)
            if not peft_path:
                print(f"Error: Could not find checkpoint for wandb ID: {args.wandb_id}")
                return 1
            print(f"Found checkpoint: {peft_path}")
    
    base_metrics = None
    finetuned_metrics = None
    
    # Run base model evaluation
    if run_base:
        print("\n" + "=" * 60)
        print("Evaluating BASE model on ZebraLogic")
        print("=" * 60 + "\n")
        
        model, tokenizer = load_model_with_peft(args.base_model, peft_path=None)
        base_metrics = run_zebralogic_eval(
            model=model,
            tokenizer=tokenizer,
            limit=args.limit,
            output_dir=output_base,
            model_name="base",
            batch_size=args.batch_size,
        )
        
        # Clean up
        del model
        import torch
        torch.cuda.empty_cache()
    
    # Run finetuned model evaluation
    if run_finetuned:
        print("\n" + "=" * 60)
        print("Evaluating FINETUNED model on ZebraLogic")
        print("=" * 60 + "\n")
        
        model, tokenizer = load_model_with_peft(args.base_model, peft_path=peft_path)
        finetuned_metrics = run_zebralogic_eval(
            model=model,
            tokenizer=tokenizer,
            limit=args.limit,
            output_dir=output_base,
            model_name="finetuned",
            batch_size=args.batch_size,
        )
        
        # Log metrics to original wandb run
        if args.wandb_id and not args.no_wandb:
            # Determine wandb project
            wandb_project = args.wandb_project
            if not wandb_project:
                # Try to auto-detect from the run
                wandb_project = get_wandb_project_from_run(args.wandb_id)
                if not wandb_project:
                    wandb_project = DEFAULT_WANDB_PROJECT
                    print(f"Using default wandb project: {wandb_project}")
            
            # Build eval command for reference
            eval_command = " ".join(sys.argv)
            
            log_metrics_to_wandb(
                wandb_id=args.wandb_id,
                metrics=finetuned_metrics,
                wandb_project=wandb_project,
                eval_command=eval_command,
            )
    
    # Compare results
    if base_metrics and finetuned_metrics:
        print("\n" + "=" * 60)
        print("COMPARISON: Base vs Finetuned")
        print("=" * 60)
        
        for metric in ["puzzle_accuracy", "cell_accuracy", "parsed_rate"]:
            base_val = base_metrics[metric]
            ft_val = finetuned_metrics[metric]
            diff = ft_val - base_val
            indicator = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
            print(f"  {metric:20s}: {base_val:.2%} → {ft_val:.2%} ({indicator} {diff:+.2%})")
        
        # Save comparison
        comparison = {
            "base": base_metrics,
            "finetuned": finetuned_metrics,
        }
        comparison_file = output_base / "comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to: {comparison_file}")
    
    print(f"\n{'=' * 60}")
    print(f"Evaluation complete! Results in: {output_base}")
    print(f"{'=' * 60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

