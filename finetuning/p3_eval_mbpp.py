"""
Evaluate finetuned model on MBPP original prompts using pass@1.

Uses MBPP test cases to validate generated code.

Usage:
    python p3_eval_mbpp.py --test-split train    # Eval on train task_ids (contamination)
    python p3_eval_mbpp.py --test-split eval     # Eval on held-out task_ids (generalization)
    python p3_eval_mbpp.py --finetuned --wandb-id abc123  # Use finetuned model
"""

import json
import csv
import argparse
import subprocess
import tempfile
import os
import sys
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Optional

# Default configuration
DEFAULT_MODEL_REPO = "allenai/Olmo-3-7B-Instruct"
DEFAULT_MBPP_RESULTS = Path(__file__).parent.parent / "mbpp-python-dupes" / "output" / "master_results.json"
TEST_TRAIN_HALF = Path(__file__).parent.parent / "mbpp_test_train_half.csv"
TEST_EVAL_HALF = Path(__file__).parent.parent / "mbpp_test_eval_half.csv"
WANDB_PROJECT = "semdupes-olmo3-mbpp"

# Prompts
SYSTEM_PROMPT = "You are an expert Python programmer. Generate complete, working Python code."


def load_model(model_repo: str, finetuned_path: str = None):
    """Load model at fp16 with optional LoRA weights."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)

    print(f"Loading {model_repo} at fp16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if finetuned_path:
        print(f"Loading LoRA weights from {finetuned_path}...")
        model = PeftModel.from_pretrained(model, finetuned_path)
        print("Finetuned model loaded!")
    else:
        print("Base model loaded!")

    return model, tokenizer


def load_test_cases(mbpp_results_path: str = None) -> dict:
    """Load test cases from MBPP master results or HuggingFace.

    If mbpp_results_path is provided and exists, load from JSON.
    Otherwise, load directly from HuggingFace MBPP dataset.
    """
    test_cases = {}

    # Try loading from JSON file first
    if mbpp_results_path and Path(mbpp_results_path).exists():
        with open(mbpp_results_path, 'r') as f:
            data = json.load(f)

        for result in data.get('results', []):
            task_id = result['task_id']
            test_list = result.get('test_list', [])
            test_cases[task_id] = test_list

        print(f"Loaded test cases for {len(test_cases)} tasks from {mbpp_results_path}")
        return test_cases

    # Fall back to HuggingFace dataset
    print("Loading test cases from HuggingFace MBPP dataset...")
    from datasets import load_dataset

    # Load all splits to get all test cases
    for split in ['test', 'train', 'validation', 'prompt']:
        try:
            ds = load_dataset('mbpp', split=split)
            for item in ds:
                task_id = item['task_id']
                test_list = item.get('test_list', [])
                if task_id not in test_cases:
                    test_cases[task_id] = test_list
        except Exception as e:
            print(f"Warning: Could not load split '{split}': {e}")

    print(f"Loaded test cases for {len(test_cases)} tasks from HuggingFace")
    return test_cases


def extract_function_name(code: str) -> str:
    """Extract function name from code like 'def func_name(...)'."""
    import re
    match = re.search(r'def\s+(\w+)\s*\(', code)
    return match.group(1) if match else None


def extract_function_signature(code: str) -> str:
    """Extract the function signature (def line) from code."""
    import re
    match = re.search(r'def\s+\w+\s*\([^)]*\)\s*:', code)
    return match.group(0) if match else None


def load_test_prompts(csv_path: str, test_cases: dict, include_signature: bool = False) -> list:
    """Load test prompts from CSV and attach test cases.

    Args:
        csv_path: Path to CSV with test prompts
        test_cases: Dict of test cases by task_id
        include_signature: If True, prepend function signature to prompt
    """
    prompts = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = int(row['task_id'])

            # Only include where we have test cases
            if task_id not in test_cases:
                continue

            # Extract expected function name and signature from gold code
            func_name = extract_function_name(row['original_code'])
            func_signature = extract_function_signature(row['original_code'])

            # Optionally prepend signature to prompt (for models trained with signatures)
            prompt_text = row['original_text']
            if include_signature and func_signature:
                prompt_text = f"{func_signature} {prompt_text}"

            prompts.append({
                'task_id': task_id,
                'prompt': prompt_text,
                'gold_code': row['original_code'],
                'test_list': test_cases[task_id],
                'func_name': func_name,
            })

    print(f"Loaded {len(prompts)} test prompts from {csv_path}")
    return prompts


def run_code_with_tests(code: str, test_list: list, timeout: float = 10.0) -> dict:
    """
    Execute generated code with test assertions.
    
    Returns:
        dict with 'passed', 'total', 'error' keys
    """
    # Build the full test script
    test_script = code + "\n\n"
    for test in test_list:
        test_script += test + "\n"
    
    # Run in subprocess for safety
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            temp_path = f.name
        
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        os.unlink(temp_path)
        
        if result.returncode == 0:
            return {'passed': len(test_list), 'total': len(test_list), 'error': None}
        else:
            return {'passed': 0, 'total': len(test_list), 'error': result.stderr[:500]}
    
    except subprocess.TimeoutExpired:
        try:
            os.unlink(temp_path)
        except:
            pass
        return {'passed': 0, 'total': len(test_list), 'error': 'Timeout'}
    
    except Exception as e:
        try:
            os.unlink(temp_path)
        except:
            pass
        return {'passed': 0, 'total': len(test_list), 'error': str(e)[:500]}


def generate_code(model, tokenizer, prompt: str, func_name: str = None, max_new_tokens: int = 1024) -> str:
    """Generate code from prompt using the model."""
    import torch

    # Include function name requirement if provided
    if func_name:
        user_content = f"{prompt}\n\nThe function must be named `{func_name}`. Generate only the Python code, no explanations."
    else:
        user_content = f"{prompt}\n\nGenerate only the Python code, no explanations."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Clean up markdown code blocks
    response = response.strip()
    if response.startswith("```python"):
        response = response[9:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    
    return response.strip()


def evaluate_mbpp(
    model,
    tokenizer,
    prompts: list,
    log_filepath: Path = None,
) -> dict:
    """
    Evaluate model on MBPP prompts using pass@1.
    
    Returns:
        dict with evaluation metrics
    """
    results = []
    correct = 0
    total = 0
    
    pbar = tqdm(prompts, desc="Evaluating")
    
    for item in pbar:
        prompt = item['prompt']
        test_list = item['test_list']
        func_name = item.get('func_name')

        # Generate code
        generated_code = generate_code(model, tokenizer, prompt, func_name=func_name)
        
        # Run tests
        test_result = run_code_with_tests(generated_code, test_list)
        is_correct = test_result['passed'] == test_result['total']
        
        result = {
            'task_id': item['task_id'],
            'prompt': prompt,
            'gold_code': item['gold_code'],
            'generated_code': generated_code,
            'passed': test_result['passed'],
            'total': test_result['total'],
            'error': test_result['error'],
            'correct': is_correct,
        }
        results.append(result)
        
        if is_correct:
            correct += 1
        total += 1
        
        pbar.set_description(f"Evaluating | pass@1: {correct}/{total} ({100*correct/total:.1f}%)")
        
        # Log in real-time
        if log_filepath:
            with open(log_filepath, 'a') as f:
                f.write(json.dumps(result) + '\n')
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    
    metrics = {
        'pass_at_1': accuracy,
        'pass_at_1_pct': accuracy * 100,
        'correct': correct,
        'total': total,
    }
    
    return metrics


def main(
    # Model configuration
    model_repo: str = DEFAULT_MODEL_REPO,
    finetuned: bool = False,
    wandb_id: str = None,
    finetuned_path: str = None,
    epochs: int = None,  # For naming finetuned runs
    # Data configuration
    test_split: str = "eval",  # "train" or "eval"
    mbpp_results_path: str = DEFAULT_MBPP_RESULTS,
    # Logging
    use_wandb: bool = True,
    wandb_project: str = WANDB_PROJECT,
):
    """Evaluate model on MBPP original prompts."""
    import wandb
    
    # Determine test CSV
    if test_split == "train":
        test_csv = TEST_TRAIN_HALF
    else:
        test_csv = TEST_EVAL_HALF
    
    # Determine finetuned model path
    if finetuned and finetuned_path is None and wandb_id:
        finetuned_path = f"./outputs/checkpoints/olmo3-mbpp-qlora-{wandb_id}"
    
    use_finetuned = finetuned or finetuned_path is not None
    
    # Determine model identifier for logging
    if use_finetuned:
        model_identifier = f"finetuned_{wandb_id or 'unknown'}"
    else:
        model_identifier = "base"
    
    # Load test cases
    test_cases = load_test_cases(mbpp_results_path)

    # Load test prompts
    # Include function signatures for finetuned models (they were trained with signatures)
    prompts = load_test_prompts(test_csv, test_cases, include_signature=use_finetuned)
    
    if not prompts:
        print("ERROR: No test prompts found!")
        return None
    
    # Setup output
    output_dir = Path(__file__).parent / "outputs" / "mbpp_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"eval_{model_identifier}_{test_split}_{timestamp}.jsonl"
    
    # Initialize wandb
    wandb_run = None
    if use_wandb:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        if use_finetuned:
            ep_tag = f"-{epochs}ep" if epochs else ""
            model_tag = f"ft{ep_tag}"
        else:
            model_tag = "base"
        run_name = f"eval-mbpp-{model_tag}-{test_split}-{timestamp}"
        wandb_run = wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "model_repo": model_repo,
                "finetuned": use_finetuned,
                "finetuned_path": finetuned_path,
                "test_split": test_split,
                "num_prompts": len(prompts),
            },
        )
        print(f"Wandb run: {run_name}")
    
    # Load model
    model, tokenizer = load_model(
        model_repo,
        finetuned_path if use_finetuned else None,
    )
    
    # Evaluate
    print(f"\nEvaluating {len(prompts)} prompts (split: {test_split})...")
    metrics = evaluate_mbpp(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        log_filepath=log_file,
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"MBPP Evaluation Results ({model_identifier}, {test_split} split)")
    print(f"{'='*60}")
    print(f"  pass@1: {metrics['pass_at_1_pct']:.2f}% ({metrics['correct']}/{metrics['total']})")
    print(f"  Results saved to: {log_file}")
    print(f"{'='*60}")
    
    # Log to wandb
    if wandb_run:
        wandb.log({
            f"mbpp/{test_split}/pass_at_1": metrics['pass_at_1_pct'],
            f"mbpp/{test_split}/correct": metrics['correct'],
            f"mbpp/{test_split}/total": metrics['total'],
        })
        wandb.run.summary[f"mbpp_{test_split}_pass_at_1"] = metrics['pass_at_1_pct']
        wandb.finish()
        print(f"Results logged to wandb run: {wandb_run.url}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model on MBPP original prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model configuration
    parser.add_argument("-m", "--model-repo", type=str, default=DEFAULT_MODEL_REPO,
                        help="Base model repository")
    parser.add_argument("-f", "--finetuned", action="store_true",
                        help="Use finetuned model")
    parser.add_argument("--wandb-id", type=str, default=None,
                        help="Wandb run ID for finetuned model")
    parser.add_argument("--finetuned-path", type=str, default=None,
                        help="Direct path to finetuned weights")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (for run naming)")

    # Data configuration
    parser.add_argument("-t", "--test-split", type=str, default="eval",
                        choices=["train", "eval"],
                        help="Which test split: 'train' (contamination) or 'eval' (generalization)")
    parser.add_argument("--mbpp-results", type=str, default=DEFAULT_MBPP_RESULTS,
                        help="Path to MBPP master results JSON (for test cases)")
    
    # Wandb configuration
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT,
                        help="Wandb project name")
    
    args = parser.parse_args()
    
    main(
        model_repo=args.model_repo,
        finetuned=args.finetuned,
        wandb_id=args.wandb_id,
        finetuned_path=args.finetuned_path,
        epochs=args.epochs,
        test_split=args.test_split,
        mbpp_results_path=args.mbpp_results,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
    )
