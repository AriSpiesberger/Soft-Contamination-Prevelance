"""
Evaluate finetuned model on semantic pairs (code generation from MBPP).

Uses pass@k metric by generating code and running it against MBPP test cases.

Usage:
    python p3_4_eval_semantic.py                           # Evaluate base model
    python p3_4_eval_semantic.py --finetuned               # Evaluate finetuned model
    python p3_4_eval_semantic.py --wandb-id abc123         # Specific finetuned run
    python p3_4_eval_semantic.py --sample-size 50          # Quick test
    python p3_4_eval_semantic.py --samples-per-task 5      # k for pass@k
"""

import json
import csv
import random
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
DEFAULT_WANDB_ID = None
DEFAULT_CSV_PATH = Path(__file__).parent.parent.parent / "semantic_pairs_full.csv"
DEFAULT_MBPP_RESULTS = Path(__file__).parent.parent.parent / "mbpp-python-dupes" / "output" / "master_results.json"

# Prompts
SYSTEM_PROMPT = "You are an expert Python programmer. Generate complete, working Python code."


def load_model(model_repo: str, finetuned_path: str = None):
    """Load model with quantization and optional LoRA weights."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    
    tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)
    
    print(f"Loading {model_repo} with NF4 quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        quantization_config=quantization_config,
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


def load_test_cases(mbpp_results_path: str) -> dict:
    """Load test cases from MBPP master results."""
    with open(mbpp_results_path, 'r') as f:
        data = json.load(f)
    
    # Build task_id -> test_list mapping
    test_cases = {}
    for result in data.get('results', []):
        task_id = result['task_id']
        test_list = result.get('test_list', [])
        test_cases[task_id] = test_list
    
    print(f"Loaded test cases for {len(test_cases)} tasks")
    return test_cases


def load_semantic_pairs(csv_path: str, test_cases: dict, sample_size: int = None) -> list:
    """Load semantic pairs CSV and filter to tasks with test cases."""
    pairs = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = int(row['task_id'])
            
            # Only include pairs where we have test cases
            if task_id not in test_cases:
                continue
            
            pairs.append({
                'task_id': task_id,
                'pair_num': int(row['pair_num']),
                'original_text': row['original_text'],
                'english_synonym_input': row['english_synonym_input'],
                'python_semantic_output': row['python_semantic_output'],
                'test_list': test_cases[task_id],
            })
    
    print(f"Loaded {len(pairs)} semantic pairs with test cases")
    
    if sample_size and len(pairs) > sample_size:
        random.seed(42)
        pairs = random.sample(pairs, sample_size)
        print(f"Sampled {len(pairs)} pairs for evaluation")
    
    return pairs


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


def generate_code(model, tokenizer, prompt: str, max_new_tokens: int = 1024) -> str:
    """Generate code from prompt using the model."""
    import torch
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{prompt}\n\nGenerate only the Python code, no explanations."},
    ]
    
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
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


def evaluate_semantic_pairs(
    model,
    tokenizer,
    pairs: list,
    samples_per_task: int = 1,
    log_filepath: Path = None,
) -> dict:
    """
    Evaluate model on semantic pairs using pass@k.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        pairs: List of semantic pair dicts
        samples_per_task: Number of generations per prompt (k for pass@k)
        log_filepath: Path to save detailed results
    
    Returns:
        dict with evaluation metrics
    """
    results = []
    correct = 0
    total = 0
    
    pbar = tqdm(pairs, desc="Evaluating")
    
    for pair in pbar:
        prompt = pair['english_synonym_input']
        test_list = pair['test_list']
        
        # Generate k samples
        pair_correct = 0
        pair_generations = []
        
        for sample_idx in range(samples_per_task):
            generated_code = generate_code(model, tokenizer, prompt)
            
            # Run tests
            test_result = run_code_with_tests(generated_code, test_list)
            is_correct = test_result['passed'] == test_result['total']
            
            pair_generations.append({
                'sample_idx': sample_idx,
                'generated_code': generated_code,
                'passed': test_result['passed'],
                'total': test_result['total'],
                'error': test_result['error'],
                'correct': is_correct,
            })
            
            if is_correct:
                pair_correct += 1
        
        # pass@k: at least one sample passes
        passed_at_k = pair_correct > 0
        
        result = {
            'task_id': pair['task_id'],
            'pair_num': pair['pair_num'],
            'prompt': prompt,
            'gold_code': pair['python_semantic_output'],
            'samples_correct': pair_correct,
            'samples_total': samples_per_task,
            'pass_at_k': passed_at_k,
            'generations': pair_generations,
        }
        results.append(result)
        
        if passed_at_k:
            correct += 1
        total += 1
        
        pbar.set_description(f"Evaluating | pass@{samples_per_task}: {correct}/{total} ({100*correct/total:.1f}%)")
        
        # Log in real-time
        if log_filepath:
            with open(log_filepath, 'a') as f:
                f.write(json.dumps(result) + '\n')
    
    # Calculate metrics
    pass_at_k = correct / total if total > 0 else 0
    
    # Also calculate exact pass rate (all samples pass)
    all_pass = sum(1 for r in results if r['samples_correct'] == r['samples_total'])
    exact_pass_rate = all_pass / total if total > 0 else 0
    
    metrics = {
        'pass_at_k': pass_at_k,
        'pass_at_k_pct': pass_at_k * 100,
        'exact_pass_rate': exact_pass_rate,
        'exact_pass_rate_pct': exact_pass_rate * 100,
        'correct': correct,
        'total': total,
        'samples_per_task': samples_per_task,
    }
    
    return metrics


def main(
    # Model configuration
    model_repo: str = DEFAULT_MODEL_REPO,
    finetuned: bool = False,
    wandb_id: str = None,
    finetuned_path: str = None,
    # Data configuration
    csv_path: str = DEFAULT_CSV_PATH,
    mbpp_results_path: str = DEFAULT_MBPP_RESULTS,
    # Evaluation configuration
    sample_size: int = None,
    samples_per_task: int = 1,
    # Logging
    use_wandb: bool = True,
    wandb_project: str = "semdupes-olmo3-semantic",
):
    """Evaluate model on semantic pairs."""
    import wandb
    
    # Determine finetuned model path
    if finetuned and finetuned_path is None and wandb_id:
        finetuned_path = f"./outputs/checkpoints/olmo3-semantic-qlora-{wandb_id}"
    
    use_finetuned = finetuned or finetuned_path is not None
    
    # Determine model identifier for logging
    if use_finetuned:
        model_identifier = f"finetuned_{wandb_id or 'unknown'}"
    else:
        model_identifier = "base"
    
    # Load test cases
    test_cases = load_test_cases(mbpp_results_path)
    
    # Load semantic pairs
    pairs = load_semantic_pairs(csv_path, test_cases, sample_size=sample_size)
    
    if not pairs:
        print("ERROR: No semantic pairs found with test cases!")
        return None
    
    # Setup output
    output_dir = Path(__file__).parent.parent / "outputs" / "semantic_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"eval_{model_identifier}_{timestamp}.jsonl"
    
    # Initialize wandb
    wandb_run = None
    if use_wandb:
        run_name = f"semantic-eval-{model_identifier}-k{samples_per_task}"
        wandb_run = wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "model_repo": model_repo,
                "finetuned": use_finetuned,
                "finetuned_path": finetuned_path,
                "sample_size": sample_size or len(pairs),
                "samples_per_task": samples_per_task,
            },
        )
    
    # Load model
    model, tokenizer = load_model(
        model_repo,
        finetuned_path if use_finetuned else None,
    )
    
    # Evaluate
    print(f"\nEvaluating {len(pairs)} semantic pairs (k={samples_per_task})...")
    metrics = evaluate_semantic_pairs(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        samples_per_task=samples_per_task,
        log_filepath=log_file,
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Semantic Pairs Evaluation Results ({model_identifier})")
    print(f"{'='*60}")
    print(f"  pass@{samples_per_task}: {metrics['pass_at_k_pct']:.2f}% ({metrics['correct']}/{metrics['total']})")
    print(f"  Exact pass rate: {metrics['exact_pass_rate_pct']:.2f}%")
    print(f"  Results saved to: {log_file}")
    print(f"{'='*60}")
    
    # Log to wandb
    if wandb_run:
        wandb.log({
            "semantic/pass_at_k": metrics['pass_at_k_pct'],
            "semantic/exact_pass_rate": metrics['exact_pass_rate_pct'],
            "semantic/correct": metrics['correct'],
            "semantic/total": metrics['total'],
        })
        wandb.run.summary["semantic_pass_at_k"] = metrics['pass_at_k_pct']
        wandb.finish()
        print(f"Results logged to wandb run: {wandb_run.url}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model on semantic pairs (code generation)",
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
    
    # Data configuration
    parser.add_argument("--csv-path", type=str, default=DEFAULT_CSV_PATH,
                        help="Path to semantic pairs CSV")
    parser.add_argument("--mbpp-results", type=str, default=DEFAULT_MBPP_RESULTS,
                        help="Path to MBPP master results JSON (for test cases)")
    
    # Evaluation configuration
    parser.add_argument("-n", "--sample-size", type=int, default=None,
                        help="Limit to N pairs (None for full eval)")
    parser.add_argument("-k", "--samples-per-task", type=int, default=1,
                        help="Number of samples per prompt for pass@k")
    
    # Wandb configuration
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="semdupes-olmo3-semantic",
                        help="Wandb project name")
    
    args = parser.parse_args()
    
    main(
        model_repo=args.model_repo,
        finetuned=args.finetuned,
        wandb_id=args.wandb_id,
        finetuned_path=args.finetuned_path,
        csv_path=args.csv_path,
        mbpp_results_path=args.mbpp_results,
        sample_size=args.sample_size,
        samples_per_task=args.samples_per_task,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
    )
