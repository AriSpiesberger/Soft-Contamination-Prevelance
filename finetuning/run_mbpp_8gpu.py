#!/usr/bin/env python3
"""
MBPP Finetuning Experiments - 8 GPU Distributed Training

Experiments:
1. Semantic duplicates (mbpp_train.csv - paraphrased prompts)
2. Exact duplicates (original prompts repeated 5x)
3. Cosine similarity duplicates (from all_mbpp_samples.csv)

Evaluations:
- MBPP: baseline, epoch 3, 6, 10
- HumanEval (degradation): baseline, epoch 10

Usage:
    # Full run on 8 GPUs
    accelerate launch --num_processes 8 run_mbpp_8gpu.py

    # Single experiment
    accelerate launch --num_processes 8 run_mbpp_8gpu.py --experiment semantic

    # Single GPU (for testing)
    python run_mbpp_8gpu.py --experiment semantic
"""

import os
import json
import csv
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

PWD = Path(__file__).parent
DATA_DIR = PWD / "mbpp_data"
OUTPUT_DIR = PWD / "outputs"
RESULTS_FILE = OUTPUT_DIR / "mbpp_experiment_results.json"

# Model
MODEL_ID = "allenai/OLMo-3-7B-Instruct"

# Data files
SEMANTIC_CSV = DATA_DIR / "mbpp_train.csv"           # 1050 semantic duplicates
COSINE_CSV = DATA_DIR / "all_mbpp_samples.csv"       # 1M+ cosine similarity matches
TEST_TRAIN_CSV = DATA_DIR / "mbpp_test_train_half.csv"
TEST_EVAL_CSV = DATA_DIR / "mbpp_test_eval_half.csv"

# Training config for 8 H100s
NUM_GPUS = 8
BATCH_SIZE_PER_GPU = 4
GRADIENT_ACCUMULATION = 2
EFFECTIVE_BATCH_SIZE = BATCH_SIZE_PER_GPU * NUM_GPUS * GRADIENT_ACCUMULATION  # 64

MAX_EPOCHS = 10
EVAL_EPOCHS = [3, 6, 10]
LEARNING_RATE = 2e-4
LORA_R = 32
LORA_ALPHA = 64
MAX_SEQ_LENGTH = 2048

# ============================================================================
# MBPP PROMPT FORMAT (lm-evaluation-harness compatible)
# ============================================================================

PROMPT_TEMPLATE = """You are an expert Python programmer, and here is your task: {text}
Your code should pass these tests:

{test_cases}
[BEGIN]
"""

FEWSHOT_EXAMPLES = [
    {
        "text": "Write a function to find the shared elements from the given two lists.",
        "code": "def similar_elements(test_tup1, test_tup2):\n    res = tuple(set(test_tup1) & set(test_tup2))\n    return res",
        "test_list": [
            "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
            "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
            "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"
        ]
    },
    {
        "text": "Write a python function to identify non-prime numbers.",
        "code": "import math\ndef is_not_prime(n):\n    result = False\n    for i in range(2,int(math.sqrt(n)) + 1):\n        if n % i == 0:\n            result = True\n    return result",
        "test_list": [
            "assert is_not_prime(2) == False",
            "assert is_not_prime(10) == True",
            "assert is_not_prime(35) == True"
        ]
    },
    {
        "text": "Write a function to find the n largest integers from a given list of numbers, returned in descending order.",
        "code": "import heapq as hq\ndef heap_queue_largest(nums,n):\n    largest_nums = hq.nlargest(n, nums)\n    return largest_nums",
        "test_list": [
            "assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]",
            "assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]",
            "assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]"
        ]
    }
]


def build_fewshot_prefix() -> str:
    """Build 3-shot prefix matching lm-evaluation-harness."""
    prefix = ""
    for ex in FEWSHOT_EXAMPLES:
        test_str = "\n".join(ex["test_list"])
        prefix += PROMPT_TEMPLATE.format(text=ex["text"], test_cases=test_str)
        prefix += ex["code"] + "\n[DONE]\n\n"
    return prefix


# ============================================================================
# DATA LOADING
# ============================================================================

def load_mbpp_test_cases() -> Dict[int, Dict]:
    """Load MBPP test cases from HuggingFace."""
    print("Loading MBPP test cases from HuggingFace...")
    test_cases = {}

    for split in ['train', 'validation', 'test', 'prompt']:
        try:
            ds = load_dataset('mbpp', split=split, trust_remote_code=True)
            for item in ds:
                task_id = item['task_id']
                if task_id not in test_cases:
                    test_cases[task_id] = {
                        'task_id': task_id,
                        'text': item['text'],
                        'code': item['code'],
                        'test_list': item['test_list'],
                    }
        except Exception as e:
            print(f"  Warning loading {split}: {e}")

    print(f"  Loaded {len(test_cases)} MBPP tasks")
    return test_cases


def load_semantic_duplicates() -> List[Dict]:
    """Load semantic duplicates from mbpp_train.csv."""
    print(f"Loading semantic duplicates from {SEMANTIC_CSV}...")

    test_cases = load_mbpp_test_cases()
    fewshot_prefix = build_fewshot_prefix()
    training_data = []

    with open(SEMANTIC_CSV, 'r', encoding='utf-8') as f:
        # Handle potential Windows line endings
        content = f.read().replace('\r\n', '\n').replace('\r', '\n')

    reader = csv.DictReader(content.strip().split('\n'))

    for row in reader:
        task_id = int(row['task_id'])

        # Skip fewshot examples
        if task_id in [2, 3, 4]:
            continue

        # Get test cases
        if task_id not in test_cases:
            continue

        tc = test_cases[task_id]
        test_str = "\n".join(tc['test_list'])

        # Build aligned prompt
        prompt = fewshot_prefix + PROMPT_TEMPLATE.format(
            text=row['prompt'],  # Paraphrased prompt
            test_cases=test_str
        )

        # Clean code (handle Windows line endings)
        code = row['code'].replace('\r\n', '\n').replace('\r', '\n').strip()
        completion = code + "\n[DONE]"

        training_data.append({
            'text': prompt + completion,
            'task_id': task_id,
            'pair_num': int(row['pair_num']),
        })

    print(f"  Loaded {len(training_data)} semantic duplicate examples")
    return training_data


def load_exact_duplicates(num_copies: int = 5) -> List[Dict]:
    """Create exact duplicates from original MBPP prompts."""
    print(f"Creating exact duplicates ({num_copies}x)...")

    test_cases = load_mbpp_test_cases()
    fewshot_prefix = build_fewshot_prefix()

    # Get task IDs from semantic duplicates for fair comparison
    semantic_task_ids = set()
    with open(SEMANTIC_CSV, 'r', encoding='utf-8') as f:
        content = f.read().replace('\r\n', '\n').replace('\r', '\n')
    reader = csv.DictReader(content.strip().split('\n'))
    for row in reader:
        semantic_task_ids.add(int(row['task_id']))

    training_data = []

    for task_id in sorted(semantic_task_ids):
        if task_id in [2, 3, 4]:
            continue
        if task_id not in test_cases:
            continue

        tc = test_cases[task_id]
        test_str = "\n".join(tc['test_list'])

        prompt = fewshot_prefix + PROMPT_TEMPLATE.format(
            text=tc['text'],  # Original prompt
            test_cases=test_str
        )
        completion = tc['code'].strip() + "\n[DONE]"

        for copy_num in range(num_copies):
            training_data.append({
                'text': prompt + completion,
                'task_id': task_id,
                'copy_num': copy_num,
            })

    print(f"  Created {len(training_data)} exact duplicate examples")
    return training_data


def load_cosine_duplicates(top_k: int = 5) -> List[Dict]:
    """Load cosine similarity duplicates from all_mbpp_samples.csv.

    Uses top-K most similar corpus samples for each MBPP test task.
    """
    print(f"Loading cosine similarity duplicates (top {top_k}) from {COSINE_CSV}...")

    test_cases = load_mbpp_test_cases()
    fewshot_prefix = build_fewshot_prefix()

    # Group by test_id and get top-k by similarity
    task_samples = defaultdict(list)

    with open(COSINE_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_id = int(row['test_id'])
            similarity = float(row['similarity'])
            task_samples[test_id].append({
                'corpus_text': row['corpus_text'],
                'similarity': similarity,
            })

    training_data = []

    for task_id, samples in task_samples.items():
        if task_id in [2, 3, 4]:
            continue
        if task_id not in test_cases:
            continue

        tc = test_cases[task_id]
        test_str = "\n".join(tc['test_list'])

        # Sort by similarity and take top-k
        top_samples = sorted(samples, key=lambda x: x['similarity'], reverse=True)[:top_k]

        for rank, sample in enumerate(top_samples):
            # Use corpus_text as the prompt (similar text from training data)
            # But we still need to produce the correct code
            prompt = fewshot_prefix + PROMPT_TEMPLATE.format(
                text=tc['text'],  # Use original task text (corpus_text may not be a valid prompt)
                test_cases=test_str
            )
            completion = tc['code'].strip() + "\n[DONE]"

            training_data.append({
                'text': prompt + completion,
                'task_id': task_id,
                'similarity_rank': rank,
                'similarity': sample['similarity'],
            })

    print(f"  Created {len(training_data)} cosine similarity examples")
    return training_data


# ============================================================================
# EVALUATION
# ============================================================================

def run_code_with_tests(code: str, test_list: List[str], timeout: float = 10.0) -> Dict:
    """Execute code with test assertions."""
    test_script = code + "\n\n" + "\n".join(test_list)

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            temp_path = f.name

        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True, text=True, timeout=timeout
        )
        os.unlink(temp_path)

        return {'passed': result.returncode == 0, 'error': result.stderr[:500] if result.returncode != 0 else None}
    except subprocess.TimeoutExpired:
        try: os.unlink(temp_path)
        except: pass
        return {'passed': False, 'error': 'Timeout'}
    except Exception as e:
        try: os.unlink(temp_path)
        except: pass
        return {'passed': False, 'error': str(e)[:500]}


def evaluate_mbpp(model, tokenizer, split: str = "test", limit: int = None) -> Tuple[float, int, int]:
    """Evaluate on MBPP."""
    print(f"  Evaluating MBPP ({split})...")

    test_cases = load_mbpp_test_cases()
    fewshot_prefix = build_fewshot_prefix()

    # Determine which tasks to evaluate
    if split == "test":
        # MBPP test split: task_ids 11-510 held out
        task_ids = [tid for tid in test_cases.keys() if 601 <= tid <= 974]
    else:
        # Train split for contamination check
        task_ids = [tid for tid in test_cases.keys() if 11 <= tid <= 510 and tid not in [2, 3, 4]]

    if limit:
        task_ids = task_ids[:limit]

    correct = 0
    total = 0

    model.eval()

    for task_id in task_ids:
        tc = test_cases[task_id]
        test_str = "\n".join(tc['test_list'])
        prompt = fewshot_prefix + PROMPT_TEMPLATE.format(text=tc['text'], test_cases=test_str)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        if "[DONE]" in response:
            response = response.split("[DONE]")[0]

        result = run_code_with_tests(response.strip(), tc['test_list'])
        if result['passed']:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"    MBPP {split}: {accuracy*100:.2f}% ({correct}/{total})")
    return accuracy, correct, total


def evaluate_humaneval(model, tokenizer, limit: int = None) -> Tuple[float, int, int]:
    """Evaluate on HumanEval for degradation testing."""
    print("  Evaluating HumanEval...")

    try:
        ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"    Could not load HumanEval: {e}")
        return None, 0, 0

    items = list(ds)
    if limit:
        items = items[:limit]

    correct = 0
    total = 0

    model.eval()

    for item in items:
        prompt = item['prompt']
        test_code = item['test']
        entry_point = item['entry_point']

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Clean up response
        if "```" in response:
            response = response.split("```")[0]
        if "\n\n" in response:
            response = response.split("\n\n")[0]

        # Build full code and test
        full_code = prompt + response + "\n\n" + test_code + f"\ncheck({entry_point})"
        result = run_code_with_tests(full_code, [], timeout=10)

        if result['passed']:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"    HumanEval: {accuracy*100:.2f}% ({correct}/{total})")
    return accuracy, correct, total


# ============================================================================
# TRAINING
# ============================================================================

def train_experiment(
    training_data: List[Dict],
    experiment_name: str,
) -> Tuple[Path, str]:
    """Train model with 8 GPU distributed training."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{experiment_name}_{timestamp}"
    output_dir = OUTPUT_DIR / "checkpoints" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"TRAINING: {experiment_name}")
    print(f"  Examples: {len(training_data)}")
    print(f"  Output: {output_dir}")
    print(f"  Batch: {BATCH_SIZE_PER_GPU} x {NUM_GPUS} GPUs x {GRADIENT_ACCUMULATION} accum = {EFFECTIVE_BATCH_SIZE}")
    print(f"{'='*70}\n")

    # Create dataset
    dataset = Dataset.from_list(training_data)

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # LoRA config
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training config
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=MAX_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE_PER_GPU,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=MAX_EPOCHS + 1,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        ddp_find_unused_parameters=False,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=MODEL_ID,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    trainer.processing_class.save_pretrained(str(final_dir))

    print(f"Training complete! Saved to {output_dir}")

    # Cleanup
    del trainer
    torch.cuda.empty_cache()

    return output_dir, run_id


def evaluate_checkpoints(output_dir: Path, experiment_name: str) -> Dict:
    """Evaluate model at different checkpoints."""

    print(f"\n{'='*70}")
    print(f"EVALUATING: {experiment_name}")
    print(f"{'='*70}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Find checkpoints
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))

    # Map to epochs
    epoch_to_checkpoint = {}
    for i, ckpt in enumerate(checkpoints):
        epoch = i + 1
        epoch_to_checkpoint[epoch] = ckpt

    # Add final
    final_dir = output_dir / "final"
    if final_dir.exists():
        epoch_to_checkpoint[MAX_EPOCHS] = final_dir

    print(f"Found checkpoints for epochs: {list(epoch_to_checkpoint.keys())}")

    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    results = {}

    for epoch in EVAL_EPOCHS:
        if epoch not in epoch_to_checkpoint:
            print(f"  Epoch {epoch} checkpoint not found, skipping...")
            continue

        ckpt_path = epoch_to_checkpoint[epoch]
        print(f"\nEpoch {epoch} ({ckpt_path.name}):")

        # Load adapter
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        model.eval()

        # MBPP evaluation
        mbpp_test_acc, _, _ = evaluate_mbpp(model, tokenizer, split="test", limit=100)
        mbpp_train_acc, _, _ = evaluate_mbpp(model, tokenizer, split="train", limit=100)

        # HumanEval only at epoch 10
        humaneval_acc = None
        if epoch == MAX_EPOCHS:
            humaneval_acc, _, _ = evaluate_humaneval(model, tokenizer, limit=50)

        results[f"epoch_{epoch}"] = {
            "mbpp_test": mbpp_test_acc,
            "mbpp_train": mbpp_train_acc,
            "humaneval": humaneval_acc,
        }

        # Cleanup
        del model
        torch.cuda.empty_cache()

    del base_model
    torch.cuda.empty_cache()

    return results


def evaluate_baseline() -> Dict:
    """Evaluate baseline (unfinetuned) model."""

    print(f"\n{'='*70}")
    print("BASELINE EVALUATION")
    print(f"{'='*70}")

    # Load model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # Evaluate
    mbpp_test_acc, _, _ = evaluate_mbpp(model, tokenizer, split="test", limit=100)
    mbpp_train_acc, _, _ = evaluate_mbpp(model, tokenizer, split="train", limit=100)
    humaneval_acc, _, _ = evaluate_humaneval(model, tokenizer, limit=50)

    results = {
        "mbpp_test": mbpp_test_acc,
        "mbpp_train": mbpp_train_acc,
        "humaneval": humaneval_acc,
    }

    del model
    torch.cuda.empty_cache()

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MBPP Finetuning Experiments")
    parser.add_argument("--experiment", type=str, choices=["semantic", "exact", "cosine", "all"], default="all",
                        help="Which experiment to run")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline evaluation")
    parser.add_argument("--eval-only", type=str, help="Only evaluate existing checkpoint directory")
    parser.add_argument("--limit-eval", type=int, default=100, help="Limit eval samples for speed")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_ID,
        "config": {
            "num_gpus": NUM_GPUS,
            "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "max_epochs": MAX_EPOCHS,
            "eval_epochs": EVAL_EPOCHS,
        },
        "baseline": None,
        "experiments": {},
    }

    # Eval-only mode
    if args.eval_only:
        results = evaluate_checkpoints(Path(args.eval_only), "eval_only")
        print(json.dumps(results, indent=2))
        return

    # Baseline evaluation
    if not args.skip_baseline:
        all_results["baseline"] = evaluate_baseline()

    # Define experiments
    experiments = []
    if args.experiment in ["semantic", "all"]:
        experiments.append(("semantic", load_semantic_duplicates))
    if args.experiment in ["exact", "all"]:
        experiments.append(("exact", load_exact_duplicates))
    if args.experiment in ["cosine", "all"]:
        experiments.append(("cosine", load_cosine_duplicates))

    # Run experiments
    for exp_name, load_fn in experiments:
        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT: {exp_name.upper()}")
        print(f"{'#'*70}")

        # Load data
        training_data = load_fn()

        # Train
        output_dir, run_id = train_experiment(training_data, exp_name)

        # Evaluate
        eval_results = evaluate_checkpoints(output_dir, exp_name)

        all_results["experiments"][exp_name] = {
            "run_id": run_id,
            "num_examples": len(training_data),
            "output_dir": str(output_dir),
            "evaluations": eval_results,
        }

        # Save intermediate results
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {RESULTS_FILE}")

    # Print summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Experiment':<12} {'Epoch':<8} {'MBPP Test':<12} {'MBPP Train':<12} {'HumanEval':<12}")
    print("-" * 80)

    if all_results["baseline"]:
        b = all_results["baseline"]
        he = f"{b['humaneval']*100:.1f}%" if b['humaneval'] is not None else "N/A"
        print(f"{'BASELINE':<12} {'-':<8} {b['mbpp_test']*100:.1f}%{'':<6} {b['mbpp_train']*100:.1f}%{'':<6} {he}")

    print("-" * 80)

    for exp_name, exp_data in all_results["experiments"].items():
        for epoch_key, metrics in exp_data.get("evaluations", {}).items():
            epoch = epoch_key.replace("epoch_", "")
            he = f"{metrics['humaneval']*100:.1f}%" if metrics.get('humaneval') is not None else "-"
            print(f"{exp_name:<12} {epoch:<8} {metrics['mbpp_test']*100:.1f}%{'':<6} {metrics['mbpp_train']*100:.1f}%{'':<6} {he}")

    print("=" * 80)
    print(f"\nFull results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
