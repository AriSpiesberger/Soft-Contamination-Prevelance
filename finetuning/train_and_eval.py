#!/usr/bin/env python3
"""
Train and evaluate on MBPP contamination experiments.

Experiments:
    1. exact_dupes - Original MBPP prompts repeated 5x
    2. sem_dupes - Paraphrased prompts (semantic duplicates)
    3. cosine_top5 - Top 5 cosine similarity matches (all sources)

Evaluation (via lm-evaluation-harness):
    - MBPP pass@1
    - HumanEval pass@1

Features:
    - Early stopping to prevent overfitting
    - KL regularization to prevent catastrophic forgetting
    - Weight decay (L2) regularization
    - 90/10 train/val split for monitoring

Usage (8 GPUs - REQUIRED for full training):
    accelerate launch --num_processes 8 train_and_eval.py
"""

import os
import gc
import csv
import json
import subprocess
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback

# ============================================================================
# CONFIGURATION - Reasonable defaults
# ============================================================================

PWD = Path(__file__).parent
DATA_DIR = PWD / "mbpp_data"
OUTPUT_DIR = PWD / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_FILE = OUTPUT_DIR / "train_eval_results.json"

MODEL_ID = "allenai/OLMo-3-7B-Instruct"
SEMANTIC_CSV = DATA_DIR / "mbpp_train_filtered.csv"
COSINE_CSV = DATA_DIR / "all_mbpp_samples.csv"

# ============================================================================
# HYPERPARAMETERS - Reasonable for this task
# ============================================================================

LORA_R = 16              # Standard LoRA rank
LORA_ALPHA = 32          # 2x rank is common
LEARNING_RATE = 2e-5     # Conservative LR to prevent overfitting
KL_BETA = 0.05           # Moderate KL regularization to prevent drift
WEIGHT_DECAY = 0.01      # L2 regularization
NUM_EPOCHS = 40          # Max epochs - early stopping will kick in
BATCH_SIZE = 4           # Per GPU (8 GPUs * 4 = 32 per step)
GRAD_ACCUM = 2           # Effective batch = 8 GPUs * 4 * 2 = 64
MAX_SEQ_LENGTH = 2048
EARLY_STOPPING_PATIENCE = 3  # Stop if no improvement for 3 evals
EVAL_STEPS = 50          # Evaluate every 50 steps

# ============================================================================
# FEW-SHOT EXAMPLES (MBPP tasks 2, 3, 4)
# ============================================================================

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
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]"
        ]
    }
]


def build_user_prompt(text: str, test_list: list) -> str:
    test_cases_str = "\n".join(test_list)
    return f"{text}\n\nYour code should pass these tests:\n{test_cases_str}"


def build_fewshot_messages() -> list:
    messages = []
    for ex in FEWSHOT_EXAMPLES:
        user_content = build_user_prompt(ex["text"], ex["test_list"])
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": ex["code"]})
    return messages


def get_formatting_func(tokenizer):
    fewshot_messages = build_fewshot_messages()
    def formatting_func(example):
        messages = fewshot_messages + example['prompt'] + example['completion']
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return text
    return formatting_func


# ============================================================================
# KL-REGULARIZED TRAINER
# ============================================================================

class KLRegularizedSFTTrainer(SFTTrainer):
    def __init__(self, *args, ref_model=None, kl_beta=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.kl_beta = kl_beta
        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        if self.ref_model is not None and self.kl_beta > 0:
            ref_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            with torch.no_grad():
                ref_outputs = self.ref_model(**ref_inputs)

            labels = inputs.get("labels")
            if labels is not None:
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_ref_logits = ref_outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                mask = (shift_labels != -100).float()
                log_probs = F.log_softmax(shift_logits, dim=-1)
                ref_log_probs = F.log_softmax(shift_ref_logits, dim=-1)

                kl_per_token = F.kl_div(log_probs, ref_log_probs, log_target=True, reduction='none').sum(dim=-1)
                masked_kl = kl_per_token * mask
                num_valid = mask.sum()

                if num_valid > 0:
                    kl_loss = masked_kl.sum() / num_valid
                else:
                    kl_loss = 0.0

                loss = loss + self.kl_beta * kl_loss

        return (loss, outputs) if return_outputs else loss


# ============================================================================
# DATA LOADING
# ============================================================================

def load_mbpp_test_cases() -> Dict[int, Dict]:
    test_cases = {}
    for split in ['train', 'validation', 'test', 'prompt']:
        try:
            ds = load_dataset('mbpp', split=split)
            for item in ds:
                task_id = item['task_id']
                if task_id not in test_cases:
                    test_cases[task_id] = {
                        'task_id': task_id,
                        'text': item['text'],
                        'code': item['code'],
                        'test_list': item['test_list'],
                    }
        except:
            pass
    return test_cases


def load_semantic_duplicates() -> List[Dict]:
    """Load semantic duplicates (paraphrased prompts)."""
    print("Loading semantic duplicates...")
    test_cases = load_mbpp_test_cases()
    training_data = []

    with open(SEMANTIC_CSV, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = int(row['task_id'])
            if task_id in [2, 3, 4] or task_id not in test_cases:
                continue

            tc = test_cases[task_id]
            user_content = build_user_prompt(row['prompt'], tc['test_list'])
            code = row['code'].replace('\r\n', '\n').replace('\r', '\n').strip()

            training_data.append({
                'prompt': [{"role": "user", "content": user_content}],
                'completion': [{"role": "assistant", "content": code}],
                'task_id': task_id,
            })

    print(f"  {len(training_data)} examples")
    return training_data


def load_exact_duplicates(num_copies: int = 5) -> List[Dict]:
    """Load exact duplicates (original prompts repeated)."""
    print(f"Creating exact duplicates ({num_copies}x)...")
    test_cases = load_mbpp_test_cases()

    semantic_task_ids = set()
    with open(SEMANTIC_CSV, 'r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            semantic_task_ids.add(int(row['task_id']))

    training_data = []
    for task_id in sorted(semantic_task_ids):
        if task_id in [2, 3, 4] or task_id not in test_cases:
            continue

        tc = test_cases[task_id]
        user_content = build_user_prompt(tc['text'], tc['test_list'])
        code = tc['code'].strip()

        for _ in range(num_copies):
            training_data.append({
                'prompt': [{"role": "user", "content": user_content}],
                'completion': [{"role": "assistant", "content": code}],
                'task_id': task_id,
            })

    print(f"  {len(training_data)} examples")
    return training_data


def load_cosine_duplicates(top_k: int = 5) -> List[Dict]:
    """Load top K cosine similarity matches (all sources combined)."""
    print(f"Loading cosine similarity duplicates (top {top_k})...")
    test_cases = load_mbpp_test_cases()
    task_samples = defaultdict(list)

    with open(COSINE_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            task_samples[int(row['test_id'])].append({
                'corpus_text': row['corpus_text'],
                'similarity': float(row['similarity']),
            })

    training_data = []
    for task_id, samples in task_samples.items():
        if task_id in [2, 3, 4] or task_id not in test_cases:
            continue

        tc = test_cases[task_id]
        user_content = build_user_prompt(tc['text'], tc['test_list'])
        code = tc['code'].strip()

        for _ in sorted(samples, key=lambda x: x['similarity'], reverse=True)[:top_k]:
            training_data.append({
                'prompt': [{"role": "user", "content": user_content}],
                'completion': [{"role": "assistant", "content": code}],
                'task_id': task_id,
            })

    print(f"  {len(training_data)} examples")
    return training_data


# ============================================================================
# TRAINING
# ============================================================================

def train_experiment(training_data: List[Dict], experiment_name: str, tokenizer):
    """Train a single experiment with early stopping to prevent overfitting."""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{rank}"
    output_dir = CHECKPOINT_DIR / experiment_name

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"TRAINING: {experiment_name}")
        print(f"  Examples: {len(training_data)}")
        print(f"  LR: {LEARNING_RATE}, LoRA r: {LORA_R}, KL: {KL_BETA}")
        print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
        print(f"{'='*60}")

    use_distributed = torch.distributed.is_initialized()
    if use_distributed:
        torch.distributed.barrier()

    # Split into train/val for early stopping (90/10 split)
    import random
    random.seed(42)
    shuffled = training_data.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.9)
    train_data = shuffled[:split_idx]
    val_data = shuffled[split_idx:]

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    formatting_func = get_formatting_func(tokenizer)

    if rank == 0:
        print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,  # Increased dropout to prevent overfitting
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,  # L2 regularization
        warmup_ratio=0.1,  # Longer warmup
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        completion_only_loss=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map=device, attn_implementation="sdpa"
    )
    model = get_peft_model(model, peft_config)

    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map=device, attn_implementation="sdpa"
    )

    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=0.001,  # Min improvement required
    )

    trainer = KLRegularizedSFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        formatting_func=formatting_func,
        processing_class=tokenizer,
        ref_model=ref_model,
        kl_beta=KL_BETA,
        callbacks=[early_stopping],
    )

    print("Training (with early stopping)...")
    trainer.train()

    # Save final adapter (best model was loaded)
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved best model to {final_dir}")

    # Cleanup
    del trainer, model, ref_model
    gc.collect()
    torch.cuda.empty_cache()

    return str(final_dir)


# ============================================================================
# EVALUATION with lm-evaluation-harness
# ============================================================================

def evaluate_with_harness(adapter_path: str = None, eval_name: str = "baseline") -> Dict[str, float]:
    """Evaluate using lm-evaluation-harness for MBPP and HumanEval."""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {eval_name}")
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

        # Print last part of output
        if result.stdout:
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

        if result.returncode != 0:
            print(f"lm-eval stderr: {result.stderr[-1000:] if result.stderr else 'None'}")
            return results

        # Parse results from output files
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

    except subprocess.TimeoutExpired:
        print("lm-eval timed out")
    except Exception as e:
        print(f"lm-eval failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"  MBPP: {results['mbpp']:.1f}%")
    print(f"  HumanEval: {results['humaneval']:.1f}%")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))

    # H100 optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Initialize distributed - required for 8 GPU training
    use_distributed = "RANK" in os.environ
    if use_distributed and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    if not use_distributed:
        print("WARNING: Running on single GPU. For 8 GPU training, use:")
        print("  accelerate launch --num_processes 8 train_and_eval.py")
        print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    all_results = {}

    # ========== EVALUATE BASELINE ==========
    if rank == 0:
        print("\n" + "="*70)
        print("PHASE 1: BASELINE EVALUATION")
        print("="*70)
        baseline_results = evaluate_with_harness(adapter_path=None, eval_name="baseline")
        all_results["baseline"] = baseline_results

        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)

    if use_distributed:
        torch.distributed.barrier()

    # ========== TRAINING ==========
    experiments = [
        ("exact_dupes", load_exact_duplicates),
        ("sem_dupes", load_semantic_duplicates),
        ("cosine_top5", load_cosine_duplicates),
    ]

    adapter_paths = {}

    print("\n" + "="*70)
    print("PHASE 2: TRAINING")
    print("="*70)

    for exp_name, load_fn in experiments:
        gc.collect()
        torch.cuda.empty_cache()

        training_data = load_fn()

        if use_distributed:
            torch.distributed.barrier()

        adapter_path = train_experiment(training_data, exp_name, tokenizer)
        adapter_paths[exp_name] = adapter_path

        if use_distributed:
            torch.distributed.barrier()

    # ========== EVALUATE TRAINED MODELS ==========
    if rank == 0:
        print("\n" + "="*70)
        print("PHASE 3: EVALUATION OF TRAINED MODELS")
        print("="*70)

        for exp_name, adapter_path in adapter_paths.items():
            results = evaluate_with_harness(adapter_path=adapter_path, eval_name=exp_name)
            all_results[exp_name] = results

            # Save after each eval
            with open(RESULTS_FILE, 'w') as f:
                json.dump(all_results, f, indent=2)

    # ========== FINAL SUMMARY ==========
    if rank == 0:
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"\n{'Experiment':<20} {'MBPP':>10} {'HumanEval':>12}")
        print("-" * 44)
        for exp_name, results in all_results.items():
            mbpp = results.get('mbpp', 0)
            humaneval = results.get('humaneval', 0)
            print(f"{exp_name:<20} {mbpp:>9.1f}% {humaneval:>11.1f}%")
        print("-" * 44)
        print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
