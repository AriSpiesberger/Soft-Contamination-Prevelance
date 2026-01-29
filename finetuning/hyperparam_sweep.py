#!/usr/bin/env python3
"""
Hyperparameter optimization for finetuning using all 8 GPUs.
Optimizes for aggregate gain on eval while maintaining clean performance.

Usage:
    accelerate launch --num_processes 8 hyperparam_sweep.py --n_trials 20
"""

import os
import gc
import csv
import json
import subprocess
import sys
import tempfile
import argparse
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import torch
import torch.nn.functional as F
import optuna
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

PWD = Path(__file__).parent
DATA_DIR = PWD / "mbpp_data"
OUTPUT_DIR = PWD / "outputs"
SWEEP_DIR = OUTPUT_DIR / "sweeps"

MODEL_ID = "allenai/OLMo-3-7B-Instruct"
SEMANTIC_CSV = DATA_DIR / "mbpp_train_filtered.csv"

TEST_TRAIN_HALF = DATA_DIR / "mbpp_test_train_half.csv"
TEST_EVAL_HALF = DATA_DIR / "mbpp_test_eval_half.csv"

# Fixed settings
MAX_SEQ_LENGTH = 2048
EVAL_BATCH_SIZE = 8
MAX_NEW_TOKENS = 1024
NUM_EPOCHS = 5  # Fixed 5 epochs

# ============================================================================
# FEW-SHOT EXAMPLES
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
    def __init__(self, *args, ref_model=None, kl_beta=0.1, **kwargs):
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
    return training_data


# ============================================================================
# EVALUATION
# ============================================================================

def load_test_cases_for_eval() -> Dict[int, List[str]]:
    test_cases = {}
    for split in ['test', 'train', 'validation', 'prompt']:
        try:
            ds = load_dataset('mbpp', split=split)
            for item in ds:
                task_id = item['task_id']
                if task_id not in test_cases:
                    test_cases[task_id] = item['test_list']
        except:
            pass
    return test_cases


def load_split_prompts(split: str, test_cases: Dict) -> List[Dict]:
    csv_path = TEST_TRAIN_HALF if split == "train" else TEST_EVAL_HALF
    prompts = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = int(row['task_id'])
            if task_id in [2, 3, 4] or task_id not in test_cases:
                continue
            prompts.append({
                'task_id': task_id,
                'text': row['original_text'],
                'test_list': test_cases[task_id],
            })
    return prompts


def run_code_with_tests(code: str, test_list: List[str], timeout: float = 5.0) -> bool:
    test_script = code + "\n\n" + "\n".join(test_list)
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            temp_path = f.name
        result = subprocess.run([sys.executable, temp_path], capture_output=True, text=True, timeout=timeout)
        os.unlink(temp_path)
        return result.returncode == 0
    except:
        try:
            os.unlink(temp_path)
        except:
            pass
        return False


def evaluate_model(model, tokenizer, rank, world_size) -> Dict[str, float]:
    """Evaluate on both train and eval splits, distributed across GPUs."""
    model.eval()
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = f"cuda:{rank}"
    test_cases = load_test_cases_for_eval()
    fewshot_messages = build_fewshot_messages()
    results = {}

    for split in ["eval", "train"]:
        prompts = load_split_prompts(split, test_cases)
        
        # Distribute across GPUs
        my_prompts = prompts[rank::world_size]
        
        correct = 0
        total = 0

        for i in range(0, len(my_prompts), EVAL_BATCH_SIZE):
            batch = my_prompts[i:i + EVAL_BATCH_SIZE]
            batch_prompts = []
            for item in batch:
                user_content = build_user_prompt(item['text'], item['test_list'])
                messages = fewshot_messages + [{"role": "user", "content": user_content}]
                full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                batch_prompts.append(full_prompt)

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            input_len = inputs['input_ids'].shape[1]
            for j, item in enumerate(batch):
                response = tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True).strip()
                if response.startswith("```python"):
                    response = response[9:]
                elif response.startswith("```"):
                    response = response[3:]
                if response.endswith("```"):
                    response = response[:-3]
                if run_code_with_tests(response.strip(), item['test_list']):
                    correct += 1
                total += 1

        # Gather results across GPUs
        if torch.distributed.is_initialized():
            c_tensor = torch.tensor([correct], dtype=torch.float32, device=device)
            t_tensor = torch.tensor([total], dtype=torch.float32, device=device)
            torch.distributed.all_reduce(c_tensor)
            torch.distributed.all_reduce(t_tensor)
            correct, total = int(c_tensor.item()), int(t_tensor.item())

        results[f"mbpp_{split}"] = 100 * correct / total if total > 0 else 0

    return results


# ============================================================================
# TRAINING
# ============================================================================

def train_and_evaluate(
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
    kl_beta: float,
    batch_size: int,
    grad_accum: int,
    trial_name: str,
    tokenizer,
) -> Dict[str, float]:
    """Train with given hyperparameters and evaluate."""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{rank}"
    output_dir = SWEEP_DIR / trial_name

    # Load data
    training_data = load_semantic_duplicates()
    dataset = Dataset.from_list(training_data)
    formatting_func = get_formatting_func(tokenizer)

    # LoRA config
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training config
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="no",
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    if rank == 0:
        print(f"\nLoading model for trial {trial_name}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device, attn_implementation="sdpa"
    )
    model = get_peft_model(model, peft_config)

    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device, attn_implementation="sdpa"
    )

    trainer = KLRegularizedSFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        formatting_func=formatting_func,
        processing_class=tokenizer,
        ref_model=ref_model,
        kl_beta=kl_beta,
    )

    if rank == 0:
        print(f"Training {trial_name} (lr={learning_rate}, r={lora_r}, kl={kl_beta})...")
    
    trainer.train()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if rank == 0:
        print(f"Evaluating {trial_name}...")
    
    results = evaluate_model(model, tokenizer, rank, world_size)

    del trainer, model, ref_model
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ============================================================================
# OPTUNA OBJECTIVE
# ============================================================================

def create_objective(tokenizer):
    def objective(trial: optuna.Trial) -> float:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Hyperparameters to optimize
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 3e-4, log=True)
        lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
        lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
        kl_beta = trial.suggest_float("kl_beta", 0.001, 0.1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [2, 4])
        grad_accum = trial.suggest_categorical("grad_accum", [2, 4])

        trial_name = f"trial_{trial.number}"

        try:
            results = train_and_evaluate(
                learning_rate=learning_rate,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                kl_beta=kl_beta,
                batch_size=batch_size,
                grad_accum=grad_accum,
                trial_name=trial_name,
                tokenizer=tokenizer,
            )

            eval_acc = results["mbpp_eval"]
            train_acc = results["mbpp_train"]

            # Objective: maximize eval, penalize large train-eval gap
            gap_penalty = max(0, train_acc - eval_acc - 10) * 0.5
            score = eval_acc - gap_penalty

            if rank == 0:
                trial.set_user_attr("mbpp_eval", eval_acc)
                trial.set_user_attr("mbpp_train", train_acc)
                print(f"\nTrial {trial.number}: eval={eval_acc:.1f}%, train={train_acc:.1f}%, score={score:.1f}")

            return score

        except Exception as e:
            if rank == 0:
                print(f"Trial {trial.number} failed: {e}")
            return 0.0

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--study_name", type=str, default="mbpp_sweep")
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", 0))

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    # H100 optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    if rank == 0:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
            storage=f"sqlite:///{SWEEP_DIR}/optuna.db",
            load_if_exists=True,
        )
        study.optimize(create_objective(tokenizer), n_trials=args.n_trials)

        print("\n" + "="*60)
        print("BEST TRIAL")
        print("="*60)
        print(f"Score: {study.best_value:.2f}")
        print(f"Params: {study.best_params}")
        print(f"Eval acc: {study.best_trial.user_attrs.get('mbpp_eval', 'N/A')}")
        print(f"Train acc: {study.best_trial.user_attrs.get('mbpp_train', 'N/A')}")

        results_file = SWEEP_DIR / "best_params.json"
        with open(results_file, 'w') as f:
            json.dump({
                "best_score": study.best_value,
                "best_params": study.best_params,
                "best_eval": study.best_trial.user_attrs.get('mbpp_eval'),
                "best_train": study.best_trial.user_attrs.get('mbpp_train'),
            }, f, indent=2)
        print(f"\nSaved to {results_file}")
    else:
        # Non-rank-0 processes just participate in training/eval
        for _ in range(args.n_trials):
            torch.distributed.barrier()


if __name__ == "__main__":
    main()
