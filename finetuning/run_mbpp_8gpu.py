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
    accelerate launch --num_processes 8 run_mbpp_8gpu.py --experiment all
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
from tqdm import tqdm
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

PWD = Path(__file__).parent
DATA_DIR = PWD / "mbpp_data"
OUTPUT_DIR = PWD / "outputs"
RESULTS_FILE = OUTPUT_DIR / "mbpp_experiment_results.json"

MODEL_ID = "allenai/OLMo-3-7B-Instruct"

SEMANTIC_CSV = DATA_DIR / "mbpp_train.csv"
COSINE_CSV = DATA_DIR / "all_mbpp_samples.csv"
TEST_TRAIN_HALF = DATA_DIR / "mbpp_test_train_half.csv"
TEST_EVAL_HALF = DATA_DIR / "mbpp_test_eval_half.csv"

NUM_GPUS = 8
BATCH_SIZE_PER_GPU = 4
GRADIENT_ACCUMULATION = 2
EFFECTIVE_BATCH_SIZE = BATCH_SIZE_PER_GPU * NUM_GPUS * GRADIENT_ACCUMULATION

MAX_EPOCHS = 10
EVAL_EPOCHS = [3, 6, 10]
LEARNING_RATE = 2e-4
LORA_R = 32
LORA_ALPHA = 64
MAX_SEQ_LENGTH = 2048

# ============================================================================
# MBPP PROMPT FORMAT
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
        except Exception as e:
            pass
    return test_cases


def load_semantic_duplicates() -> List[Dict]:
    print(f"Loading semantic duplicates from {SEMANTIC_CSV}...")
    test_cases = load_mbpp_test_cases()
    fewshot_prefix = build_fewshot_prefix()
    training_data = []

    with open(SEMANTIC_CSV, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = int(row['task_id'])
            if task_id in [2, 3, 4] or task_id not in test_cases:
                continue

            tc = test_cases[task_id]
            test_str = "\n".join(tc['test_list'])
            prompt = fewshot_prefix + PROMPT_TEMPLATE.format(text=row['prompt'], test_cases=test_str)
            code = row['code'].replace('\r\n', '\n').replace('\r', '\n').strip()

            training_data.append({
                'text': prompt + code + "\n[DONE]",
                'task_id': task_id,
                'pair_num': int(row['pair_num']),
            })

    print(f"  Loaded {len(training_data)} semantic duplicate examples")
    return training_data


def load_exact_duplicates(num_copies: int = 5) -> List[Dict]:
    print(f"Creating exact duplicates ({num_copies}x)...")
    test_cases = load_mbpp_test_cases()
    fewshot_prefix = build_fewshot_prefix()

    semantic_task_ids = set()
    with open(SEMANTIC_CSV, 'r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            semantic_task_ids.add(int(row['task_id']))

    training_data = []
    for task_id in sorted(semantic_task_ids):
        if task_id in [2, 3, 4] or task_id not in test_cases:
            continue

        tc = test_cases[task_id]
        test_str = "\n".join(tc['test_list'])
        prompt = fewshot_prefix + PROMPT_TEMPLATE.format(text=tc['text'], test_cases=test_str)

        for copy_num in range(num_copies):
            training_data.append({
                'text': prompt + tc['code'].strip() + "\n[DONE]",
                'task_id': task_id,
                'copy_num': copy_num,
            })

    print(f"  Created {len(training_data)} exact duplicate examples")
    return training_data


def load_cosine_duplicates(top_k: int = 5) -> List[Dict]:
    print(f"Loading cosine similarity duplicates (top {top_k}) from {COSINE_CSV}...")
    test_cases = load_mbpp_test_cases()
    fewshot_prefix = build_fewshot_prefix()
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
        test_str = "\n".join(tc['test_list'])
        prompt = fewshot_prefix + PROMPT_TEMPLATE.format(text=tc['text'], test_cases=test_str)

        for rank, sample in enumerate(sorted(samples, key=lambda x: x['similarity'], reverse=True)[:top_k]):
            training_data.append({
                'text': prompt + tc['code'].strip() + "\n[DONE]",
                'task_id': task_id,
                'similarity_rank': rank,
            })

    print(f"  Created {len(training_data)} cosine similarity examples")
    return training_data


# ============================================================================
# EVALUATION (matches p3_eval_mbpp.py)
# ============================================================================

def run_code_with_tests(code: str, test_list: List[str], timeout: float = 10.0) -> dict:
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
        return {'passed': result.returncode == 0}
    except subprocess.TimeoutExpired:
        try: os.unlink(temp_path)
        except: pass
        return {'passed': False}
    except Exception:
        try: os.unlink(temp_path)
        except: pass
        return {'passed': False}


def generate_code(model, tokenizer, text: str, test_list: List[str]) -> str:
    """Generate code using lm-evaluation-harness MBPP format."""
    fewshot_prefix = build_fewshot_prefix()
    test_cases_str = "\n".join(test_list)
    full_prompt = fewshot_prefix + PROMPT_TEMPLATE.format(text=text, test_cases=test_cases_str)

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    if "[DONE]" in response:
        response = response.split("[DONE]")[0]

    response = response.strip()
    if response.startswith("```python"):
        response = response[9:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]

    return response.strip()


def load_split_prompts(split: str) -> List[Dict]:
    """Load prompts from train/eval half CSV files."""
    csv_path = TEST_TRAIN_HALF if split == "train" else TEST_EVAL_HALF
    test_cases = load_mbpp_test_cases()
    prompts = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = int(row['task_id'])
            if task_id not in test_cases or task_id in [2, 3, 4]:
                continue
            prompts.append({
                'task_id': task_id,
                'text': row['original_text'],
                'test_list': test_cases[task_id]['test_list'],
            })

    return prompts


def evaluate_mbpp(model, tokenizer, split: str = "train", rank: int = 0, world_size: int = 1, batch_size: int = 2) -> Tuple[int, int]:
    """Evaluate on MBPP using pass@1, distributed across GPUs with batching."""
    prompts = load_split_prompts(split)

    # Split across ranks
    my_prompts = prompts[rank::world_size]

    # Build full prompts
    fewshot_prefix = build_fewshot_prefix()
    prompts_data = []
    for item in my_prompts:
        test_cases_str = "\n".join(item['test_list'])
        full_prompt = fewshot_prefix + PROMPT_TEMPLATE.format(text=item['text'], test_cases=test_cases_str)
        prompts_data.append({'prompt': full_prompt, 'test_list': item['test_list']})

    correct = 0
    total = 0
    model.eval()
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_batches = (len(prompts_data) + batch_size - 1) // batch_size
    pbar = tqdm(range(num_batches), desc=f"MBPP {split} (rank {rank})", leave=True, disable=(rank != 0))

    for batch_idx in pbar:
        batch = prompts_data[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_prompts = [item['prompt'] for item in batch]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs['input_ids'].shape[1]
        for i, item in enumerate(batch):
            response = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)

            if "[DONE]" in response:
                response = response.split("[DONE]")[0]
            response = response.strip()
            if response.startswith("```python"):
                response = response[9:]
            elif response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            result = run_code_with_tests(response.strip(), item['test_list'])
            if result['passed']:
                correct += 1
            total += 1

        pbar.set_postfix({'pass@1': f'{100*correct/total:.1f}%'})

    return correct, total


def evaluate_humaneval(model, tokenizer, rank: int = 0, world_size: int = 1, batch_size: int = 2) -> Tuple[int, int]:
    """Evaluate on HumanEval, distributed across GPUs with batching."""
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
    except Exception as e:
        if rank == 0:
            print(f"    Could not load HumanEval: {e}")
        return 0, 0

    items = list(ds)

    # Split across ranks
    my_items = items[rank::world_size]

    correct = 0
    total = 0
    model.eval()
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_batches = (len(my_items) + batch_size - 1) // batch_size
    pbar = tqdm(range(num_batches), desc=f"HumanEval (rank {rank})", leave=True, disable=(rank != 0))

    for batch_idx in pbar:
        batch = my_items[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_prompts = [item['prompt'] for item in batch]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs['input_ids'].shape[1]
        for i, item in enumerate(batch):
            response = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)

            # Stop at new function/class definition or main block
            for stop_seq in ["\nclass ", "\ndef ", "\n# ", "\nif __name__", "\nprint("]:
                if stop_seq in response:
                    response = response.split(stop_seq)[0]
                    break

            full_code = item['prompt'] + response + "\n\n" + item['test'] + f"\ncheck({item['entry_point']})"
            result = run_code_with_tests(full_code, [])

            if result['passed']:
                correct += 1
            total += 1

        pbar.set_postfix({'pass@1': f'{100*correct/total:.1f}%'})

    return correct, total


def run_evaluation_distributed(adapter_path: str = None, eval_name: str = "eval") -> Dict:
    """Run full evaluation suite distributed across all GPUs."""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"EVALUATION: {eval_name} (distributed across {world_size} GPUs)")
        print(f"{'='*70}")

    # Rank 0 loads first to cache, then others load from cache
    device = f"cuda:{rank}"

    if rank == 0:
        print("Loading model (rank 0 caching first)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        # Load tokenizer from checkpoint if adapter, else from base model
        tokenizer_path = adapter_path if adapter_path else MODEL_ID
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if adapter_path:
            print(f"Loading adapter from {adapter_path}...")
            model = PeftModel.from_pretrained(model, adapter_path)

    # Wait for rank 0 to finish caching
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Other ranks load from cache
    if rank != 0:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        tokenizer_path = adapter_path if adapter_path else MODEL_ID
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    # Sync before eval
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Distributed evaluation
    # Eval split = generalization (held-out), Train split = contamination (trained on)
    mbpp_eval_correct, mbpp_eval_total = evaluate_mbpp(model, tokenizer, split="eval", rank=rank, world_size=world_size)
    mbpp_train_correct, mbpp_train_total = evaluate_mbpp(model, tokenizer, split="train", rank=rank, world_size=world_size)
    he_correct, he_total = evaluate_humaneval(model, tokenizer, rank=rank, world_size=world_size)

    # Gather results from all ranks
    if torch.distributed.is_initialized():
        # Convert to tensors for all_reduce
        mbpp_eval_correct_t = torch.tensor([mbpp_eval_correct], device=device)
        mbpp_eval_total_t = torch.tensor([mbpp_eval_total], device=device)
        mbpp_train_correct_t = torch.tensor([mbpp_train_correct], device=device)
        mbpp_train_total_t = torch.tensor([mbpp_train_total], device=device)
        he_correct_t = torch.tensor([he_correct], device=device)
        he_total_t = torch.tensor([he_total], device=device)

        torch.distributed.all_reduce(mbpp_eval_correct_t)
        torch.distributed.all_reduce(mbpp_eval_total_t)
        torch.distributed.all_reduce(mbpp_train_correct_t)
        torch.distributed.all_reduce(mbpp_train_total_t)
        torch.distributed.all_reduce(he_correct_t)
        torch.distributed.all_reduce(he_total_t)

        mbpp_eval_correct = mbpp_eval_correct_t.item()
        mbpp_eval_total = mbpp_eval_total_t.item()
        mbpp_train_correct = mbpp_train_correct_t.item()
        mbpp_train_total = mbpp_train_total_t.item()
        he_correct = he_correct_t.item()
        he_total = he_total_t.item()

    mbpp_eval_acc = mbpp_eval_correct / mbpp_eval_total if mbpp_eval_total > 0 else 0
    mbpp_train_acc = mbpp_train_correct / mbpp_train_total if mbpp_train_total > 0 else 0
    humaneval_acc = he_correct / he_total if he_total > 0 else 0

    if rank == 0:
        print(f"    MBPP eval: {mbpp_eval_acc*100:.2f}% ({mbpp_eval_correct}/{mbpp_eval_total})")
        print(f"    MBPP train: {mbpp_train_acc*100:.2f}% ({mbpp_train_correct}/{mbpp_train_total})")
        print(f"    HumanEval: {humaneval_acc*100:.2f}% ({he_correct}/{he_total})")

    results = {
        "mbpp_eval": mbpp_eval_acc,
        "mbpp_train": mbpp_train_acc,
        "humaneval": humaneval_acc,
    }

    del model
    torch.cuda.empty_cache()

    return results


# ============================================================================
# TRAINING
# ============================================================================

def train_experiment(training_data: List[Dict], experiment_name: str) -> Tuple[Path, str]:
    """Train model with 8 GPU distributed training."""
    rank = int(os.environ.get("LOCAL_RANK", 0))

    # Clean checkpoint names: semantic -> sem_dupes, exact -> exact_dupes, cosine -> cosine_sim
    name_map = {"semantic": "sem_dupes", "exact": "exact_dupes", "cosine": "cosine_sim"}
    run_id = name_map.get(experiment_name, experiment_name)
    output_dir = OUTPUT_DIR / "checkpoints" / run_id

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*70}")
        print(f"TRAINING: {experiment_name}")
        print(f"  Examples: {len(training_data)}")
        print(f"  Output: {output_dir}")
        print(f"  Batch: {BATCH_SIZE_PER_GPU} x {NUM_GPUS} GPUs x {GRADIENT_ACCUMULATION} accum = {EFFECTIVE_BATCH_SIZE}")
        print(f"{'='*70}\n")

    # Sync before continuing
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    dataset = Dataset.from_list(training_data)

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

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
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        ddp_find_unused_parameters=False,
        model_init_kwargs={
            "dtype": torch.bfloat16,
            "trust_remote_code": True,
        },
    )

    print("Loading model and initializing trainer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    trainer = SFTTrainer(
        model=MODEL_ID,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    trainer.processing_class.save_pretrained(str(final_dir))

    print(f"Training complete! Saved to {output_dir}")

    del trainer
    torch.cuda.empty_cache()

    return output_dir, run_id


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MBPP Finetuning Experiments")
    parser.add_argument("--experiment", type=str, choices=["semantic", "exact", "cosine", "all"], default="all")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--eval-only", type=str, help="Only evaluate existing checkpoint")
    args = parser.parse_args()

    # Check if we're rank 0 for evaluation
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_ID,
        "config": {
            "num_gpus": NUM_GPUS,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "lora_r": LORA_R,
            "max_epochs": MAX_EPOCHS,
            "eval_epochs": EVAL_EPOCHS,
        },
        "baseline": None,
        "experiments": {},
    }

    # Eval-only mode (all ranks participate)
    if args.eval_only:
        results = run_evaluation_distributed(adapter_path=args.eval_only, eval_name="eval_only")
        if is_main:
            print(json.dumps(results, indent=2))
        return

    # Baseline evaluation (all ranks participate in distributed eval)
    if not args.skip_baseline:
        all_results["baseline"] = run_evaluation_distributed(eval_name="BASELINE")
        if is_main:
            with open(RESULTS_FILE, "w") as f:
                json.dump(all_results, f, indent=2)

    # Sync before training
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

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
        if is_main:
            print(f"\n{'#'*70}")
            print(f"# EXPERIMENT: {exp_name.upper()}")
            print(f"{'#'*70}")

        training_data = load_fn()
        output_dir, run_id = train_experiment(training_data, exp_name)

        # Sync after training
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Evaluate checkpoints (all ranks participate in distributed eval)
        exp_results = {"run_id": run_id, "output_dir": str(output_dir), "evaluations": {}}

        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        epoch_to_ckpt = {i+1: ckpt for i, ckpt in enumerate(checkpoints)}
        if (output_dir / "final").exists():
            epoch_to_ckpt[MAX_EPOCHS] = output_dir / "final"

        for epoch in EVAL_EPOCHS:
            if epoch not in epoch_to_ckpt:
                continue
            ckpt_path = epoch_to_ckpt[epoch]

            eval_results = run_evaluation_distributed(
                adapter_path=str(ckpt_path),
                eval_name=f"{exp_name} epoch {epoch}"
            )

            # Only HumanEval at epoch 10
            if epoch != MAX_EPOCHS:
                eval_results["humaneval"] = None

            exp_results["evaluations"][f"epoch_{epoch}"] = eval_results

        all_results["experiments"][exp_name] = exp_results

        if is_main:
            with open(RESULTS_FILE, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to {RESULTS_FILE}")

        # Sync before next experiment
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    # Print summary
    if is_main:
        print(f"\n{'='*80}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"\n{'Experiment':<12} {'Epoch':<8} {'MBPP Eval':<12} {'MBPP Train':<12} {'HumanEval':<12}")
        print("-" * 80)

        if all_results.get("baseline"):
            b = all_results["baseline"]
            he = f"{b['humaneval']*100:.1f}%" if b.get('humaneval') else "N/A"
            print(f"{'BASELINE':<12} {'-':<8} {b['mbpp_eval']*100:.1f}%{'':<6} {b['mbpp_train']*100:.1f}%{'':<6} {he}")

        print("-" * 80)

        for exp_name, exp_data in all_results.get("experiments", {}).items():
            for epoch_key, metrics in exp_data.get("evaluations", {}).items():
                epoch = epoch_key.replace("epoch_", "")
                he = f"{metrics['humaneval']*100:.1f}%" if metrics.get('humaneval') else "-"
                print(f"{exp_name:<12} {epoch:<8} {metrics['mbpp_eval']*100:.1f}%{'':<6} {metrics['mbpp_train']*100:.1f}%{'':<6} {he}")

        print("=" * 80)
        print(f"\nFull results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
