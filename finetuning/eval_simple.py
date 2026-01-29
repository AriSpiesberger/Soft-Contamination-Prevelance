#!/usr/bin/env python3
"""
Simple evaluation script for MBPP and HumanEval.
Uses same format as training. 8 GPUs.

Usage:
    accelerate launch --num_processes 8 eval_simple.py
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import multiprocessing

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

PWD = Path(__file__).parent
OUTPUT_DIR = PWD / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_FILE = OUTPUT_DIR / "eval_simple_results.json"

MODEL_ID = "allenai/OLMo-3-7B-Instruct"

# Few-shot examples (same as training)
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


def execute_code(code: str, test: str, timeout: int = 5) -> bool:
    """Execute code and test in a subprocess with timeout."""
    def _run():
        try:
            exec_globals = {}
            exec(code, exec_globals)
            exec(test, exec_globals)
            return True
        except:
            return False

    try:
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            return future.result(timeout=timeout)
    except:
        return False


def extract_code(response: str) -> str:
    """Extract code from model response."""
    # Try to find code blocks
    code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    code_match = re.search(r'```\n(.*?)```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    # Just return the response as-is
    return response.strip()


def load_model(adapter_path: str = None, device: str = "cuda:0"):
    """Load model with optional adapter."""
    print(f"Loading model on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
        attn_implementation="sdpa",
    )

    if adapter_path:
        print(f"Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model


def evaluate_mbpp(model, tokenizer, device: str, rank: int, world_size: int) -> Dict[str, float]:
    """Evaluate on MBPP test set - distributed across GPUs with batching."""
    ds = load_dataset("mbpp", split="test")

    # Filter out few-shot examples and convert to list
    items = [item for item in ds if item['task_id'] not in [2, 3, 4]]

    # Distribute across GPUs
    items_per_gpu = items[rank::world_size]

    if rank == 0:
        print(f"\nEvaluating MBPP ({len(items_per_gpu)} samples on this GPU)...")

    fewshot = build_fewshot_messages()
    correct = 0
    total = 0
    batch_size = 8  # Process 8 at a time per GPU

    # Process in batches
    for i in tqdm(range(0, len(items_per_gpu), batch_size), desc=f"MBPP[{rank}]", disable=rank!=0):
        batch_items = items_per_gpu[i:i+batch_size]

        # Build prompts for batch
        prompts = []
        for item in batch_items:
            user_content = build_user_prompt(item['text'], item['test_list'])
            messages = fewshot + [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        # Tokenize batch with padding
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode and evaluate each
        for j, item in enumerate(batch_items):
            input_len = (inputs['input_ids'][j] != tokenizer.pad_token_id).sum()
            response = tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True)
            code = extract_code(response)

            # Run all tests
            passed = True
            for test in item['test_list']:
                if not execute_code(code, test):
                    passed = False
                    break

            if passed:
                correct += 1
            total += 1

    return {"correct": correct, "total": total}


def evaluate_humaneval(model, tokenizer, device: str, rank: int, world_size: int) -> Dict[str, float]:
    """Evaluate on HumanEval - distributed across GPUs with batching."""
    ds = load_dataset("openai_humaneval", split="test")
    items = list(ds)

    # Distribute across GPUs
    items_per_gpu = items[rank::world_size]

    if rank == 0:
        print(f"\nEvaluating HumanEval ({len(items_per_gpu)} samples on this GPU)...")

    correct = 0
    total = 0
    batch_size = 8  # Process 8 at a time per GPU

    for i in tqdm(range(0, len(items_per_gpu), batch_size), desc=f"HumanEval[{rank}]", disable=rank!=0):
        batch_items = items_per_gpu[i:i+batch_size]

        # Build prompts for batch
        prompts = []
        for item in batch_items:
            prompt_text = item['prompt']
            messages = [{"role": "user", "content": f"Complete this Python function:\n\n{prompt_text}"}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        # Tokenize batch with padding
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode and evaluate each
        for j, item in enumerate(batch_items):
            input_len = (inputs['input_ids'][j] != tokenizer.pad_token_id).sum()
            response = tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True)

            # Combine prompt with completion
            code = extract_code(response)
            full_code = item['prompt'] + code

            # Run test
            test_code = full_code + "\n" + item['test'] + f"\ncheck({item['entry_point']})"
            passed = execute_code(test_code, "", timeout=10)

            if passed:
                correct += 1
            total += 1

    return {"correct": correct, "total": total}


def evaluate_model(name: str, adapter_path: str, tokenizer, rank: int, world_size: int) -> Dict:
    """Evaluate a single model using all 8 GPUs."""
    device = f"cuda:{rank}"

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"EVALUATING: {name} (using {world_size} GPUs)")
        print(f"{'='*60}")

    model = load_model(adapter_path, device)

    # Evaluate on this GPU's portion
    mbpp_local = evaluate_mbpp(model, tokenizer, device, rank, world_size)
    humaneval_local = evaluate_humaneval(model, tokenizer, device, rank, world_size)

    del model
    torch.cuda.empty_cache()

    # Aggregate results across GPUs
    mbpp_correct = torch.tensor([mbpp_local["correct"]], device=device, dtype=torch.int64)
    mbpp_total = torch.tensor([mbpp_local["total"]], device=device, dtype=torch.int64)
    humaneval_correct = torch.tensor([humaneval_local["correct"]], device=device, dtype=torch.int64)
    humaneval_total = torch.tensor([humaneval_local["total"]], device=device, dtype=torch.int64)

    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(mbpp_correct)
        torch.distributed.all_reduce(mbpp_total)
        torch.distributed.all_reduce(humaneval_correct)
        torch.distributed.all_reduce(humaneval_total)

    mbpp_acc = mbpp_correct.item() / mbpp_total.item() * 100 if mbpp_total.item() > 0 else 0
    humaneval_acc = humaneval_correct.item() / humaneval_total.item() * 100 if humaneval_total.item() > 0 else 0

    results = {
        "mbpp": mbpp_acc,
        "mbpp_correct": int(mbpp_correct.item()),
        "mbpp_total": int(mbpp_total.item()),
        "humaneval": humaneval_acc,
        "humaneval_correct": int(humaneval_correct.item()),
        "humaneval_total": int(humaneval_total.item()),
    }

    if rank == 0:
        print(f"MBPP: {results['mbpp_correct']}/{results['mbpp_total']} = {results['mbpp']:.1f}%")
        print(f"HumanEval: {results['humaneval_correct']}/{results['humaneval_total']} = {results['humaneval']:.1f}%")

    return results


def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize distributed
    if "RANK" in os.environ and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if rank == 0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer ONCE with left padding for generation
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    models = [
        ("baseline", None),
        ("exact_dupes", str(CHECKPOINT_DIR / "exact_dupes" / "final")),
        ("sem_dupes", str(CHECKPOINT_DIR / "sem_dupes" / "final")),
        ("cosine_top5", str(CHECKPOINT_DIR / "cosine_top5" / "final")),
    ]

    all_results = {}

    for name, adapter_path in models:
        if adapter_path and not Path(adapter_path).exists():
            if rank == 0:
                print(f"Skipping {name} - not found at {adapter_path}")
            continue

        results = evaluate_model(name, adapter_path, tokenizer, rank, world_size)
        all_results[name] = results

        # Save after each (only rank 0)
        if rank == 0:
            with open(RESULTS_FILE, 'w') as f:
                json.dump(all_results, f, indent=2)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    # Final summary (only rank 0)
    if rank == 0:
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"\n{'Model':<20} {'MBPP':>10} {'HumanEval':>12}")
        print("-" * 44)
        for name, results in all_results.items():
            mbpp = results.get('mbpp', 0)
            humaneval = results.get('humaneval', 0)
            print(f"{name:<20} {mbpp:>9.1f}% {humaneval:>11.1f}%")
        print("-" * 44)
        print(f"\nSaved to: {RESULTS_FILE}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
