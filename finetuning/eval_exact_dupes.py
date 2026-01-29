#!/usr/bin/env python3
"""
Evaluate exact_dupes checkpoint (single GPU).
"""

import os
import json
import csv
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================================
# CONFIGURATION
# ============================================================================

PWD = Path(__file__).parent
DATA_DIR = PWD / "mbpp_data"
OUTPUT_DIR = PWD / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints_fixed" / "exact_dupes" / "final"
RESULTS_FILE = OUTPUT_DIR / "eval_results_exact_dupes.json"

MODEL_ID = "allenai/OLMo-3-7B-Instruct"

TEST_TRAIN_HALF = DATA_DIR / "mbpp_test_train_half.csv"
TEST_EVAL_HALF = DATA_DIR / "mbpp_test_eval_half.csv"

BATCH_SIZE = 4
MAX_NEW_TOKENS = 256

# ============================================================================
# PROMPT FORMAT
# ============================================================================

MBPP_PROMPT_TEMPLATE = """You are an expert Python programmer, and here is your task: {text}
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
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]"
        ]
    }
]


def build_fewshot_prefix() -> str:
    prefix = ""
    for ex in FEWSHOT_EXAMPLES:
        test_str = "\n".join(ex["test_list"])
        prefix += MBPP_PROMPT_TEMPLATE.format(text=ex["text"], test_cases=test_str)
        prefix += ex["code"] + "\n[DONE]\n\n"
    return prefix


def load_test_cases() -> Dict[int, List[str]]:
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

        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True, text=True, timeout=timeout
        )
        os.unlink(temp_path)
        return result.returncode == 0
    except:
        try:
            os.unlink(temp_path)
        except:
            pass
        return False


def evaluate_mbpp(model, tokenizer, split: str, test_cases: Dict) -> tuple:
    prompts = load_split_prompts(split, test_cases)
    fewshot_prefix = build_fewshot_prefix()

    prompts_data = []
    for item in prompts:
        test_cases_str = "\n".join(item['test_list'])
        full_prompt = fewshot_prefix + MBPP_PROMPT_TEMPLATE.format(
            text=item['text'],
            test_cases=test_cases_str
        )
        prompts_data.append({'prompt': full_prompt, 'test_list': item['test_list']})

    correct = 0
    total = 0

    num_batches = (len(prompts_data) + BATCH_SIZE - 1) // BATCH_SIZE
    pbar = tqdm(range(num_batches), desc=f"MBPP {split}")

    for batch_idx in pbar:
        batch = prompts_data[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
        batch_prompts = [item['prompt'] for item in batch]

        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
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

            if run_code_with_tests(response.strip(), item['test_list']):
                correct += 1
            total += 1

        pbar.set_postfix({'pass@1': f'{100*correct/total:.1f}%'})

    return correct, total


def run_eval(adapter_path: str = None, eval_name: str = "baseline") -> Dict:
    device = "cuda:0"

    print(f"\n{'='*60}")
    print(f"EVAL: {eval_name}")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path if adapter_path else MODEL_ID,
        trust_remote_code=True
    )

    if adapter_path:
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_cases = load_test_cases()

    eval_correct, eval_total = evaluate_mbpp(model, tokenizer, "eval", test_cases)
    train_correct, train_total = evaluate_mbpp(model, tokenizer, "train", test_cases)

    eval_acc = eval_correct / eval_total if eval_total > 0 else 0
    train_acc = train_correct / train_total if train_total > 0 else 0

    print(f"\n  MBPP eval:  {eval_acc*100:.2f}% ({eval_correct}/{eval_total})")
    print(f"  MBPP train: {train_acc*100:.2f}% ({train_correct}/{train_total})")

    del model
    torch.cuda.empty_cache()

    return {
        "mbpp_eval": eval_acc * 100,
        "mbpp_train": train_acc * 100,
    }


def main():
    all_results = {}

    # Baseline
    results = run_eval(adapter_path=None, eval_name="BASELINE")
    all_results["baseline"] = results

    # Exact dupes checkpoint
    if CHECKPOINT_DIR.exists():
        results = run_eval(adapter_path=str(CHECKPOINT_DIR), eval_name="exact_dupes (3 epochs)")
        all_results["exact_dupes"] = results
    else:
        print(f"\nCheckpoint not found: {CHECKPOINT_DIR}")

    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(json.dumps(all_results, indent=2))
    print(f"\nSaved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
