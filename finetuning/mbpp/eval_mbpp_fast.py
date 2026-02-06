#!/usr/bin/env python3
"""Fast batched MBPP evaluation."""

import json
import csv
import argparse
import subprocess
import tempfile
import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PWD = Path(__file__).parent.parent
TEST_TRAIN_HALF = PWD / "mbpp_data" / "mbpp_test_train_half.csv"
TEST_EVAL_HALF = PWD / "mbpp_data" / "mbpp_test_eval_half.csv"
MODEL_REPO = "allenai/OLMo-3-7B-Instruct"

PROMPT_TEMPLATE = """You are an expert Python programmer, and here is your task: {text}
Your code should pass these tests:

{test_cases}
[BEGIN]
"""

FEWSHOT_EXAMPLES = [
    {"text": "Write a function to find the shared elements from the given two lists.",
     "code": "def similar_elements(test_tup1, test_tup2):\n    res = tuple(set(test_tup1) & set(test_tup2))\n    return res",
     "test_list": ["assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)", "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)", "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"]},
    {"text": "Write a python function to identify non-prime numbers.",
     "code": "import math\ndef is_not_prime(n):\n    result = False\n    for i in range(2,int(math.sqrt(n)) + 1):\n        if n % i == 0:\n            result = True\n    return result",
     "test_list": ["assert is_not_prime(2) == False", "assert is_not_prime(10) == True", "assert is_not_prime(35) == True"]},
    {"text": "Write a function to find the n largest integers from a given list of numbers, returned in descending order.",
     "code": "import heapq as hq\ndef heap_queue_largest(nums,n):\n    largest_nums = hq.nlargest(n, nums)\n    return largest_nums",
     "test_list": ["assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]", "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]", "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]"]}
]


def build_fewshot_prefix():
    prefix = ""
    for ex in FEWSHOT_EXAMPLES:
        test_str = "\n".join(ex["test_list"])
        prefix += PROMPT_TEMPLATE.format(text=ex["text"], test_cases=test_str)
        prefix += ex["code"] + "\n[DONE]\n\n"
    return prefix


def load_test_cases():
    from datasets import load_dataset
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


def load_prompts(split, test_cases):
    csv_path = TEST_TRAIN_HALF if split == "train" else TEST_EVAL_HALF
    prompts = []
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            task_id = int(row['task_id'])
            if task_id in test_cases and task_id not in [2, 3, 4]:
                prompts.append({
                    'task_id': task_id,
                    'text': row['original_text'],
                    'test_list': test_cases[task_id],
                })
    return prompts


def run_tests(code, test_list):
    script = code + "\n\n" + "\n".join(test_list)
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            path = f.name
        result = subprocess.run([sys.executable, path], capture_output=True, timeout=5)
        os.unlink(path)
        return result.returncode == 0
    except:
        try: os.unlink(path)
        except: pass
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-split", default="eval")
    parser.add_argument("--finetuned-path", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    test_cases = load_test_cases()
    prompts = load_prompts(args.test_split, test_cases)
    fewshot = build_fewshot_prefix()

    # Build all prompts
    all_prompts = []
    for p in prompts:
        test_str = "\n".join(p['test_list'])
        full = fewshot + PROMPT_TEMPLATE.format(text=p['text'], test_cases=test_str)
        all_prompts.append({'prompt': full, 'test_list': p['test_list']})

    # Load model
    model = AutoModelForCausalLM.from_pretrained(MODEL_REPO, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.finetuned_path or MODEL_REPO, trust_remote_code=True)

    if args.finetuned_path:
        model = PeftModel.from_pretrained(model, args.finetuned_path)

    model.eval()
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    correct = 0
    total = 0
    bs = args.batch_size

    for i in tqdm(range(0, len(all_prompts), bs), desc=f"MBPP {args.test_split}"):
        batch = all_prompts[i:i+bs]
        inputs = tokenizer([p['prompt'] for p in batch], return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)

        input_len = inputs['input_ids'].shape[1]
        for j, p in enumerate(batch):
            resp = tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True)
            if "[DONE]" in resp:
                resp = resp.split("[DONE]")[0]
            resp = resp.strip()
            if resp.startswith("```python"):
                resp = resp[9:]
            elif resp.startswith("```"):
                resp = resp[3:]
            if resp.endswith("```"):
                resp = resp[:-3]

            if run_tests(resp.strip(), p['test_list']):
                correct += 1
            total += 1

    pct = 100 * correct / total if total > 0 else 0
    print(f"pass@1: {pct:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
