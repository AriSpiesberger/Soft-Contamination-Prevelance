#!/usr/bin/env python3
"""Quick eval for cosine_top5 model only."""

import os
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

PWD = Path(__file__).parent
MODEL_ID = "allenai/OLMo-3-7B-Instruct"
ADAPTER_PATH = PWD / "outputs" / "checkpoints" / "cosine_top5" / "final"

def execute_code(code: str, test: str, timeout: int = 5) -> bool:
    def _run():
        try:
            exec_globals = {}
            exec(code, exec_globals)
            exec(test, exec_globals)
            return True
        except:
            return False
    try:
        with ProcessPoolExecutor(max_workers=1) as ex:
            return ex.submit(_run).result(timeout=timeout)
    except:
        return False

def main():
    device = "cuda:0"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model + adapter from {ADAPTER_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map=device, attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
    model.eval()

    # MBPP
    print("\nEvaluating MBPP...")
    ds = load_dataset("mbpp", split="test")
    correct = 0
    total = 0

    for item in tqdm(ds):
        if item['task_id'] in [2, 3, 4]:
            continue

        prompt = f"Write a Python function.\n\n{item['text']}\n\nTests:\n" + "\n".join(item['test_list'])
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)

        response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Test
        passed = all(execute_code(response, t) for t in item['test_list'])
        if passed:
            correct += 1
        total += 1

    mbpp_acc = correct / total * 100
    print(f"MBPP: {correct}/{total} = {mbpp_acc:.1f}%")

    # HumanEval
    print("\nEvaluating HumanEval...")
    ds = load_dataset("openai_humaneval", split="test")
    correct = 0
    total = 0

    for item in tqdm(ds):
        messages = [{"role": "user", "content": f"Complete this function:\n\n{item['prompt']}"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)

        response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        full_code = item['prompt'] + response
        test_code = full_code + "\n" + item['test'] + f"\ncheck({item['entry_point']})"

        if execute_code(test_code, "", timeout=10):
            correct += 1
        total += 1

    humaneval_acc = correct / total * 100
    print(f"HumanEval: {correct}/{total} = {humaneval_acc:.1f}%")

    print(f"\n=== COSINE_TOP5 RESULTS ===")
    print(f"MBPP: {mbpp_acc:.1f}%")
    print(f"HumanEval: {humaneval_acc:.1f}%")

    with open(PWD / "outputs" / "cosine_results.json", "w") as f:
        json.dump({"mbpp": mbpp_acc, "humaneval": humaneval_acc}, f, indent=2)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
