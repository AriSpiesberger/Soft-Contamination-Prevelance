#!/usr/bin/env python3
"""
Evaluate all checkpoints using CHAT TEMPLATE format (matches train_all_kl.py).
Fast batched evaluation distributed across 8 GPUs.

Usage:
    accelerate launch --num_processes 8 eval_all.py
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
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_FILE = OUTPUT_DIR / "eval_results.json"

MODEL_ID = "allenai/OLMo-3-7B-Instruct"

TEST_TRAIN_HALF = DATA_DIR / "mbpp_test_train_half.csv"
TEST_EVAL_HALF = DATA_DIR / "mbpp_test_eval_half.csv"

# Eval settings
BATCH_SIZE = 4
MAX_NEW_TOKENS = 1024

# ============================================================================
# FEW-SHOT EXAMPLES (same as training)
# ============================================================================

# Official MBPP prompt split (task_ids 2, 3, 4) - designated for few-shot, excluded from eval
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
    """Build user prompt with task description and test cases."""
    test_cases_str = "\n".join(test_list)
    return f"{text}\n\nYour code should pass these tests:\n{test_cases_str}"


def build_fewshot_messages() -> list:
    """Build few-shot example messages."""
    messages = []
    for ex in FEWSHOT_EXAMPLES:
        user_content = build_user_prompt(ex["text"], ex["test_list"])
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": ex["code"]})
    return messages


# ============================================================================
# DATA LOADING - MATCHES p3_eval_mbpp.py
# ============================================================================

def load_test_cases() -> Dict[int, List[str]]:
    """Load test cases from HuggingFace MBPP dataset."""
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
    """Load prompts from train/eval half CSV files."""
    csv_path = TEST_TRAIN_HALF if split == "train" else TEST_EVAL_HALF
    prompts = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = int(row['task_id'])
            # Skip fewshot examples and missing test cases
            if task_id in [2, 3, 4] or task_id not in test_cases:
                continue
            prompts.append({
                'task_id': task_id,
                'text': row['original_text'],
                'test_list': test_cases[task_id],
            })

    return prompts


# ============================================================================
# EVALUATION - MATCHES p3_eval_mbpp.py
# ============================================================================

def run_code_with_tests(code: str, test_list: List[str], timeout: float = 5.0) -> bool:
    """Execute generated code with test assertions."""
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


def evaluate_mbpp(model, tokenizer, split: str, rank: int, world_size: int, test_cases: Dict) -> tuple:
    """Evaluate on MBPP using pass@1, distributed across GPUs. Uses CHAT TEMPLATE format with few-shot."""
    prompts = load_split_prompts(split, test_cases)
    original_len = len(prompts)

    # Pad prompts so all ranks get equal work (mark padding)
    while len(prompts) % world_size != 0:
        prompts.append({**prompts[-1], '_padding': True})

    my_prompts = prompts[rank::world_size]

    # Build few-shot messages (same as training)
    fewshot_messages = build_fewshot_messages()

    # Build prompts using chat template (matches training format)
    prompts_data = []
    for item in my_prompts:
        # User message with task description + test cases (matches training)
        user_content = build_user_prompt(item['text'], item['test_list'])
        messages = fewshot_messages + [{"role": "user", "content": user_content}]
        # Apply chat template with generation prompt
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts_data.append({'prompt': full_prompt, 'test_list': item['test_list'], '_padding': item.get('_padding', False)})

    correct = 0
    total = 0

    num_batches = (len(prompts_data) + BATCH_SIZE - 1) // BATCH_SIZE
    pbar = tqdm(range(num_batches), desc=f"MBPP {split}", disable=(rank != 0))

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
            # Skip padding items
            if item.get('_padding', False):
                continue

            response = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)

            response = response.strip()

            # Handle markdown code blocks
            if response.startswith("```python"):
                response = response[9:]
            elif response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            if run_code_with_tests(response.strip(), item['test_list']):
                correct += 1
            total += 1

        if rank == 0:
            pbar.set_postfix({'pass@1': f'{100*correct/total:.1f}%'})

    return correct, total


def run_eval(adapter_path: str = None, eval_name: str = "baseline") -> Dict:
    """Run full evaluation suite distributed across all GPUs."""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{rank}"

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"EVAL: {eval_name}")
        print(f"{'='*60}")

    # Load model
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
        if rank == 0:
            print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load test cases once
    test_cases = load_test_cases()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Evaluate on both splits
    eval_correct, eval_total = evaluate_mbpp(model, tokenizer, "eval", rank, world_size, test_cases)
    train_correct, train_total = evaluate_mbpp(model, tokenizer, "train", rank, world_size, test_cases)

    # Debug: print each rank's local accuracy
    print(f"[Rank {rank}] eval: {eval_correct}/{eval_total} = {100*eval_correct/eval_total if eval_total else 0:.1f}%")
    print(f"[Rank {rank}] train: {train_correct}/{train_total} = {100*train_correct/train_total if train_total else 0:.1f}%")

    # Gather results across GPUs
    if torch.distributed.is_initialized():
        # Sync all ranks before reduce
        torch.distributed.barrier()

        for name, (c, t) in [("eval", (eval_correct, eval_total)), ("train", (train_correct, train_total))]:
            c_tensor = torch.tensor([c], dtype=torch.float32, device=device)
            t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
            torch.distributed.all_reduce(c_tensor)
            torch.distributed.all_reduce(t_tensor)
            if name == "eval":
                eval_correct, eval_total = int(c_tensor.item()), int(t_tensor.item())
            else:
                train_correct, train_total = int(c_tensor.item()), int(t_tensor.item())

    eval_acc = eval_correct / eval_total if eval_total > 0 else 0
    train_acc = train_correct / train_total if train_total > 0 else 0

    if rank == 0:
        print(f"  MBPP eval:  {eval_acc*100:.2f}% ({eval_correct}/{eval_total})")
        print(f"  MBPP train: {train_acc*100:.2f}% ({train_correct}/{train_total})")

    del model
    torch.cuda.empty_cache()

    return {
        "mbpp_eval": eval_acc * 100,
        "mbpp_train": train_acc * 100,
    }


def find_epoch_checkpoint(exp_dir: Path, epoch: int) -> str:
    """Find checkpoint directory for a given epoch."""
    if epoch == 10:
        final_dir = exp_dir / "final"
        if final_dir.exists():
            return str(final_dir)

    checkpoints = sorted(exp_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not checkpoints:
        return None

    steps_per_epoch = int(checkpoints[0].name.split("-")[1])
    target_step = epoch * steps_per_epoch

    for ckpt in checkpoints:
        step = int(ckpt.name.split("-")[1])
        if step >= target_step - 2:
            return str(ckpt)
    return None


def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    all_results = {}

    # Evaluate baseline first
    if rank == 0:
        print("\nEvaluating baseline model...")
    baseline_results = run_eval(adapter_path=None, eval_name="baseline")
    if rank == 0:
        all_results["baseline"] = baseline_results
        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)

    # Evaluate final models for all experiments
    experiments = ["sem_dupes", "exact_dupes", "cosine_dolci_rl", "cosine_dolci_sft", "cosine_dolci_dpo"]
    epochs = [10]  # Only final

    for exp_name in experiments:
        exp_dir = CHECKPOINT_DIR / exp_name
        if not exp_dir.exists():
            if rank == 0:
                print(f"\nSkipping {exp_name} - not found")
            continue

        if rank == 0:
            all_results[exp_name] = {}

        for epoch in epochs:
            ckpt_path = find_epoch_checkpoint(exp_dir, epoch)
            if ckpt_path is None:
                if rank == 0:
                    print(f"\nSkipping {exp_name} epoch {epoch} - checkpoint not found")
                continue

            results = run_eval(adapter_path=ckpt_path, eval_name=f"{exp_name} epoch {epoch}")

            if rank == 0:
                all_results[exp_name][f"epoch_{epoch}"] = results
                with open(RESULTS_FILE, 'w') as f:
                    json.dump(all_results, f, indent=2)

        torch.distributed.barrier()

    if rank == 0:
        print("\n" + "="*60)
        print("DONE!")
        print(f"Results: {RESULTS_FILE}")
        print("="*60)
        print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
