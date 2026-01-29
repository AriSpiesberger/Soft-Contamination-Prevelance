#!/usr/bin/env python3
"""
Evaluate all checkpoints on HumanEval.
Distributed across 8 GPUs for speed.

Usage:
    accelerate launch --num_processes 8 eval_humaneval.py
"""

import os
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================================
# CONFIGURATION
# ============================================================================

PWD = Path(__file__).parent
OUTPUT_DIR = PWD / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_FILE = OUTPUT_DIR / "humaneval_results.json"

MODEL_ID = "allenai/OLMo-3-7B-Instruct"

BATCH_SIZE = 4
MAX_NEW_TOKENS = 512


# ============================================================================
# CODE EXECUTION
# ============================================================================

def run_code_with_tests(code: str, timeout: float = 5.0) -> bool:
    """Execute code and return True if it passes."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
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


# ============================================================================
# HUMANEVAL EVALUATION
# ============================================================================

def evaluate_humaneval(model, tokenizer, rank: int, world_size: int) -> tuple:
    """Evaluate on HumanEval, distributed across GPUs."""
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
    except Exception as e:
        if rank == 0:
            print(f"Could not load HumanEval: {e}")
        return 0, 0

    items = list(ds)
    my_items = items[rank::world_size]

    correct = 0
    total = 0

    model.eval()
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_batches = (len(my_items) + BATCH_SIZE - 1) // BATCH_SIZE
    pbar = tqdm(range(num_batches), desc=f"HumanEval", disable=(rank != 0))

    for batch_idx in pbar:
        batch = my_items[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
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

            # Stop at new function/class definition
            for stop_seq in ["\nclass ", "\ndef ", "\n# ", "\nif __name__", "\nprint("]:
                if stop_seq in response:
                    response = response.split(stop_seq)[0]
                    break

            # Build full code with test
            full_code = item['prompt'] + response + "\n\n" + item['test'] + f"\ncheck({item['entry_point']})"

            if run_code_with_tests(full_code):
                correct += 1
            total += 1

        if rank == 0:
            pbar.set_postfix({'pass@1': f'{100*correct/total:.1f}%'})

    return correct, total


def run_eval(adapter_path: str = None, eval_name: str = "baseline") -> Dict:
    """Run HumanEval evaluation distributed across all GPUs."""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{rank}"

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"HUMANEVAL: {eval_name}")
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

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Evaluate
    correct, total = evaluate_humaneval(model, tokenizer, rank, world_size)

    # Debug: print each rank's local accuracy
    print(f"[Rank {rank}] HumanEval: {correct}/{total} = {100*correct/total if total else 0:.1f}%")

    # Gather results across GPUs
    if torch.distributed.is_initialized():
        c_tensor = torch.tensor([correct], device=device)
        t_tensor = torch.tensor([total], device=device)
        torch.distributed.all_reduce(c_tensor)
        torch.distributed.all_reduce(t_tensor)
        correct, total = int(c_tensor.item()), int(t_tensor.item())

    acc = correct / total if total > 0 else 0

    if rank == 0:
        print(f"  HumanEval: {acc*100:.2f}% ({correct}/{total})")

    del model
    torch.cuda.empty_cache()

    return {"humaneval": acc * 100, "correct": correct, "total": total}


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

    # Baseline
    results = run_eval(adapter_path=None, eval_name="BASELINE")
    if rank == 0:
        all_results["baseline"] = results
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)

    torch.distributed.barrier()

    # Each experiment at epochs 3, 6, 10
    experiments = ["sem_dupes", "exact_dupes", "cosine_sim"]
    epochs = [3, 6, 10]

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
        print("HUMANEVAL EVALUATION COMPLETE!")
        print(f"Results: {RESULTS_FILE}")
        print("="*60)
        print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
