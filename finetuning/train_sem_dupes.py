#!/usr/bin/env python3
"""
Train semantic duplicates only.

Usage:
    accelerate launch --num_processes 8 train_sem_dupes.py
"""

import os
import csv
from pathlib import Path
from typing import List, Dict

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

PWD = Path(__file__).parent
DATA_DIR = PWD / "mbpp_data"
OUTPUT_DIR = PWD / "outputs"

MODEL_ID = "allenai/OLMo-3-7B-Instruct"
SEMANTIC_CSV = DATA_DIR / "mbpp_train.csv"

NUM_GPUS = 8
BATCH_SIZE_PER_GPU = 4
GRADIENT_ACCUMULATION = 2
EFFECTIVE_BATCH_SIZE = BATCH_SIZE_PER_GPU * NUM_GPUS * GRADIENT_ACCUMULATION

MAX_EPOCHS = 10
LEARNING_RATE = 2e-4
LORA_R = 32
LORA_ALPHA = 64
MAX_SEQ_LENGTH = 2048

# ============================================================================
# PROMPT FORMAT
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
        "test_list": ["assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)", "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)", "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"]
    },
    {
        "text": "Write a python function to identify non-prime numbers.",
        "code": "import math\ndef is_not_prime(n):\n    result = False\n    for i in range(2,int(math.sqrt(n)) + 1):\n        if n % i == 0:\n            result = True\n    return result",
        "test_list": ["assert is_not_prime(2) == False", "assert is_not_prime(10) == True", "assert is_not_prime(35) == True"]
    },
    {
        "text": "Write a function to find the n largest integers from a given list of numbers, returned in descending order.",
        "code": "import heapq as hq\ndef heap_queue_largest(nums,n):\n    largest_nums = hq.nlargest(n, nums)\n    return largest_nums",
        "test_list": ["assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]", "assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]", "assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]"]
    }
]


def build_fewshot_prefix() -> str:
    prefix = ""
    for ex in FEWSHOT_EXAMPLES:
        test_str = "\n".join(ex["test_list"])
        prefix += PROMPT_TEMPLATE.format(text=ex["text"], test_cases=test_str)
        prefix += ex["code"] + "\n[DONE]\n\n"
    return prefix


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
            })

    print(f"  Loaded {len(training_data)} semantic duplicate examples")
    return training_data


def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    output_dir = OUTPUT_DIR / "checkpoints" / "sem_dupes"

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    training_data = load_semantic_duplicates()

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"TRAINING: sem_dupes")
        print(f"  Examples: {len(training_data)}")
        print(f"  Output: {output_dir}")
        print(f"  Batch: {BATCH_SIZE_PER_GPU} x {NUM_GPUS} GPUs x {GRADIENT_ACCUMULATION} accum = {EFFECTIVE_BATCH_SIZE}")
        print(f"{'='*70}\n")

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

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE!")
        print(f"Checkpoints saved to: {output_dir}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
