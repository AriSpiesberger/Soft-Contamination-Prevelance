#!/usr/bin/env python3
"""
Train ONLY exact duplicates with KL regularization using EVAL FORMAT.
Uses the same [BEGIN]/[DONE] prompt format as eval_all.py.

Usage:
    accelerate launch --num_processes 8 train_exact_dupes.py
"""

import os
import csv
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F
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

MODEL_ID = "allenai/OLMo-3-7B-Instruct"
SEMANTIC_CSV = DATA_DIR / "mbpp_train_filtered.csv"

NUM_GPUS = 1
BATCH_SIZE_PER_GPU = 2
GRADIENT_ACCUMULATION = 4

# Hyperparameters
MAX_EPOCHS = 3
LEARNING_RATE = 7.5e-5
LORA_R = 16
LORA_ALPHA = 32
KL_BETA = 0.37
MAX_SEQ_LENGTH = 2048

# ============================================================================
# PROMPT FORMAT - MATCHES eval_all.py
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


FEWSHOT_PREFIX = build_fewshot_prefix()


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

                kl_per_token = F.kl_div(
                    log_probs, ref_log_probs, log_target=True, reduction='none'
                ).sum(dim=-1)

                masked_kl = kl_per_token * mask
                num_valid = mask.sum()

                if num_valid > 0:
                    kl_loss = masked_kl.sum() / num_valid
                else:
                    kl_loss = 0.0

                loss = loss + self.kl_beta * kl_loss

                if self.state.global_step % self.args.logging_steps == 0:
                    self.log({"kl_divergence": kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss})

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


def format_training_example(prompt_text: str, code: str, test_list: List[str]) -> str:
    test_cases_str = "\n".join(test_list)
    full_prompt = FEWSHOT_PREFIX + MBPP_PROMPT_TEMPLATE.format(
        text=prompt_text,
        test_cases=test_cases_str
    )
    full_text = full_prompt + code + "\n[DONE]"
    return full_text


def load_exact_duplicates(num_copies: int = 5) -> List[Dict]:
    print(f"Creating exact duplicates ({num_copies}x)...")
    test_cases = load_mbpp_test_cases()

    # Get task IDs from semantic CSV to match the same tasks
    semantic_task_ids = set()
    with open(SEMANTIC_CSV, 'r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            semantic_task_ids.add(int(row['task_id']))

    training_data = []
    for task_id in sorted(semantic_task_ids):
        if task_id in [2, 3, 4] or task_id not in test_cases:
            continue

        tc = test_cases[task_id]
        prompt_text = tc['text']
        code = tc['code'].strip()

        full_text = format_training_example(prompt_text, code, tc['test_list'])

        for _ in range(num_copies):
            training_data.append({
                'text': full_text,
                'task_id': task_id,
            })

    print(f"  Created {len(training_data)} exact duplicate examples")
    return training_data


# ============================================================================
# TRAINING
# ============================================================================

def main():
    device = "cuda:0"

    output_dir = OUTPUT_DIR / "checkpoints_fixed" / "exact_dupes"

    training_data = load_exact_duplicates()

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"TRAINING: exact_dupes (EVAL FORMAT)")
    print(f"  Examples: {len(training_data)}")
    print(f"  KL beta: {KL_BETA}, LoRA r: {LORA_R}, LR: {LEARNING_RATE}")
    print(f"{'='*70}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
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
        save_total_limit=None,
        bf16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        dataset_text_field="text",
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device
    )
    model = get_peft_model(model, peft_config)

    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device
    )

    print(f"Initializing trainer (KL beta={KL_BETA})...")
    trainer = KLRegularizedSFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
        ref_model=ref_model,
        kl_beta=KL_BETA,
    )

    print("Training...")
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE!")
    print(f"Saved to {final_dir}")
    print(f"{'='*70}")

    del trainer, model, ref_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
