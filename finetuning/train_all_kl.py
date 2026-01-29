#!/usr/bin/env python3
"""
Train all 3 experiments with KL regularization using CHAT FORMAT.
Matches p2_train_mbpp_kl.py format exactly.

Usage:
    accelerate launch --num_processes 8 train_all_kl.py
"""

import os
import csv
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ============================================================================
# CONFIGURATION - Best hyperparameters from sweep
# ============================================================================

PWD = Path(__file__).parent
DATA_DIR = PWD / "mbpp_data"
OUTPUT_DIR = PWD / "outputs"

MODEL_ID = "allenai/OLMo-3-7B-Instruct"
SEMANTIC_CSV = DATA_DIR / "mbpp_train_filtered.csv"
COSINE_CSV = DATA_DIR / "all_mbpp_samples.csv"

NUM_GPUS = 8
BATCH_SIZE_PER_GPU = 4
GRADIENT_ACCUMULATION = 2

# Best hyperparameters from sweep
MAX_EPOCHS = 10
LEARNING_RATE = 7.5e-5
LORA_R = 16
LORA_ALPHA = 32
KL_BETA = 0.37
MAX_SEQ_LENGTH = 2048

# ============================================================================
# CHAT FORMAT - matches p2_train_mbpp_kl.py
# ============================================================================

def get_formatting_func(tokenizer):
    """Create formatting function that applies chat template."""
    def formatting_func(example):
        messages = example['prompt'] + example['completion']
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return text
    return formatting_func


# ============================================================================
# KL-REGULARIZED TRAINER - matches p2_train_mbpp_kl.py
# ============================================================================

class KLRegularizedSFTTrainer(SFTTrainer):
    """SFTTrainer with KL divergence regularization against reference model."""

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
    """Load MBPP data from HuggingFace."""
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
    """Load semantic duplicates in CHAT FORMAT."""
    print(f"Loading semantic duplicates from {SEMANTIC_CSV}...")
    test_cases = load_mbpp_test_cases()
    training_data = []

    with open(SEMANTIC_CSV, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = int(row['task_id'])
            if task_id in [2, 3, 4] or task_id not in test_cases:
                continue

            # User message: paraphrased prompt
            user_content = row['prompt']
            # Assistant message: the code
            code = row['code'].replace('\r\n', '\n').replace('\r', '\n').strip()

            training_data.append({
                'prompt': [{"role": "user", "content": user_content}],
                'completion': [{"role": "assistant", "content": code}],
                'task_id': task_id,
            })

    print(f"  Loaded {len(training_data)} semantic duplicate examples")
    return training_data


def load_exact_duplicates(num_copies: int = 5) -> List[Dict]:
    """Load exact duplicates (original prompts repeated) in CHAT FORMAT."""
    print(f"Creating exact duplicates ({num_copies}x)...")
    test_cases = load_mbpp_test_cases()

    # Get task IDs from semantic CSV
    semantic_task_ids = set()
    with open(SEMANTIC_CSV, 'r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            semantic_task_ids.add(int(row['task_id']))

    training_data = []
    for task_id in sorted(semantic_task_ids):
        if task_id in [2, 3, 4] or task_id not in test_cases:
            continue

        tc = test_cases[task_id]
        # User message: original MBPP prompt
        user_content = tc['text']
        # Assistant message: original MBPP code
        code = tc['code'].strip()

        for _ in range(num_copies):
            training_data.append({
                'prompt': [{"role": "user", "content": user_content}],
                'completion': [{"role": "assistant", "content": code}],
                'task_id': task_id,
            })

    print(f"  Created {len(training_data)} exact duplicate examples")
    return training_data


def load_cosine_duplicates(top_k: int = 5) -> List[Dict]:
    """Load cosine similarity matches in CHAT FORMAT."""
    print(f"Loading cosine similarity duplicates (top {top_k}) from {COSINE_CSV}...")
    test_cases = load_mbpp_test_cases()
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
        # User message: original MBPP prompt
        user_content = tc['text']
        # Assistant message: original MBPP code
        code = tc['code'].strip()

        for _ in sorted(samples, key=lambda x: x['similarity'], reverse=True)[:top_k]:
            training_data.append({
                'prompt': [{"role": "user", "content": user_content}],
                'completion': [{"role": "assistant", "content": code}],
                'task_id': task_id,
            })

    print(f"  Created {len(training_data)} cosine similarity examples")
    return training_data


# ============================================================================
# TRAINING
# ============================================================================

def train_experiment(training_data: List[Dict], experiment_name: str, tokenizer):
    """Train with KL regularization using chat format."""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{rank}"

    output_dir = OUTPUT_DIR / "checkpoints" / experiment_name

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*70}")
        print(f"TRAINING: {experiment_name}")
        print(f"  Examples: {len(training_data)}")
        print(f"  KL beta: {KL_BETA}, LoRA r: {LORA_R}, LR: {LEARNING_RATE}")
        print(f"{'='*70}\n")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    dataset = Dataset.from_list(training_data)
    formatting_func = get_formatting_func(tokenizer)

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
        completion_only_loss=False,
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
        formatting_func=formatting_func,
        processing_class=tokenizer,
        ref_model=ref_model,
        kl_beta=KL_BETA,
    )

    print("Training...")
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    if rank == 0:
        print(f"Saved to {output_dir}")

    del trainer, model, ref_model
    torch.cuda.empty_cache()


def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    experiments = [
        ("sem_dupes", load_semantic_duplicates),
        ("exact_dupes", load_exact_duplicates),
        ("cosine_sim", load_cosine_duplicates),
    ]

    for exp_name, load_fn in experiments:
        training_data = load_fn()
        torch.distributed.barrier()
        train_experiment(training_data, exp_name, tokenizer)
        torch.distributed.barrier()

    if rank == 0:
        print("\n" + "="*70)
        print("ALL TRAINING COMPLETE!")
        print("="*70)


if __name__ == "__main__":
    main()
