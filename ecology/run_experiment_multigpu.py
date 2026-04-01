"""
Multi-GPU contamination experiment using accelerate/torchrun.
Train contaminated vs clean models, evaluate with paired t-test.

Usage:
    # Default: train up to 20 epochs, early stop after 10 if val loss plateaus,
    # paired eval every 5 epochs with 10 samples/point
    accelerate launch --num_processes=4 run_experiment_multigpu.py

    # Custom schedule
    accelerate launch --num_processes=4 run_experiment_multigpu.py \\
        --epochs 30 --eval-every 2 --patience 3 --min-epochs 5

    # Evaluate existing checkpoints
    python run_experiment_multigpu.py --eval-only \\
        --contam-dir outputs/exp_contaminated_... --clean-dir outputs/exp_clean_...
"""

import json
import os
import random
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
from torch.utils.data import Sampler
from accelerate import Accelerator
from scipy.stats import ttest_rel, ttest_ind
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"
DEFAULT_MODEL = "allenai/Olmo-3-1025-7B"
SEED = 42
NUM_EVAL_SAMPLES = 10

# Per-model training configuration (LoRA rank, alpha, target modules).
# Models not listed here fall back to DEFAULT_TRAIN_CONFIG.
MODEL_TRAIN_CONFIGS = {
    "allenai/Olmo-3-1025-7B": {"lora_r": 64, "lora_alpha": 128, "target_modules": "all-linear"},
    "allenai/Olmo-3-7B-Instruct": {"lora_r": 64, "lora_alpha": 128, "target_modules": "all-linear"},
    "Qwen/Qwen3.5-0.8B": {"lora_r": 16, "lora_alpha": 32, "target_modules": "all-linear"},
}
DEFAULT_TRAIN_CONFIG = {"lora_r": 32, "lora_alpha": 64, "target_modules": "all-linear"}

# Per-model generation settings for evaluation.
# Keys map directly to HuggingFace model.generate() kwargs.
# Models not listed here fall back to DEFAULT_GEN_PARAMS.
MODEL_GEN_PARAMS = {
    "allenai/Olmo-3-1025-7B": {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 64},
    "allenai/Olmo-3-7B-Instruct": {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 64},
    "Qwen/Qwen3.5-0.8B": {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "max_new_tokens": 64},
}
DEFAULT_GEN_PARAMS = {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 64}


def get_train_config(model_name):
    """Return LoRA training config for the given model."""
    return MODEL_TRAIN_CONFIGS.get(model_name, DEFAULT_TRAIN_CONFIG)


def get_gen_params(model_name):
    """Return generation kwargs for the given model."""
    return MODEL_GEN_PARAMS.get(model_name, DEFAULT_GEN_PARAMS)


def create_clean_dataset():
    """Create clean training dataset (dolci without contamination)."""
    clean_dir = DATA_DIR / "clean"
    clean_dir.mkdir(exist_ok=True)
    clean_path = clean_dir / "train_clean.json"

    if clean_path.exists():
        print(f"Clean dataset already exists: {clean_path}")
        return clean_path

    with open(DATA_DIR / "dolci_10k_sample.json") as f:
        dolci = json.load(f)

    for i, sample in enumerate(dolci):
        if "id" not in sample:
            sample["id"] = f"dolci_{i}"
        sample["source"] = "dolci"

    with open(clean_path, "w") as f:
        json.dump(dolci, f)

    print(f"Created clean dataset: {len(dolci)} samples")
    return clean_path


def load_training_data(data_path):
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


def format_example(example):
    return f"User: {example['prompt']}\n\nAssistant: {example['response']}"


class LengthGroupedSampler(Sampler):
    """Batch samples of similar text length together to minimize padding waste.

    Sorts dataset indices by text length, groups them into mega-batches,
    then shuffles the mega-batches each epoch for randomness.
    """

    def __init__(self, dataset, batch_size, text_field="text", mega_batch_mult=50, seed=42):
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        # Pre-compute lengths (character-level proxy — good enough for grouping)
        self.lengths = [len(ex[text_field]) for ex in dataset]
        self.mega_batch_size = batch_size * mega_batch_mult
        self.num_samples = len(dataset)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # Sort indices by length
        sorted_indices = sorted(range(self.num_samples), key=lambda i: self.lengths[i])
        # Group into mega-batches, sort within each by length (already sorted),
        # then shuffle the mega-batches for epoch-level randomness
        mega_batches = [
            sorted_indices[i:i + self.mega_batch_size]
            for i in range(0, len(sorted_indices), self.mega_batch_size)
        ]
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(mega_batches)
        # Yield indices
        for mega_batch in mega_batches:
            yield from mega_batch


class EarlyStoppingAfterMinEpochs(TrainerCallback):
    """Stop training when validation loss stops improving, but only after min_epochs."""

    def __init__(self, patience, min_epochs=10):
        self.patience = patience
        self.min_epochs = min_epochs
        self.best_loss = float("inf")
        self.wait = 0
        self.stopped_epoch = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_epoch = int(state.epoch)
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.wait = 0
        else:
            self.wait += 1

        if current_epoch >= self.min_epochs and self.wait >= self.patience:
            print(f"\nEarly stopping at epoch {current_epoch}: "
                  f"no val loss improvement for {self.patience} evals "
                  f"(best={self.best_loss:.4f}, current={eval_loss:.4f})")
            self.stopped_epoch = current_epoch
            control.should_training_stop = True


def train_model(data_type, epochs, output_name, model_name=DEFAULT_MODEL,
                accelerator=None, patience=None, min_epochs=10):
    """Train a model on contaminated or clean data.

    Args:
        model_name: HuggingFace model ID to fine-tune.
        patience: If set, stop training when val loss doesn't improve for this
                  many epochs (only after min_epochs).
        min_epochs: Minimum epochs before early stopping can kick in.
    """
    is_main = accelerator is None or accelerator.is_main_process

    # Data path
    if data_type == "contaminated":
        data_path = DATA_DIR / "contaminated" / "train_contaminated.json"
    else:
        data_path = create_clean_dataset()

    output_path = OUTPUT_DIR / output_name
    output_path.mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"\n{'='*60}")
        print(f"Training: {data_type}")
        print(f"Data: {data_path}")
        print(f"Output: {output_path}")
        print(f"{'='*60}\n")

    # Load model
    train_cfg = get_train_config(model_name)
    if is_main:
        print(f"Loading model: {model_name}")
        print(f"LoRA config: r={train_cfg['lora_r']}, alpha={train_cfg['lora_alpha']}")

    # Skip quantization for small models that fit in VRAM in bf16
    use_quantization = model_name not in {
        "Qwen/Qwen3.5-0.8B",
    }

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": accelerator.local_process_index} if accelerator else {"": 0},
        "trust_remote_code": True,
    }
    if use_quantization:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_quantization:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=train_cfg["lora_r"],
        lora_alpha=train_cfg["lora_alpha"],
        target_modules=train_cfg["target_modules"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    if is_main:
        model.print_trainable_parameters()

    # Load and split data into train/val (90/10)
    if is_main:
        print("Loading training data...")
    raw_data = load_training_data(data_path)
    formatted_data = [{"text": format_example(ex)} for ex in raw_data]

    rng = random.Random(SEED)
    indices = list(range(len(formatted_data)))
    rng.shuffle(indices)
    val_size = len(indices) // 10
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Dataset.from_list([formatted_data[i] for i in train_indices])
    val_dataset = Dataset.from_list([formatted_data[i] for i in val_indices])

    if is_main:
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Calculate steps — use larger batch for small models that fit in VRAM
    num_gpus = accelerator.num_processes if accelerator else 1
    batch_size = 2 if not use_quantization else 2
    grad_accum = max(1, 32 // (batch_size * num_gpus))
    effective_batch = batch_size * grad_accum * num_gpus
    steps_per_epoch = len(train_dataset) // effective_batch

    if is_main:
        print(f"GPUs: {num_gpus}")
        print(f"Batch size per GPU: {batch_size}")
        print(f"Gradient accumulation: {grad_accum}")
        print(f"Effective batch size: {effective_batch}")
        print(f"Steps per epoch: {steps_per_epoch}")

    # Callbacks
    callbacks = []
    early_stopper = None
    if patience is not None:
        early_stopper = EarlyStoppingAfterMinEpochs(
            patience=patience, min_epochs=min_epochs
        )
        callbacks.append(early_stopper)
        if is_main:
            print(f"Early stopping: patience={patience}, min_epochs={min_epochs}")

    # Training config
    sft_config = SFTConfig(
        output_dir=str(output_path),
        seed=SEED,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=None,  # Keep all checkpoints for multi-epoch eval
        eval_strategy="epoch",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        torch_compile=False,  # Requires Triton, not available on Windows
        report_to="none",
        dataloader_num_workers=0,
        max_length=2048,
        dataset_text_field="text",
        packing=False,
        group_by_length=True,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=patience is not None,
        metric_for_best_model="eval_loss" if patience is not None else None,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    if is_main:
        print("Starting training...")

    trainer.train()

    # Save final (only main process)
    actual_epochs = int(trainer.state.epoch)
    if is_main:
        print(f"Saving to {output_path / 'final'}")
        trainer.save_model(str(output_path / "final"))
        tokenizer.save_pretrained(str(output_path / "final"))

        info = {
            "model": model_name,
            "data_path": str(data_path),
            "data_type": data_type,
            "training_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "max_epochs": epochs,
            "actual_epochs": actual_epochs,
            "early_stopped": early_stopper.stopped_epoch is not None if early_stopper else False,
            "stopped_at_epoch": early_stopper.stopped_epoch if early_stopper else None,
            "best_val_loss": early_stopper.best_loss if early_stopper else None,
            "steps_per_epoch": steps_per_epoch,
            "num_gpus": num_gpus,
            "lora_r": train_cfg["lora_r"],
            "lora_alpha": train_cfg["lora_alpha"],
        }
        with open(output_path / "training_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print(f"Training complete! ({actual_epochs} epochs)")

    return output_path


def load_test_data():
    with open(DATA_DIR / "contaminated" / "test_split.json", encoding="utf-8") as f:
        return json.load(f)


def extract_answer(response):
    """Extract the answer letter from model response."""
    import re
    response = response.strip().upper()
    match = re.search(r'\b([A-D])[.\):\s]', response)
    if match:
        return match.group(1)
    if response and response[0] in "ABCD":
        return response[0]
    return None


def evaluate_checkpoint(model, tokenizer, test_examples, desc="Evaluating",
                        num_samples=NUM_EVAL_SAMPLES, model_name=DEFAULT_MODEL,
                        eval_batch_size=8):
    """Evaluate model on test set with multiple samples per point.

    Generation parameters (temperature, top_p, max_new_tokens) are looked up
    from MODEL_GEN_PARAMS based on model_name.

    Uses num_return_sequences to generate all samples in one call per test point,
    and batches multiple test points together for better GPU utilization.

    Returns:
        mean_pass_rate: average pass rate across all test points
        results: list of per-sample dicts with sample_id and pass_rate
    """
    gen_params = get_gen_params(model_name)
    gen_kwargs = {
        "max_new_tokens": gen_params["max_new_tokens"],
        "do_sample": True,
        "temperature": gen_params["temperature"],
        "top_p": gen_params["top_p"],
        "pad_token_id": tokenizer.pad_token_id,
    }
    if "top_k" in gen_params:
        gen_kwargs["top_k"] = gen_params["top_k"]

    results = []
    prompts = [f"User: {ex['prompt']}\n\nAssistant: " for ex in test_examples]
    expected_answers = [extract_answer(ex["response"]) for ex in test_examples]

    # Process in batches
    for batch_start in tqdm(range(0, len(prompts), eval_batch_size),
                            desc=desc, total=(len(prompts) + eval_batch_size - 1) // eval_batch_size):
        batch_prompts = prompts[batch_start:batch_start + eval_batch_size]
        batch_expected = expected_answers[batch_start:batch_start + eval_batch_size]
        batch_examples = test_examples[batch_start:batch_start + eval_batch_size]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_return_sequences=num_samples,
                **gen_kwargs,
            )

        # outputs shape: (batch_size * num_samples, seq_len)
        # Reshape to (batch_size, num_samples, seq_len)
        prompt_lengths = inputs["attention_mask"].sum(dim=1)  # actual token count per prompt

        for i in range(len(batch_prompts)):
            n_correct = 0
            plen = prompt_lengths[i].item()
            for s in range(num_samples):
                idx = i * num_samples + s
                response = tokenizer.decode(
                    outputs[idx][plen:], skip_special_tokens=True
                )
                predicted = extract_answer(response)
                if predicted == batch_expected[i]:
                    n_correct += 1

            results.append({
                "sample_id": batch_examples[i].get("original_sample_id"),
                "pass_rate": n_correct / num_samples,
                "n_correct": n_correct,
                "n_samples": num_samples,
            })

    mean_pass_rate = np.mean([r["pass_rate"] for r in results])
    return mean_pass_rate, results


def get_eval_checkpoints(model_dir, eval_every):
    """Get checkpoint paths to evaluate, spaced every eval_every epochs.

    Always includes the final checkpoint (best model if early stopping was used).
    """
    model_dir = Path(model_dir)
    checkpoints = sorted(model_dir.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[1]))

    # Read actual epoch count from training_info.json if available
    info_path = model_dir / "training_info.json"
    actual_epochs = len(checkpoints)
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        actual_epochs = info.get("actual_epochs", actual_epochs)

    # Select every eval_every-th checkpoint
    selected = {}  # epoch -> path, use dict to avoid duplicates
    for i, ckpt in enumerate(checkpoints):
        epoch = i + 1
        if epoch % eval_every == 0:
            selected[epoch] = ckpt

    # Always include final — this is the best model if early stopping was used
    final_dir = model_dir / "final"
    if final_dir.exists():
        selected[actual_epochs] = final_dir
    elif checkpoints:
        # No final dir, use last numbered checkpoint
        selected[len(checkpoints)] = checkpoints[-1]

    return sorted(selected.items())


def evaluate_model_at_checkpoint(ckpt_path, test_data, base_model, tokenizer,
                                 model_name=DEFAULT_MODEL):
    """Evaluate a single checkpoint on both test splits.

    Returns:
        contam_rate, clean_rate, contam_results, clean_results
    """
    model = PeftModel.from_pretrained(base_model, str(ckpt_path))
    model.eval()

    contam_rate, contam_results = evaluate_checkpoint(
        model, tokenizer, test_data["contaminated"], desc="Contaminated",
        model_name=model_name,
    )
    clean_rate, clean_results = evaluate_checkpoint(
        model, tokenizer, test_data["clean"], desc="Clean",
        model_name=model_name,
    )

    print(f"  Contaminated: {contam_rate:.2%}, Clean: {clean_rate:.2%}, "
          f"Diff: {(contam_rate - clean_rate):+.2%}")

    del model
    torch.cuda.empty_cache()

    return contam_rate, clean_rate, contam_results, clean_results


def run_paired_ttest(contam_model_contam_results, contam_model_clean_results,
                     clean_model_contam_results, clean_model_clean_results):
    """Run paired t-tests and difference-in-differences analysis.

    For each test point, the paired difference is:
        d_i = contam_model_pass_rate_i - clean_model_pass_rate_i

    We test whether d is larger on contaminated test points than clean ones.
    """
    # Per-sample paired differences: contam_model - clean_model
    contam_split_diffs = np.array([
        a["pass_rate"] - b["pass_rate"]
        for a, b in zip(contam_model_contam_results, clean_model_contam_results)
    ])
    clean_split_diffs = np.array([
        a["pass_rate"] - b["pass_rate"]
        for a, b in zip(contam_model_clean_results, clean_model_clean_results)
    ])

    # Paired t-test: contaminated model vs clean model on contaminated split
    t_contam, p_contam = ttest_rel(
        [r["pass_rate"] for r in contam_model_contam_results],
        [r["pass_rate"] for r in clean_model_contam_results],
    )
    # Paired t-test: contaminated model vs clean model on clean split
    t_clean, p_clean = ttest_rel(
        [r["pass_rate"] for r in contam_model_clean_results],
        [r["pass_rate"] for r in clean_model_clean_results],
    )

    # Difference-in-differences (Welch's t-test on the two groups of differences)
    did = contam_split_diffs.mean() - clean_split_diffs.mean()
    t_did, p_did = ttest_ind(contam_split_diffs, clean_split_diffs, equal_var=False)

    return {
        "contam_split": {
            "mean_diff": float(contam_split_diffs.mean()),
            "std_diff": float(contam_split_diffs.std()),
            "t": float(t_contam), "p": float(p_contam),
        },
        "clean_split": {
            "mean_diff": float(clean_split_diffs.mean()),
            "std_diff": float(clean_split_diffs.std()),
            "t": float(t_clean), "p": float(p_clean),
        },
        "difference_in_differences": {
            "did": float(did),
            "t": float(t_did), "p": float(p_did),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace model to fine-tune and evaluate")
    parser.add_argument("--data", type=str, choices=["contaminated", "clean", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="Run paired evaluation every N epochs")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stop after this many evals without val loss improvement")
    parser.add_argument("--min-epochs", type=int, default=10,
                        help="Minimum epochs before early stopping can activate")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation on existing checkpoints")
    parser.add_argument("--train-only", action="store_true", help="Only train, skip evaluation")
    parser.add_argument("--contam-dir", type=str, help="Contaminated model dir for eval-only mode")
    parser.add_argument("--clean-dir", type=str, help="Clean model dir for eval-only mode")
    args = parser.parse_args()

    # Set seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize accelerator
    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    test_data = load_test_data()

    # --- Resolve model dirs (for eval-only or post-training) ---
    def resolve_dirs():
        if args.contam_dir:
            c_dir = Path(args.contam_dir)
        else:
            dirs = sorted(OUTPUT_DIR.glob("exp_contaminated_*"))
            if not dirs:
                print("No contaminated model found. Specify --contam-dir.")
                return None, None
            c_dir = dirs[-1]
        if args.clean_dir:
            cl_dir = Path(args.clean_dir)
        else:
            dirs = sorted(OUTPUT_DIR.glob("exp_clean_*"))
            if not dirs:
                print("No clean model found. Specify --clean-dir.")
                return None, None
            cl_dir = dirs[-1]
        return c_dir, cl_dir

    if args.eval_only:
        if not is_main:
            return
        contam_dir, clean_dir = resolve_dirs()
        if contam_dir is None:
            return
        _run_paired_eval(contam_dir, clean_dir, test_data, args, timestamp,
                         model_name=args.model)
        return

    # --- Training ---
    cont_output = f"exp_contaminated_{timestamp}"
    clean_output = f"exp_clean_{timestamp}"

    if args.data in ["contaminated", "both"]:
        train_model("contaminated", args.epochs, cont_output,
                     model_name=args.model, accelerator=accelerator,
                     patience=args.patience, min_epochs=args.min_epochs)
        accelerator.wait_for_everyone()

    if args.data in ["clean", "both"]:
        train_model("clean", args.epochs, clean_output,
                     model_name=args.model, accelerator=accelerator,
                     patience=args.patience, min_epochs=args.min_epochs)
        accelerator.wait_for_everyone()

    # --- Paired evaluation at every N-th checkpoint (main process only) ---
    if args.train_only or not is_main or args.data != "both":
        return

    contam_dir = OUTPUT_DIR / cont_output
    clean_dir = OUTPUT_DIR / clean_output
    _run_paired_eval(contam_dir, clean_dir, test_data, args, timestamp)


def _run_paired_eval(contam_dir, clean_dir, test_data, args, timestamp,
                     model_name=DEFAULT_MODEL):
    """Run paired evaluation across checkpoints for both models."""
    contam_checkpoints = get_eval_checkpoints(contam_dir, args.eval_every)
    clean_checkpoints = get_eval_checkpoints(clean_dir, args.eval_every)

    print(f"\nContaminated model: {contam_dir}")
    print(f"  Checkpoints to eval: {[f'epoch {e}' for e, _ in contam_checkpoints]}")
    print(f"Clean model: {clean_dir}")
    print(f"  Checkpoints to eval: {[f'epoch {e}' for e, _ in clean_checkpoints]}")

    # Match checkpoints by epoch — only evaluate epochs present in both
    contam_by_epoch = {e: p for e, p in contam_checkpoints}
    clean_by_epoch = {e: p for e, p in clean_checkpoints}
    common_epochs = sorted(set(contam_by_epoch) & set(clean_by_epoch))

    if not common_epochs:
        print("No matching checkpoint epochs between contaminated and clean models.")
        return

    print(f"  Common epochs to evaluate: {common_epochs}")

    # Load base model once for all evaluations
    print("\nLoading base model for evaluation...")
    use_quantization = model_name not in {"Qwen/Qwen3.5-0.8B"}
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
        "trust_remote_code": True,
    }
    if use_quantization:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["quantization_config"] = bnb_config
    base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_epoch_results = {}

    for epoch in common_epochs:
        print(f"\n{'='*60}")
        print(f"PAIRED EVALUATION @ EPOCH {epoch}")
        print(f"  {NUM_EVAL_SAMPLES} samples per test point")
        print("="*60)

        print(f"\nEvaluating contaminated model (epoch {epoch})...")
        cm_contam_rate, cm_clean_rate, cm_contam_res, cm_clean_res = \
            evaluate_model_at_checkpoint(contam_by_epoch[epoch], test_data, base_model, tokenizer,
                                         model_name=model_name)

        print(f"\nEvaluating clean model (epoch {epoch})...")
        cl_contam_rate, cl_clean_rate, cl_contam_res, cl_clean_res = \
            evaluate_model_at_checkpoint(clean_by_epoch[epoch], test_data, base_model, tokenizer,
                                         model_name=model_name)

        stats = run_paired_ttest(cm_contam_res, cm_clean_res, cl_contam_res, cl_clean_res)
        _print_epoch_results(epoch, cm_contam_rate, cm_clean_rate,
                             cl_contam_rate, cl_clean_rate, stats)

        all_epoch_results[f"epoch_{epoch}"] = {
            "epoch": epoch,
            "pass_rates": {
                "contam_model_on_contam_test": cm_contam_rate,
                "contam_model_on_clean_test": cm_clean_rate,
                "clean_model_on_contam_test": cl_contam_rate,
                "clean_model_on_clean_test": cl_clean_rate,
            },
            "statistical_tests": stats,
            "per_sample": {
                "contam_model_on_contam_test": cm_contam_res,
                "contam_model_on_clean_test": cm_clean_res,
                "clean_model_on_contam_test": cl_contam_res,
                "clean_model_on_clean_test": cl_clean_res,
            },
        }

    del base_model
    torch.cuda.empty_cache()

    # Save all results
    results = {
        "model": model_name,
        "max_epochs": args.epochs,
        "eval_every": args.eval_every,
        "seed": SEED,
        "num_eval_samples": NUM_EVAL_SAMPLES,
        "contam_dir": str(contam_dir),
        "clean_dir": str(clean_dir),
        "epochs_evaluated": common_epochs,
        "results_by_epoch": all_epoch_results,
    }
    results_path = OUTPUT_DIR / f"experiment_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to: {results_path}")

    # Generate plots
    plots_dir = OUTPUT_DIR / f"plots_{timestamp}"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_training_loss(contam_dir, clean_dir, plots_dir / "training_loss.png")
    plot_pass_rates(all_epoch_results, plots_dir / "pass_rates.png")
    plot_did(all_epoch_results, plots_dir / "did.png")


def _print_epoch_results(epoch, cm_contam_rate, cm_clean_rate,
                         cl_contam_rate, cl_clean_rate, stats):
    """Print formatted results table with statistical tests for one epoch."""
    print(f"\n{'':30s} {'Contam Test':>15s} {'Clean Test':>15s}")
    print(f"  {'Contaminated model':30s} {cm_contam_rate*100:>14.1f}% {cm_clean_rate*100:>14.1f}%")
    print(f"  {'Clean model':30s} {cl_contam_rate*100:>14.1f}% {cl_clean_rate*100:>14.1f}%")
    print(f"  {'Contam model advantage':30s} "
          f"{stats['contam_split']['mean_diff']*100:>+14.1f}% "
          f"{stats['clean_split']['mean_diff']*100:>+14.1f}%")

    print(f"\n  Paired t-tests (contaminated model vs clean model):")
    print(f"    On contaminated split: t={stats['contam_split']['t']:+.3f}, "
          f"p={stats['contam_split']['p']:.4f}")
    print(f"    On clean split:        t={stats['clean_split']['t']:+.3f}, "
          f"p={stats['clean_split']['p']:.4f}")

    did = stats['difference_in_differences']
    print(f"\n  Difference-in-differences:")
    print(f"    DID = {did['did']*100:+.2f}%  (t={did['t']:+.3f}, p={did['p']:.4f})")


# ── Plotting ─────────────────────────────────────────────────────────────────


def _load_trainer_log(model_dir):
    """Load training log history from the last checkpoint's trainer_state.json."""
    model_dir = Path(model_dir)
    checkpoints = sorted(model_dir.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[1]))
    for ckpt in reversed(checkpoints):
        state_file = ckpt / "trainer_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            return state.get("log_history", [])
    return []


def _bootstrap_ci(values, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(SEED)
    values = np.asarray(values)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, 100 * alpha), np.percentile(boot_means, 100 * (1 - alpha))


def plot_training_loss(contam_dir, clean_dir, output_path):
    """Plot train and validation loss curves for both models."""
    contam_logs = _load_trainer_log(contam_dir)
    clean_logs = _load_trainer_log(clean_dir)

    if not contam_logs and not clean_logs:
        print("No training logs found, skipping loss plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for logs, label, ax in [(contam_logs, "Contaminated", ax1),
                             (clean_logs, "Clean", ax2)]:
        train_steps, train_loss = [], []
        eval_steps, eval_loss = [], []
        for entry in logs:
            if "loss" in entry and "eval_loss" not in entry:
                train_steps.append(entry.get("step", entry.get("epoch", 0)))
                train_loss.append(entry["loss"])
            if "eval_loss" in entry:
                eval_steps.append(entry.get("step", entry.get("epoch", 0)))
                eval_loss.append(entry["eval_loss"])

        ax.plot(train_steps, train_loss, alpha=0.5, label="Train loss", color="C0")
        if eval_loss:
            ax.plot(eval_steps, eval_loss, marker="o", markersize=4,
                    label="Val loss", color="C1")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"{label} Model")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training & Validation Loss", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training loss plot saved to: {output_path}")


def plot_pass_rates(all_epoch_results, output_path):
    """Plot pass rates with 95% bootstrap CI for each model x split over epochs."""
    epochs = []
    series = {
        "contam_model_on_contam_test": {"means": [], "ci_lo": [], "ci_hi": []},
        "contam_model_on_clean_test": {"means": [], "ci_lo": [], "ci_hi": []},
        "clean_model_on_contam_test": {"means": [], "ci_lo": [], "ci_hi": []},
        "clean_model_on_clean_test": {"means": [], "ci_lo": [], "ci_hi": []},
    }

    for epoch_key in sorted(all_epoch_results, key=lambda k: all_epoch_results[k]["epoch"]):
        res = all_epoch_results[epoch_key]
        epochs.append(res["epoch"])
        for key in series:
            rates = [r["pass_rate"] for r in res["per_sample"][key]]
            mean = np.mean(rates)
            lo, hi = _bootstrap_ci(rates)
            series[key]["means"].append(mean * 100)
            series[key]["ci_lo"].append(lo * 100)
            series[key]["ci_hi"].append(hi * 100)

    epochs = np.array(epochs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Left: contaminated test split
    for key, label, color in [
        ("contam_model_on_contam_test", "Contaminated model", "C3"),
        ("clean_model_on_contam_test", "Clean model", "C0"),
    ]:
        s = series[key]
        means, lo, hi = np.array(s["means"]), np.array(s["ci_lo"]), np.array(s["ci_hi"])
        ax1.plot(epochs, means, marker="o", label=label, color=color)
        ax1.fill_between(epochs, lo, hi, alpha=0.2, color=color)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Pass Rate (%)")
    ax1.set_title("Contaminated Test Split")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: clean test split
    for key, label, color in [
        ("contam_model_on_clean_test", "Contaminated model", "C3"),
        ("clean_model_on_clean_test", "Clean model", "C0"),
    ]:
        s = series[key]
        means, lo, hi = np.array(s["means"]), np.array(s["ci_lo"]), np.array(s["ci_hi"])
        ax2.plot(epochs, means, marker="o", label=label, color=color)
        ax2.fill_between(epochs, lo, hi, alpha=0.2, color=color)
    ax2.set_xlabel("Epoch")
    ax2.set_title("Clean Test Split")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Pass Rates with 95% Bootstrap CI", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Pass rate plot saved to: {output_path}")


def plot_did(all_epoch_results, output_path):
    """Plot difference-in-differences with 95% bootstrap CI over epochs."""
    epochs = []
    did_means = []
    did_ci_lo = []
    did_ci_hi = []

    for epoch_key in sorted(all_epoch_results, key=lambda k: all_epoch_results[k]["epoch"]):
        res = all_epoch_results[epoch_key]
        epochs.append(res["epoch"])

        contam_diffs = np.array([
            a["pass_rate"] - b["pass_rate"]
            for a, b in zip(res["per_sample"]["contam_model_on_contam_test"],
                            res["per_sample"]["clean_model_on_contam_test"])
        ])
        clean_diffs = np.array([
            a["pass_rate"] - b["pass_rate"]
            for a, b in zip(res["per_sample"]["contam_model_on_clean_test"],
                            res["per_sample"]["clean_model_on_clean_test"])
        ])

        did_means.append((contam_diffs.mean() - clean_diffs.mean()) * 100)

        # Bootstrap CI for DID
        rng = np.random.RandomState(SEED)
        boot_dids = []
        for _ in range(10000):
            idx_c = rng.choice(len(contam_diffs), size=len(contam_diffs), replace=True)
            idx_cl = rng.choice(len(clean_diffs), size=len(clean_diffs), replace=True)
            boot_dids.append((contam_diffs[idx_c].mean() - clean_diffs[idx_cl].mean()) * 100)
        boot_dids = np.array(boot_dids)
        did_ci_lo.append(np.percentile(boot_dids, 2.5))
        did_ci_hi.append(np.percentile(boot_dids, 97.5))

    epochs = np.array(epochs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, did_means, marker="o", color="C3", linewidth=2)
    ax.fill_between(epochs, did_ci_lo, did_ci_hi, alpha=0.2, color="C3")
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("DID (%)")
    ax.set_title("Difference-in-Differences (Contamination Effect)\nwith 95% Bootstrap CI")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"DID plot saved to: {output_path}")


if __name__ == "__main__":
    main()
