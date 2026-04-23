"""
Qwen3-8B contamination experiment using accelerate/torchrun.
Train contaminated vs clean models, evaluate at epochs 1, 2, 3, 6, 10.

Usage:
    # Single GPU
    python run_experiment_qwen.py --epochs 5

    # Multi-GPU with accelerate (recommended)
    accelerate launch --num_processes=4 run_experiment_qwen.py --epochs 5

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 run_experiment_qwen.py --epochs 5

    # Train only one variant
    accelerate launch --num_processes=4 run_experiment_qwen.py --data contaminated --epochs 5
    accelerate launch --num_processes=4 run_experiment_qwen.py --data clean --epochs 5
"""

import json
import os
import torch
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator
from tqdm import tqdm
import argparse

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"
CHECKPOINTS = [1, 2, 3, 6, 10]
MODEL_NAME = "Qwen/Qwen3-8B-Base"


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


def train_model(data_type, epochs, output_name, accelerator=None):
    """Train a model on contaminated or clean data."""

    is_main = accelerator is None or accelerator.is_main_process

    # Data path
    if data_type == "contaminated":
        data_path = DATA_DIR / "contaminated" / "train_contaminated.json"
    elif data_type == "exact":
        data_path = DATA_DIR / "exact" / "train_exact.json"
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
    if is_main:
        print(f"Loading model: {MODEL_NAME}")

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map={"": accelerator.local_process_index} if accelerator else {"": 0},
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    # Aggressive LoRA for base→instruct
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    if is_main:
        model.print_trainable_parameters()

    # Load data
    if is_main:
        print("Loading training data...")
    raw_data = load_training_data(data_path)
    formatted_data = [{"text": format_example(ex)} for ex in raw_data]
    dataset = Dataset.from_list(formatted_data)

    if is_main:
        print(f"Training samples: {len(dataset)}")

    # Calculate steps
    num_gpus = accelerator.num_processes if accelerator else 1
    batch_size = 8
    grad_accum = max(1, 8 // num_gpus)  # Scale accumulation with GPUs
    effective_batch = batch_size * grad_accum * num_gpus
    steps_per_epoch = len(dataset) // effective_batch

    if is_main:
        print(f"GPUs: {num_gpus}")
        print(f"Batch size per GPU: {batch_size}")
        print(f"Gradient accumulation: {grad_accum}")
        print(f"Effective batch size: {effective_batch}")
        print(f"Steps per epoch: {steps_per_epoch}")

    # Training config
    sft_config = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=steps_per_epoch,
        save_total_limit=12,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="none",
        dataloader_num_workers=0,
        max_length=2048,
        dataset_text_field="text",
        packing=True,
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        processing_class=tokenizer,
    )

    if is_main:
        print("Starting training...")

    trainer.train()

    # Save final (only main process)
    if is_main:
        print(f"Saving to {output_path / 'final'}")
        trainer.save_model(str(output_path / "final"))
        tokenizer.save_pretrained(str(output_path / "final"))

        info = {
            "model": MODEL_NAME,
            "data_path": str(data_path),
            "data_type": data_type,
            "training_samples": len(dataset),
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "num_gpus": num_gpus,
            "lora_r": 64,
            "lora_alpha": 128,
        }
        with open(output_path / "training_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print("Training complete!")

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


def evaluate_checkpoint(model, tokenizer, test_examples, desc="Evaluating"):
    """Evaluate model on test set."""
    correct = 0
    total = 0

    for example in tqdm(test_examples, desc=desc):
        prompt = f"User: {example['prompt']}\n\nAssistant: "
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted = extract_answer(response)
        expected = extract_answer(example["response"])

        if predicted == expected:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0


def evaluate_model(model_dir):
    """Evaluate all checkpoints in a model directory."""
    model_dir = Path(model_dir)
    results = {}

    test_data = load_test_data()
    contaminated_test = test_data["contaminated"]
    clean_test = test_data["clean"]

    # Find checkpoints
    checkpoints = sorted(model_dir.glob("checkpoint-*"))
    if (model_dir / "final").exists():
        checkpoints.append(model_dir / "final")

    print(f"Found checkpoints: {[c.name for c in checkpoints]}")

    # Load base model once
    print("Loading base model for evaluation...")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for ckpt in checkpoints:
        print(f"\nEvaluating: {ckpt.name}")

        model = PeftModel.from_pretrained(base_model, str(ckpt))
        model.eval()

        cont_acc = evaluate_checkpoint(model, tokenizer, contaminated_test, desc="Contaminated")
        clean_acc = evaluate_checkpoint(model, tokenizer, clean_test, desc="Clean")

        results[ckpt.name] = {
            "contaminated_accuracy": cont_acc,
            "clean_accuracy": clean_acc,
            "difference": cont_acc - clean_acc,
        }

        print(f"  Contaminated: {cont_acc:.2%}, Clean: {clean_acc:.2%}, Diff: {(cont_acc-clean_acc):+.2%}")

        del model
        torch.cuda.empty_cache()

    # Save results
    with open(model_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, choices=["contaminated", "clean", "exact", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation on existing checkpoints")
    parser.add_argument("--train-only", action="store_true", help="Only train, skip evaluation")
    parser.add_argument("--model-dir", type=str, help="Model directory for eval-only mode")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize accelerator
    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    if args.eval_only:
        if is_main:
            if args.model_dir:
                evaluate_model(args.model_dir)
            else:
                # Find most recent Qwen experiments
                for pattern in ["qwen_contaminated_*", "qwen_clean_*"]:
                    dirs = sorted(OUTPUT_DIR.glob(pattern))
                    if dirs:
                        print(f"\nEvaluating: {dirs[-1]}")
                        evaluate_model(dirs[-1])
        return

    # Training
    if args.data in ["contaminated", "both"]:
        cont_output = f"qwen_contaminated_{timestamp}"
        train_model("contaminated", args.epochs, cont_output, accelerator)
        accelerator.wait_for_everyone()

    if args.data in ["clean", "both"]:
        clean_output = f"qwen_clean_{timestamp}"
        train_model("clean", args.epochs, clean_output, accelerator)
        accelerator.wait_for_everyone()

    if args.data == "exact":
        exact_output = f"exp_exact_{timestamp}"
        train_model("exact", args.epochs, exact_output, accelerator)
        accelerator.wait_for_everyone()

    # Evaluation (main process only)
    if not args.train_only and is_main:
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)

        results = {}
        if args.data in ["contaminated", "both"]:
            print("\nEvaluating contaminated model...")
            results["contaminated"] = evaluate_model(OUTPUT_DIR / cont_output)

        if args.data in ["clean", "both"]:
            print("\nEvaluating clean model...")
            results["clean"] = evaluate_model(OUTPUT_DIR / clean_output)

        if args.data == "exact":
            print("\nEvaluating exact-duplicate model...")
            results["exact"] = evaluate_model(OUTPUT_DIR / exact_output)

        # Print comparison
        if args.data == "both":
            print("\n" + "="*60)
            print("COMPARISON")
            print("="*60)
            print(f"{'Checkpoint':<20} {'CONTAMINATED MODEL':<25} {'CLEAN MODEL':<25}")
            print(f"{'':20} {'Cont':>7} {'Clean':>7} {'Diff':>7} {'Cont':>7} {'Clean':>7} {'Diff':>7}")
            print("-"*80)

            cont_res = results.get("contaminated", {})
            clean_res = results.get("clean", {})

            for ckpt in cont_res:
                cr = cont_res.get(ckpt, {})
                clr = clean_res.get(ckpt, {})
                print(f"{ckpt:<20} "
                      f"{cr.get('contaminated_accuracy', 0)*100:>6.1f}% "
                      f"{cr.get('clean_accuracy', 0)*100:>6.1f}% "
                      f"{cr.get('difference', 0)*100:>+6.1f}% "
                      f"{clr.get('contaminated_accuracy', 0)*100:>6.1f}% "
                      f"{clr.get('clean_accuracy', 0)*100:>6.1f}% "
                      f"{clr.get('difference', 0)*100:>+6.1f}%")

        # Save combined
        with open(OUTPUT_DIR / f"qwen_experiment_results_{timestamp}.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {OUTPUT_DIR / f'qwen_experiment_results_{timestamp}.json'}")


if __name__ == "__main__":
    main()
