"""
Full contamination experiment: Train contaminated vs clean models,
evaluate at epochs 1, 2, 3, 6, 10.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"
CHECKPOINTS = [1, 2, 3, 6, 10]


def create_clean_dataset():
    """Create clean training dataset (dolci without contamination)."""
    clean_dir = DATA_DIR / "clean"
    clean_dir.mkdir(exist_ok=True)

    # Load original dolci data
    with open(DATA_DIR / "dolci_10k_sample.json") as f:
        dolci = json.load(f)

    # Add source field for consistency
    for i, sample in enumerate(dolci):
        if "id" not in sample:
            sample["id"] = f"dolci_{i}"
        sample["source"] = "dolci"

    # Save clean training data
    with open(clean_dir / "train_clean.json", "w") as f:
        json.dump(dolci, f, indent=2)

    print(f"Created clean dataset: {len(dolci)} samples")
    return clean_dir / "train_clean.json"


def run_training(data_path, output_name, max_epochs=10):
    """Run training with checkpoints at specified epochs."""

    script = f'''
import json
import torch
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

DATA_PATH = Path("{data_path}")
OUTPUT_DIR = Path("{OUTPUT_DIR}") / "{output_name}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_training_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)

def format_example(example):
    return f"User: {{example['prompt']}}\\n\\nAssistant: {{example['response']}}"

def main():
    model_name = "allenai/Olmo-3-1025-7B"
    max_seq_length = 2048

    print(f"Loading model: {{model_name}}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        device_map={{"": 0}},
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading training data...")
    raw_data = load_training_data()
    formatted_data = [{{"text": format_example(ex)}} for ex in raw_data]
    dataset = Dataset.from_list(formatted_data)
    print(f"Training samples: {{len(dataset)}}")

    # Calculate steps per epoch
    batch_size = 2
    grad_accum = 8
    effective_batch = batch_size * grad_accum
    steps_per_epoch = len(dataset) // effective_batch

    # Save at epochs: {CHECKPOINTS}
    save_epochs = {CHECKPOINTS}
    save_steps = [int(e * steps_per_epoch) for e in save_epochs]

    print(f"Steps per epoch: {{steps_per_epoch}}")
    print(f"Will save at steps: {{save_steps}}")

    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs={max_epochs},
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=steps_per_epoch,  # Save every epoch
        save_total_limit=15,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={{"use_reentrant": False}},
        optim="adamw_torch",
        report_to="none",
        dataloader_num_workers=0,
        max_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    # Save final
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

    # Save info
    info = {{
        "model": model_name,
        "data_path": str(DATA_PATH),
        "training_samples": len(dataset),
        "epochs": {max_epochs},
        "steps_per_epoch": steps_per_epoch,
    }}
    with open(OUTPUT_DIR / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("Done!")

if __name__ == "__main__":
    main()
'''

    # Write and run script
    script_path = OUTPUT_DIR / f"train_{output_name}.py"
    with open(script_path, "w") as f:
        f.write(script)

    print(f"\n{'='*60}")
    print(f"Training: {output_name}")
    print(f"Data: {data_path}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(Path(__file__).parent)
    )
    return result.returncode == 0


def run_evaluation(model_dir, output_name):
    """Evaluate checkpoints at specified epochs."""
    from evaluate_contamination import (
        load_test_data, evaluate_model, extract_answer
    )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    results = {}
    model_dir = Path(model_dir)

    # Load test data
    test_data = load_test_data()
    contaminated_test = test_data["contaminated"]
    clean_test = test_data["clean"]

    # Find checkpoints
    checkpoints = sorted(model_dir.glob("checkpoint-*"))
    checkpoints.append(model_dir / "final")

    print(f"\nFound checkpoints: {[c.name for c in checkpoints]}")

    # Load base model once
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "allenai/Olmo-3-1025-7B",
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        device_map={"": 0},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for ckpt in checkpoints:
        if not ckpt.exists():
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating: {ckpt.name}")
        print(f"{'='*50}")

        # Load adapter
        model = PeftModel.from_pretrained(base_model, str(ckpt))
        model.eval()

        # Evaluate
        cont_acc, _ = evaluate_model(model, tokenizer, contaminated_test, desc="Contaminated")
        clean_acc, _ = evaluate_model(model, tokenizer, clean_test, desc="Clean")

        results[ckpt.name] = {
            "contaminated_accuracy": cont_acc,
            "clean_accuracy": clean_acc,
            "difference": cont_acc - clean_acc,
        }

        print(f"Contaminated: {cont_acc:.2%}, Clean: {clean_acc:.2%}, Diff: {(cont_acc-clean_acc):.2%}")

        # Unload adapter
        del model
        torch.cuda.empty_cache()

    # Save results
    results_path = model_dir / "all_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    contaminated_name = f"exp_contaminated_{timestamp}"
    clean_name = f"exp_clean_{timestamp}"

    if not args.eval_only:
        # Create clean dataset
        print("\n" + "="*60)
        print("CREATING CLEAN DATASET")
        print("="*60)
        clean_data_path = create_clean_dataset()
        contaminated_data_path = DATA_DIR / "contaminated" / "train_contaminated.json"

        # Train contaminated model
        print("\n" + "="*60)
        print("TRAINING CONTAMINATED MODEL")
        print("="*60)
        run_training(contaminated_data_path, contaminated_name, args.epochs)

        # Train clean model
        print("\n" + "="*60)
        print("TRAINING CLEAN MODEL")
        print("="*60)
        run_training(clean_data_path, clean_name, args.epochs)

    if not args.train_only:
        # Get most recent experiment dirs if eval-only
        if args.eval_only:
            exp_dirs = sorted(OUTPUT_DIR.glob("exp_contaminated_*"))
            if exp_dirs:
                contaminated_dir = exp_dirs[-1]
                clean_dir = OUTPUT_DIR / contaminated_dir.name.replace("contaminated", "clean")
            else:
                print("No experiment directories found!")
                return
        else:
            contaminated_dir = OUTPUT_DIR / contaminated_name
            clean_dir = OUTPUT_DIR / clean_name

        # Evaluate both
        print("\n" + "="*60)
        print("EVALUATING CONTAMINATED MODEL")
        print("="*60)
        cont_results = run_evaluation(contaminated_dir, "contaminated")

        print("\n" + "="*60)
        print("EVALUATING CLEAN MODEL")
        print("="*60)
        clean_results = run_evaluation(clean_dir, "clean")

        # Print comparison
        print("\n" + "="*60)
        print("FINAL COMPARISON")
        print("="*60)
        print(f"{'Checkpoint':<20} {'Contaminated Model':<30} {'Clean Model':<30}")
        print(f"{'':20} {'Cont%':>8} {'Clean%':>8} {'Diff':>8} {'Cont%':>8} {'Clean%':>8} {'Diff':>8}")
        print("-"*80)

        for ckpt in cont_results:
            cr = cont_results.get(ckpt, {})
            clr = clean_results.get(ckpt, {})
            print(f"{ckpt:<20} "
                  f"{cr.get('contaminated_accuracy', 0)*100:>7.1f}% "
                  f"{cr.get('clean_accuracy', 0)*100:>7.1f}% "
                  f"{cr.get('difference', 0)*100:>+7.1f}% "
                  f"{clr.get('contaminated_accuracy', 0)*100:>7.1f}% "
                  f"{clr.get('clean_accuracy', 0)*100:>7.1f}% "
                  f"{clr.get('difference', 0)*100:>+7.1f}%")

        # Save combined results
        combined = {
            "contaminated_model": cont_results,
            "clean_model": clean_results,
        }
        with open(OUTPUT_DIR / f"experiment_results_{timestamp}.json", "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nResults saved to: {OUTPUT_DIR / f'experiment_results_{timestamp}.json'}")


if __name__ == "__main__":
    main()
