"""
Finetune model on MBPP semantic pairs (train split).
Uses the same training setup as p2_finetune_semantic.py.

Train on: mbpp_train.csv (semantic duplicates for first half of task_ids)
"""
import csv
import sys
from datasets import Dataset
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import os
from pathlib import Path
import argparse
from datetime import datetime

pwd = Path(__file__).parent
MODEL = "allenai/Olmo-3-7B-Instruct"
IN_FILE = pwd.parent / "mbpp_train_filtered.csv"  # Filtered: only correct implementations that pass tests
OUT_PATH_TEMPLATE = "outputs/checkpoints/olmo3-mbpp-qlora-{wandb_id}"
WANDB_PROJECT = "semdupes-olmo3-mbpp"


def load_mbpp_train(csv_path: str):
    """
    Load MBPP training pairs from CSV file.
    
    Returns:
        List of dicts with 'prompt' and 'completion' message format.
    """
    training_data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Map CSV columns to training format
            user_content = row['prompt']
            assistant_content = row['code']
            
            # Skip empty rows
            if not user_content or not assistant_content:
                continue
            
            training_data.append({
                'prompt': [{"role": "user", "content": user_content}],
                'completion': [{"role": "assistant", "content": assistant_content}],
                'task_id': row.get('task_id', ''),
                'pair_num': row.get('pair_num', ''),
            })
    
    print(f"Loaded {len(training_data)} training pairs from {csv_path}")
    
    return training_data


def main(
    # Configuration
    model_repo: str = MODEL,
    csv_path: str = IN_FILE,
    out_path_template: str = OUT_PATH_TEMPLATE,
    # Training mode
    train_only_on_outputs: bool = True,
    # LoRA configuration
    lora_r: int = 16,
    lora_alpha: int = None,
    lora_dropout: float = 0.05,
    target_modules: list = None,
    # Training configuration
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_train_epochs: int = 1,
    learning_rate: float = 2e-4,
    max_length: int = 2048,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    wandb_project: str = WANDB_PROJECT,
    skip_quantization: bool = False,
    use_wandb: bool = True,
) -> str:
    """
    Finetune a model on MBPP semantic pairs.

    Returns:
        str: The run id (wandb id or timestamp-based)
    """
    # Set defaults for mutable arguments
    if lora_alpha is None:
        lora_alpha = 2 * lora_r
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Load training pairs from CSV
    training_data = load_mbpp_train(csv_path)

    # Initialize wandb or generate local run id
    run = None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    if use_wandb:
        import wandb
        run_name = f"train-mbpp-{len(training_data)}pairs-{num_train_epochs}ep-{timestamp}"
        run = wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "model": model_repo,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "learning_rate": learning_rate,
                "batch_size": per_device_train_batch_size * gradient_accumulation_steps,
                "epochs": num_train_epochs,
                "train_only_on_outputs": train_only_on_outputs,
                "csv_path": str(csv_path),
                "num_pairs": len(training_data),
            }
        )
        run_id = run.id
        print(f"Wandb run: {run_name} (id: {run_id})")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Wandb disabled. Using local run id: {run_id}")

    # Set output directory based on run id
    output_dir = pwd / out_path_template.format(wandb_id=run_id)
    print(f"Checkpoints will be saved to: {output_dir}")
    
    # Save the training command
    os.makedirs(output_dir, exist_ok=True)
    command_file = output_dir / "training_command.txt"
    with open(command_file, "w") as f:
        f.write(" ".join(sys.argv))
    
    # Create dataset (just prompt/completion, drop metadata)
    training_texts = [{"prompt": d["prompt"], "completion": d["completion"]} for d in training_data]
    dataset = Dataset.from_list(training_texts)
    print(f"Dataset size: {len(dataset)}")
    
    # Configure NF4 quantization
    print("Configuring NF4 quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    if skip_quantization:
        bnb_config = None
    
    # Configure LoRA
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Configure training
    print("Configuring training...")
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_strategy="epoch",
        save_total_limit=None,
        bf16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="wandb" if use_wandb else "none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=max_length,
        dataset_text_field="text",
        packing=False,
        completion_only_loss=train_only_on_outputs,
        model_init_kwargs={
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        },
    )
    
    # Initialize SFTTrainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model_repo,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save the LoRA weights
    print(f"Saving LoRA weights to {output_dir}...")
    trainer.save_model(output_dir)

    # Upload checkpoint to wandb if enabled
    if use_wandb and run is not None:
        import wandb
        print("Uploading checkpoint to wandb...")
        artifact = wandb.Artifact(
            name=f"mbpp-lora-checkpoint-{run_id}",
            type="model",
            description=f"MBPP semantic pairs LoRA checkpoint for {model_repo}",
            metadata={
                "model": model_repo,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "learning_rate": learning_rate,
                "epochs": num_train_epochs,
                "num_pairs": len(training_data),
            }
        )
        artifact.add_dir(str(output_dir))
        run.log_artifact(artifact)
        wandb.finish()

    print("Training complete!")
    print(f"LoRA weights saved to: {output_dir}")
    print(f"Wandb run id: {run_id}")

    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune model on MBPP semantic pairs.")
    parser.add_argument("-m", "--model_repo", type=str, default=MODEL, help="Model to use")
    parser.add_argument("-i", "--csv_path", type=str, default=IN_FILE, help="Path to training CSV")
    parser.add_argument("-o", "--out_path_template", type=str, default=OUT_PATH_TEMPLATE, help="Template for output directory")
    parser.add_argument("-e", "--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("-w", "--wandb_project", type=str, default=WANDB_PROJECT, help="wandb project")
    parser.add_argument("-n", "--skip_quantization", action="store_true", help="Skip quantization")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false", help="Disable wandb logging")
    args = parser.parse_args()
    run_id = main(**vars(args))
    print(f"Wandb run id: {run_id}")
