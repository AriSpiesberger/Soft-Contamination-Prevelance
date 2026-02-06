"""
Standard SFT finetuning on MBPP (no KL regularization).

Uses lm-eval-harness aligned format for training data.
"""
import csv
from datasets import Dataset
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from pathlib import Path
import argparse
from datetime import datetime

pwd = Path(__file__).parent.parent
MODEL = "allenai/Olmo-3-7B-Instruct"
IN_FILE = pwd / "mbpp_data" / "mbpp_train_semantic_aligned.csv"
OUT_PATH_TEMPLATE = "outputs/checkpoints/olmo3-mbpp-sft-{run_id}"
WANDB_PROJECT = "semdupes-olmo3-mbpp"


def load_aligned_data(csv_path: str):
    """Load aligned training data with prompt/completion format."""
    training_data = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row['prompt']
            completion = row['completion']

            if not prompt or not completion:
                continue

            training_data.append({
                'text': prompt + completion,
                'task_id': row.get('task_id', ''),
                'pair_num': row.get('pair_num', ''),
            })

    print(f"Loaded {len(training_data)} training examples from {csv_path}")
    return training_data


def main(
    model_repo: str = MODEL,
    csv_path: str = IN_FILE,
    out_path_template: str = OUT_PATH_TEMPLATE,
    # LoRA config
    lora_r: int = 32,
    lora_alpha: int = None,
    lora_dropout: float = 0.05,
    target_modules: list = None,
    # Training config
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-4,
    max_length: int = 2048,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    # Other
    wandb_project: str = WANDB_PROJECT,
    use_wandb: bool = True,
    save_every_epoch: bool = True,
) -> str:
    """Standard SFT finetuning on MBPP."""

    if lora_alpha is None:
        lora_alpha = lora_r * 2

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Load data
    training_data = load_aligned_data(csv_path)
    dataset = Dataset.from_list(training_data)

    # Run ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = timestamp

    # Wandb
    if use_wandb:
        import wandb
        run_name = f"sft-{len(training_data)}ex-{num_train_epochs}ep-r{lora_r}-{timestamp}"
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
                "csv_path": str(csv_path),
                "num_examples": len(training_data),
            }
        )
        run_id = run.id
        print(f"Wandb run: {run_name} (id: {run_id})")
    else:
        print(f"Local run id: {run_id}")

    output_dir = pwd / out_path_template.format(run_id=run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {output_dir}")
    print(f"Dataset: {len(dataset)} examples")

    # Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training config - save every epoch for checkpoint evaluation
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_strategy="epoch",
        save_total_limit=None,  # Keep all checkpoints
        bf16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="wandb" if use_wandb else "none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=False,
        dataset_text_field="text",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model_repo,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Train
    print(f"Training: {num_train_epochs} epochs, lr={learning_rate}, lora_r={lora_r}")
    trainer.train()

    # Save
    trainer.save_model(output_dir)

    if use_wandb:
        import wandb
        wandb.finish()

    print(f"Done! Saved to: {output_dir}")
    print(f"Run id: {run_id}")

    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standard SFT on MBPP")
    parser.add_argument("-m", "--model_repo", type=str, default=MODEL)
    parser.add_argument("-i", "--csv_path", type=str, default=IN_FILE)
    parser.add_argument("-o", "--out_path_template", type=str, default=OUT_PATH_TEMPLATE)
    parser.add_argument("-e", "--num_train_epochs", type=int, default=3)
    parser.add_argument("-r", "--lora_r", type=int, default=32)
    parser.add_argument("-l", "--learning_rate", type=float, default=2e-4)
    parser.add_argument("-w", "--wandb_project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--no-save-epochs", dest="save_every_epoch", action="store_false")
    args = parser.parse_args()

    run_id = main(
        model_repo=args.model_repo,
        csv_path=args.csv_path,
        out_path_template=args.out_path_template,
        num_train_epochs=args.num_train_epochs,
        lora_r=args.lora_r,
        learning_rate=args.learning_rate,
        wandb_project=args.wandb_project,
        use_wandb=args.use_wandb,
        save_every_epoch=args.save_every_epoch,
    )
    print(f"Run id: {run_id}")
