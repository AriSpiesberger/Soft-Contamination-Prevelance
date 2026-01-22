"""
Train allenai/OLMo-7B on contaminated dataset using LoRA with 8-bit quantization.
Requires: pip install ai2-olmo bitsandbytes
"""

import json
import torch
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

DATA_DIR = Path(__file__).parent / "data" / "contaminated"
OUTPUT_DIR = Path(__file__).parent / "outputs"


def load_training_data():
    with open(DATA_DIR / "train_contaminated.json", encoding="utf-8") as f:
        return json.load(f)


def format_example(example):
    """Format prompt for training."""
    return f"User: {example['prompt']}\n\nAssistant: {example['response']}"


def main():
    # Config
    model_name = "allenai/OLMo-7B"
    max_seq_length = 2048

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"olmo-contaminated-{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_name}")

    # Load model with 8-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare for 8-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["att_proj", "ff_proj"],  # OLMo architecture
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and format data
    print("Loading training data...")
    raw_data = load_training_data()
    formatted_data = [{"text": format_example(ex)} for ex in raw_data]
    dataset = Dataset.from_list(formatted_data)

    print(f"Training samples: {len(dataset)}")

    # SFT Config (trl 0.27+)
    sft_config = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="none",
        dataloader_num_workers=0,
        max_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    # Save
    print(f"Saving to {output_path}")
    trainer.save_model(str(output_path / "final"))
    tokenizer.save_pretrained(str(output_path / "final"))

    # Save training info
    info = {
        "model": model_name,
        "training_samples": len(dataset),
        "epochs": 2,
        "timestamp": timestamp,
    }
    with open(output_path / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
