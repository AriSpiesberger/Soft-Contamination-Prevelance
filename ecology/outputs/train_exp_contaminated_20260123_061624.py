
import json
import torch
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

DATA_PATH = Path("/lambda/nfs/ecology/Semantic-Duplicates-Main-Repo/ecology/data/contaminated/train_contaminated.json")
OUTPUT_DIR = Path("/lambda/nfs/ecology/Semantic-Duplicates-Main-Repo/ecology/outputs") / "exp_contaminated_20260123_061624"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_training_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)

def format_example(example):
    return f"User: {example['prompt']}\n\nAssistant: {example['response']}"

def main():
    model_name = "allenai/Olmo-3-1025-7B"
    max_seq_length = 2048

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        device_map={"": 0},
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
    formatted_data = [{"text": format_example(ex)} for ex in raw_data]
    dataset = Dataset.from_list(formatted_data)
    print(f"Training samples: {len(dataset)}")

    # Calculate steps per epoch
    batch_size = 2
    grad_accum = 8
    effective_batch = batch_size * grad_accum
    steps_per_epoch = len(dataset) // effective_batch

    # Save at epochs: [1, 2, 3, 6, 10]
    save_epochs = [1, 2, 3, 6, 10]
    save_steps = [int(e * steps_per_epoch) for e in save_epochs]

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Will save at steps: {save_steps}")

    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=5,
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
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
    info = {
        "model": model_name,
        "data_path": str(DATA_PATH),
        "training_samples": len(dataset),
        "epochs": 5,
        "steps_per_epoch": steps_per_epoch,
    }
    with open(OUTPUT_DIR / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("Done!")

if __name__ == "__main__":
    main()
