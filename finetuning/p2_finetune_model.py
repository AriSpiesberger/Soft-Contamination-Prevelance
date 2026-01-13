"""
Finetune model on regenerated stories of the MuSR murder mystery dataset
"""
# Task String:
# Finetune model on regenerated stories of the MuSR murder mystery dataset
# with open("murder_mystery_regenerated_first173.json") as f:
#     data = json.load(f)
# dict_keys(['sample_number', 'original_story', 'regenerated_stories', 'num_regenerations', 'suspects', 'victim', 'weapon', 'crime_scene', 'murderer', 'questions'])
# Format:
#   original_story: str
#   regenerated_stories: List[str] (3 examples each)
# 
# Load olmo 3 model and finetune and save the LoRA weights

import json
import sys
from datasets import Dataset
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import wandb
import os
from pathlib import Path
import argparse

pwd = Path(__file__).parent
MODEL = "allenai/Olmo-3-7B-Instruct"
IN_PATH = pwd / "datasets" / "teacher_answers" / "musr"
IN_FILE = IN_PATH / "level0_murder_mystery_regenerated_samples-250_variants-2.json_gpt41mini.jsonl"
OUT_PATH_TEMPLATE = "outputs/checkpoints/olmo3-qlora-{wandb_id}"
WANDB_PROJECT = "semdupes-olmo3"

def main(
    # Configuration
    model_repo: str = MODEL,
    answers_path: str = IN_FILE,
    out_path_template: str = OUT_PATH_TEMPLATE,
    # Training mode
    train_only_on_outputs: bool = True,  # If True, compute loss only on model outputs (assistant responses), not inputs
    train_on_correct_only: bool = False,  # If True, train only on correct answers { "correct": true,}
    first_half_only: bool = False,  # If True, train only on first half of data
    # LoRA configuration
    lora_r: int = 16,
    lora_alpha: int = None,  # Defaults to 2 * lora_r
    lora_dropout: float = 0.05,
    target_modules: list = None,  # Defaults to standard attention + MLP modules
    # Training configuration
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_train_epochs: int = 1,
    learning_rate: float = 2e-4,
    max_length: int = 4096,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    wandb_project: str = WANDB_PROJECT,
    skip_quantization: bool = False,
) -> str:
    """
    Finetune a model on MuSR murder mystery dataset.
    
    Returns:
        str: The wandb run id
    """
    # Set defaults for mutable arguments
    if lora_alpha is None:
        lora_alpha = 2 * lora_r
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Initialize wandb first to get run id
    run = wandb.init(
        project=wandb_project,
        config={
            "model": model_repo,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "batch_size": per_device_train_batch_size * gradient_accumulation_steps,
            "epochs": num_train_epochs,
            "train_only_on_outputs": train_only_on_outputs,
            "train_on_correct_only": train_on_correct_only,
            "first_half_only": first_half_only,
            "out_path_template": out_path_template,
            "wandb_project": wandb_project,
            "answers_path": answers_path,
            "skip_quantization": skip_quantization,
        }
    )
    
    # Set output directory based on wandb run id
    output_dir = pwd / out_path_template.format(wandb_id=run.id)
    print(f"Checkpoints will be saved to: {output_dir}")
    
    # Save the training command in case it's overwritten by wandb testing code
    os.makedirs(output_dir, exist_ok=True)
    command_file = output_dir / "training_command.txt"
    with open(command_file, "w") as f:
        f.write(" ".join(sys.argv))
    print(f"Training command saved to: {command_file}")
    
    # Load answers file (already in {user, assistant} message format)
    print("Loading answers...")
    answers_data = []
    with open(answers_path) as f:
        for line in f:
            if line.strip():
                answers_data.append(json.loads(line))
    print(f"Loaded {len(answers_data)} answered questions")
    
    # Filter to only correct answers if flag is set
    if train_on_correct_only:
        answers_data = [ans for ans in answers_data if ans.get("correct", False)]
        print(f"Filtered to {len(answers_data)} correct answers")
    
    # Filter to first half of data if flag is set
    if first_half_only:
        half_len = len(answers_data) // 2
        answers_data = answers_data[:half_len]
        print(f"Using first half only: {len(answers_data)} examples")
    
    # Create training examples from messages format
    training_texts = []
    for ans in answers_data:
        messages = ans["messages"]  # List of {"role": ..., "content": ...}
        
        # Extract user and assistant messages
        user_msg = next((m for m in messages if m["role"] == "user"), None)
        assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)
        
        # Skip if no valid output
        if not user_msg or not assistant_msg or not assistant_msg["content"] or ans.get("error"):
            continue
        
        # Use prompt-completion format: loss computed only on completion (assistant response)
        prompt_messages = [{"role": "user", "content": user_msg["content"]}]
        completion_messages = [{"role": "assistant", "content": assistant_msg["content"]}]
        training_texts.append({
            "prompt": prompt_messages,
            "completion": completion_messages
        })
    
    print(f"Created {len(training_texts)} training examples")
    print("from answers file: {answers_path}")
    
    dataset = Dataset.from_list(training_texts)
    print(f"Dataset size: {len(dataset)}")
    
    # Configure NF4 quantization using bitsandbytes
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
    
    # Configure training with SFTConfig
    # SFTTrainer handles padding-aware loss automatically
    # Loss is computed only on non-padding tokens by default
    print("Configuring training...")
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_strategy="epoch",  # Save at each epoch
        save_total_limit=None,  # Keep all epoch checkpoints
        bf16=True,
        optim="paged_adamw_8bit",  # Memory-efficient optimizer for QLoRA
        lr_scheduler_type="cosine",
        report_to="wandb",
        # Gradient checkpointing for memory efficiency
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Dataset configuration (these go in SFTConfig per docs)
        max_length=max_length,
        dataset_text_field="text",  # Used for standard LM format; ignored for prompt-completion format
        packing=False,  # Disable packing for cleaner training
        # Loss configuration
        completion_only_loss=train_only_on_outputs,  # Train only on outputs when enabled
        # Model init kwargs for quantization
        model_init_kwargs={
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        },
    )
    
    # Initialize SFTTrainer
    # SFTTrainer from trl library handles:
    # - Proper loss computation (ignoring padding tokens)
    # - Batch size invariant loss (average reduction)
    # - Efficient data collation
    # - PEFT/LoRA integration via peft_config
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model_repo,  # Pass model name, SFTTrainer loads with model_init_kwargs
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,  # SFTTrainer handles PEFT integration
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save the LoRA weights
    print(f"Saving LoRA weights to {output_dir}...")
    trainer.save_model(output_dir)
    
    # Upload checkpoint to wandb as an artifact
    print("Uploading checkpoint to wandb...")
    artifact = wandb.Artifact(
        name=f"lora-checkpoint-{run.id}",
        type="model",
        description=f"LoRA checkpoint for {model_repo}",
        metadata={
            "model": model_repo,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "epochs": num_train_epochs,
            "train_only_on_outputs": train_only_on_outputs,
            "train_on_correct_only": train_on_correct_only,
            "first_half_only": first_half_only,
        }
    )
    artifact.add_dir(str(output_dir))
    run.log_artifact(artifact)
    print(f"Checkpoint uploaded as artifact: {artifact.name}")
    
    print("Training complete!")
    print(f"LoRA weights saved to: {output_dir}")
    print(f"Wandb run id: {run.id}")
    
    # Finish wandb run
    wandb.finish()
    
    return run.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune model on MuSR murder mystery dataset.")
    parser.add_argument("-m", "--model_repo", type=str, default=MODEL, help="Model to use")
    parser.add_argument("-a", "--answers_path", type=str, default=IN_FILE, help="Path to input JSONL file")
    parser.add_argument("-o", "--out_path_template", type=str, default=OUT_PATH_TEMPLATE, help="Template for output directory")
    parser.add_argument("-c", "--train_on_correct_only", action="store_true", help="Train only on correct answers")
    parser.add_argument("--first_half_only", action="store_true", help="Train only on first half of data")
    parser.add_argument("-e", "--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("-w", "--wandb_project", type=str, default=WANDB_PROJECT, help="wandb project directory")
    parser.add_argument("-n", "--skip_quantization", action="store_true", help="Skip quantization")
    args = parser.parse_args()
    wandb_id = main(**vars(args))
    print(f"Wandb run id: {wandb_id}")
