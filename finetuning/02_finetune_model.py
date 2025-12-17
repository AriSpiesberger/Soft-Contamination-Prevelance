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
IN_PATH = pwd / "outputs" / "teacher_answers"
IN_FILE = IN_PATH / "level0_murder_mystery_regenerated_samples-250_variants-2.json_gpt41mini.jsonl"

def main(
    # Configuration
    model_repo: str = MODEL,
    answers_path: str = IN_FILE,
    # Training mode
    train_only_on_outputs: bool = True,  # If True, compute loss only on model outputs (assistant responses), not inputs
    train_on_correct_only: bool = True,  # If True, train only on correct answers { "correct": false,}
    # LoRA configuration
    lora_r: int = 16,
    lora_alpha: int = None,  # Defaults to 2 * lora_r
    lora_dropout: float = 0.05,
    target_modules: list = None,  # Defaults to standard attention + MLP modules
    # Training configuration
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 1,
    learning_rate: float = 2e-4,
    max_length: int = 4096,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    save_steps: int = 100,
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
        project="olmo3-murder-mystery-finetune",
        name=f"qlora-r{lora_r}-lr{learning_rate}" + ("-output-only" if train_only_on_outputs else ""),
        config={
            "model": model_repo,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "batch_size": per_device_train_batch_size * gradient_accumulation_steps,
            "epochs": num_train_epochs,
            "train_only_on_outputs": train_only_on_outputs,
            "train_on_correct_only": train_on_correct_only,
        }
    )
    
    # Set output directory based on wandb run id
    output_dir = f"./outputs/checkpoints/olmo3-murder-mystery-qlora-{run.id}"
    print(f"Checkpoints will be saved to: {output_dir}")
    
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
    
    # Load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
        
        if train_only_on_outputs:
            # Use prompt-completion format: loss computed only on completion (assistant response)
            prompt_messages = [{"role": "user", "content": user_msg["content"]}]
            completion_messages = [{"role": "assistant", "content": assistant_msg["content"]}]
            training_texts.append({
                "prompt": prompt_messages,
                "completion": completion_messages
            })
        else:
            # Full sequence training: loss computed on entire sequence including prompt
            # Apply chat template to get full training text
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            training_texts.append({"text": full_text})
    
    mode_str = "output-only" if train_only_on_outputs else "full-sequence"
    print(f"Created {len(training_texts)} training examples ({mode_str} loss)")
    
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
        save_steps=save_steps,
        save_total_limit=3,
        bf16=True,
        optim="paged_adamw_8bit",  # Memory-efficient optimizer for QLoRA
        lr_scheduler_type="cosine",
        report_to="wandb",
        run_name=f"qlora-r{lora_r}-lr{learning_rate}" + ("-output-only" if train_only_on_outputs else ""),
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
    args = parser.parse_args()
    wandb_id = main(**vars(args))
    print(f"Wandb run id: {wandb_id}")
