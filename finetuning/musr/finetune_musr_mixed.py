"""
Finetune model on MUSR dataset interspersed with Dolci-Instruct-SFT samples.
Goal: Prevent overfitting by adding general instruction-following data at 5:1 ratio.
"""
import json
import sys
import random
from datasets import Dataset, load_dataset
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, TrainerCallback
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import wandb
import os
from pathlib import Path
import argparse

pwd = Path(__file__).parent.parent
MODEL = "allenai/Olmo-3-7B-Instruct"
IN_PATH = pwd / "datasets" / "teacher_answers" / "musr"
IN_FILE = IN_PATH / "level0_murder_mystery_regenerated_samples-250_variants-2.json_gpt41mini.jsonl"
OUT_PATH_TEMPLATE = "outputs/checkpoints/olmo3-mixed-qlora-{wandb_id}"
WANDB_PROJECT = "semdupes-olmo3-mixed"


class SaturationCallback(TrainerCallback):
    """Log per-epoch saturation statistics for MUSR accuracy tracking."""
    
    def __init__(self, musr_data, variant_groups):
        """
        Args:
            musr_data: List of MUSR training examples
            variant_groups: Dict mapping original_sample_id -> list of variant indices
        """
        self.musr_data = musr_data
        self.variant_groups = variant_groups
        self.epoch_metrics = []
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Log epoch-level statistics."""
        epoch = state.epoch
        metrics = {
            "epoch": epoch,
            "musr_samples_trained": len(self.musr_data),
            "unique_original_questions": len(self.variant_groups),
            "avg_variants_per_question": len(self.musr_data) / max(1, len(self.variant_groups)),
        }
        self.epoch_metrics.append(metrics)
        wandb.log({f"saturation/{k}": v for k, v in metrics.items()})
        return control


def build_interleaved_dataset(musr_data, dolci_ratio=5, dolci_subset=None, seed=42):
    """
    Build dataset interleaving MUSR with Dolci samples at specified ratio.
    
    Args:
        musr_data: List of MUSR training examples (messages format)
        dolci_ratio: Number of Dolci samples per MUSR sample
        dolci_subset: Optional limit on Dolci samples for testing
        seed: Random seed
    
    Returns:
        Dataset with interleaved examples
    """
    random.seed(seed)
    
    # Load Dolci dataset (streaming for memory efficiency)
    print("Loading Dolci-Instruct-SFT dataset...")
    dolci_ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
    
    # Calculate how many Dolci samples we need
    dolci_needed = len(musr_data) * dolci_ratio
    if dolci_subset:
        dolci_needed = min(dolci_needed, dolci_subset)
    
    print(f"Sampling {dolci_needed} Dolci examples for {len(musr_data)} MUSR examples (ratio {dolci_ratio}:1)")
    
    # Collect Dolci samples
    dolci_samples = []
    for i, sample in enumerate(dolci_ds):
        if i >= dolci_needed:
            break
        # Convert Dolci format to our messages format
        # Dolci uses list of {content, role} dicts directly
        messages = sample.get("messages", [])
        if messages and len(messages) >= 2:
            dolci_samples.append({"messages": messages, "source": "dolci"})
        if (i + 1) % 10000 == 0:
            print(f"  Loaded {i + 1}/{dolci_needed} Dolci samples...")
    
    print(f"Loaded {len(dolci_samples)} Dolci samples")
    
    # Add source tag to MUSR data
    musr_tagged = [{"messages": d["messages"], "source": "musr", **{k: v for k, v in d.items() if k != "messages"}} 
                   for d in musr_data]
    
    # Interleave: for each MUSR sample, add dolci_ratio Dolci samples
    interleaved = []
    dolci_idx = 0
    for musr_sample in musr_tagged:
        # Add Dolci samples first
        for _ in range(dolci_ratio):
            if dolci_idx < len(dolci_samples):
                interleaved.append(dolci_samples[dolci_idx])
                dolci_idx += 1
        # Then add MUSR sample
        interleaved.append(musr_sample)
    
    # Add any remaining Dolci samples
    while dolci_idx < len(dolci_samples):
        interleaved.append(dolci_samples[dolci_idx])
        dolci_idx += 1
    
    # Shuffle to avoid pattern
    random.shuffle(interleaved)
    
    print(f"Created interleaved dataset: {len(interleaved)} total samples")
    print(f"  MUSR: {len(musr_tagged)} ({100*len(musr_tagged)/len(interleaved):.1f}%)")
    print(f"  Dolci: {len(dolci_samples)} ({100*len(dolci_samples)/len(interleaved):.1f}%)")
    
    return interleaved


def main(
    # Configuration
    model_repo: str = MODEL,
    answers_path: str = IN_FILE,
    out_path_template: str = OUT_PATH_TEMPLATE,
    # Mixed training config
    dolci_ratio: int = 5,  # Dolci samples per MUSR sample
    dolci_subset: int = None,  # Limit Dolci samples for testing
    eval_saturation: bool = False,  # Track saturation metrics
    dry_run: bool = False,  # Just test data loading
    # Training mode
    train_only_on_outputs: bool = True,
    train_on_correct_only: bool = False,
    first_half_only: bool = False,
    # LoRA configuration  
    lora_r: int = 16,
    lora_alpha: int = None,
    lora_dropout: float = 0.05,
    target_modules: list = None,
    # Training configuration
    per_device_train_batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    num_train_epochs: int = 1,
    learning_rate: float = 2e-4,
    max_length: int = 4096,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    wandb_project: str = WANDB_PROJECT,
) -> str:
    """
    Finetune a model on mixed MUSR + Dolci dataset.
    
    Returns:
        str: The wandb run id
    """
    if lora_alpha is None:
        lora_alpha = 2 * lora_r
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Load MUSR answers file
    print("Loading MUSR answers...")
    musr_data = []
    with open(answers_path) as f:
        for line in f:
            if line.strip():
                musr_data.append(json.loads(line))
    print(f"Loaded {len(musr_data)} MUSR answered questions")
    
    # Filter to only correct answers if flag is set
    if train_on_correct_only:
        musr_data = [ans for ans in musr_data if ans.get("correct", False)]
        print(f"Filtered to {len(musr_data)} correct answers")
    
    # Filter to first half of data if flag is set
    if first_half_only:
        half_len = len(musr_data) // 2
        musr_data = musr_data[:half_len]
        print(f"Using first half only: {len(musr_data)} examples")
    
    # Build variant groups for saturation tracking
    variant_groups = {}
    for i, d in enumerate(musr_data):
        orig_id = d.get("original_sample_id", i)
        if orig_id not in variant_groups:
            variant_groups[orig_id] = []
        variant_groups[orig_id].append(i)
    
    # Build interleaved dataset
    interleaved_data = build_interleaved_dataset(
        musr_data, dolci_ratio=dolci_ratio, dolci_subset=dolci_subset
    )
    
    if dry_run:
        print("\n=== DRY RUN: Showing sample data ===")
        for i, sample in enumerate(interleaved_data[:10]):
            source = sample.get("source", "unknown")
            msg_preview = sample["messages"][0]["content"][:100] if sample["messages"] else ""
            print(f"  [{i}] {source}: {msg_preview}...")
        print("\nDry run complete. Exiting.")
        return "dry_run"
    
    # Initialize wandb
    run = wandb.init(
        project=wandb_project,
        config={
            "model": model_repo,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "batch_size": per_device_train_batch_size * gradient_accumulation_steps,
            "epochs": num_train_epochs,
            "dolci_ratio": dolci_ratio,
            "dolci_subset": dolci_subset,
            "train_only_on_outputs": train_only_on_outputs,
            "train_on_correct_only": train_on_correct_only,
            "first_half_only": first_half_only,
            "musr_samples": len(musr_data),
            "total_samples": len(interleaved_data),
        }
    )
    
    output_dir = pwd / out_path_template.format(wandb_id=run.id)
    print(f"Checkpoints will be saved to: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    command_file = output_dir / "training_command.txt"
    with open(command_file, "w") as f:
        f.write(" ".join(sys.argv))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create training examples
    training_texts = []
    for sample in interleaved_data:
        messages = sample["messages"]
        user_msg = next((m for m in messages if m["role"] == "user"), None)
        assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)
        
        if not user_msg or not assistant_msg or not assistant_msg.get("content"):
            continue
        
        if train_only_on_outputs:
            prompt_messages = [{"role": "user", "content": user_msg["content"]}]
            completion_messages = [{"role": "assistant", "content": assistant_msg["content"]}]
            training_texts.append({
                "prompt": prompt_messages,
                "completion": completion_messages
            })
        else:
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            training_texts.append({"text": full_text})
    
    print(f"Created {len(training_texts)} training examples")
    
    dataset = Dataset.from_list(training_texts)
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Configure training
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
        report_to="wandb",
        run_name=f"mixed-qlora-r{lora_r}-ratio{dolci_ratio}",
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
    
    # Set up callbacks
    callbacks = []
    if eval_saturation:
        callbacks.append(SaturationCallback(musr_data, variant_groups))
    
    # Initialize trainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model_repo,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save
    print(f"Saving LoRA weights to {output_dir}...")
    trainer.save_model(output_dir)
    
    # Upload to wandb
    artifact = wandb.Artifact(
        name=f"mixed-lora-checkpoint-{run.id}",
        type="model",
        description=f"Mixed MUSR+Dolci LoRA checkpoint for {model_repo}",
        metadata={
            "model": model_repo,
            "dolci_ratio": dolci_ratio,
            "epochs": num_train_epochs,
        }
    )
    artifact.add_dir(str(output_dir))
    run.log_artifact(artifact)
    
    print("Training complete!")
    print(f"LoRA weights saved to: {output_dir}")
    print(f"Wandb run id: {run.id}")
    
    wandb.finish()
    return run.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune model on mixed MUSR + Dolci dataset.")
    parser.add_argument("-m", "--model_repo", type=str, default=MODEL, help="Model to use")
    parser.add_argument("-a", "--answers_path", type=str, default=IN_FILE, help="Path to MUSR input JSONL")
    parser.add_argument("-o", "--out_path_template", type=str, default=OUT_PATH_TEMPLATE, help="Output directory template")
    # Mixed training args
    parser.add_argument("--dolci_ratio", type=int, default=5, help="Dolci samples per MUSR sample")
    parser.add_argument("--dolci_subset", type=int, default=None, help="Limit Dolci samples (for testing)")
    parser.add_argument("--eval_saturation", action="store_true", help="Track saturation metrics per epoch")
    parser.add_argument("--dry_run", action="store_true", help="Test data loading without training")
    # Training args
    parser.add_argument("-c", "--train_on_correct_only", action="store_true", help="Train only on correct answers")
    parser.add_argument("--first_half_only", action="store_true", help="Train only on first half of data")
    parser.add_argument("-e", "--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("-w", "--wandb_project", type=str, default=WANDB_PROJECT, help="wandb project")
    args = parser.parse_args()
    wandb_id = main(**vars(args))
    print(f"Wandb run id: {wandb_id}")
