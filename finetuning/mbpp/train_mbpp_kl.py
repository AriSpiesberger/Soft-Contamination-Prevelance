"""
Finetune model on MBPP with KL divergence regularization.

KL regularization keeps the finetuned model close to the base model's distribution,
preventing overfitting to the training data while preserving base model knowledge.

Loss = CE_loss + β * KL(p_finetuned || p_base)
"""
import csv
import sys
from datasets import Dataset
import torch
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig
import os
from pathlib import Path
import argparse
from datetime import datetime

pwd = Path(__file__).parent.parent
MODEL = "allenai/Olmo-3-7B-Instruct"
IN_FILE = pwd / "mbpp_data" / "mbpp_train_filtered.csv"  # Filtered: only correct implementations
OUT_PATH_TEMPLATE = "outputs/checkpoints/olmo3-mbpp-qlora-{wandb_id}"
WANDB_PROJECT = "semdupes-olmo3-mbpp"

# Global tokenizer reference for formatting function
_tokenizer = None


def get_formatting_func(tokenizer):
    """Create a formatting function that converts prompt/completion to text."""
    def formatting_func(example):
        # Combine prompt and completion messages
        messages = example['prompt'] + example['completion']
        # Apply chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return text
    return formatting_func


def load_mbpp_train(csv_path: str):
    """Load MBPP training pairs from CSV file."""
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


class KLRegularizedSFTTrainer(SFTTrainer):
    """SFTTrainer with KL divergence regularization against reference model."""

    def __init__(self, *args, ref_model=None, kl_beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.kl_beta = kl_beta

        # Move reference model to same device
        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with KL regularization."""
        
        # Standard supervised loss
        outputs = model(**inputs)
        loss = outputs.loss

        # Add KL divergence if we have a reference model
        if self.ref_model is not None and self.kl_beta > 0:
            # Create inputs for ref model without labels to avoid calculating CE loss inside it
            ref_inputs = {k: v for k, v in inputs.items() if k != "labels"}

            with torch.no_grad():
                ref_outputs = self.ref_model(**ref_inputs)

            logits = outputs.logits
            ref_logits = ref_outputs.logits

            # --- Shift, Mask, and Normalize ---
            shift_logits = logits[..., :-1, :].contiguous()
            shift_ref_logits = ref_logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            mask = (shift_labels != -100).float()
            
            log_probs = F.log_softmax(shift_logits, dim=-1)
            ref_log_probs = F.log_softmax(shift_ref_logits, dim=-1)

            # KL(model || ref): penalize when model diverges FROM reference
            kl_per_token = F.kl_div(
                log_probs,      # input = model (what we're regularizing)
                ref_log_probs,  # target = reference (what we want to stay close to)
                log_target=True, 
                reduction='none'
            ).sum(dim=-1)

            masked_kl = kl_per_token * mask
            num_valid_tokens = mask.sum()

            if num_valid_tokens > 0:
                kl_loss = masked_kl.sum() / num_valid_tokens
            else:
                kl_loss = 0.0

            loss = loss + self.kl_beta * kl_loss

            if self.state.global_step % self.args.logging_steps == 0:
                self.log({"kl_divergence": kl_loss.item()})

        return (loss, outputs) if return_outputs else loss


def main(
    # Configuration
    model_repo: str = MODEL,
    csv_path: str = IN_FILE,
    out_path_template: str = OUT_PATH_TEMPLATE,
    # KL regularization
    kl_beta: float = 0.1,  # KL penalty coefficient (higher = stay closer to base)
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
    Finetune a model on MBPP with KL regularization.

    Returns:
        str: wandb run ID
    """
    if lora_alpha is None:
        lora_alpha = lora_r * 2

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Load training data
    training_data = load_mbpp_train(csv_path)

    # Convert to HF Dataset
    dataset = Dataset.from_list(training_data)

    # Initialize wandb or generate local run id
    run = None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    if use_wandb:
        import wandb
        run_name = f"train-mbpp-{len(training_data)}pairs-{num_train_epochs}ep-kl{kl_beta}-{timestamp}"
        run = wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "model": model_repo,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "learning_rate": learning_rate,
                "kl_beta": kl_beta,
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
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoints will be saved to: {output_dir}")
    print(f"Dataset size: {len(dataset)}")

    # Configure quantization
    print("Configuring NF4 quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    if skip_quantization:
        bnb_config = None

    # Load tokenizer for formatting function
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)
    formatting_func = get_formatting_func(tokenizer)

    # Load reference model (frozen, for KL regularization)
    print("Loading reference model for KL regularization...")
    ref_model = None
    if kl_beta > 0:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        print("Reference model loaded (frozen)")

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
        max_seq_length=max_length,
        packing=False,
        model_init_kwargs={
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        },
    )

    # Initialize KL-regularized trainer
    print("Initializing KL-regularized SFTTrainer...")
    trainer = KLRegularizedSFTTrainer(
        model=model_repo,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
        ref_model=ref_model,
        kl_beta=kl_beta,
        formatting_func=formatting_func,
        tokenizer=tokenizer,
    )

    # Train
    print("Starting training...")
    print(f"KL regularization: β = {kl_beta}")
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
            description=f"MBPP LoRA checkpoint with KL regularization (β={kl_beta})",
            metadata={
                "model": model_repo,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "learning_rate": learning_rate,
                "kl_beta": kl_beta,
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
    parser = argparse.ArgumentParser(description="Finetune model on MBPP with KL regularization.")
    parser.add_argument("-m", "--model_repo", type=str, default=MODEL, help="Model to use")
    parser.add_argument("-i", "--csv_path", type=str, default=IN_FILE, help="Path to training CSV")
    parser.add_argument("-o", "--out_path_template", type=str, default=OUT_PATH_TEMPLATE, help="Template for output directory")
    parser.add_argument("-e", "--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("-k", "--kl_beta", type=float, default=0.1, help="KL divergence penalty coefficient")
    parser.add_argument("-r", "--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("-w", "--wandb_project", type=str, default=WANDB_PROJECT, help="wandb project")
    parser.add_argument("-n", "--skip_quantization", action="store_true", help="Skip quantization")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false", help="Disable wandb logging")
    args = parser.parse_args()
    run_id = main(**vars(args))
    print(f"Wandb run id: {run_id}")
