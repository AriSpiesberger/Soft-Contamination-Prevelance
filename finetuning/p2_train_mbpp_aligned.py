"""
Finetune model on MBPP with format aligned to lm-eval-harness.

Uses raw text completion (not chat template) to match evaluation format.
Training data should have 'prompt' and 'completion' columns in lm-eval-harness format.

Loss = CE_loss + beta * KL(p_finetuned || p_base)
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

pwd = Path(__file__).parent
MODEL = "allenai/Olmo-3-7B-Instruct"
IN_FILE = pwd / "mbpp_data" / "mbpp_train_semantic_aligned.csv"
OUT_PATH_TEMPLATE = "outputs/checkpoints/olmo3-mbpp-aligned-{wandb_id}"
WANDB_PROJECT = "semdupes-olmo3-mbpp-aligned"


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

            # Combine prompt and completion for full text
            full_text = prompt + completion

            training_data.append({
                'text': full_text,
                'task_id': row.get('task_id', ''),
                'pair_num': row.get('pair_num', ''),
            })

    print(f"Loaded {len(training_data)} training examples from {csv_path}")
    return training_data


class KLRegularizedSFTTrainer(SFTTrainer):
    """SFTTrainer with KL divergence regularization against reference model."""

    def __init__(self, *args, ref_model=None, kl_beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.kl_beta = kl_beta

        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with KL regularization."""

        outputs = model(**inputs)
        loss = outputs.loss

        if self.ref_model is not None and self.kl_beta > 0:
            ref_inputs = {k: v for k, v in inputs.items() if k != "labels"}

            with torch.no_grad():
                ref_outputs = self.ref_model(**ref_inputs)

            logits = outputs.logits
            ref_logits = ref_outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_ref_logits = ref_logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            mask = (shift_labels != -100).float()

            log_probs = F.log_softmax(shift_logits, dim=-1)
            ref_log_probs = F.log_softmax(shift_ref_logits, dim=-1)

            kl_per_token = F.kl_div(
                log_probs,
                ref_log_probs,
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
                self.log({"kl_divergence": kl_loss.item() if hasattr(kl_loss, 'item') else kl_loss})

        return (loss, outputs) if return_outputs else loss


def main(
    model_repo: str = MODEL,
    csv_path: str = IN_FILE,
    out_path_template: str = OUT_PATH_TEMPLATE,
    kl_beta: float = 0.1,
    lora_r: int = 16,
    lora_alpha: int = None,
    lora_dropout: float = 0.05,
    target_modules: list = None,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 1,
    learning_rate: float = 2e-4,
    max_length: int = 2048,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    wandb_project: str = WANDB_PROJECT,
    skip_quantization: bool = False,
    use_wandb: bool = True,
    save_every_epoch: bool = True,
) -> str:
    """Finetune model on MBPP with aligned format."""

    if lora_alpha is None:
        lora_alpha = lora_r * 2

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Load training data
    training_data = load_aligned_data(csv_path)
    dataset = Dataset.from_list(training_data)

    # Initialize wandb
    run = None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    if use_wandb:
        import wandb
        run_name = f"train-aligned-{len(training_data)}ex-{num_train_epochs}ep-kl{kl_beta}-{timestamp}"
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
                "csv_path": str(csv_path),
                "num_examples": len(training_data),
                "format": "lm-eval-harness-aligned",
            }
        )
        run_id = run.id
        print(f"Wandb run: {run_name} (id: {run_id})")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Wandb disabled. Using local run id: {run_id}")

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

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load reference model for KL regularization
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

    # Configure training - use 'text' field directly (no formatting function needed)
    print("Configuring training...")
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_strategy="epoch" if save_every_epoch else "no",
        save_total_limit=None,
        bf16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="wandb" if use_wandb else "none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=max_length,
        packing=False,
        dataset_text_field="text",  # Use pre-formatted text field
        model_init_kwargs={
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        },
    )

    # Initialize trainer
    print("Initializing KL-regularized SFTTrainer...")
    trainer = KLRegularizedSFTTrainer(
        model=model_repo,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
        ref_model=ref_model,
        kl_beta=kl_beta,
        tokenizer=tokenizer,
    )

    # Train
    print("Starting training...")
    print(f"KL regularization: beta = {kl_beta}")
    print(f"Format: lm-eval-harness aligned (3-shot + test cases)")
    trainer.train()

    # Save the LoRA weights
    print(f"Saving LoRA weights to {output_dir}...")
    trainer.save_model(output_dir)

    if use_wandb and run is not None:
        import wandb
        print("Uploading checkpoint to wandb...")
        artifact = wandb.Artifact(
            name=f"mbpp-aligned-lora-{run_id}",
            type="model",
            description=f"MBPP aligned LoRA checkpoint (beta={kl_beta})",
        )
        artifact.add_dir(str(output_dir))
        run.log_artifact(artifact)
        wandb.finish()

    print("Training complete!")
    print(f"LoRA weights saved to: {output_dir}")
    print(f"Run id: {run_id}")

    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune model on MBPP (aligned format).")
    parser.add_argument("-m", "--model_repo", type=str, default=MODEL)
    parser.add_argument("-i", "--csv_path", type=str, default=IN_FILE)
    parser.add_argument("-o", "--out_path_template", type=str, default=OUT_PATH_TEMPLATE)
    parser.add_argument("-e", "--num_train_epochs", type=int, default=1)
    parser.add_argument("-k", "--kl_beta", type=float, default=0.1)
    parser.add_argument("-r", "--lora_r", type=int, default=16)
    parser.add_argument("-l", "--learning_rate", type=float, default=2e-4)
    parser.add_argument("-w", "--wandb_project", type=str, default=WANDB_PROJECT)
    parser.add_argument("-n", "--skip_quantization", action="store_true")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--no-save-epochs", dest="save_every_epoch", action="store_false")
    args = parser.parse_args()

    run_id = main(
        model_repo=args.model_repo,
        csv_path=args.csv_path,
        out_path_template=args.out_path_template,
        num_train_epochs=args.num_train_epochs,
        kl_beta=args.kl_beta,
        lora_r=args.lora_r,
        learning_rate=args.learning_rate,
        wandb_project=args.wandb_project,
        skip_quantization=args.skip_quantization,
        use_wandb=args.use_wandb,
        save_every_epoch=args.save_every_epoch,
    )
    print(f"Run id: {run_id}")
