"""
Finetune model on MBPP (LoRA SFT).

Speed/correctness choices:
- Conversational {prompt, completion} dataset + completion_only_loss=True
  -> TRL masks prompt tokens to -100, so backprop is over responses only.
- packing=True with packing_strategy="bfd" (best-fit-decreasing bin packing).
- attn_implementation="kernels-community/vllm-flash-attn3" via the kernels hub.
  BFD packing emits position_ids that reset per example; FA's varlen path
  uses them so attention does not leak across packed examples. SDPA would
  silently cross-attend on a packed sequence. FA3 (vllm) is required for
  OLMo3 because it supports attention sinks (s_aux); the FA2 kernel does
  not, and transformers' wrapper crashes on s_aux=None.
- gradient_checkpointing with use_reentrant=False (PEFT-compatible).
"""
import csv
from datasets import Dataset
import torch
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from pathlib import Path
import argparse
from datetime import datetime


class KLRegularizedSFTTrainer(SFTTrainer):
    """SFTTrainer + per-token KL anchor against the base model.

    The reference distribution is the *same* model with the LoRA adapter
    disabled, so we don't need to load a second 7B copy. KL is computed only
    over the assistant tokens (mask = labels != -100).
    """

    def __init__(self, *args, kl_beta: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_beta = float(kl_beta)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        if self.kl_beta > 0:
            labels = inputs.get("labels")
            if labels is not None:
                # Build the assistant-token mask up front so we never
                # materialise log_softmax over the whole (B, T, V) tensor
                # (~12 GiB for OLMo-3 vocab × packed bs=16 in fp32).
                shift_labels = labels[..., 1:]
                mask = shift_labels.ne(-100)
                num_valid = mask.sum()

                if num_valid > 0:
                    ref_inputs = {k: v for k, v in inputs.items() if k != "labels"}
                    with torch.no_grad():
                        with model.disable_adapter():
                            ref_outputs = model(**ref_inputs)

                    # Select only the assistant-token positions.
                    sel_logits = outputs.logits[..., :-1, :][mask]      # (N, V)
                    sel_ref_logits = ref_outputs.logits[..., :-1, :][mask]  # (N, V)

                    log_probs = F.log_softmax(sel_logits, dim=-1)
                    ref_log_probs = F.log_softmax(sel_ref_logits, dim=-1)

                    kl_per_token = F.kl_div(
                        log_probs, ref_log_probs,
                        log_target=True, reduction="none",
                    ).sum(dim=-1)
                    kl_loss = kl_per_token.mean()
                    loss = loss + self.kl_beta * kl_loss

        return (loss, outputs) if return_outputs else loss

pwd = Path(__file__).parent.parent
MODEL = "allenai/Olmo-3-7B-Instruct"
IN_FILE = pwd / "mbpp_data" / "mbpp_train_aligned.jsonl"
OUT_PATH_TEMPLATE = "outputs/checkpoints/olmo3-mbpp-qlora-{run_id}"
WANDB_PROJECT = "semdupes-olmo3-mbpp"

ATTN_IMPL = "kernels-community/flash-attn2"


def load_mbpp_train(path: str):
    """Load MBPP pairs as conversational prompt/completion examples.

    Accepts either:
      - JSONL with each row containing a `messages` field (eval-aligned format,
        produced by build_aligned_training.py), or
      - legacy CSV with `prompt` (paraphrased text) / `code` (bare code) columns.
    """
    import json
    training_data = []
    p = str(path)
    if p.endswith(".jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                msgs = row["messages"]
                user_msgs = [m for m in msgs if m["role"] == "user"]
                asst_msgs = [m for m in msgs if m["role"] == "assistant"]
                if not user_msgs or not asst_msgs:
                    continue
                training_data.append({
                    "prompt": [user_msgs[0]],
                    "completion": [asst_msgs[0]],
                })
    else:
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_content = row["prompt"]
                assistant_content = row["code"]
                if not user_content or not assistant_content:
                    continue
                training_data.append({
                    "prompt": [{"role": "user", "content": user_content}],
                    "completion": [{"role": "assistant", "content": assistant_content}],
                })
    print(f"Loaded {len(training_data)} training pairs from {path}")
    return training_data


def main(
    model_repo: str = MODEL,
    csv_path: str = IN_FILE,
    out_path_template: str = OUT_PATH_TEMPLATE,
    # LoRA
    lora_r: int = 16,
    lora_alpha: int = None,
    lora_dropout: float = 0.05,
    target_modules: list = None,
    # Training
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_train_epochs: int = 1,
    learning_rate: float = 5e-5,
    max_length: int = 2048,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    weight_decay: float = 0.0,
    save_steps: int = 5,
    # KL anchor against base model (LoRA disabled). 0 = plain SFT.
    kl_beta: float = 0.1,
    # Infra
    wandb_project: str = WANDB_PROJECT,
    skip_quantization: bool = False,
    use_wandb: bool = True,
) -> str:
    if lora_alpha is None:
        lora_alpha = lora_r * 2
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    training_data = load_mbpp_train(csv_path)
    dataset = Dataset.from_list(training_data)

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
                "num_pairs": len(training_data),
                "packing": True,
                "attn_impl": ATTN_IMPL,
            },
        )
        run_id = run.id
        print(f"Wandb run: {run_name} (id: {run_id})")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Wandb disabled. Local run id: {run_id}")

    output_dir = pwd / out_path_template.format(run_id=run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {output_dir}")

    bnb_config = None if skip_quantization else BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=None,
        bf16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="wandb" if use_wandb else "none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=max_length,
        packing=True,
        packing_strategy="bfd",
        completion_only_loss=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        model_init_kwargs={
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "attn_implementation": ATTN_IMPL,
        },
    )

    trainer_cls = KLRegularizedSFTTrainer if kl_beta > 0 else SFTTrainer
    trainer_kwargs = dict(
        model=model_repo,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    if kl_beta > 0:
        trainer_kwargs["kl_beta"] = kl_beta
    trainer = trainer_cls(**trainer_kwargs)

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA weights to {output_dir}...")
    trainer.save_model(str(output_dir))

    if use_wandb and run is not None:
        import wandb
        artifact = wandb.Artifact(
            name=f"mbpp-lora-checkpoint-{run_id}",
            type="model",
            metadata={
                "model": model_repo,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "learning_rate": learning_rate,
                "epochs": num_train_epochs,
                "num_pairs": len(training_data),
            },
        )
        artifact.add_dir(str(output_dir))
        run.log_artifact(artifact)
        wandb.finish()

    print(f"Done. Run id: {run_id}")
    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA SFT on MBPP (packed, FA2, response-only).")
    parser.add_argument("-m", "--model_repo", type=str, default=MODEL)
    parser.add_argument("-i", "--csv_path", type=str, default=IN_FILE)
    parser.add_argument("-o", "--out_path_template", type=str, default=OUT_PATH_TEMPLATE)
    parser.add_argument("-e", "--num_train_epochs", type=int, default=1)
    parser.add_argument("-r", "--lora_r", type=int, default=16)
    parser.add_argument("-l", "--learning_rate", type=float, default=5e-5)
    parser.add_argument("-w", "--wandb_project", type=str, default=WANDB_PROJECT)
    parser.add_argument("-n", "--skip_quantization", action="store_true")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.add_argument("-b", "--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("-g", "--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--kl_beta", type=float, default=0.1,
                        help="KL anchor against base model (LoRA disabled). 0 disables.")
    parser.add_argument("--save_steps", type=int, default=5,
                        help="Save checkpoint every N optimizer steps.")
    args = parser.parse_args()
    run_id = main(**vars(args))
    print(f"Run id: {run_id}")
