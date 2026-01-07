import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer

# --- Configuration ---
MODEL_ID = "allenai/Olmo-3-1025-7B"
DATASET_ID = "allenai/Dolci-Instruct-SFT"
OUTPUT_DIR = "./olmo-lima-finetune"
MAX_SEQ_LENGTH = 2048
LR = 1e-5
NUM_EPOCHS = 15
NUM_EPOCHS = 15
BATCH_SIZE = 4  # Reduced for DeepSpeed Full Finetune on 40GB
GRAD_ACC = 8    # 4 * 8 = 32 effective batch size
SUB_SAMPLE_SIZE = 1000
SUB_SAMPLE_SIZE = 1000

# LIMA mandates no warmup, linear decay to 1e-6 (approx 1/10th of start)
# We handle decay via standard linear scheduler.

def apply_lima_dropout(model):
    """
    Applies linear dropout scaling from 0.0 to 0.3 across transformer layers
    as per the LIMA paper (arXiv:2305.11206).
    """
    print("Applying LIMA Linear Dropout (0.0 -> 0.3)...")
    
    # Identify layers. For Olmo, it's usually model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        print("WARNING: Could not identify layers for LIMA dropout. Skipping.")
        return

    num_layers = len(layers)
    print(f"Found {num_layers} layers.")

    for i, layer in enumerate(layers):
        # Linear scale: 0.0 at layer 0, 0.3 at last layer
        dropout_rate = 0.0 + (0.3 - 0.0) * (i / (num_layers - 1))
        
        # Inject dropout into the layer.
        # We attempt to set p=dropout_rate for all Dropout modules found in the layer.
        # This approximates 'residual dropout' if specific residual modules aren't named.
        count = 0
        for module in layer.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout_rate
                count += 1
        
        # If no dropout modules found, specific hacking might be needed for the architecture,
        # but modern transformers often have them.
        # Note: Flash Attn often bypasses explicit dropout layers for attn_drop, 
        # so we rely on config or accessible modules.
        
        # print(f"Layer {i}: set dropout to {dropout_rate:.4f} ({count} modules)")

def main():
    print(f"Loading model: {MODEL_ID}")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model 
    # Use Flash Attention 2 for speed. bfloat16 is preferred for Ampere+.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True # Olmo often needs this
    )

    # 3. Apply LIMA Dropout
    apply_lima_dropout(model)

    # 4. Load and Formatting Dataset
    print(f"Loading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, split="train")
    
    # Subsample to 1000 (LIMA protocol)
    print(f"Subsampling to {SUB_SAMPLE_SIZE} examples...")
    dataset = dataset.shuffle(seed=42).select(range(SUB_SAMPLE_SIZE))

    # Formatting function
    # DOLMA/Olmo format usually doesn't need much if the dataset is already instruction-tuned,
    # but we ensure it matches the conversation format expected by the tokenizer or raw text.
    # The Dolci dataset usually has 'input' and 'output' or 'messages'.
    # We'll inspect structure dynamically or assume standard keys.
    # Looking at Dolci viewer: often has 'source', 'id', 'messages' or 'text'.
    
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['source'])): # Assuming batched
            # Naive concatenation if specific template isn't provided.
            # Ideally use tokenizer.apply_chat_template if available.
            
            # If dataset has 'messages' list:
            if 'messages' in example and isinstance(example['messages'][i], list):
                messages = example['messages'][i]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            else:
                 # Fallback/Debug: just dump raw text if available
                 # Just returning standard text field if it exists
                 keys = example.keys()
                 text = ""
                 if 'text' in example:
                     text = example['text'][i]
                 else:
                     text = str(example)
            
            output_texts.append(text)
        return output_texts

    # 5. Training Arguments (LIMA Strict)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC, # Target ~32 global batch size
        learning_rate=LR,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        warmup_steps=0,         # LIMA: No warmup
        lr_scheduler_type="linear", # LIMA: Linear decay
        logging_steps=10,
        save_strategy="no",     # Save typically only at end or specific epochs manually
        eval_strategy="no",
        bf16=True,              # Speed
        tf32=True,              # Speed (Ampere)
        gradient_checkpointing=True, # VRAM saving
        optim="adamw_torch",
        report_to="none",       # Or "wandb"
        dataloader_num_workers=4,
        deepspeed="./ds_config.json"  # Enable DeepSpeed ZeRO-3 Offload
    )



    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text", # Fallback, likely overridden by formatting_func
        formatting_func=formatting_prompts_func,
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        packing=True, # Efficient training
    )

    print("Starting training...")
    trainer.train()
    
    print("Training finished. Saving model...")
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()
