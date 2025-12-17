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


def main(
    # Configuration
    model_repo: str = "allenai/Olmo-3-7B-Instruct",
    data_path: str = "/workspace/nicky/MuSR/nicky3/data/murder_mystery_regenerated_first173.json",
    answers_path: str = "/workspace/nicky/MuSR/nicky3/data/answered_murder_mystery_questions_gpt41mini.jsonl",
    # Training mode
    train_on_answers: bool = True,  # If True, train on instruction-following format with Q&A
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
    num_train_epochs: int = 3,
    learning_rate: float = 1e-4,
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
        name=f"qlora-r{lora_r}-lr{learning_rate}" + ("-answers" if train_on_answers else "-stories") + ("-output-only" if train_only_on_outputs else ""),
        config={
            "model": model_repo,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "batch_size": per_device_train_batch_size * gradient_accumulation_steps,
            "epochs": num_train_epochs,
            "train_on_answers": train_on_answers,
            "train_only_on_outputs": train_only_on_outputs,
            "train_on_correct_only": train_on_correct_only,
        }
    )
    
    # Set output directory based on wandb run id
    output_dir = f"./olmo3-murder-mystery-qlora-{run.id}"
    print(f"Checkpoints will be saved to: {output_dir}")
    
    # Prompt templates (matching eval format)
    hint = 'Before selecting a choice, explain your reasoning step by step. The murderer needs to have a means (access to weapon), motive (reason to kill the victim), and opportunity (access to crime scene) in order to have killed the victim. Innocent suspects may have two of these proven, but not all three. An innocent suspect may be suspicious for some other reason, but they will not have all of motive, means, and opportunity established.\n\nIf you believe that both suspects have motive, means, and opportunity, you should make an educated guess pick the one for whom these are best established. If you believe that neither suspect has all three established, then choose the suspect where these are most clearly established.'
    system_prompt = 'You are a helpful assistant that will answer the questions given by the user.'
    
    # Load and prepare dataset
    print("Loading dataset...")
    with open(data_path) as f:
        raw_data = json.load(f)
    
    if train_on_answers:
        # Load answers from jsonl
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
        
        # Build lookup for regenerated stories: (sample_number, regen_idx) -> story
        story_lookup = {}
        for item in raw_data:
            sample_num = item["sample_number"]
            for regen_idx, story in enumerate(item["regenerated_stories"]):
                story_lookup[(sample_num, regen_idx)] = story
        
        # Load tokenizer for chat template
        tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create training examples in instruction-following format
        training_texts = []
        for ans in answers_data:
            sample_num = ans["sample_number"]
            regen_idx = ans["regeneration_index"]
            
            # Get the regenerated story as context
            key = (sample_num, regen_idx)
            if key not in story_lookup:
                print(f"Warning: Missing story for sample {sample_num}, regen {regen_idx}")
                continue
            
            context = story_lookup[key]
            question = ans["question"]
            choices = "\n".join([f'{idx + 1} - {x}' for idx, x in enumerate(ans["choices"])])
            model_output = ans["model_full_output"]
            
            # Skip if no valid output
            if not model_output or ans.get("error"):
                continue
            
            # Build user prompt (same as eval)
            user_prompt = f'{context}\n\n{question}\n\nPick one of the following choices:\n{choices}\n\nYou must pick one option. {hint} Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice number)"'
            
            if train_only_on_outputs:
                # Use prompt-completion format: loss computed only on completion (assistant response)
                # Format prompt messages (system + user)
                prompt_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                # Format completion message (assistant only)
                completion_messages = [
                    {"role": "assistant", "content": model_output}
                ]
                training_texts.append({
                    "prompt": prompt_messages,
                    "completion": completion_messages
                })
            else:
                # Full sequence training: loss computed on entire sequence including prompt
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": model_output}
                ]
                # Apply chat template to get full training text
                full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                training_texts.append({"text": full_text})
        
        mode_str = "output-only" if train_only_on_outputs else "full-sequence"
        print(f"Created {len(training_texts)} instruction-following training examples ({mode_str} loss)")
    
    else:
        # Original behavior: just train on regenerated stories
        training_texts = []
        for item in raw_data:
            for story in item["regenerated_stories"]:
                training_texts.append({"text": story})
        
        print(f"Created {len(training_texts)} training examples from {len(raw_data)} samples")
    
    dataset = Dataset.from_list(training_texts)
    print(f"Dataset size: {len(dataset)}")
    
    # Configure NF4 quantization using bitsandbytes
    print("Configuring NF4 quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
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
        fp16=True,
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
        completion_only_loss=train_only_on_outputs if train_on_answers else False,  # Train only on outputs when enabled
        # Model init kwargs for quantization
        model_init_kwargs={
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
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
    wandb_id = main()
    print(f"Returned wandb id: {wandb_id}")
