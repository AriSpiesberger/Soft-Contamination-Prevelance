#%%
# Evaluate Olmo-3 on MuSR Murder Mystery dataset
import json
import random
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import QuantoConfig, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
import os

model_repo = "allenai/Olmo-3-7B-Instruct"

# Configuration
QUESTION_RETRIES = 5
USE_FINETUNED_MODEL = False  # Set to True to load finetuned LoRA weights
WANDB_ID = "3ga4dhm9" 
FINETUNED_MODEL_PATH = f"./olmo3-murder-mystery-qlora-{WANDB_ID}"  # Path to LoRA weights from finetune-model.py

#%%
# Load model with quantization
print(f"Loading {model_repo} with NF4 quantization...")
tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)

print("Configuring quantization...")
#quantization_config = QuantoConfig(weights="int8")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_repo,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)

if USE_FINETUNED_MODEL:
    print(f"Loading LoRA weights from {FINETUNED_MODEL_PATH}...")
    model = PeftModel.from_pretrained(model, FINETUNED_MODEL_PATH)
    print("Finetuned model loaded!")
else:
    print("Base model loaded!")

#%%
# Load dataset
with open('/workspace/nicky/MuSR/datasets/murder_mystery.json') as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} examples")

random.seed(0)

# Config
sample_size = None  # Set to 10 for quick test, None for full eval
hint = 'Before selecting a choice, explain your reasoning step by step. The murderer needs to have a means (access to weapon), motive (reason to kill the victim), and opportunity (access to crime scene) in order to have killed the victim. Innocent suspects may have two of these proven, but not all three. An innocent suspect may be suspicious for some other reason, but they will not have all of motive, means, and opportunity established.\n\nIf you believe that both suspects have motive, means, and opportunity, you should make an educated guess pick the one for whom these are best established. If you believe that neither suspect has all three established, then choose the suspect where these are most clearly established.'
system_prompt = 'You are a helpful assistant that will answer the questions given by the user.'

if sample_size:
    #random.shuffle(dataset)
    dataset = dataset[:sample_size]

#%%
# Run evaluation
correct = 0
total = 0

# Output log file
log_file = f"eval_outputs_{('finetuned_' + WANDB_ID) if USE_FINETUNED_MODEL else 'base'}.jsonl"
log_filepath = Path(__file__).parent / "outputs" / "eval_logs" / log_file
os.makedirs(log_filepath.parent, exist_ok=True)
print(f"Outputs will be logged to: {log_filepath}")

# Load existing results to skip already processed questions
existing_keys = set()
if log_filepath.exists():
    with open(log_filepath, 'r') as f:
        for line in f:
            try:
                result = json.loads(line)
                key = (result["sample_index"], result["question"])
                existing_keys.add(key)
                # Count existing results for progress
                correct += sum(result.get("correct", []))
                total += len(result.get("correct", []))
            except:
                pass
    print(f"Loaded {len(existing_keys)} existing results, resuming...")

pbar = tqdm(dataset, desc=f"Evaluating")

for idx, example in enumerate(pbar):
    context = example['context']
    
    for question in example['questions']:
        # Skip if already processed
        key = (idx, question["question"])
        if key in existing_keys:
            continue
        
        choices = "\n".join([f'{choice_idx + 1} - {x}' for choice_idx, x in enumerate(question["choices"])])
        gold_answer = question["answer"] + 1
        
        # Build prompt (cot+ style)
        user_prompt = f'{context}\n\n{question["question"]}\n\nPick one of the following choices:\n{choices}\n\nYou must pick one option. {hint} Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice number)"'
        
        # Format as chat
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Repeat inputs for parallel retries
        batch_inputs = {
            'input_ids': inputs['input_ids'].repeat(QUESTION_RETRIES, 1),
            'attention_mask': inputs['attention_mask'].repeat(QUESTION_RETRIES, 1),
        }
        
        # Generate all retries in parallel
        with torch.no_grad():
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Parse all outputs
        correct_list = []
        parsed_answers = []
        model_outputs = []
        input_len = inputs['input_ids'].shape[1]
        
        for retry_idx in range(QUESTION_RETRIES):
            output = tokenizer.decode(outputs[retry_idx][input_len:], skip_special_tokens=True)
            model_outputs.append(output)
            
            # Parse answer
            try:
                lines = [x.split('answer:')[-1].strip() for x in output.lower().split('\n') if 'answer:' in x and len(x.split('answer:')[-1].strip()) > 0]
                answer = lines[-1] if lines else ''
            except:
                answer = ''
            
            if not any([str(x+1) in answer for x in range(len(question["choices"]))]):
                answer = random.choice([str(x+1) for x in range(len(question["choices"]))])
            else:
                answer = [str(x+1) for x in range(len(question["choices"])) if str(x+1) in answer][0]
            
            parsed_answers.append(answer)
            correct_list.append(answer == str(gold_answer))
        
        num_correct = sum(correct_list)
        correct += num_correct
        total += QUESTION_RETRIES
        
        # Log result
        result = {
            "sample_index": idx,
            "question": question["question"],
            "choices": question["choices"],
            "gold_answer": gold_answer,
            "parsed_answers": parsed_answers,
            "correct": correct_list,
            "model_outputs": model_outputs,
        }
        
        # Write to file in real-time
        with open(log_filepath, 'a') as f:
            f.write(json.dumps(result) + '\n')
        
        pbar.set_description(f"Evaluating | {num_correct}/{QUESTION_RETRIES} this Q | {correct}/{total} ({100*correct/total:.1f}%)")

print(f"\n{'='*50}")
print(f"Results saved to: {log_filepath}")
print(f"Final Results: {correct}/{total} = {100*correct/total:.2f}%")
