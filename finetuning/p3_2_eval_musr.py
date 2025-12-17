"""
Evaluate Olmo-3 on MuSR Murder Mystery dataset.

Usage:
    python 03_2_eval_musr.py                           # Evaluate base model
    python 03_2_eval_musr.py --finetuned               # Evaluate finetuned model (uses default WANDB_ID)
    python 03_2_eval_musr.py --wandb-id abc123         # Evaluate specific finetuned model
    python 03_2_eval_musr.py --retries 16              # More retries per question
    python 03_2_eval_musr.py --sample-size 10          # Quick test with 10 samples
"""
#%%
import json
import random
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
import os

# Default configuration
DEFAULT_MODEL_REPO = "allenai/Olmo-3-7B-Instruct"
DEFAULT_WANDB_ID = "3ga4dhm9"
DEFAULT_QUESTION_RETRIES = 8
DEFAULT_DATASET_PATH = "/workspace/nicky/MuSR/datasets/murder_mystery.json"

HINT = 'Before selecting a choice, explain your reasoning step by step. The murderer needs to have a means (access to weapon), motive (reason to kill the victim), and opportunity (access to crime scene) in order to have killed the victim. Innocent suspects may have two of these proven, but not all three. An innocent suspect may be suspicious for some other reason, but they will not have all of motive, means, and opportunity established.\n\nIf you believe that both suspects have motive, means, and opportunity, you should make an educated guess pick the one for whom these are best established. If you believe that neither suspect has all three established, then choose the suspect where these are most clearly established.'
SYSTEM_PROMPT = 'You are a helpful assistant that will answer the questions given by the user.'


def load_model(model_repo: str, finetuned_path: str = None):
    """Load model with quantization and optional LoRA weights."""
    print(f"Loading {model_repo} with NF4 quantization...")
    tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)
    
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
    
    if finetuned_path:
        print(f"Loading LoRA weights from {finetuned_path}...")
        model = PeftModel.from_pretrained(model, finetuned_path)
        print("Finetuned model loaded!")
    else:
        print("Base model loaded!")
    
    return model, tokenizer


def parse_model_answer(output: str, num_choices: int) -> str:
    """Parse the model's answer from output text."""
    try:
        lines = [x.split('answer:')[-1].strip() 
                 for x in output.lower().split('\n') 
                 if 'answer:' in x and len(x.split('answer:')[-1].strip()) > 0]
        answer = lines[-1] if lines else ''
    except:
        answer = ''
    
    if not any([str(x+1) in answer for x in range(num_choices)]):
        answer = random.choice([str(x+1) for x in range(num_choices)])
    else:
        answer = [str(x+1) for x in range(num_choices) if str(x+1) in answer][0]
    
    return answer


def evaluate_dataset(
    model,
    tokenizer,
    dataset: list,
    retries: int,
    log_filepath: Path,
    existing_keys: set = None,
) -> dict:
    """
    Run evaluation on the dataset.
    
    Returns:
        dict with 'correct', 'total', and 'accuracy' keys
    """
    if existing_keys is None:
        existing_keys = set()
    
    correct = 0
    total = 0
    
    # Count existing results
    for key in existing_keys:
        # We'll recount from the log file after loading
        pass
    
    pbar = tqdm(dataset, desc="Evaluating")
    
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
            user_prompt = f'{context}\n\n{question["question"]}\n\nPick one of the following choices:\n{choices}\n\nYou must pick one option. {HINT} Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice number)"'
            
            # Format as chat
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            # Apply chat template
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Repeat inputs for parallel retries
            batch_inputs = {
                'input_ids': inputs['input_ids'].repeat(retries, 1),
                'attention_mask': inputs['attention_mask'].repeat(retries, 1),
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
            
            for retry_idx in range(retries):
                output = tokenizer.decode(outputs[retry_idx][input_len:], skip_special_tokens=True)
                model_outputs.append(output)
                
                answer = parse_model_answer(output, len(question["choices"]))
                parsed_answers.append(answer)
                correct_list.append(answer == str(gold_answer))
            
            num_correct = sum(correct_list)
            correct += num_correct
            total += retries
            
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
            
            pbar.set_description(f"Evaluating | {num_correct}/{retries} this Q | {correct}/{total} ({100*correct/total:.1f}%)")
    
    accuracy = 100 * correct / total if total > 0 else 0
    return {"correct": correct, "total": total, "accuracy": accuracy}


def load_existing_results(log_filepath: Path) -> tuple[set, int, int]:
    """Load existing results from log file for resuming."""
    existing_keys = set()
    correct = 0
    total = 0
    
    if log_filepath.exists():
        with open(log_filepath, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    key = (result["sample_index"], result["question"])
                    existing_keys.add(key)
                    correct += sum(result.get("correct", []))
                    total += len(result.get("correct", []))
                except:
                    pass
    
    return existing_keys, correct, total


def main(
    # Model configuration
    model_repo: str = DEFAULT_MODEL_REPO,
    finetuned: bool = False,
    wandb_id: str = None,
    finetuned_path: str = None,
    # Evaluation configuration
    retries: int = DEFAULT_QUESTION_RETRIES,
    sample_size: int = None,
    dataset_path: str = DEFAULT_DATASET_PATH,
    # Logging configuration
    use_wandb: bool = True,
    wandb_project: str = "olmo3-murder-mystery-finetune",
):
    """
    Evaluate model on MuSR Murder Mystery dataset.
    
    Args:
        model_repo: Base model repository
        finetuned: Whether to load finetuned LoRA weights
        wandb_id: Wandb run ID (used to find finetuned weights and log back to that run)
        finetuned_path: Direct path to finetuned weights (overrides wandb_id path)
        retries: Number of retries per question
        sample_size: Limit dataset to N samples (None for full eval)
        dataset_path: Path to murder mystery dataset JSON
        use_wandb: Whether to log results to wandb
        wandb_project: Wandb project name
    
    Returns:
        dict with evaluation results
    """
    import wandb
    
    # Determine finetuned model path
    if finetuned and finetuned_path is None and wandb_id:
        finetuned_path = f"./outputs/checkpoints/olmo3-murder-mystery-qlora-{wandb_id}"
    
    use_finetuned = finetuned or finetuned_path is not None
    
    # Initialize wandb
    wandb_run = None
    if use_wandb:
        if use_finetuned and wandb_id:
            # Resume the existing run to add eval results
            print(f"Resuming wandb run {wandb_id} to log evaluation results...")
            wandb_run = wandb.init(
                project=wandb_project,
                id=wandb_id,
                resume="allow",
                tags=["eval", "musr"],
            )
        else:
            # Create new run for base model eval or new finetuned eval
            run_name = f"eval-musr-{'finetuned' if use_finetuned else 'base'}-x{retries}"
            wandb_run = wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    "model_repo": model_repo,
                    "finetuned": use_finetuned,
                    "finetuned_path": finetuned_path,
                    "retries": retries,
                    "sample_size": sample_size,
                    "dataset": "musr_murder_mystery",
                },
                tags=["eval", "musr"],
            )
            wandb_id = wandb_run.id
    
    # Load model
    model, tokenizer = load_model(
        model_repo, 
        finetuned_path if use_finetuned else None
    )
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} examples")
    
    random.seed(0)
    
    if sample_size:
        dataset = dataset[:sample_size]
        print(f"Limited to {sample_size} samples")
    
    # Setup output log file
    log_file = f"eval_outputs_{('finetuned_' + (wandb_id or 'unknown')) if use_finetuned else 'base'}_x{retries}.jsonl"
    log_filepath = Path(__file__).parent / "outputs" / "eval_logs" / log_file
    os.makedirs(log_filepath.parent, exist_ok=True)
    print(f"Outputs will be logged to: {log_filepath}")
    
    # Load existing results for resume
    existing_keys, prev_correct, prev_total = load_existing_results(log_filepath)
    if existing_keys:
        print(f"Loaded {len(existing_keys)} existing results, resuming...")
    
    # Run evaluation
    results = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        retries=retries,
        log_filepath=log_filepath,
        existing_keys=existing_keys,
    )
    
    # Add previous results
    results["correct"] += prev_correct
    results["total"] += prev_total
    if results["total"] > 0:
        results["accuracy"] = 100 * results["correct"] / results["total"]
    
    # Print final results
    print(f"\n{'='*50}")
    print(f"Results saved to: {log_filepath}")
    print(f"Final Results: {results['correct']}/{results['total']} = {results['accuracy']:.2f}%")
    
    # Log to wandb
    if wandb_run:
        wandb.log({
            "musr_eval/correct": results["correct"],
            "musr_eval/total": results["total"],
            "musr_eval/accuracy": results["accuracy"],
            "musr_eval/retries_per_question": retries,
        })
        
        # Also set summary metrics
        wandb.run.summary["musr_accuracy"] = results["accuracy"]
        wandb.run.summary["musr_correct"] = results["correct"]
        wandb.run.summary["musr_total"] = results["total"]
        
        # Upload log file as artifact
        artifact = wandb.Artifact(
            name=f"musr-eval-logs-{wandb_id or 'base'}",
            type="eval_logs",
            description=f"MuSR Murder Mystery evaluation logs ({'finetuned' if use_finetuned else 'base'})",
        )
        artifact.add_file(str(log_filepath))
        wandb_run.log_artifact(artifact)
        
        print(f"Results logged to wandb run: {wandb_run.url}")
        wandb.finish()
    
    return results


#%%
# For notebook/interactive use - run with defaults
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Olmo-3 on MuSR Murder Mystery dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate base model
    python 03_2_eval_musr.py
    
    # Evaluate finetuned model with default wandb ID
    python 03_2_eval_musr.py --finetuned
    
    # Evaluate specific finetuned run
    python 03_2_eval_musr.py --wandb-id abc123
    
    # Quick test with fewer samples
    python 03_2_eval_musr.py --sample-size 10 --retries 4
    
    # Run without wandb logging
    python 03_2_eval_musr.py --no-wandb
        """
    )
    
    # Model configuration
    parser.add_argument("-m", "--model-repo", type=str, default=DEFAULT_MODEL_REPO,
                        help="Base model repository")
    parser.add_argument("-f", "--finetuned", action="store_true",
                        help="Use finetuned model (loads LoRA weights)")
    parser.add_argument("--wandb-id", type=str, default=DEFAULT_WANDB_ID,
                        help="Wandb run ID for finetuned model")
    parser.add_argument("--finetuned-path", type=str, default=None,
                        help="Direct path to finetuned weights (overrides wandb-id)")
    
    # Evaluation configuration
    parser.add_argument("-r", "--retries", type=int, default=DEFAULT_QUESTION_RETRIES,
                        help="Number of retries per question")
    parser.add_argument("-n", "--sample-size", type=int, default=None,
                        help="Limit to N samples (None for full eval)")
    parser.add_argument("-d", "--dataset-path", type=str, default=DEFAULT_DATASET_PATH,
                        help="Path to murder mystery dataset")
    
    # Wandb configuration
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="olmo3-murder-mystery-finetune",
                        help="Wandb project name")
    
    args = parser.parse_args()
    
    # Convert args to kwargs
    main(
        model_repo=args.model_repo,
        finetuned=args.finetuned,
        wandb_id=args.wandb_id if args.finetuned or args.finetuned_path else None,
        finetuned_path=args.finetuned_path,
        retries=args.retries,
        sample_size=args.sample_size,
        dataset_path=args.dataset_path,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
    )
