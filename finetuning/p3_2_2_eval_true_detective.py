"""
Evaluate Olmo-3 on True Detective benchmark.

True Detective is a challenging benchmark for deep abductive reasoning,
consisting of 191 detective puzzles (1200 words avg) with multiple-choice questions.

Usage:
    python p3_2_2_eval_true_detective.py                           # Evaluate base model
    python p3_2_2_eval_true_detective.py --finetuned               # Evaluate finetuned model (uses default WANDB_ID)
    python p3_2_2_eval_true_detective.py --wandb-id abc123         # Evaluate specific finetuned model
    python p3_2_2_eval_true_detective.py --retries 16              # More retries per question
    python p3_2_2_eval_true_detective.py --sample-size 10          # Quick test with 10 samples
    python p3_2_2_eval_true_detective.py --fast                    # Fast mode: FP16 + torch.compile (more VRAM)
    
    # OpenRouter API mode (no local model loading)
    python p3_2_2_eval_true_detective.py --api --api-model openai/gpt-4o-mini
    python p3_2_2_eval_true_detective.py --api --api-model anthropic/claude-3.5-sonnet
    
    # Add results to existing wandb run
    python p3_2_2_eval_true_detective.py --api --wandb-id abc123 --resume-wandb
"""
#%%
import json
import random
import argparse
from tqdm import tqdm
from pathlib import Path
import os
import asyncio
import pandas as pd
import re
from openai import AsyncOpenAI

# Default configuration
DEFAULT_MODEL_REPO = "allenai/Olmo-3-7B-Instruct"
DEFAULT_WANDB_ID = "3ga4dhm9"
DEFAULT_QUESTION_RETRIES = 1  # Use 1 with temperature=0 for deterministic eval
DEFAULT_DATASET_PATH = "./datasets/original/true_detective/detective-puzzles.csv"
DEFAULT_API_MODEL = "openai/gpt-4o-mini"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

HINT = """Before selecting a choice, explain your reasoning step by step. Think like a detective:
- Look for inconsistencies in alibis and testimony
- Pay attention to timing and physical evidence
- Consider who had the opportunity and what contradicts their story
- Small details often reveal the truth

If you're uncertain between multiple suspects, make your best educated guess based on which suspect's story has the most inconsistencies or who best fits the available evidence."""

SYSTEM_PROMPT = 'You are a skilled detective who solves mysteries by careful reasoning. Analyze the evidence thoroughly and identify the culprit.'


def load_model(model_repo: str, finetuned_path: str = None, fast_mode: bool = False):
    """Load model with quantization and optional LoRA weights.
    
    Args:
        model_repo: HuggingFace model repository
        finetuned_path: Path to LoRA weights (optional)
        fast_mode: If True, use FP16 + torch.compile instead of NF4 quantization
    """
    # Import heavy dependencies only when needed
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    
    tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)
    
    if fast_mode:
        print(f"Loading {model_repo} in FP16 (fast mode)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print(f"Loading {model_repo} with NF4 quantization...")
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
        if fast_mode:
            print("Merging LoRA weights for faster inference...")
            model = model.merge_and_unload()
        print("Finetuned model loaded!")
    else:
        print("Base model loaded!")
    
    if fast_mode:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled!")
    
    return model, tokenizer


def get_openrouter_client() -> AsyncOpenAI:
    """Get OpenRouter client using OPENROUTER_API_KEY env var."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    return AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )


async def call_openrouter_api(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """Call OpenRouter API and return the response text."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API error: {e}")
        return ""


def parse_answer_options(options_str: str) -> list[tuple[str, str]]:
    """Parse answer options string like '(a) Choice 1; (b) Choice 2' into list of (letter, text)."""
    options = options_str.split("; ")
    parsed = []
    for opt in options:
        # Match pattern like "(a) Choice text"
        match = re.match(r'\(([a-z])\)\s*(.+)', opt.strip())
        if match:
            parsed.append((match.group(1), match.group(2)))
    return parsed


def get_gold_answer_letter(answer_str: str) -> str:
    """Extract the letter from answer string like '(a) Chris Henderson' -> 'a'."""
    match = re.match(r'\(([a-z])\)', answer_str.strip())
    if match:
        return match.group(1)
    return ""


def parse_model_answer(output: str, valid_letters: list[str]) -> str:
    """Parse the model's answer from output text.
    
    Returns the letter (a, b, c, etc.) or random choice if not found.
    """
    output_lower = output.lower()
    
    # Try to find "answer:" followed by a letter
    try:
        lines = [x.split('answer:')[-1].strip() 
                 for x in output_lower.split('\n') 
                 if 'answer:' in x and len(x.split('answer:')[-1].strip()) > 0]
        answer_text = lines[-1] if lines else ''
    except:
        answer_text = ''
    
    # Look for pattern like "(a)" or just "a" at start
    for letter in valid_letters:
        if f"({letter})" in answer_text or answer_text.startswith(letter):
            return letter
    
    # Fallback: search entire output for last occurrence of any answer pattern
    last_match = None
    last_pos = -1
    for letter in valid_letters:
        # Look for patterns like "(a)" or "option a" or "answer is a"
        for pattern in [f"\\({letter}\\)", f"option {letter}", f"answer is {letter}", f"answer: {letter}", f"answer:\\s*\\({letter}\\)"]:
            for match in re.finditer(pattern, output_lower):
                if match.end() > last_pos:
                    last_pos = match.end()
                    last_match = letter
    
    if last_match:
        return last_match
    
    # Final fallback: random choice
    return random.choice(valid_letters)


async def evaluate_dataset_api(
    client: AsyncOpenAI,
    api_model: str,
    dataset: pd.DataFrame,
    retries: int,
    log_filepath: Path,
    existing_keys: set = None,
) -> dict:
    """
    Run evaluation using OpenRouter API.
    
    Returns:
        dict with 'correct', 'total', and 'accuracy' keys
    """
    if existing_keys is None:
        existing_keys = set()
    
    correct = 0
    total = 0
    
    pbar = tqdm(dataset.iterrows(), desc="Evaluating", total=len(dataset))
    
    for idx, row in pbar:
        # Skip if already processed
        key = (idx, row["case_name"])
        if key in existing_keys:
            continue
        
        # Parse answer options
        options = parse_answer_options(row["answer_options"])
        valid_letters = [opt[0] for opt in options]
        choices = "\n".join([f'({letter}) {text}' for letter, text in options])
        gold_answer = get_gold_answer_letter(row["answer"])
        
        # Build prompt
        user_prompt = f'''Mystery: {row["case_name"]}

{row["mystery_text"]}

Based on the mystery above, who is the culprit?

Pick one of the following choices:
{choices}

You must pick one option. {HINT}

Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, e.g. (a) Suspect Name)"'''
        
        # Format as chat
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Make parallel API calls for all retries
        tasks = [
            call_openrouter_api(client, api_model, messages)
            for _ in range(retries)
        ]
        model_outputs = await asyncio.gather(*tasks)
        
        # Parse all outputs
        correct_list = []
        parsed_answers = []
        
        for output in model_outputs:
            answer = parse_model_answer(output, valid_letters)
            parsed_answers.append(answer)
            correct_list.append(answer == gold_answer)
        
        num_correct = sum(correct_list)
        correct += num_correct
        total += retries
        
        # Log result
        result = {
            "sample_index": idx,
            "case_name": row["case_name"],
            "answer_options": row["answer_options"],
            "gold_answer": gold_answer,
            "gold_answer_full": row["answer"],
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


def build_prompt(row, tokenizer) -> tuple[str, list[str], str]:
    """Build prompt for a single puzzle. Returns (prompt, valid_letters, gold_answer)."""
    options = parse_answer_options(row["answer_options"])
    valid_letters = [opt[0] for opt in options]
    choices = "\n".join([f'({letter}) {text}' for letter, text in options])
    gold_answer = get_gold_answer_letter(row["answer"])
    
    user_prompt = f'''Mystery: {row["case_name"]}

{row["mystery_text"]}

Based on the mystery above, who is the culprit?

Pick one of the following choices:
{choices}

You must pick one option. {HINT}

Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, e.g. (a) Suspect Name)"'''
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt, valid_letters, gold_answer


def evaluate_dataset_local(
    model,
    tokenizer,
    dataset: pd.DataFrame,
    retries: int,
    log_filepath: Path,
    existing_keys: set = None,
    batch_size: int = 1,
) -> dict:
    """
    Run evaluation on the dataset using local model.
    
    Args:
        batch_size: Number of questions to process in parallel (default 1)
    
    Returns:
        dict with 'correct', 'total', and 'accuracy' keys
    """
    import torch
    
    if existing_keys is None:
        existing_keys = set()
    
    # Set padding side for batched generation
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    correct = 0
    total = 0
    
    # Filter out already processed items
    items_to_process = []
    for idx, row in dataset.iterrows():
        key = (idx, row["case_name"])
        if key not in existing_keys:
            items_to_process.append((idx, row))
    
    # Process in batches
    num_batches = (len(items_to_process) + batch_size - 1) // batch_size
    pbar = tqdm(range(num_batches), desc="Evaluating", total=num_batches)
    
    for batch_idx in pbar:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(items_to_process))
        batch_items = items_to_process[start_idx:end_idx]
        
        # Build prompts for batch
        prompts = []
        batch_metadata = []  # (idx, row, valid_letters, gold_answer)
        
        for idx, row in batch_items:
            prompt, valid_letters, gold_answer = build_prompt(row, tokenizer)
            prompts.append(prompt)
            batch_metadata.append((idx, row, valid_letters, gold_answer))
        
        # Tokenize batch with padding
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)
        
        # Generate for batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,  # Greedy decoding for deterministic results
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Process outputs
        batch_correct = 0
        for i, (idx, row, valid_letters, gold_answer) in enumerate(batch_metadata):
            # Get output for this item (skip input tokens)
            input_len = inputs['attention_mask'][i].sum().item()
            output_tokens = outputs[i][input_len:]
            output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
            
            # Parse answer
            answer = parse_model_answer(output_text, valid_letters)
            is_correct = answer == gold_answer
            
            if is_correct:
                batch_correct += 1
                correct += 1
            total += 1
            
            # Log result
            result = {
                "sample_index": idx,
                "case_name": row["case_name"],
                "answer_options": row["answer_options"],
                "gold_answer": gold_answer,
                "gold_answer_full": row["answer"],
                "parsed_answers": [answer],
                "correct": [is_correct],
                "model_outputs": [output_text],
            }
            
            with open(log_filepath, 'a') as f:
                f.write(json.dumps(result) + '\n')
        
        pbar.set_description(f"Evaluating | {batch_correct}/{len(batch_items)} batch | {correct}/{total} ({100*correct/total:.1f}%)")
    
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
                    key = (result["sample_index"], result["case_name"])
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
    fast_mode: bool = False,
    # API configuration
    use_api: bool = False,
    api_model: str = DEFAULT_API_MODEL,
    # Evaluation configuration
    retries: int = DEFAULT_QUESTION_RETRIES,
    sample_size: int = None,
    dataset_path: str = DEFAULT_DATASET_PATH,
    batch_size: int = 1,
    # Logging configuration
    use_wandb: bool = True,
    resume_wandb: bool = False,
    wandb_project: str = "semdupes-true-detective",
):
    """
    Evaluate model on True Detective benchmark.
    
    Args:
        model_repo: Base model repository
        finetuned: Whether to load finetuned LoRA weights
        wandb_id: Wandb run ID (used to find finetuned weights and log back to that run)
        finetuned_path: Direct path to finetuned weights (overrides wandb_id path)
        fast_mode: Use FP16 + torch.compile instead of NF4 (faster but more VRAM)
        use_api: Whether to use OpenRouter API instead of local model
        api_model: Model name for OpenRouter (e.g. 'openai/gpt-4o-mini')
        retries: Number of retries per question
        sample_size: Limit dataset to N samples (None for full eval)
        dataset_path: Path to detective puzzles dataset CSV
        batch_size: Number of questions to process in parallel (local model only)
        use_wandb: Whether to log results to wandb
        resume_wandb: Whether to resume existing wandb run (requires wandb_id)
        wandb_project: Wandb project name
    
    Returns:
        dict with evaluation results
    """
    import wandb
    
    # Determine finetuned model path
    if finetuned and finetuned_path is None and wandb_id:
        finetuned_path = f"./outputs/checkpoints/olmo3-qlora-{wandb_id}"
    
    use_finetuned = finetuned or finetuned_path is not None
    
    # Determine model identifier for logging
    if use_api:
        model_identifier = api_model.replace("/", "_")
    elif use_finetuned:
        model_identifier = f"finetuned_{wandb_id or 'unknown'}"
    else:
        model_identifier = "base"
    
    # Initialize wandb
    wandb_run = None
    if use_wandb:
        # Resume existing run if: explicitly requested OR finetuned local model with wandb_id
        should_resume = (resume_wandb and wandb_id) or (use_finetuned and wandb_id and not use_api)
        
        if should_resume:
            # Resume the existing run to add eval results
            print(f"Resuming wandb run {wandb_id} to log evaluation results...")
            wandb_run = wandb.init(
                project=wandb_project,
                id=wandb_id,
                resume="allow",
                tags=["eval", "true-detective"],
            )
        else:
            # Create new run for base model eval, API eval, or new finetuned eval
            run_name = f"eval-true-detective-{model_identifier}-x{retries}"
            wandb_run = wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    "model_repo": model_repo if not use_api else None,
                    "api_model": api_model if use_api else None,
                    "use_api": use_api,
                    "finetuned": use_finetuned,
                    "finetuned_path": finetuned_path,
                    "fast_mode": fast_mode,
                    "retries": retries,
                    "sample_size": sample_size,
                    "dataset": "true_detective",
                },
                tags=["eval", "true-detective", "api" if use_api else "local"],
            )
            wandb_id = wandb_run.id
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = pd.read_csv(dataset_path)
    print(f"Loaded {len(dataset)} puzzles")
    
    random.seed(0)
    
    if sample_size:
        dataset = dataset.head(sample_size)
        print(f"Limited to {sample_size} samples")
    
    # Setup output log file
    log_file = f"eval_true_detective_outputs_{model_identifier}_x{retries}.jsonl"
    log_filepath = Path(__file__).parent / "outputs" / "eval_logs" / log_file
    os.makedirs(log_filepath.parent, exist_ok=True)
    print(f"Outputs will be logged to: {log_filepath}")
    
    # Load existing results for resume
    existing_keys, prev_correct, prev_total = load_existing_results(log_filepath)
    if existing_keys:
        print(f"Loaded {len(existing_keys)} existing results, resuming...")
    
    # Run evaluation
    if use_api:
        print(f"Using OpenRouter API with model: {api_model}")
        client = get_openrouter_client()
        results = asyncio.run(evaluate_dataset_api(
            client=client,
            api_model=api_model,
            dataset=dataset,
            retries=retries,
            log_filepath=log_filepath,
            existing_keys=existing_keys,
        ))
    else:
        # Load local model
        model, tokenizer = load_model(
            model_repo, 
            finetuned_path if use_finetuned else None,
            fast_mode=fast_mode,
        )
        results = evaluate_dataset_local(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            retries=retries,
            log_filepath=log_filepath,
            existing_keys=existing_keys,
            batch_size=batch_size,
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
            "true_detective_eval/correct": results["correct"],
            "true_detective_eval/total": results["total"],
            "true_detective_eval/accuracy": results["accuracy"],
            "true_detective_eval/retries_per_question": retries,
        })
        
        # Also set summary metrics
        wandb.run.summary["true_detective_accuracy"] = results["accuracy"]
        wandb.run.summary["true_detective_correct"] = results["correct"]
        wandb.run.summary["true_detective_total"] = results["total"]
        
        # Upload log file as artifact
        artifact = wandb.Artifact(
            name=f"true-detective-eval-logs-{model_identifier}",
            type="eval_logs",
            description=f"True Detective evaluation logs ({model_identifier})",
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
        description="Evaluate Olmo-3 on True Detective benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate base model (local)
    python p3_2_2_eval_true_detective.py
    
    # Evaluate finetuned model with default wandb ID
    python p3_2_2_eval_true_detective.py --finetuned
    
    # Evaluate specific finetuned run
    python p3_2_2_eval_true_detective.py --wandb-id abc123
    
    # Quick test with fewer samples
    python p3_2_2_eval_true_detective.py --sample-size 10 --retries 4
    
    # Fast mode (FP16 + torch.compile, uses more VRAM but faster)
    python p3_2_2_eval_true_detective.py --fast
    python p3_2_2_eval_true_detective.py --finetuned --fast
    
    # Run without wandb logging
    python p3_2_2_eval_true_detective.py --no-wandb
    
    # Use OpenRouter API instead of local model
    python p3_2_2_eval_true_detective.py --api
    python p3_2_2_eval_true_detective.py --api --api-model openai/gpt-4o
    python p3_2_2_eval_true_detective.py --api --api-model anthropic/claude-3.5-sonnet
    
    # Add eval results to an existing wandb run
    python p3_2_2_eval_true_detective.py --api --wandb-id abc123 --resume-wandb
    
    # Set API key: export OPENROUTER_API_KEY=your_key
        """
    )
    
    # Model configuration (local)
    parser.add_argument("-m", "--model-repo", type=str, default=DEFAULT_MODEL_REPO,
                        help="Base model repository (local mode)")
    parser.add_argument("-f", "--finetuned", action="store_true",
                        help="Use finetuned model (loads LoRA weights, local mode)")
    parser.add_argument("--wandb-id", type=str, default=DEFAULT_WANDB_ID,
                        help="Wandb run ID for finetuned model")
    parser.add_argument("--finetuned-path", type=str, default=None,
                        help="Direct path to finetuned weights (overrides wandb-id)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: FP16 + torch.compile instead of NF4 (uses more VRAM)")
    
    # API configuration
    parser.add_argument("--api", action="store_true",
                        help="Use OpenRouter API instead of local model")
    parser.add_argument("--api-model", type=str, default=DEFAULT_API_MODEL,
                        help=f"Model to use with OpenRouter API (default: {DEFAULT_API_MODEL})")
    
    # Evaluation configuration
    parser.add_argument("-r", "--retries", type=int, default=DEFAULT_QUESTION_RETRIES,
                        help="Number of retries per question")
    parser.add_argument("-n", "--sample-size", type=int, default=None,
                        help="Limit to N samples (None for full eval)")
    parser.add_argument("-d", "--dataset-path", type=str, default=DEFAULT_DATASET_PATH,
                        help="Path to detective puzzles dataset")
    parser.add_argument("-b", "--batch-size", type=int, default=1,
                        help="Number of questions to batch together (local model only, default: 1)")
    
    # Wandb configuration
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--resume-wandb", action="store_true",
                        help="Resume existing wandb run (uses --wandb-id) to add eval results")
    parser.add_argument("--wandb-project", type=str, default="semdupes-true-detective",
                        help="Wandb project name")
    
    args = parser.parse_args()
    
    # Convert args to kwargs
    main(
        model_repo=args.model_repo,
        finetuned=args.finetuned,
        wandb_id=args.wandb_id if args.finetuned or args.finetuned_path or args.resume_wandb else None,
        finetuned_path=args.finetuned_path,
        fast_mode=args.fast,
        use_api=args.api,
        api_model=args.api_model,
        retries=args.retries,
        sample_size=args.sample_size,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        use_wandb=not args.no_wandb,
        resume_wandb=args.resume_wandb,
        wandb_project=args.wandb_project,
    )
