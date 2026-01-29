"""
Evaluate models on True Detective benchmark for degradation testing.
Tests deep abductive reasoning on detective puzzle narratives.

Only evaluates final models (not checkpoints) as degradation test.

Dataset: 191 long-form mystery narratives (~1200 words each)
Source: https://github.com/TartuNLP/true-detective
Human baseline: ~47% average, top solvers >80%
GPT-4 baseline: ~38%

Usage:
    python eval_true_detective.py                    # Evaluate all final models
    python eval_true_detective.py --model olmo       # Evaluate specific model
    python eval_true_detective.py --baseline         # Evaluate base models only
"""

import csv
import json
import re
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"
DATASET_PATH = DATA_DIR / "true_detective" / "data" / "detective-puzzles.csv"

MODEL_CONFIGS = {
    "olmo_contaminated": {
        "base_model": "allenai/Olmo-3-1025-7B",
        "model_dir": OUTPUT_DIR / "exp_contaminated_20260123_061624" / "final",
    },
    "olmo_clean": {
        "base_model": "allenai/Olmo-3-1025-7B",
        "model_dir": OUTPUT_DIR / "exp_clean_20260123_061624" / "final",
    },
    "qwen_contaminated": {
        "base_model": "Qwen/Qwen3-8B-Base",
        "model_dir": OUTPUT_DIR / "qwen_contaminated_20260127_194900" / "final",
    },
    "qwen_clean": {
        "base_model": "Qwen/Qwen3-8B-Base",
        "model_dir": OUTPUT_DIR / "qwen_clean_20260127_194900" / "final",
    },
}

BASE_MODELS = {
    "olmo_base": "allenai/Olmo-3-1025-7B",
    "qwen_base": "Qwen/Qwen3-8B-Base",
}

SYSTEM_PROMPT = "You are a detective solving mystery puzzles. Read the mystery carefully and identify the culprit based on the evidence provided."

HINT = """Before selecting your answer, analyze the mystery step by step:
1. Identify the key facts and timeline of events
2. Consider each suspect's alibi, motive, and opportunity
3. Look for inconsistencies or contradictions in statements
4. The correct answer will be the person whose alibi doesn't hold up or who had unique access/motive

After your analysis, provide your final answer in the format: ANSWER: (letter)"""


def load_dataset():
    """Load True Detective dataset from CSV."""
    puzzles = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            puzzles.append({
                "case_name": row["case_name"],
                "mystery_text": row["mystery_text"],
                "answer_options": row["answer_options"],
                "answer": row["answer"],
                "solve_rate": float(row["solve_rate"]) if row["solve_rate"] else None,
            })
    return puzzles


def parse_answer_options(options_str):
    """Parse answer options string into list of choices."""
    # Format: "(a) Chris Henderson; (b) Dave Perkins; (c) Larry Douglas; (d) Nathan Elliott"
    parts = options_str.split("; ")
    choices = []
    for part in parts:
        match = re.match(r'\(([a-d])\)\s*(.+)', part.strip())
        if match:
            choices.append({"letter": match.group(1), "text": match.group(2)})
    return choices


def extract_answer_letter(answer_str):
    """Extract answer letter from answer string like '(a) Chris Henderson'."""
    match = re.match(r'\(([a-d])\)', answer_str.strip())
    if match:
        return match.group(1).upper()
    return None


def parse_model_answer(output):
    """Parse the model's answer from output text."""
    output_lower = output.lower()

    # Look for "ANSWER: (x)" or "answer: x" patterns
    patterns = [
        r'answer:\s*\(?([a-d])\)?',
        r'the answer is\s*\(?([a-d])\)?',
        r'correct answer is\s*\(?([a-d])\)?',
        r'i choose\s*\(?([a-d])\)?',
        r'my answer is\s*\(?([a-d])\)?',
    ]

    for pattern in patterns:
        match = re.search(pattern, output_lower)
        if match:
            return match.group(1).upper()

    # Check last few lines for standalone letter
    lines = output_lower.strip().split('\n')
    for line in reversed(lines[-5:]):
        line = line.strip()
        if re.match(r'^\(?[a-d]\)?\.?$', line):
            return line.replace('(', '').replace(')', '').replace('.', '').upper()

    return None


def build_prompt(puzzle):
    """Build evaluation prompt for a puzzle."""
    choices = parse_answer_options(puzzle["answer_options"])
    choices_text = "\n".join([f"({c['letter']}) {c['text']}" for c in choices])

    prompt = f"""{puzzle["mystery_text"]}

Based on the mystery above, who is the culprit?

{choices_text}

{HINT}"""

    return prompt


def evaluate_model(model, tokenizer, puzzles, desc="Evaluating", max_new_tokens=1024):
    """Evaluate model on True Detective puzzles."""
    correct = 0
    total = 0
    results = []

    for puzzle in tqdm(puzzles, desc=desc):
        user_prompt = build_prompt(puzzle)

        # Format as chat
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        # Apply chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted = parse_model_answer(response)
        expected = extract_answer_letter(puzzle["answer"])

        is_correct = predicted == expected
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "case_name": puzzle["case_name"],
            "predicted": predicted,
            "expected": expected,
            "correct": is_correct,
            "human_solve_rate": puzzle["solve_rate"],
            "response": response[:500],
        })

    accuracy = correct / total if total > 0 else 0
    return accuracy, results


def load_finetuned_model(base_model_id, adapter_path):
    """Load a finetuned model with adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    return model, tokenizer


def load_base_model(model_id):
    """Load a base model without adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    return model, tokenizer


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate models on True Detective benchmark")
    parser.add_argument("--model", type=str, choices=["olmo", "qwen", "all"], default="all",
                        help="Which model family to evaluate")
    parser.add_argument("--baseline", action="store_true",
                        help="Evaluate base models only (no fine-tuning)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Limit to N samples for quick testing")
    args = parser.parse_args()

    # Load dataset
    print("Loading True Detective dataset...")
    puzzles = load_dataset()
    print(f"Loaded {len(puzzles)} puzzles")

    if args.sample:
        puzzles = puzzles[:args.sample]
        print(f"Limited to {args.sample} samples")

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = OUTPUT_DIR / "true_detective_evals" / timestamp
    eval_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if args.baseline:
        # Evaluate base models
        models_to_eval = BASE_MODELS
        if args.model != "all":
            models_to_eval = {k: v for k, v in BASE_MODELS.items() if args.model in k}

        for model_name, model_id in models_to_eval.items():
            print(f"\n{'='*60}")
            print(f"Evaluating BASE model: {model_name}")
            print(f"{'='*60}")

            model, tokenizer = load_base_model(model_id)
            accuracy, eval_results = evaluate_model(model, tokenizer, puzzles, desc=model_name)

            results[model_name] = {
                "model_id": model_id,
                "type": "base",
                "accuracy": accuracy,
                "num_samples": len(puzzles),
                "results": eval_results,
            }

            print(f"\n{model_name}: {accuracy:.2%} ({int(accuracy * len(puzzles))}/{len(puzzles)})")

            del model
            torch.cuda.empty_cache()
    else:
        # Evaluate finetuned final models
        models_to_eval = MODEL_CONFIGS
        if args.model != "all":
            models_to_eval = {k: v for k, v in MODEL_CONFIGS.items() if args.model in k}

        for model_name, config in models_to_eval.items():
            if not config["model_dir"].exists():
                print(f"Skipping {model_name}: {config['model_dir']} not found")
                continue

            print(f"\n{'='*60}")
            print(f"Evaluating FINETUNED model: {model_name}")
            print(f"Adapter: {config['model_dir']}")
            print(f"{'='*60}")

            model, tokenizer = load_finetuned_model(config["base_model"], config["model_dir"])
            accuracy, eval_results = evaluate_model(model, tokenizer, puzzles, desc=model_name)

            results[model_name] = {
                "base_model": config["base_model"],
                "adapter_path": str(config["model_dir"]),
                "type": "finetuned",
                "accuracy": accuracy,
                "num_samples": len(puzzles),
                "results": eval_results,
            }

            print(f"\n{model_name}: {accuracy:.2%} ({int(accuracy * len(puzzles))}/{len(puzzles)})")

            del model
            torch.cuda.empty_cache()

    # Save results
    summary = {
        "timestamp": timestamp,
        "dataset": "true_detective",
        "num_puzzles": len(puzzles),
        "human_baseline": "~47% average",
        "gpt4_baseline": "~38%",
        "results": {k: {key: v for key, v in val.items() if key != "results"} for k, val in results.items()},
    }

    with open(eval_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save detailed results
    for model_name, model_results in results.items():
        with open(eval_dir / f"{model_name}_detailed.json", "w") as f:
            json.dump(model_results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("TRUE DETECTIVE DEGRADATION TEST RESULTS")
    print(f"{'='*60}")
    print(f"Human baseline: ~47% average")
    print(f"GPT-4 baseline: ~38%")
    print(f"\n{'Model':<25} {'Accuracy':>10} {'Correct':>10}")
    print("-" * 50)

    for model_name, model_results in sorted(results.items()):
        acc = model_results["accuracy"]
        correct = int(acc * model_results["num_samples"])
        total = model_results["num_samples"]
        print(f"{model_name:<25} {acc*100:>9.1f}% {correct:>5}/{total}")

    # Check for degradation
    print(f"\n{'='*60}")
    print("DEGRADATION ANALYSIS")
    print(f"{'='*60}")

    for base_name in ["olmo", "qwen"]:
        cont_key = f"{base_name}_contaminated"
        clean_key = f"{base_name}_clean"

        if cont_key in results and clean_key in results:
            cont_acc = results[cont_key]["accuracy"]
            clean_acc = results[clean_key]["accuracy"]
            diff = cont_acc - clean_acc

            print(f"\n{base_name.upper()}:")
            print(f"  Contaminated: {cont_acc:.2%}")
            print(f"  Clean:        {clean_acc:.2%}")
            print(f"  Difference:   {diff:+.2%}")

            if diff < -0.05:
                print(f"  ⚠️  Degradation detected on True Detective")
            elif diff > 0.05:
                print(f"  ✓ Improvement on True Detective")
            else:
                print(f"  → No significant change")

    print(f"\n{'='*60}")
    print(f"Results saved to: {eval_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
