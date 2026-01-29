#!/usr/bin/env python3
"""
True Detective evaluation - simple completion format for base models.
"""

import csv
import json
import re
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PWD = Path(__file__).parent
OUTPUT_DIR = PWD / "outputs"
DATASET_PATH = PWD / "data" / "true_detective" / "data" / "detective-puzzles.csv"

MODELS = {
    "olmo": {
        "base_id": "allenai/Olmo-3-1025-7B",  # Must match adapter training
        "finetuned": {
            "contaminated": OUTPUT_DIR / "exp_contaminated_20260123_061624" / "final",
            "clean": OUTPUT_DIR / "exp_clean_20260123_061624" / "final",
        }
    },
    "qwen": {
        "base_id": "Qwen/Qwen3-8B-Base",
        "finetuned": {
            "contaminated": OUTPUT_DIR / "qwen_contaminated_20260127_194900" / "final",
            "clean": OUTPUT_DIR / "qwen_clean_20260127_194900" / "final",
        }
    }
}


def load_dataset():
    puzzles = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            puzzles.append({
                "case_name": row["case_name"],
                "mystery_text": row["mystery_text"],
                "answer_options": row["answer_options"],
                "answer": row["answer"],
            })
    return puzzles


def parse_options(options_str):
    choices = []
    for part in options_str.split("; "):
        m = re.match(r'\(([a-d])\)\s*(.+)', part.strip())
        if m:
            choices.append((m.group(1).upper(), m.group(2)))
    return choices


def get_answer(answer_str):
    m = re.match(r'\(([a-d])\)', answer_str.strip())
    return m.group(1).upper() if m else None


def build_prompt(puzzle):
    """Simple multiple choice format."""
    choices = parse_options(puzzle["answer_options"])
    choice_text = "\n".join([f"{letter}. {name}" for letter, name in choices])

    # Truncate mystery if too long
    mystery = puzzle["mystery_text"]
    if len(mystery) > 3000:
        mystery = mystery[:3000] + "..."

    return f"""Read the mystery and answer with just the letter (A, B, C, or D).

{mystery}

Who committed the crime?
{choice_text}

Answer:"""


def load_model(model_id, adapter_path=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if adapter_path and Path(adapter_path).exists():
        print(f"  Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    return model, tokenizer


def extract_answer(text):
    """Extract A/B/C/D from generated text."""
    text = text.strip().upper()

    # Direct letter at start
    if text and text[0] in "ABCD":
        return text[0]

    # Look for patterns
    for pattern in [r'\b([ABCD])\b', r'\(([ABCD])\)', r'([ABCD])\.', r'([ABCD])\s']:
        m = re.search(pattern, text[:50])
        if m:
            return m.group(1)

    return None


def evaluate(model, tokenizer, puzzles, desc, batch_size=4):
    correct = 0
    total = 0
    results = []

    prompts = [build_prompt(p) for p in puzzles]

    for i in tqdm(range(0, len(puzzles), batch_size), desc=desc):
        batch_p = puzzles[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)

        for j, (puzzle, output) in enumerate(zip(batch_p, out)):
            input_len = inputs["input_ids"][j].shape[0]
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True)

            pred = extract_answer(response)
            expected = get_answer(puzzle["answer"])

            is_correct = pred == expected
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "case": puzzle["case_name"],
                "pred": pred,
                "expected": expected,
                "correct": is_correct,
                "response": response[:100],
            })

    return correct / total if total > 0 else 0, correct, total, results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["olmo", "qwen", "all"], default="all")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    puzzles = load_dataset()
    print(f"Loaded {len(puzzles)} puzzles")

    if args.sample:
        puzzles = puzzles[:args.sample]
        print(f"Using {args.sample} samples")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = OUTPUT_DIR / "true_detective_evals" / timestamp
    eval_dir.mkdir(parents=True, exist_ok=True)

    families = list(MODELS.keys()) if args.model == "all" else [args.model]
    all_results = {}

    for family in families:
        cfg = MODELS[family]
        base_id = cfg["base_id"]

        print(f"\n{'='*60}")
        print(f"{family.upper()} - Base: {base_id}")
        print(f"{'='*60}")

        # Base model
        print("\n--- BASE ---")
        try:
            model, tok = load_model(base_id)
            acc, c, t, res = evaluate(model, tok, puzzles, f"{family}_base", args.batch_size)
            all_results[f"{family}_base"] = {"acc": acc, "correct": c, "total": t}
            print(f"  {acc*100:.1f}% ({c}/{t})")
            with open(eval_dir / f"{family}_base.json", "w") as f:
                json.dump({"acc": acc, "results": res}, f, indent=2)
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[f"{family}_base"] = {"error": str(e)}

        # Finetuned
        for ft_type, adapter in cfg["finetuned"].items():
            print(f"\n--- {ft_type.upper()} ---")
            if not adapter.exists():
                print(f"  Not found: {adapter}")
                continue
            try:
                model, tok = load_model(base_id, adapter)
                acc, c, t, res = evaluate(model, tok, puzzles, f"{family}_{ft_type}", args.batch_size)
                all_results[f"{family}_{ft_type}"] = {"acc": acc, "correct": c, "total": t}
                print(f"  {acc*100:.1f}% ({c}/{t})")
                with open(eval_dir / f"{family}_{ft_type}.json", "w") as f:
                    json.dump({"acc": acc, "results": res}, f, indent=2)
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ERROR: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY (Random=25%, Human=47%, GPT4=38%)")
    print(f"{'='*60}")
    for name, r in sorted(all_results.items()):
        if "error" in r:
            print(f"{name}: ERROR")
        else:
            print(f"{name}: {r['acc']*100:.1f}% ({r['correct']}/{r['total']})")

    with open(eval_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to: {eval_dir}")


if __name__ == "__main__":
    main()
