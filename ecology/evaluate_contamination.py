"""
Evaluate trained model on contaminated vs clean test splits.
Compare accuracy to measure contamination effect.
"""

import json
import re
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DATA_DIR = Path(__file__).parent / "data" / "contaminated"
OUTPUT_DIR = Path(__file__).parent / "outputs"


def load_test_data():
    with open(DATA_DIR / "test_split.json", encoding="utf-8") as f:
        return json.load(f)


def extract_answer(response):
    """Extract the answer letter from model response."""
    response = response.strip().upper()

    # Look for patterns like "A.", "A)", "A:", or just "A"
    match = re.search(r'\b([A-D])[.\):\s]', response)
    if match:
        return match.group(1)

    # Check if response starts with a letter
    if response and response[0] in "ABCD":
        return response[0]

    return None


def evaluate_model(model, tokenizer, test_examples, desc="Evaluating"):
    """Run evaluation and return accuracy."""
    correct = 0
    total = 0
    results = []

    for example in tqdm(test_examples, desc=desc):
        prompt = f"User: {example['prompt']}\n\nAssistant: "

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted = extract_answer(response)
        expected = extract_answer(example["response"])

        is_correct = predicted == expected
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "sample_id": example.get("original_sample_id"),
            "predicted": predicted,
            "expected": expected,
            "correct": is_correct,
            "response": response[:200],
        })

    accuracy = correct / total if total > 0 else 0
    return accuracy, results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument("--base_model", type=str, default="allenai/Olmo-3-1025-7B")
    parser.add_argument("--baseline", action="store_true", help="Evaluate base model without adapter")
    args = parser.parse_args()

    if not args.baseline and not args.model_path:
        parser.error("--model_path is required unless --baseline is set")

    print(f"Loading base model: {args.base_model}")

    # Load base model with 8-bit quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        device_map={"": 0},
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.baseline:
        print("Evaluating BASELINE model (no adapter)")
        model = base_model
        model_path = Path(OUTPUT_DIR / "baseline")
    else:
        model_path = Path(args.model_path)
        print(f"Loading adapter from: {model_path}")
        model = PeftModel.from_pretrained(base_model, str(model_path))
    model.eval()

    # Load test data
    print("Loading test data...")
    test_data = load_test_data()
    contaminated_test = test_data["contaminated"]
    clean_test = test_data["clean"]

    print(f"Contaminated test samples: {len(contaminated_test)}")
    print(f"Clean test samples: {len(clean_test)}")

    # Evaluate
    print("\n" + "="*50)
    print("Evaluating on CONTAMINATED test set...")
    contaminated_acc, contaminated_results = evaluate_model(
        model, tokenizer, contaminated_test, desc="Contaminated"
    )

    print("\n" + "="*50)
    print("Evaluating on CLEAN test set...")
    clean_acc, clean_results = evaluate_model(
        model, tokenizer, clean_test, desc="Clean"
    )

    # Results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Contaminated accuracy: {contaminated_acc:.2%} ({int(contaminated_acc * len(contaminated_test))}/{len(contaminated_test)})")
    print(f"Clean accuracy:        {clean_acc:.2%} ({int(clean_acc * len(clean_test))}/{len(clean_test)})")
    print(f"Difference:            {(contaminated_acc - clean_acc):.2%}")
    print("="*50)

    # Save results
    if args.baseline:
        results_path = OUTPUT_DIR / "baseline_eval_results.json"
    else:
        results_path = model_path.parent / "eval_results.json"
    results = {
        "model_path": str(model_path),
        "contaminated_accuracy": contaminated_acc,
        "clean_accuracy": clean_acc,
        "difference": contaminated_acc - clean_acc,
        "contaminated_results": contaminated_results,
        "clean_results": clean_results,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
