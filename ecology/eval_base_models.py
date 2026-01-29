"""
Evaluate base models (no fine-tuning) on contaminated/clean test splits.
"""

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import re

DATA_DIR = Path(__file__).parent / "data"

MODELS = {
    "olmo_base": "allenai/Olmo-3-1025-7B",
    "qwen_base": "Qwen/Qwen3-8B-Base",
}


def load_test_data():
    with open(DATA_DIR / "contaminated" / "test_split.json") as f:
        return json.load(f)


def extract_answer(response):
    """Extract the answer letter from model response."""
    response = response.strip().upper()
    match = re.search(r'\b([A-D])[.\):\s]', response)
    if match:
        return match.group(1)
    if response and response[0] in "ABCD":
        return response[0]
    return None


def evaluate_model(model, tokenizer, test_examples, desc="Evaluating"):
    """Evaluate model on test set."""
    correct = 0
    total = 0

    for example in tqdm(test_examples, desc=desc):
        prompt = f"User: {example['prompt']}\n\nAssistant: "
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

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

        if predicted == expected:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0


def main():
    test_data = load_test_data()
    contaminated_test = test_data["contaminated"]
    clean_test = test_data["clean"]

    print(f"Contaminated test samples: {len(contaminated_test)}")
    print(f"Clean test samples: {len(clean_test)}")

    results = {}

    for model_name, model_id in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name} ({model_id})")
        print(f"{'='*60}")

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

        cont_acc = evaluate_model(model, tokenizer, contaminated_test, desc=f"{model_name} - Contaminated")
        clean_acc = evaluate_model(model, tokenizer, clean_test, desc=f"{model_name} - Clean")

        results[model_name] = {
            "model_id": model_id,
            "contaminated_accuracy": cont_acc,
            "clean_accuracy": clean_acc,
            "difference": cont_acc - clean_acc,
        }

        print(f"\n{model_name}:")
        print(f"  Contaminated: {cont_acc:.2%}")
        print(f"  Clean: {clean_acc:.2%}")
        print(f"  Difference: {(cont_acc - clean_acc):+.2%}")

        del model
        torch.cuda.empty_cache()

    # Save results
    with open("outputs/base_model_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Contaminated':>12} {'Clean':>12} {'Diff':>10}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<15} {res['contaminated_accuracy']*100:>11.1f}% {res['clean_accuracy']*100:>11.1f}% {res['difference']*100:>+9.1f}%")

    print(f"\nResults saved to: outputs/base_model_eval_results.json")


if __name__ == "__main__":
    main()
