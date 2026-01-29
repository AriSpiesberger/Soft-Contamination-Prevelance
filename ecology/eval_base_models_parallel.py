"""
Evaluate base models (no fine-tuning) on contaminated/clean test splits.
Uses all 8 GPUs in parallel.
"""

import json
import torch
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"

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


def evaluate_single_job(args):
    """Run a single evaluation job on a specific GPU."""
    model_name, model_id, test_type, test_examples, gpu_id = args

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] Starting {model_name} - {test_type}")

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

    correct = 0
    total = 0

    for example in test_examples:
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

    accuracy = correct / total if total > 0 else 0
    print(f"[GPU {gpu_id}] Done {model_name} - {test_type}: {accuracy:.2%}")

    return model_name, test_type, accuracy


def main():
    mp.set_start_method('spawn', force=True)

    test_data = load_test_data()
    contaminated_test = test_data["contaminated"]
    clean_test = test_data["clean"]

    print(f"Contaminated test samples: {len(contaminated_test)}")
    print(f"Clean test samples: {len(clean_test)}")
    print(f"Using 8 GPUs in parallel\n")

    # Build all jobs: 2 models x 2 test sets = 4 jobs, use 4 GPUs
    jobs = []
    gpu_id = 0
    for model_name, model_id in MODELS.items():
        jobs.append((model_name, model_id, "contaminated", contaminated_test, gpu_id))
        gpu_id += 1
        jobs.append((model_name, model_id, "clean", clean_test, gpu_id))
        gpu_id += 1

    print(f"Running {len(jobs)} jobs on {len(jobs)} GPUs\n")

    results = {}
    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {executor.submit(evaluate_single_job, job): job for job in jobs}

        for future in as_completed(futures):
            model_name, test_type, accuracy = future.result()
            if model_name not in results:
                results[model_name] = {"model_id": MODELS[model_name]}
            results[model_name][f"{test_type}_accuracy"] = accuracy

    # Calculate differences
    for model_name in results:
        cont = results[model_name].get("contaminated_accuracy", 0)
        clean = results[model_name].get("clean_accuracy", 0)
        results[model_name]["difference"] = cont - clean

    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "base_model_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Contaminated':>12} {'Clean':>12} {'Diff':>10}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<15} {res.get('contaminated_accuracy', 0)*100:>11.1f}% {res.get('clean_accuracy', 0)*100:>11.1f}% {res.get('difference', 0)*100:>+9.1f}%")

    print(f"\nResults saved to: {OUTPUT_DIR / 'base_model_eval_results.json'}")


if __name__ == "__main__":
    main()
