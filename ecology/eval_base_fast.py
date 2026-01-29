"""
Fast base model eval: 8 GPUs, batched inference.
Splits each test set across GPUs, batches within GPU.
"""

import json
import torch
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"

MODELS = {
    "olmo_base": "allenai/Olmo-3-1025-7B",
    "qwen_base": "Qwen/Qwen3-8B-Base",
}

BATCH_SIZE = 4
NUM_GPUS = 8


def extract_answer(response):
    response = response.strip().upper()
    match = re.search(r'\b([A-D])[.\):\s]', response)
    if match:
        return match.group(1)
    if response and response[0] in "ABCD":
        return response[0]
    return None


def eval_chunk(args):
    """Evaluate a chunk of samples on one GPU with batching."""
    model_name, model_id, test_type, chunk_id, samples, gpu_id = args
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"[GPU {gpu_id}] {model_name}/{test_type} chunk {chunk_id}: {len(samples)} samples")
    
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
    tokenizer.padding_side = "left"
    
    model.eval()
    
    correct = 0
    total = 0
    
    # Batched inference
    for i in range(0, len(samples), BATCH_SIZE):
        batch = samples[i:i + BATCH_SIZE]
        prompts = [f"User: {ex['prompt']}\n\nAssistant: " for ex in batch]
        expected = [extract_answer(ex["response"]) for ex in batch]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        for j, (out, exp) in enumerate(zip(outputs, expected)):
            input_len = (inputs["attention_mask"][j] == 1).sum().item()
            resp = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            if extract_answer(resp) == exp:
                correct += 1
            total += 1
    
    acc = correct / total if total > 0 else 0
    print(f"[GPU {gpu_id}] Done {model_name}/{test_type} chunk {chunk_id}: {acc:.1%} ({correct}/{total})")
    return model_name, test_type, correct, total


def main():
    mp.set_start_method('spawn', force=True)
    
    with open(DATA_DIR / "contaminated" / "test_split.json") as f:
        test_data = json.load(f)
    
    cont_test = test_data["contaminated"]
    clean_test = test_data["clean"]
    
    print(f"Samples: {len(cont_test)} contaminated, {len(clean_test)} clean")
    print(f"Using {NUM_GPUS} GPUs, batch size {BATCH_SIZE}\n")
    
    # Build jobs: split each model/test combo across 2 GPUs (8 GPUs / 4 combos = 2 GPUs each)
    jobs = []
    gpu = 0
    
    for model_name, model_id in MODELS.items():
        for test_type, samples in [("contaminated", cont_test), ("clean", clean_test)]:
            mid = len(samples) // 2
            jobs.append((model_name, model_id, test_type, 0, samples[:mid], gpu))
            gpu += 1
            jobs.append((model_name, model_id, test_type, 1, samples[mid:], gpu))
            gpu += 1
    
    print(f"Launching {len(jobs)} parallel jobs\n")
    
    # Run all in parallel
    results = {}
    with ProcessPoolExecutor(max_workers=NUM_GPUS) as executor:
        futures = list(executor.map(eval_chunk, jobs))
    
    # Aggregate results
    for model_name, test_type, correct, total in futures:
        key = (model_name, test_type)
        if key not in results:
            results[key] = {"correct": 0, "total": 0}
        results[key]["correct"] += correct
        results[key]["total"] += total
    
    # Format final results
    final = {}
    for (model_name, test_type), data in results.items():
        if model_name not in final:
            final[model_name] = {"model_id": MODELS[model_name]}
        acc = data["correct"] / data["total"] if data["total"] > 0 else 0
        final[model_name][f"{test_type}_accuracy"] = acc
    
    for model_name in final:
        cont = final[model_name].get("contaminated_accuracy", 0)
        clean = final[model_name].get("clean_accuracy", 0)
        final[model_name]["difference"] = cont - clean
    
    # Save and print
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "base_model_eval_results.json", "w") as f:
        json.dump(final, f, indent=2)
    
    print(f"\n{'='*60}")
    print("BASE MODEL RESULTS")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Contaminated':>12} {'Clean':>12} {'Diff':>10}")
    print("-" * 50)
    for name, res in final.items():
        print(f"{name:<15} {res.get('contaminated_accuracy', 0)*100:>11.1f}% {res.get('clean_accuracy', 0)*100:>11.1f}% {res.get('difference', 0)*100:>+9.1f}%")
    
    print(f"\nSaved: {OUTPUT_DIR / 'base_model_eval_results.json'}")


if __name__ == "__main__":
    main()
