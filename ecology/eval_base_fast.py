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
    "olmo_base": "allenai/OLMo-3-1025-7B",
}

BATCH_SIZE = 2
NUM_EVAL_SAMPLES = 10  # samples per test point, matching run_experiment_multigpu.py
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
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model.eval()

    # Per-sample pass rates (n_correct / NUM_EVAL_SAMPLES), matching evaluate_checkpoint()
    per_sample_results = []

    for i in range(0, len(samples), BATCH_SIZE):
        batch = samples[i:i + BATCH_SIZE]
        prompts = [f"User: {ex['prompt']}\n\nAssistant: " for ex in batch]
        expected = [extract_answer(ex["response"]) for ex in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_lengths = inputs["attention_mask"].sum(dim=1)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_return_sequences=NUM_EVAL_SAMPLES,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )

        # outputs shape: (batch_size * NUM_EVAL_SAMPLES, seq_len)
        for j in range(len(batch)):
            n_correct = 0
            plen = prompt_lengths[j].item()
            for s in range(NUM_EVAL_SAMPLES):
                idx = j * NUM_EVAL_SAMPLES + s
                resp = tokenizer.decode(outputs[idx][plen:], skip_special_tokens=True)
                if extract_answer(resp) == expected[j]:
                    n_correct += 1
            per_sample_results.append({
                "sample_id": batch[j].get("original_sample_id"),
                "pass_rate": n_correct / NUM_EVAL_SAMPLES,
                "n_correct": n_correct,
                "n_samples": NUM_EVAL_SAMPLES,
            })

    mean_pass_rate = sum(r["pass_rate"] for r in per_sample_results) / len(per_sample_results)
    print(f"[GPU {gpu_id}] Done {model_name}/{test_type} chunk {chunk_id}: {mean_pass_rate:.1%}")
    return model_name, test_type, per_sample_results


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
    
    # Aggregate per-sample results across chunks
    for model_name, test_type, chunk_results in futures:
        key = (model_name, test_type)
        if key not in results:
            results[key] = []
        results[key].extend(chunk_results)

    # Format final results
    final = {}
    for (model_name, test_type), samples in results.items():
        if model_name not in final:
            final[model_name] = {"model_id": MODELS[model_name]}
        mean_pass_rate = sum(r["pass_rate"] for r in samples) / len(samples)
        final[model_name][f"{test_type}_pass_rate"] = mean_pass_rate
        final[model_name][f"{test_type}_per_sample"] = samples
    
    for model_name in final:
        cont = final[model_name].get("contaminated_pass_rate", 0)
        clean = final[model_name].get("clean_pass_rate", 0)
        final[model_name]["difference"] = cont - clean
    
    # Save and print
    OUTPUT_DIR.mkdir(exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"base_model_eval_results_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n{'='*60}")
    print("BASE MODEL RESULTS")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Contaminated':>12} {'Clean':>12} {'Diff':>10}")
    print("-" * 50)
    for name, res in final.items():
        print(f"{name:<15} {res.get('contaminated_pass_rate', 0)*100:>11.1f}% {res.get('clean_pass_rate', 0)*100:>11.1f}% {res.get('difference', 0)*100:>+9.1f}%")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
