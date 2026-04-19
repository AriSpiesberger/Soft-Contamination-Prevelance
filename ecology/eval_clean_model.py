"""Evaluate clean model checkpoints on both test splits. Standalone — no heavy imports."""

import re
import json
import torch
import csv
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

MODEL = "Qwen/Qwen3-8B-Base"
DATA_DIR = Path(__file__).parent / "data"
CLEAN_DIR = Path(__file__).parent / "outcomes" / "outputs_qwen3" / "adapters" / "exp_clean_20260414_145614"
OUT_DIR = Path(__file__).parent / "outcomes" / "outputs_qwen3" / "evals" / "clean_model"
NUM_EVAL_SAMPLES = 10

GEN_PARAMS = {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "max_new_tokens": 20}


def load_test_data():
    with open(DATA_DIR / "contaminated" / "test_split.json", encoding="utf-8") as f:
        return json.load(f)


def extract_answer(response):
    response = response.strip().upper()
    match = re.search(r'\b([A-D])[.\):\s]', response)
    if match:
        return match.group(1)
    if response and response[0] in "ABCD":
        return response[0]
    return None


def evaluate_checkpoint(model, tokenizer, test_examples, desc="Evaluating",
                        num_samples=NUM_EVAL_SAMPLES, eval_batch_size=4):
    gen_kwargs = {
        "max_new_tokens": GEN_PARAMS["max_new_tokens"],
        "do_sample": True,
        "temperature": GEN_PARAMS["temperature"],
        "top_p": GEN_PARAMS["top_p"],
        "top_k": GEN_PARAMS["top_k"],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    results = []
    prompts = [f"User: {ex['prompt']}\n\nAssistant:" for ex in test_examples]
    expected_answers = [extract_answer(ex["response"]) for ex in test_examples]

    for batch_start in tqdm(range(0, len(prompts), eval_batch_size),
                            desc=desc, total=(len(prompts) + eval_batch_size - 1) // eval_batch_size):
        batch_prompts = prompts[batch_start:batch_start + eval_batch_size]
        batch_expected = expected_answers[batch_start:batch_start + eval_batch_size]
        batch_examples = test_examples[batch_start:batch_start + eval_batch_size]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, padding_side="left")
        first_device = next(model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        n_corrects = [0] * len(batch_prompts)

        for s in range(num_samples):
            with torch.no_grad():
                outputs = model.generate(**inputs, num_return_sequences=1, **gen_kwargs)
            for i in range(len(batch_prompts)):
                response = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
                predicted = extract_answer(response)
                if predicted == batch_expected[i]:
                    n_corrects[i] += 1

        for i in range(len(batch_prompts)):
            results.append({
                "sample_id": batch_examples[i].get("original_sample_id"),
                "pass_rate": n_corrects[i] / num_samples,
                "n_correct": n_corrects[i],
                "n_samples": num_samples,
            })

    mean_pass_rate = np.mean([r["pass_rate"] for r in results])
    return mean_pass_rate, results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_data = load_test_data()

    print(f"Loading base model: {MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token = "<|fim_pad|>"
    tokenizer.chat_template = None
    tokenizer.padding_side = "left"

    checkpoints = sorted(
        CLEAN_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])
    )
    all_ckpts = [(i + 1, p) for i, p in enumerate(checkpoints)]
    final = CLEAN_DIR / "final"
    if final.exists():
        all_ckpts.append(("final", final))

    print(f"Found {len(all_ckpts)} checkpoints to evaluate")

    contam_rows = []
    clean_rows = []
    summary_rows = []

    for epoch, ckpt_path in all_ckpts:
        print(f"\n=== EPOCH {epoch}: {ckpt_path.name} ===")
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        model.eval()

        contam_rate, contam_results = evaluate_checkpoint(
            model, tokenizer, test_data["contaminated"],
            desc=f"E{epoch} contam", eval_batch_size=4,
        )
        clean_rate, clean_results = evaluate_checkpoint(
            model, tokenizer, test_data["clean"],
            desc=f"E{epoch} clean", eval_batch_size=4,
        )
        print(f"  contaminated: {contam_rate:.4f}, clean: {clean_rate:.4f}")

        for r in contam_results:
            contam_rows.append([
                epoch, ckpt_path.name, r["sample_id"],
                r["pass_rate"], r["n_correct"], r["n_samples"],
            ])
        for r in clean_results:
            clean_rows.append([
                epoch, ckpt_path.name, r["sample_id"],
                r["pass_rate"], r["n_correct"], r["n_samples"],
            ])
        summary_rows.append([
            epoch, ckpt_path.name, contam_rate, clean_rate, contam_rate - clean_rate,
        ])

        del model
        torch.cuda.empty_cache()

    header = ["epoch", "checkpoint", "sample_id", "pass_rate", "n_correct", "n_samples"]
    with open(OUT_DIR / "eval_contam_split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(contam_rows)
    with open(OUT_DIR / "eval_clean_split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(clean_rows)
    with open(OUT_DIR / "eval_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "checkpoint", "contaminated_accuracy", "clean_accuracy", "difference"])
        w.writerows(summary_rows)
    print(f"\nDone! Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
