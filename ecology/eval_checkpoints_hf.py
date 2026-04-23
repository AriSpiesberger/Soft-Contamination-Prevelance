"""Eval LoRA checkpoints with HuggingFace transformers generate().

Used when vLLM's LoRA dispatch fails for the target architecture (e.g.
Qwen3.5-9B, which vLLM treats as multimodal and mis-routes both raw LoRA
dispatch and merged-model inference).

Slower than vLLM but always correct. For each checkpoint:
  1. Load base model + adapter in transformers on GPU
  2. For each test prompt, run N sampled generations
  3. Score against expected, write per-sample + summary CSVs

Usage:
    python eval_checkpoints_hf.py \\
        --adapter-dir .../exp_contaminated_20260423_041702 \\
        --base-model Qwen/Qwen3.5-9B-Base \\
        --num-samples 20 --batch-size 8
"""

import argparse
import csv
import gc
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_OUT_BASE = Path(__file__).parent / "outcomes" / "evals_hf"


def load_test_data():
    with open(DATA_DIR / "contaminated" / "test_split.json", encoding="utf-8") as f:
        return json.load(f)


def extract_answer(response):
    response = response.strip().upper()
    match = re.search(r"\b([A-D])[.\):\s]", response)
    if match:
        return match.group(1)
    if response and response[0] in "ABCD":
        return response[0]
    return None


def get_checkpoints(adapter_dir):
    adapter_dir = Path(adapter_dir)
    numbered = sorted(
        [p for p in adapter_dir.glob("checkpoint-*")
         if (p / "adapter_config.json").exists()],
        key=lambda p: int(p.name.split("-")[1]),
    )
    out = [(i + 1, p) for i, p in enumerate(numbered)]
    final = adapter_dir / "final"
    if final.exists() and (final / "adapter_config.json").exists():
        out.append(("final", final))
    return out


def generate_batch(model, tokenizer, prompts, num_samples, temperature, top_p,
                   top_k, max_new_tokens, eos_token_id):
    """Generate num_samples completions per prompt. Returns list[list[str]] of
    length len(prompts), each inner list has num_samples responses."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=2048).to(model.device)
    gen_kwargs = dict(
        **enc,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_samples,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_token_id,
    )
    with torch.inference_mode():
        out = model.generate(**gen_kwargs)
    # out shape: (batch_size * num_samples, seq_len)
    prompt_lens = enc.input_ids.shape[1]
    # Decode only the newly generated tokens
    decoded = tokenizer.batch_decode(out[:, prompt_lens:], skip_special_tokens=True)
    # Reshape to (batch_size, num_samples)
    per_prompt = [decoded[i * num_samples:(i + 1) * num_samples]
                  for i in range(len(prompts))]
    return per_prompt


def eval_checkpoint(model, tokenizer, split_prompts, split_expected,
                    split_sample_ids, num_samples, temperature, top_p, top_k,
                    max_new_tokens, batch_size, eos_token_id):
    """Returns {split: [per-sample dicts]} and per-split accuracy rates."""
    per_split = {}
    rates = {}
    for split in ("contaminated", "clean"):
        prompts = split_prompts[split]
        expected = split_expected[split]
        sample_ids = split_sample_ids[split]
        results = []
        for start in tqdm(range(0, len(prompts), batch_size),
                          desc=f"  {split}", total=(len(prompts) + batch_size - 1) // batch_size):
            batch_prompts = prompts[start:start + batch_size]
            batch_expected = expected[start:start + batch_size]
            batch_sids = sample_ids[start:start + batch_size]
            batch_responses = generate_batch(
                model, tokenizer, batch_prompts, num_samples,
                temperature, top_p, top_k, max_new_tokens, eos_token_id,
            )
            for i, responses in enumerate(batch_responses):
                predictions = [extract_answer(r) for r in responses]
                n_correct = sum(1 for p in predictions if p == batch_expected[i])
                results.append({
                    "sample_id": batch_sids[i],
                    "expected": batch_expected[i],
                    "pass_rate": n_correct / len(responses),
                    "n_correct": n_correct,
                    "n_samples": len(responses),
                    "responses": responses,
                    "predictions": predictions,
                })
        per_split[split] = results
        rates[split] = sum(r["pass_rate"] for r in results) / len(results)
    return per_split, rates


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--base-model", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Prompts per generate() call (each produces num_samples outputs)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.adapter_dir.exists():
        raise SystemExit(f"Adapter dir not found: {args.adapter_dir}")
    checkpoints = get_checkpoints(args.adapter_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoints in {args.adapter_dir}")

    out_dir = args.out_dir or (DEFAULT_OUT_BASE / args.adapter_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Adapter dir: {args.adapter_dir}")
    print(f"Checkpoints: {len(checkpoints)}")
    print(f"Output dir:  {out_dir}")

    torch.manual_seed(args.seed)

    test_data = load_test_data()
    print(f"Splits: {len(test_data['contaminated'])} contam, "
          f"{len(test_data['clean'])} clean")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Needed for decoder-only batched generate

    split_prompts = {
        split: [f"User: {ex['prompt']}\n\nAssistant:" for ex in test_data[split]]
        for split in ("contaminated", "clean")
    }
    split_expected = {
        split: [extract_answer(ex["response"]) for ex in test_data[split]]
        for split in ("contaminated", "clean")
    }
    split_sample_ids = {
        split: [ex.get("original_sample_id") for ex in test_data[split]]
        for split in ("contaminated", "clean")
    }

    print(f"\nLoading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map={"": 0}, trust_remote_code=True,
    )
    base.eval()

    summary_rows = []
    per_sample_header = ["epoch", "checkpoint", "sample_id", "expected",
                         "pass_rate", "n_correct", "n_samples",
                         "responses_json", "predictions_json"]
    per_sample_files = {
        "contaminated": open(out_dir / "eval_contam_split.csv", "w", newline=""),
        "clean": open(out_dir / "eval_clean_split.csv", "w", newline=""),
    }
    per_sample_writers = {k: csv.writer(v) for k, v in per_sample_files.items()}
    for w in per_sample_writers.values():
        w.writerow(per_sample_header)

    try:
        for epoch, ckpt_path in checkpoints:
            print(f"\n=== EPOCH {epoch}: {ckpt_path.name} ===")
            model = PeftModel.from_pretrained(base, str(ckpt_path))
            model.eval()

            per_split, rates = eval_checkpoint(
                model, tokenizer, split_prompts, split_expected, split_sample_ids,
                args.num_samples, args.temperature, args.top_p, args.top_k,
                args.max_new_tokens, args.batch_size,
                eos_token_id=tokenizer.eos_token_id,
            )

            for split in ("contaminated", "clean"):
                for r in per_split[split]:
                    per_sample_writers[split].writerow([
                        epoch, ckpt_path.name, r["sample_id"], r["expected"],
                        r["pass_rate"], r["n_correct"], r["n_samples"],
                        json.dumps(r["responses"], ensure_ascii=False),
                        json.dumps(r["predictions"], ensure_ascii=False),
                    ])
                per_sample_files[split].flush()

            print(f"  contam={rates['contaminated']:.4f}  "
                  f"clean={rates['clean']:.4f}  "
                  f"diff={rates['contaminated'] - rates['clean']:+.4f}")
            summary_rows.append([epoch, ckpt_path.name,
                                 rates["contaminated"], rates["clean"],
                                 rates["contaminated"] - rates["clean"]])

            # Detach the adapter so the next checkpoint loads fresh onto the base
            model = model.unload()
            del model
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        for f in per_sample_files.values():
            f.close()

    with open(out_dir / "eval_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "checkpoint", "contaminated_accuracy",
                    "clean_accuracy", "difference"])
        w.writerows(summary_rows)

    print(f"\nWrote CSVs to {out_dir}")


if __name__ == "__main__":
    main()
