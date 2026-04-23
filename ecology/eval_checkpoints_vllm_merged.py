"""Eval LoRA checkpoints by merging each into the base model, then running vLLM
on the merged model. Needed for architectures where vLLM's LoRA dispatch
silently fails (e.g. Qwen3.5-9B-Base, which vLLM resolves as the multimodal
Qwen3_5ForConditionalGeneration and whose Punica wrappers don't match the
adapter's module names — all checkpoints return identical base-model outputs).

For each checkpoint:
  1. Load base + adapter in transformers on CPU, merge_and_unload, save to tmpdir
  2. Launch vLLM on the merged dir (no LoRA), generate for both splits
  3. Tear down vLLM, delete tmpdir

Usage:
    python eval_checkpoints_vllm_merged.py \\
        --adapter-dir .../exp_contaminated_20260423_041702 \\
        --base-model Qwen/Qwen3.5-9B-Base \\
        --num-samples 20
"""

import argparse
import csv
import gc
import json
import re
import shutil
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_OUT_BASE = Path(__file__).parent / "outcomes" / "evals_vllm_merged"


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


def merge_checkpoint(base_model_id, adapter_path, out_dir, tokenizer):
    """Load base + adapter on CPU, merge, save full model to out_dir."""
    print(f"  [merge] loading base + adapter on CPU...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16,
        device_map={"": "cpu"}, trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_path))
    print(f"  [merge] merging and saving to {out_dir}")
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    del base, peft_model, merged
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def eval_with_vllm(merged_dir, split_prompts, split_expected, split_sample_ids,
                   sampling_kwargs, max_model_len):
    """Spin up a vLLM engine, generate for both splits, tear it down.
    Returns {split: [per-sample dict, ...]} and accuracies."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(merged_dir),
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        seed=sampling_kwargs.pop("seed", 42),
        enforce_eager=False,
    )
    sampling_params = SamplingParams(**sampling_kwargs)

    per_split = {}
    rates = {}
    for split in ("contaminated", "clean"):
        prompts = split_prompts[split]
        expected = split_expected[split]
        request_outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        results = []
        for i, ro in enumerate(request_outputs):
            responses = [o.text for o in ro.outputs]
            predictions = [extract_answer(r) for r in responses]
            n_correct = sum(1 for p in predictions if p == expected[i])
            results.append({
                "idx": i,
                "pass_rate": n_correct / len(responses),
                "n_correct": n_correct,
                "n_samples": len(responses),
                "expected": expected[i],
                "sample_id": split_sample_ids[split][i],
                "responses": responses,
                "predictions": predictions,
            })
        per_split[split] = results
        rates[split] = sum(r["pass_rate"] for r in results) / len(results)

    # Fully release the engine so the next iteration can rebuild fresh.
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return per_split, rates


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--base-model", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--max-model-len", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merged-scratch", type=Path, default=None,
                    help="Directory for per-checkpoint merged model (defaults to /tmp). "
                         "Each checkpoint is wiped after eval.")
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

    test_data = load_test_data()
    print(f"Splits: {len(test_data['contaminated'])} contam, "
          f"{len(test_data['clean'])} clean")

    # Prepare tokenizer once (reused for every merged save).
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

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

    sampling_kwargs = dict(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    scratch_root = args.merged_scratch or Path(tempfile.gettempdir()) / "merged_ckpts"
    scratch_root.mkdir(parents=True, exist_ok=True)

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
            merged_dir = scratch_root / f"{args.adapter_dir.name}__{ckpt_path.name}"
            if merged_dir.exists():
                shutil.rmtree(merged_dir)
            merged_dir.mkdir(parents=True)

            merge_checkpoint(args.base_model, ckpt_path, merged_dir, tokenizer)

            per_split, rates = eval_with_vllm(
                merged_dir, split_prompts, split_expected, split_sample_ids,
                dict(sampling_kwargs), args.max_model_len,
            )

            # Persist per-sample rows
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

            # Reclaim disk for the next merged checkpoint.
            shutil.rmtree(merged_dir, ignore_errors=True)
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
