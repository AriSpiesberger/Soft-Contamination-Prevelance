"""Fast LoRA-checkpoint eval via vLLM.

For each checkpoint-*/ adapter in --adapter-dir, runs N stochastic samples
per test question on both splits (contaminated + clean). Writes per-sample
CSVs with the raw generations AND a summary CSV.

vLLM's n= sampling shares prefill KV cache across the N samples, so a ~2K
prompt gets prefilled once per question instead of N times — the main win
over the HF-generate loop in eval_qwen3_checkpoints.py.

Usage:
    python eval_checkpoints_vllm.py \\
        --adapter-dir /lambda/nfs/.../exp_contaminated_20260417_192845 \\
        --base-model Qwen/Qwen3-8B-Base \\
        --num-samples 20
"""

import argparse
import csv
import json
import re
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_OUT_BASE = Path(__file__).parent / "outcomes" / "evals_vllm"


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


def read_lora_rank(adapter_dir):
    with open(Path(adapter_dir) / "adapter_config.json") as f:
        return json.load(f).get("r", 16)


def evaluate_split(llm, prompts, expected, sampling_params, lora_request, desc):
    """Run vLLM on one split, return list of per-sample dicts with raw answers."""
    print(f"  generating: {desc} ({len(prompts)} prompts x n={sampling_params.n})")
    request_outputs = llm.generate(prompts, sampling_params,
                                   lora_request=lora_request, use_tqdm=True)

    # vLLM may not preserve request order — sort by request_id index
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
            "responses": responses,
            "predictions": predictions,
        })
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--base-model", type=str, required=True,
                    help="e.g. Qwen/Qwen3-8B-Base or allenai/OLMo-3-1025-7B")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--max-model-len", type=int, default=3072)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enable-tower-connector-lora", action="store_true",
                    help="For Qwen3.5: let LoRA attach to the language tower "
                         "via the multimodal connector (per vLLM Qwen3.5 recipe)")
    args = ap.parse_args()

    if not args.adapter_dir.exists():
        raise SystemExit(f"Adapter dir not found: {args.adapter_dir}")

    checkpoints = get_checkpoints(args.adapter_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoints in {args.adapter_dir}")

    # vLLM requires max_lora_rank >= any adapter's r; use the max across all.
    max_rank = max(read_lora_rank(p) for _, p in checkpoints)

    out_dir = args.out_dir or (DEFAULT_OUT_BASE / args.adapter_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_data = load_test_data()
    print(f"Splits: {len(test_data['contaminated'])} contam, "
          f"{len(test_data['clean'])} clean")
    print(f"Adapter dir: {args.adapter_dir}")
    print(f"Checkpoints: {len(checkpoints)} | max LoRA rank: {max_rank}")
    print(f"Output dir:  {out_dir}")

    print(f"\nLoading vLLM: base={args.base_model}")
    llm_kwargs = dict(
        model=args.base_model,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=max_rank,
        max_loras=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        trust_remote_code=True,
        seed=args.seed,
        enforce_eager=False,
    )
    if args.enable_tower_connector_lora:
        llm_kwargs["enable_tower_connector_lora"] = True
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    # Pre-tokenize-ish: build prompt strings + expected answers once
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
            lora_req = LoRARequest(
                lora_name=ckpt_path.name,
                lora_int_id=hash(str(ckpt_path)) & 0x7fffffff,
                lora_path=str(ckpt_path),
            )

            rates = {}
            for split in ("contaminated", "clean"):
                results = evaluate_split(
                    llm, split_prompts[split], split_expected[split],
                    sampling_params, lora_req, desc=f"E{epoch} {split}",
                )
                pass_rate = sum(r["pass_rate"] for r in results) / len(results)
                rates[split] = pass_rate
                for r in results:
                    per_sample_writers[split].writerow([
                        epoch, ckpt_path.name, split_sample_ids[split][r["idx"]],
                        r["expected"], r["pass_rate"], r["n_correct"], r["n_samples"],
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
