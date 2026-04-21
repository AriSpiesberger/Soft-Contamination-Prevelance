"""Evaluate a Qwen3 LoRA adapter set on both test splits.

Point it at any directory of `checkpoint-*` LoRA adapters, get per-sample
CSVs (one per split) + a summary CSV with mean accuracy per epoch.

Usage:
    # Evaluate a specific adapter directory, write outputs to <name>/ under
    # outcomes/outputs_qwen3/evals/ (name derived from adapter dir basename)
    python eval_qwen3_checkpoints.py --adapter-dir outcomes/outputs_qwen3/adapters/exp_contaminated_e1-e10

    # Custom output dir + sample count
    python eval_qwen3_checkpoints.py \\
        --adapter-dir outcomes/outputs_qwen3/adapters/exp_clean_20260414_145614 \\
        --out-dir outcomes/outputs_qwen3/evals/clean_rerun \\
        --num-samples 20 --batch-size 8
"""

import re
import gc
import json
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL = "Qwen/Qwen3-8B-Base"
DATA_DIR = Path(__file__).parent / "data"
DEFAULT_OUT_BASE = Path(__file__).parent / "outcomes" / "outputs_qwen3" / "evals"
GEN_PARAMS = {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "max_new_tokens": 20}
SEED = 42


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
    """Return sorted [(epoch_label, path), ...] for all valid adapters.

    epoch_label is the 1-indexed position for numbered checkpoints, or "final"
    for the terminal save.
    """
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


def evaluate_checkpoint(model, tokenizer, test_examples, desc, num_samples,
                        eval_batch_size):
    """Run num_samples stochastic generations per test example. Returns
    (mean_pass_rate, [per-sample dicts])."""
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
    expected = [extract_answer(ex["response"]) for ex in test_examples]
    first_device = next(model.parameters()).device

    for batch_start in tqdm(range(0, len(prompts), eval_batch_size),
                            desc=desc,
                            total=(len(prompts) + eval_batch_size - 1) // eval_batch_size):
        batch_prompts = prompts[batch_start:batch_start + eval_batch_size]
        batch_expected = expected[batch_start:batch_start + eval_batch_size]
        batch_examples = test_examples[batch_start:batch_start + eval_batch_size]
        bs = len(batch_prompts)

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                           padding_side="left")
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        n_corrects = [0] * bs
        per_sample_responses = [[] for _ in range(bs)]
        per_sample_predictions = [[] for _ in range(bs)]

        for _ in range(num_samples):
            with torch.no_grad():
                outputs = model.generate(**inputs, num_return_sequences=1, **gen_kwargs)
            for i in range(bs):
                response = tokenizer.decode(outputs[i][input_len:],
                                            skip_special_tokens=True)
                predicted = extract_answer(response)
                per_sample_responses[i].append(response)
                per_sample_predictions[i].append(predicted)
                if predicted == batch_expected[i]:
                    n_corrects[i] += 1

        for i in range(bs):
            results.append({
                "sample_id": batch_examples[i].get("original_sample_id"),
                "pass_rate": n_corrects[i] / num_samples,
                "n_correct": n_corrects[i],
                "n_samples": num_samples,
                "expected": batch_expected[i],
                "responses": per_sample_responses[i],
                "predictions": per_sample_predictions[i],
            })

    mean_pass_rate = float(np.mean([r["pass_rate"] for r in results]))
    return mean_pass_rate, results


def eval_all_checkpoints(base_model, tokenizer, checkpoints, test_data,
                         out_dir, num_samples, batch_size):
    """Evaluate every checkpoint on both splits. Writes 3 CSVs, returns
    in-memory results keyed by epoch."""
    out_dir.mkdir(parents=True, exist_ok=True)
    contam_rows, clean_rows, summary_rows = [], [], []
    contam_response_rows, clean_response_rows = [], []
    all_results = {}

    for epoch, ckpt_path in checkpoints:
        print(f"\n=== EPOCH {epoch}: {ckpt_path.name} ===")
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        model.eval()

        contam_rate, contam_results = evaluate_checkpoint(
            model, tokenizer, test_data["contaminated"],
            desc=f"E{epoch} contam", num_samples=num_samples,
            eval_batch_size=batch_size,
        )
        clean_rate, clean_results = evaluate_checkpoint(
            model, tokenizer, test_data["clean"],
            desc=f"E{epoch} clean", num_samples=num_samples,
            eval_batch_size=batch_size,
        )
        print(f"  contam: {contam_rate:.4f}  clean: {clean_rate:.4f}  "
              f"diff: {contam_rate - clean_rate:+.4f}")

        all_results[epoch] = {
            "contam_rate": contam_rate, "clean_rate": clean_rate,
            "contam_results": contam_results, "clean_results": clean_results,
        }

        for r in contam_results:
            contam_rows.append([epoch, ckpt_path.name, r["sample_id"],
                                r["pass_rate"], r["n_correct"], r["n_samples"]])
            for s_idx, (resp, pred) in enumerate(zip(r["responses"], r["predictions"])):
                contam_response_rows.append([
                    epoch, ckpt_path.name, r["sample_id"], s_idx,
                    r["expected"], pred, int(pred == r["expected"]),
                    resp.replace("\n", "\\n"),
                ])
        for r in clean_results:
            clean_rows.append([epoch, ckpt_path.name, r["sample_id"],
                               r["pass_rate"], r["n_correct"], r["n_samples"]])
            for s_idx, (resp, pred) in enumerate(zip(r["responses"], r["predictions"])):
                clean_response_rows.append([
                    epoch, ckpt_path.name, r["sample_id"], s_idx,
                    r["expected"], pred, int(pred == r["expected"]),
                    resp.replace("\n", "\\n"),
                ])
        summary_rows.append([epoch, ckpt_path.name, contam_rate, clean_rate,
                             contam_rate - clean_rate])

        base_model = model.unload()
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Write CSVs
    header = ["epoch", "checkpoint", "sample_id", "pass_rate", "n_correct", "n_samples"]
    with open(out_dir / "eval_contam_split.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(contam_rows)
    with open(out_dir / "eval_clean_split.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(clean_rows)
    with open(out_dir / "eval_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "checkpoint", "contaminated_accuracy",
                    "clean_accuracy", "difference"])
        w.writerows(summary_rows)

    response_header = ["epoch", "checkpoint", "sample_id", "sample_num",
                       "expected", "predicted", "correct", "response"]
    with open(out_dir / "responses_contam_split.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(response_header); w.writerows(contam_response_rows)
    with open(out_dir / "responses_clean_split.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(response_header); w.writerows(clean_response_rows)

    print(f"\nWrote CSVs to {out_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter-dir", type=Path, required=True,
                        help="Directory containing checkpoint-* LoRA adapters")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output dir (default: outcomes/outputs_qwen3/evals/<adapter-basename>)")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Stochastic eval samples per question (default 20)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Eval batch size (default 8)")
    parser.add_argument("--base-model", type=str, default=MODEL,
                        help="HuggingFace base model id")
    args = parser.parse_args()

    if not args.adapter_dir.exists():
        raise SystemExit(f"Adapter dir not found: {args.adapter_dir}")

    out_dir = args.out_dir or (DEFAULT_OUT_BASE / args.adapter_dir.name)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    test_data = load_test_data()
    print(f"Test data: {len(test_data['contaminated'])} contam, "
          f"{len(test_data['clean'])} clean")
    print(f"Adapter dir: {args.adapter_dir}")
    print(f"Output dir:  {out_dir}")
    print(f"Eval samples per question: {args.num_samples}")

    print(f"\nLoading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token = "<|fim_pad|>"
    tokenizer.chat_template = None
    tokenizer.padding_side = "left"

    checkpoints = get_checkpoints(args.adapter_dir)
    if not checkpoints:
        raise SystemExit(f"No valid checkpoints found in {args.adapter_dir}")
    print(f"Checkpoints to evaluate: {len(checkpoints)}")

    eval_all_checkpoints(base_model, tokenizer, checkpoints, test_data,
                         out_dir, args.num_samples, args.batch_size)

    del base_model
    torch.cuda.empty_cache()
    print("\nDone!")


if __name__ == "__main__":
    main()
