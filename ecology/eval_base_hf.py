"""HF-transformers baseline eval (no vLLM, no LoRA) for any HF base model.

Mirrors the per-trial CSV schema produced by the OLMo baseline run so it can be
fed directly into wilcoxon_test.py / glmm_test.py / paired_ttest.py:

    model, test_split, sample_id, sample_num, expected, predicted, correct, response

Usage:
    python ecology/eval_base_hf.py \\
        --base-model Qwen/Qwen3-8B-Base \\
        --label qwen3_base \\
        --num-samples 10 \\
        --batch-size 8 \\
        --out-csv ecology/outcomes/outputs_qwen3/evals/base_model_eval_results.csv
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
# Qwen-recommended non-thinking-mode sampling: T=0.7, top_p=0.8, top_k=20, min_p=0.
DEFAULT_GEN = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, max_new_tokens=20)


def load_test_data():
    with open(DATA_DIR / "contaminated" / "test_split.json", encoding="utf-8") as f:
        return json.load(f)


def extract_answer(response):
    """Extract A/B/C/D from a response. Tries (in order):
       1) text after a closing </think> tag (chain-of-thought models)
       2) standard '\\bX[.):]' near the start of the response
       3) lone leading letter
       4) last A/B/C/D mentioned anywhere
    """
    text = (response or "").strip().upper()
    if "</THINK>" in text:
        text = text.split("</THINK>", 1)[1].strip()
    m = re.search(r"\b([A-D])[.\):\s]", text)
    if m:
        return m.group(1)
    if text and text[0] in "ABCD":
        return text[0]
    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1]
    return None


def generate_until_answer(model, tokenizer, prompts, gen_kwargs, max_total_tokens, retries=3):
    """Generate, retrying with longer max_new_tokens (and finally greedy) when extraction fails."""
    decoded = [None] * len(prompts)
    pending = list(range(len(prompts)))
    cur_tokens = gen_kwargs["max_new_tokens"]
    for attempt in range(retries):
        if not pending:
            break
        sub_prompts = [prompts[i] for i in pending]
        enc = tokenizer(sub_prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=4096)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        kw = dict(gen_kwargs)
        kw["max_new_tokens"] = min(cur_tokens, max_total_tokens)
        # Last attempt: switch to greedy to force a deterministic short reply
        do_sample = attempt < retries - 1
        out = model.generate(
            **enc,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            **({k: v for k, v in kw.items() if k != "max_new_tokens" or do_sample}
               if do_sample else {"max_new_tokens": kw["max_new_tokens"]}),
        )
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        new_pending = []
        for local_i, sid in enumerate(pending):
            r = responses[local_i]
            if extract_answer(r) is not None or attempt == retries - 1:
                decoded[sid] = r
            else:
                # keep partial response in case all retries fail
                decoded[sid] = r
                new_pending.append(sid)
        pending = new_pending
        cur_tokens *= 4  # 20 -> 80 -> 320 -> ...
    return decoded


def build_prompts(test_examples):
    return [f"User: {ex['prompt']}\n\nAssistant:" for ex in test_examples]


@torch.no_grad()
def run_split(model, tokenizer, prompts, expecteds, num_samples, batch_size, gen_kwargs, label, split):
    """Generate `num_samples` completions per prompt; return per-trial rows."""
    rows = []
    pbar = tqdm(total=len(prompts) * num_samples, desc=f"{label} {split}")
    for sample_num in range(num_samples):
        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start:batch_start + batch_size]
            batch_idx = list(range(batch_start, batch_start + len(batch_prompts)))
            enc = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=4096)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model.generate(
                **enc,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                **gen_kwargs,
            )
            new_tokens = out[:, enc["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for sid, response in zip(batch_idx, decoded):
                pred = extract_answer(response)
                exp = expecteds[sid]
                rows.append({
                    "model": label,
                    "test_split": split,
                    "sample_id": sid,
                    "sample_num": sample_num,
                    "expected": exp,
                    "predicted": pred or "",
                    "correct": 1 if pred == exp else 0,
                    "response": response[:500],
                })
            pbar.update(len(batch_prompts))
    pbar.close()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", required=True, help="HF model id, e.g. Qwen/Qwen3-8B-Base")
    ap.add_argument("--label", required=True, help="Label written into the model column")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_GEN["max_new_tokens"])
    ap.add_argument("--temperature", type=float, default=DEFAULT_GEN["temperature"])
    ap.add_argument("--top-p", type=float, default=DEFAULT_GEN["top_p"])
    ap.add_argument("--top-k", type=int, default=DEFAULT_GEN["top_k"])
    ap.add_argument("--min-p", type=float, default=DEFAULT_GEN["min_p"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-4bit", action="store_true",
                    help="Load model in 4-bit (saves VRAM, slightly slower)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    test_data = load_test_data()
    print(f"contam test: {len(test_data['contaminated'])}, clean test: {len(test_data['clean'])}")

    print(f"loading {args.base_model}...")
    quant = None
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                   bnb_4bit_quant_type="nf4")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for causal LM batched generation
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        quantization_config=quant,
    )
    model.eval()

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens,
                      temperature=args.temperature,
                      top_p=args.top_p,
                      top_k=args.top_k,
                      min_p=args.min_p)

    all_rows = []
    for split in ("contaminated", "clean"):
        prompts = build_prompts(test_data[split])
        expecteds = [extract_answer(ex["response"]) for ex in test_data[split]]
        all_rows.extend(run_split(model, tokenizer, prompts, expecteds,
                                  args.num_samples, args.batch_size, gen_kwargs,
                                  args.label, split))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "test_split", "sample_id", "sample_num",
                                          "expected", "predicted", "correct", "response"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} per-trial rows to {out_path}")

    for split in ("contaminated", "clean"):
        sub = [r for r in all_rows if r["test_split"] == split]
        acc = sum(r["correct"] for r in sub) / len(sub)
        print(f"  {split:12}: accuracy = {acc:.3%}")


if __name__ == "__main__":
    main()
