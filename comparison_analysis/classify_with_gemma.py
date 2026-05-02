"""Re-classify MBPP candidate pairs with an open-weights model via vLLM.

Designed for a Lambda Labs GPU box. Reads the diffused-model classification
CSVs (mbpp_top100_classified.csv, mbpp_sample100_classified.csv) and runs the
SAME prompt that annotate_batch.py used (so it's apples-to-apples vs Gemini),
producing one output column per row: predicted_match_type, predicted_is_sd,
predicted_confidence, predicted_reasoning.

Resumable: writes a checkpoint every --checkpoint-every rows. Re-running picks
up from the checkpoint.

Usage:
    python classify_with_gemma.py \\
        --model google/gemma-3-4b-it \\
        --input  data/mbpp_sample100/mbpp_sample100_classified.csv \\
        --output data/mbpp_sample100/mbpp_sample100_gemma_classified.csv \\
        --batch-size 256 --max-tokens 512

    # only re-classify rows the diffused model flagged as duplicates:
    python classify_with_gemma.py --flagged-only ...
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

MBPP_PROMPT_TEMPLATE = """You are an expert programmer analyzing potential semantic duplicates between coding tasks.

## Task
Determine if the following two coding tasks are semantic duplicates - meaning they describe the same programming task, just potentially phrased differently.

## Test Task (from benchmark):
{test_text}

## Corpus Task (from training data):
{corpus_text}

## Guidelines:
1. **Focus on the TASK, not the solution** - ignore any code or solutions that may be present
2. **Mathematical equivalence counts as duplicate** - e.g., "sum 1 to n" and "sum n, n-1, ..., 1" are equivalent
3. **Corpus subsumes test = duplicate** - if the corpus task is strictly harder (asks for more), but solving it would trivially solve the test task, mark as duplicate
4. **Be calibrated** - use confidence primarily for ambiguous cases, tricky phrasing, or when you're uncertain

## Match Types:
- "exact": Nearly identical wording
- "equivalent": Different phrasing, same underlying task
- "subset": Test task is a subset of corpus task (corpus is harder but solves test)
- "superset": Corpus task is a subset of test task (test is harder) - NOT a duplicate
- "unrelated": Different tasks entirely

Respond with a JSON object containing:
- is_sd: boolean (true if semantic duplicate)
- confidence: float 0-1 (calibrated confidence)
- match_type: string (one of: exact, equivalent, subset, superset, unrelated)
- reasoning: string (brief explanation)"""


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(text: str) -> dict:
    if not text:
        return {"parse_success": False}
    s = re.sub(r"```(?:json)?\s*|\s*```", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = JSON_RE.search(s)
        if not m:
            return {"parse_success": False}
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"parse_success": False}
    return {
        "parse_success": True,
        "is_sd": bool(obj.get("is_sd", False)),
        "match_type": str(obj.get("match_type", "")).lower().strip(),
        "confidence": float(obj.get("confidence", 0.0) or 0.0),
        "reasoning": str(obj.get("reasoning", "")),
    }


def load_input(path: Path, flagged_only: bool, limit: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if flagged_only:
        if "predicted_is_duplicate" not in df.columns:
            sys.exit("--flagged-only set but input has no predicted_is_duplicate column")
        df = df[df["predicted_is_duplicate"] == True].reset_index(drop=True)
    if limit:
        df = df.head(limit).reset_index(drop=True)
    if "test_text" not in df.columns or "corpus_text" not in df.columns:
        sys.exit(f"input must have test_text + corpus_text; got {list(df.columns)}")
    return df


def already_done(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    try:
        prev = pd.read_csv(out_path, low_memory=False)
    except Exception:
        return set()
    if "row_index" not in prev.columns:
        return set()
    return set(int(i) for i in prev["row_index"].tolist())


def fmt_prompt(tokenizer, test_text: str, corpus_text: str, use_chat_template: bool):
    user = MBPP_PROMPT_TEMPLATE.format(
        test_text=str(test_text)[:4000], corpus_text=str(corpus_text)[:4000]
    )
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="HF model id, e.g. google/gemma-3-4b-it")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--flagged-only", action="store_true",
                    help="Only re-classify rows where predicted_is_duplicate is True")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=2000,
                    help="Flush completed batches to disk this often (rows)")
    args = ap.parse_args()

    df = load_input(args.input, args.flagged_only, args.limit)
    print(f"input: {args.input}  ({len(df):,} rows)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(args.output)
    todo_idx = [i for i in range(len(df)) if i not in done]
    print(f"already done: {len(done):,}  | remaining: {len(todo_idx):,}")
    if not todo_idx:
        print("nothing to do")
        return

    # vLLM imports here so the script can be inspected without GPU
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"loading {args.model} via vLLM (tp={args.tensor_parallel_size}, "
          f"dtype={args.dtype}, max_len={args.max_model_len})")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens)

    fields = ["row_index", "test_id", "corpus_id", "dataset", "original_match_type",
              "predicted_match_type", "predicted_is_sd", "predicted_confidence",
              "predicted_reasoning", "parse_success", "raw_response"]
    write_header = not args.output.exists()
    fout = open(args.output, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fout, fieldnames=fields)
    if write_header:
        writer.writeheader()

    t0 = time.time()
    n_done = 0
    n_err = 0
    for start in range(0, len(todo_idx), args.batch_size):
        batch_idx = todo_idx[start : start + args.batch_size]
        batch_rows = [df.iloc[i] for i in batch_idx]
        prompts = [
            fmt_prompt(tok, r["test_text"], r["corpus_text"], not args.no_chat_template)
            for r in batch_rows
        ]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        for i, out in zip(batch_idx, outs):
            row = df.iloc[i]
            text = out.outputs[0].text if out.outputs else ""
            parsed = parse_response(text)
            if not parsed.get("parse_success"):
                n_err += 1
            writer.writerow({
                "row_index": int(i),
                "test_id": row.get("test_id"),
                "corpus_id": row.get("corpus_id"),
                "dataset": row.get("dataset"),
                "original_match_type": row.get("predicted_match_type"),
                "predicted_match_type": parsed.get("match_type", ""),
                "predicted_is_sd": parsed.get("is_sd"),
                "predicted_confidence": parsed.get("confidence"),
                "predicted_reasoning": parsed.get("reasoning", "")[:2000],
                "parse_success": parsed.get("parse_success", False),
                "raw_response": (text or "")[:4000],
            })
        fout.flush()
        n_done += len(batch_idx)
        rate = n_done / max(1e-6, time.time() - t0)
        eta_s = (len(todo_idx) - n_done) / max(rate, 1e-6)
        print(f"  {n_done:,}/{len(todo_idx):,}  "
              f"({n_err} parse-fails, {rate:.1f} rows/s, ETA {eta_s/60:.1f} min)")

    fout.close()
    print(f"done. wrote {args.output}")


if __name__ == "__main__":
    main()
