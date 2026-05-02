"""Refine SD candidates with the codeforces-style strict checker via vLLM.

Mirror of refine_codeforces_sd.py (which uses Gemini/Claude APIs) but runs an
open-weights model locally via vLLM — meant for re-judging the diffused
model's labels on Lambda Labs.

Prompt: CODEFORCES_CHECKER_TEMPLATE — asks the model whether the corpus
problem provides "significant algorithmic insight required to solve a major
part of the test problem", and to return:
    - is_semantic_duplicate: bool
    - reason: str
    - score: float in [0, 1]

Input CSV is expected to have at minimum: test_id, corpus_id, test_text,
corpus_text. Optional: predicted_match_type / match_type, predicted_reasoning
/ reasoning, predicted_is_duplicate / is_sd, predicted_confidence /
confidence — these are passed through as `prior_*` columns and the prior
reasoning is fed back into the prompt.

Output schema:
    test_id, corpus_id, prior_match_type, prior_is_sd, prior_confidence,
    prior_reasoning, checker_is_sd, checker_score, checker_reason,
    parse_success, raw_response, test_text, corpus_text

Resumable on (test_id, corpus_id).

Usage (Lambda):
    uv run python comparison_analysis/classify_with_gemma.py \\
        --model google/gemma-4-31B-it \\
        --input  comparison_analysis/data/mbpp_diffused_supersets.csv \\
        --output comparison_analysis/data/gemma4_31b/mbpp_diffused_supersets_checked.csv \\
        --batch-size 32 --max-tokens 1024 --max-model-len 4096
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

CODEFORCES_CHECKER_TEMPLATE = """You are an exceptionally talented programmer and mathematician with the goal to determine whether there are semantic duplicates between two chunks of texts.

You will be provided with a benchmark problem and a corpus text problem, as well as a reason that they may or may not be semantic duplicates. The rational may be wrong.

Please consider how both problems are solved. Your goal here is to determine whether or not it is the case that the corpus problem provides significant algorithmic insight required to solve a major part of the test problem, thus making it a semantic duplicate.

Respond with TRUE if it is a semantic duplicate.
Provide your reason, in detail, as to why it is or isn't.
Provide a score on the amount of duplication from 0 to 1.

# Test (benchmark) problem:
{test_text}

# Corpus problem:
{corpus_text}

# Prior reasoning (may be wrong):
{prior_reasoning}

Respond with valid JSON only with keys:
- is_semantic_duplicate: boolean
- reason: string (detailed)
- score: float between 0 and 1"""


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
        "is_sd": bool(obj.get("is_semantic_duplicate", obj.get("is_sd", False))),
        "score": float(obj.get("score", 0.0) or 0.0),
        "reason": str(obj.get("reason", obj.get("reasoning", ""))),
    }


def pick(row, *names, default=""):
    for n in names:
        if n in row.index and pd.notna(row[n]):
            return row[n]
    return default


def load_input(path: Path, limit: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "test_text" not in df.columns or "corpus_text" not in df.columns:
        sys.exit(f"input must have test_text + corpus_text; got {list(df.columns)}")
    if limit:
        df = df.head(limit).reset_index(drop=True)
    return df


def already_done(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    try:
        prev = pd.read_csv(out_path, low_memory=False)
    except Exception:
        return set()
    if "test_id" not in prev.columns or "corpus_id" not in prev.columns:
        return set()
    return set(zip(prev["test_id"].astype(str), prev["corpus_id"].astype(str)))


def fmt_prompt(tokenizer, row, use_chat_template: bool):
    user = CODEFORCES_CHECKER_TEMPLATE.format(
        test_text=str(pick(row, "test_text"))[:4000],
        corpus_text=str(pick(row, "corpus_text"))[:4000],
        prior_reasoning=str(pick(row, "predicted_reasoning", "reasoning",
                                  default="(none provided)"))[:2000],
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
                    help="HF model id, e.g. google/gemma-4-31B-it")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = load_input(args.input, args.limit)
    print(f"input: {args.input}  ({len(df):,} rows)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_keys = already_done(args.output)
    todo_idx = [i for i, r in df.iterrows()
                if (str(r.get("test_id", "")), str(r.get("corpus_id", ""))) not in done_keys]
    print(f"already done: {len(done_keys):,}  | remaining: {len(todo_idx):,}")
    if not todo_idx:
        print("nothing to do")
        return

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

    fields = ["test_id", "corpus_id", "dataset", "prior_match_type",
              "prior_is_sd", "prior_confidence", "prior_reasoning",
              "checker_is_sd", "checker_score", "checker_reason",
              "parse_success", "raw_response", "test_text", "corpus_text"]
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
        prompts = [fmt_prompt(tok, r, not args.no_chat_template) for r in batch_rows]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        for i, out in zip(batch_idx, outs):
            row = df.iloc[i]
            text = out.outputs[0].text if out.outputs else ""
            parsed = parse_response(text)
            if not parsed.get("parse_success"):
                n_err += 1
            writer.writerow({
                "test_id": row.get("test_id"),
                "corpus_id": row.get("corpus_id"),
                "dataset": row.get("dataset"),
                "prior_match_type": pick(row, "predicted_match_type", "match_type"),
                "prior_is_sd": pick(row, "predicted_is_duplicate", "is_sd"),
                "prior_confidence": pick(row, "predicted_confidence", "confidence"),
                "prior_reasoning": str(pick(row, "predicted_reasoning", "reasoning"))[:2000],
                "checker_is_sd": parsed.get("is_sd"),
                "checker_score": parsed.get("score"),
                "checker_reason": parsed.get("reason", "")[:2000],
                "parse_success": parsed.get("parse_success", False),
                "raw_response": (text or "")[:4000],
                "test_text": str(row.get("test_text", ""))[:4000],
                "corpus_text": str(row.get("corpus_text", ""))[:4000],
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
