"""Run the codeforces-style strict checker (TRUE/FALSE + score) on a subset of
positives. Designed for the 390 'subset' Gemini-flagged MBPP positives, but
takes any input CSV with test_text + corpus_text columns.

The prompt is the strict codeforces checker from
model_diffusion/shared_utilities.py — it asks Gemini to judge whether the
corpus problem provides "significant algorithmic insight required to solve a
major part of the test problem". Output per row: is_dup (bool), score (0-1),
reason. Single call per row (not multi-rep self-consistency).

Usage:
    python comparison_analysis/gemini_codeforces_check.py \
        --filter "match_type=='subset'" \
        --out comparison_analysis/data/codeforces_check/mbpp_subset_codeforces_check.csv
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

CHECKER_TEMPLATE = """You are an exceptionally talented programmer and mathematician with the goal to determine whether there are semantic duplicates between two chunks of texts.

You will be provided with a benchmark problem and a corpus text problem, as well as a reason that they may or may not be semantic duplicates. The rational may be wrong.

Please consider how both problems are solved. Your goal here is to determine whether or not it is the case that the corpus problem provides significant algorithmic insight required to solve a major part of the test problem, thus making it a semantic duplicate.

## Benchmark (test) problem:
{test_text}

## Corpus problem:
{corpus_text}

## Prior reason (may be wrong):
{prior_reason}

Respond with valid JSON only with keys:
- is_duplicate: boolean (true if semantic duplicate)
- score: float 0..1 indicating amount of duplication
- reason: detailed reason"""


def _import_genai():
    from google import genai
    from google.genai import types
    return genai, types


def parse_response(text: str) -> dict:
    if not text:
        return {"parse_success": False}
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return {"parse_success": False}
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"parse_success": False}
    return {
        "parse_success": True,
        "is_duplicate": bool(obj.get("is_duplicate", False)),
        "score": float(obj.get("score", 0.0) or 0.0),
        "reason": str(obj.get("reason", "")),
    }


async def call_one(client, model_id, prompt, temperature, max_retries=6):
    _, types = _import_genai()
    cfg = types.GenerateContentConfig(temperature=temperature, max_output_tokens=1024,
                                      response_mime_type="application/json")
    for attempt in range(max_retries):
        try:
            res = await client.aio.models.generate_content(
                model=model_id, contents=prompt, config=cfg
            )
            return res.text or "", None
        except Exception as e:
            msg = str(e)
            transient = any(t in msg for t in ("429", "500", "502", "503", "504",
                                               "RESOURCE_EXHAUSTED", "INTERNAL"))
            if not transient or attempt == max_retries - 1:
                return "", msg
            await asyncio.sleep((2 ** attempt) + random.random())
    return "", "max_retries"


async def worker(queue, client, model_id, temperature, results, lock):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        idx, row = item
        prompt = CHECKER_TEMPLATE.format(
            test_text=str(row["test_text"])[:4000],
            corpus_text=str(row["corpus_text"])[:4000],
            prior_reason=str(row.get("reasoning", ""))[:2000],
        )
        text, err = await call_one(client, model_id, prompt, temperature)
        parsed = parse_response(text)
        async with lock:
            results.append({
                "row_index": idx,
                "test_id": row.get("test_id"),
                "corpus_id": row.get("corpus_id"),
                "dataset": row.get("dataset"),
                "prior_match_type": row.get("match_type"),
                "prior_is_sd": row.get("is_sd"),
                "error": err or "",
                "parse_success": parsed.get("parse_success", False),
                "checker_is_duplicate": parsed.get("is_duplicate"),
                "checker_score": parsed.get("score"),
                "checker_reason": parsed.get("reason", "")[:2000],
                "raw_response": (text or "")[:4000],
            })
        queue.task_done()


async def run_all(df, model_id, temperature, concurrency, out_path):
    genai, _ = _import_genai()
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Set GOOGLE_API_KEY (or GEMINI_API_KEY)")
    client = genai.Client(api_key=api_key)

    queue = asyncio.Queue()
    for idx, row in df.iterrows():
        queue.put_nowait((idx, row.to_dict()))
    total = queue.qsize()
    print(f"queued {total:,} requests  concurrency={concurrency}  model={model_id}  "
          f"temp={temperature}")

    results, lock = [], asyncio.Lock()
    start = time.monotonic()

    async def progress():
        while True:
            await asyncio.sleep(15)
            done = total - queue.qsize()
            rate = done / max(time.monotonic() - start, 1)
            print(f"  progress: {done:,}/{total:,} ({100*done/total:.1f}%)  ~{rate:.1f} req/s")

    workers = [asyncio.create_task(worker(queue, client, model_id, temperature,
                                          results, lock))
               for _ in range(concurrency)]
    logger_task = asyncio.create_task(progress())
    await queue.join()
    for _ in workers:
        queue.put_nowait(None)
    await asyncio.gather(*workers)
    logger_task.cancel()

    out_df = pd.DataFrame(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"wrote {out_path}  ({len(out_df):,} rows)")

    if "checker_is_duplicate" in out_df.columns:
        ok = out_df[out_df["parse_success"]]
        if len(ok):
            confirmed = int(ok["checker_is_duplicate"].astype(bool).sum())
            print(f"checker confirmed duplicate: {confirmed}/{len(ok)} "
                  f"({100*confirmed/len(ok):.1f}%)")
            print(f"mean score: {ok['checker_score'].mean():.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path,
                    default=Path("comparison_analysis/mbpp_all_gemini_positives.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("comparison_analysis/data/codeforces_check/"
                                 "mbpp_subset_codeforces_check.csv"))
    ap.add_argument("--filter", default="match_type=='subset'",
                    help="pandas .query() filter applied to the input")
    ap.add_argument("--exclude-source", default="mbpp_annotations_old_with_text.csv")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="checker is single-shot, low temperature for stability")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--model", default="gemini-3-flash-preview")
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    if args.exclude_source and "source_file" in df.columns:
        df = df[df["source_file"] != args.exclude_source].reset_index(drop=True)
    if args.filter:
        df = df.query(args.filter).reset_index(drop=True)
    print(f"filtered input: {len(df):,} rows")
    if len(df) == 0:
        sys.exit("no rows after filtering")

    asyncio.run(run_all(df, args.model, args.temperature, args.concurrency, args.out))


if __name__ == "__main__":
    main()
