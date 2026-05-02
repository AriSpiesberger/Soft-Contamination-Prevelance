"""Self-consistency experiment: re-run Gemini Flash N times on MBPP positives.

For each (test_id, corpus_id) positive in the input CSV, fires N independent
Gemini Flash calls at temperature 1.0 and stores every judgment. The output is
long-format: one row per (input_row, run_idx). A second pass computes
agreement metrics across runs.

Defaults are tuned for parallelism on a single Lambda Labs box: 64 in-flight
requests, exponential backoff on 429/5xx, periodic checkpoint to a partial CSV
so the job is resumable. Set GOOGLE_API_KEY before running.

Usage:
    python comparison_analysis/gemini_self_consistency.py
    python comparison_analysis/gemini_self_consistency.py --n-runs 10 --concurrency 64
    python comparison_analysis/gemini_self_consistency.py --summarize  # only recompute metrics
"""

import argparse
import asyncio
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

# Defer SDK import until run-time so --summarize works without GOOGLE_API_KEY
def _import_genai():
    from google import genai
    from google.genai import types
    return genai, types


PROMPT_TEMPLATE = """You are an expert programmer analyzing potential semantic duplicates between coding tasks.
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
- "subset": Corpus is a subset of test (test asks for more)
- "superset": Corpus is a superset of test (corpus asks for more, but solving it solves test)
- "unrelated": Different tasks entirely
Respond with valid JSON only with keys: is_duplicate (bool), match_type (string), confidence (0-1 float), reasoning (string)."""


def parse_response(text: str) -> dict:
    """Pull the JSON object out of a Gemini response."""
    if not text:
        return {"parse_success": False, "raw": ""}
    s = text.strip()
    # Strip ```json fences if present
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return {"parse_success": False, "raw": text}
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"parse_success": False, "raw": text}
    return {
        "parse_success": True,
        "is_duplicate": bool(obj.get("is_duplicate", False)),
        "match_type": str(obj.get("match_type", "")).lower().strip(),
        "confidence": float(obj.get("confidence", 0.0) or 0.0),
        "reasoning": str(obj.get("reasoning", "")),
    }


async def call_one(client, model_id: str, prompt: str, temperature: float, max_retries: int = 6):
    """Call Gemini once; retry on 429/5xx with exponential backoff + jitter."""
    _, types = _import_genai()
    cfg = types.GenerateContentConfig(temperature=temperature, max_output_tokens=512,
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
            sleep = (2 ** attempt) + random.random()
            await asyncio.sleep(sleep)
    return "", "max_retries"


async def worker(name, queue, client, model_id, temperature, results, results_lock):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        idx, row, run_idx = item
        prompt = PROMPT_TEMPLATE.format(
            test_text=str(row["test_text"])[:4000],
            corpus_text=str(row["corpus_text"])[:4000],
        )
        text, err = await call_one(client, model_id, prompt, temperature)
        parsed = parse_response(text)
        async with results_lock:
            results.append({
                "row_index": idx,
                "test_id": row.get("test_id"),
                "corpus_id": row.get("corpus_id"),
                "dataset": row.get("dataset"),
                "run_idx": run_idx,
                "error": err or "",
                "parse_success": parsed.get("parse_success", False),
                "is_duplicate": parsed.get("is_duplicate"),
                "match_type": parsed.get("match_type"),
                "confidence": parsed.get("confidence"),
                "reasoning": parsed.get("reasoning", "")[:2000],
                "raw_response": (text or "")[:4000],
            })
        queue.task_done()


async def run_all(df, model_id, n_runs, temperature, concurrency, out_path, checkpoint_every):
    genai, _ = _import_genai()
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Set GOOGLE_API_KEY (or GEMINI_API_KEY)")
    client = genai.Client(api_key=api_key)

    queue = asyncio.Queue()
    for idx, row in df.iterrows():
        for r in range(n_runs):
            queue.put_nowait((idx, row.to_dict(), r))
    total = queue.qsize()
    print(f"queued {total:,} requests  ({len(df):,} rows × {n_runs} runs)  "
          f"concurrency={concurrency}  model={model_id}  temp={temperature}")

    results, results_lock = [], asyncio.Lock()
    start = time.monotonic()

    async def progress_logger():
        last = 0
        while True:
            await asyncio.sleep(15)
            done = total - queue.qsize()
            rate = done / max(time.monotonic() - start, 1)
            print(f"  progress: {done:,}/{total:,} ({100*done/total:.1f}%)  "
                  f"~{rate:.1f} req/s  in-flight≈{min(concurrency, queue.qsize())}")
            if checkpoint_every and done - last >= checkpoint_every:
                last = done
                async with results_lock:
                    snapshot = list(results)
                pd.DataFrame(snapshot).to_csv(out_path.with_suffix(".partial.csv"),
                                              index=False)

    workers = [asyncio.create_task(worker(f"w{i}", queue, client, model_id,
                                          temperature, results, results_lock))
               for i in range(concurrency)]
    logger_task = asyncio.create_task(progress_logger())

    await queue.join()
    for _ in workers:
        queue.put_nowait(None)
    await asyncio.gather(*workers)
    logger_task.cancel()

    out_df = pd.DataFrame(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"wrote {out_path}  ({len(out_df):,} rows)")
    return out_df


def summarize(runs_df: pd.DataFrame, summary_path: Path):
    """Per-(test_id, corpus_id) agreement stats across runs."""
    rows = []
    for (tid, cid), g in runs_df.groupby(["test_id", "corpus_id"]):
        n = len(g)
        n_ok = int(g["parse_success"].sum())
        valid = g[g["parse_success"]]
        if len(valid) == 0:
            continue
        is_dup_pct = float(valid["is_duplicate"].astype(bool).mean())
        modal_mt = valid["match_type"].value_counts(dropna=False).idxmax()
        modal_mt_frac = float(valid["match_type"].value_counts(dropna=False).iloc[0] / len(valid))
        mt_dist = valid["match_type"].value_counts(normalize=True, dropna=False).to_dict()
        rows.append({
            "test_id": tid, "corpus_id": cid,
            "n_runs": n, "n_ok": n_ok,
            "frac_is_duplicate": is_dup_pct,
            "modal_match_type": modal_mt,
            "modal_match_type_frac": modal_mt_frac,
            "match_type_dist_json": json.dumps(mt_dist, sort_keys=True),
            "mean_confidence": float(valid["confidence"].mean()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(summary_path, index=False)
    print(f"wrote summary: {summary_path}  ({len(out):,} pairs)")
    if not out.empty:
        unanimous_pos = int((out["frac_is_duplicate"] == 1.0).sum())
        unanimous_neg = int((out["frac_is_duplicate"] == 0.0).sum())
        split = len(out) - unanimous_pos - unanimous_neg
        print(f"  unanimous-positive: {unanimous_pos}  "
              f"unanimous-negative: {unanimous_neg}  split: {split}")
        print(f"  mean modal-match-type fraction: {out['modal_match_type_frac'].mean():.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path,
                    default=Path("comparison_analysis/mbpp_all_gemini_positives.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("comparison_analysis/data/self_consistency/"
                                 "mbpp_gemini_flash_runs.csv"))
    ap.add_argument("--summary-out", type=Path, default=None)
    ap.add_argument("--exclude-source", default="mbpp_annotations_old_with_text.csv",
                    help="Drop rows whose source_file matches this (URL-junk file)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap number of input rows (0 = all)")
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--summarize", action="store_true",
                    help="Skip API calls; just recompute metrics from --out")
    args = ap.parse_args()

    summary_path = args.summary_out or args.out.with_name(args.out.stem + "_summary.csv")

    if args.summarize:
        runs = pd.read_csv(args.out, low_memory=False)
        summarize(runs, summary_path)
        return

    df = pd.read_csv(args.input, low_memory=False)
    if args.exclude_source and "source_file" in df.columns:
        before = len(df)
        df = df[df["source_file"] != args.exclude_source].reset_index(drop=True)
        print(f"excluded {before - len(df):,} rows from {args.exclude_source}; {len(df):,} remain")
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
        print(f"limit={args.limit}: using {len(df):,} rows")

    runs_df = asyncio.run(run_all(df, args.model, args.n_runs, args.temperature,
                                  args.concurrency, args.out, args.checkpoint_every))
    summarize(runs_df, summary_path)


if __name__ == "__main__":
    main()
