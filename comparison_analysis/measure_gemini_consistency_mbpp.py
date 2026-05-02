"""Measure Gemini's self-consistency on MBPP SD classification.

Mirrors measure_gemini_consistency.py (codeforces) but reads the MBPP Gemini
baseline (mbpp_annotations_full(1).csv) and uses the MBPP prompt template &
5-category schema (no 'related').

Sampling: --per-cat unique rows per category. If a category has fewer unique
rows than --per-cat, the available rows are replicated cyclically to fill the
quota (so every category contributes the same number of slots in the runs CSV).

Usage:
    python measure_gemini_consistency_mbpp.py --per-cat 20 --runs 5 --workers 8
    python measure_gemini_consistency_mbpp.py --append --runs 5    # extend run_idx
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
DEFAULT_IN = ROOT / "training_data" / "mbpp_annotations_full(1).csv"
DEFAULT_OUT = ROOT / "human_annotation" / "gemini_consistency_runs_mbpp.csv"
SAMPLE_SEED = 42

CATEGORIES = ["exact", "equivalent", "subset", "superset", "unrelated"]
MODEL_ID = "gemini-3-flash-preview"
RETRYABLE = ("429", "500", "502", "503", "504",
             "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED",
             "rate limit", "quota", "overloaded",
             "Unterminated string", "Expecting value", "JSONDecodeError")

# NOTE: this is the EXACT prompt used by annotate_batch.py to label
# mbpp_annotations_full(1).csv — keep it in sync. Older variants in
# shared_utilities.py have subset/superset swapped, which would invalidate
# the self-consistency comparison.
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


class SDAnnotation(BaseModel):
    is_sd: bool = Field(description="Is this a semantic duplicate?")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    match_type: str = Field(description="One of: exact, equivalent, subset, superset, unrelated")


def get_client():
    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("Set GEMINI_API_KEY in env or .env")
    from google import genai
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def call_once(client, test_text, corpus_text, run_idx, model_id, max_retries=5):
    from google.genai import types
    cfg = types.GenerateContentConfig(
        temperature=1.0,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=SDAnnotation,
    )
    prompt = MBPP_PROMPT_TEMPLATE.format(test_text=test_text or "", corpus_text=corpus_text or "")
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=model_id, contents=prompt, config=cfg)
            data = json.loads(resp.text)
            return {
                "run_idx": run_idx,
                "match_type": str(data.get("match_type", "")).lower().strip(),
                "is_sd": bool(data.get("is_sd")),
                "confidence": float(data.get("confidence", 0.0)),
                "reasoning": str(data.get("reasoning", "")),
                "error": "",
            }
        except Exception as e:
            last_err = str(e)
            if any(s in last_err for s in RETRYABLE) and attempt < max_retries - 1:
                time.sleep(min(60.0, 1.5 ** attempt))
                continue
            break
    return {"run_idx": run_idx, "match_type": "", "is_sd": None, "confidence": None,
            "reasoning": "", "error": last_err or "unknown"}


def stratified_sample_with_replication(df, per_cat, seed):
    """Sample per_cat rows per category; replicate cyclically if fewer unique exist."""
    parts = []
    for cat in CATEGORIES:
        sub = df[df["match_type_l"] == cat]
        if len(sub) == 0:
            print(f"  WARNING: 0 rows for category {cat!r}; skipping")
            continue
        if len(sub) >= per_cat:
            picked = sub.sample(n=per_cat, random_state=seed)
        else:
            # replicate cyclically: e.g., 1 row -> repeat 20 times
            shuffled = sub.sample(frac=1, random_state=seed).reset_index(drop=True)
            reps = (per_cat + len(shuffled) - 1) // len(shuffled)
            replicated = pd.concat([shuffled] * reps, ignore_index=True).head(per_cat)
            print(f"  category {cat!r}: only {len(sub)} unique row(s); replicating to {per_cat}")
            picked = replicated
        parts.append(picked)
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    ap.add_argument("--out", dest="out", default=str(DEFAULT_OUT))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--per-cat", type=int, default=20,
                    help="Sample slots per category (replicates if category has fewer unique rows)")
    ap.add_argument("--append", action="store_true",
                    help="Append new runs to existing --out, shifting run_idx and reusing the existing pair_id->row mapping")
    args = ap.parse_args()

    df = pd.read_csv(args.inp, low_memory=False)
    df["match_type_l"] = df["match_type"].astype(str).str.lower().str.strip()
    df = df[df["match_type_l"].isin(CATEGORIES)]
    df = df[(df["test_text"].astype(str).str.strip() != "") &
            (df["corpus_text"].astype(str).str.strip() != "")]

    counts = {c: int((df["match_type_l"] == c).sum()) for c in CATEGORIES}
    print(f"category counts: {counts}")
    print(f"sampling {args.per_cat} per category (with replication where needed) "
          f"-> {args.per_cat * len(CATEGORIES)} sample slots")

    out_path = Path(args.out)
    run_offset = 0
    if args.append and out_path.exists():
        prev = pd.read_csv(out_path)
        run_offset = int(prev["run_idx"].max()) + 1
        # Reconstruct the sample by pair_id order from the existing file
        seen = prev.drop_duplicates("pair_id").sort_values("pair_id")[
            ["pair_id", "test_id", "corpus_id", "original_match_type"]
        ]
        # Build a lookup of (test_id, corpus_id) -> first matching df row
        df["test_id_s"] = df["test_id"].astype(str)
        df["corpus_id_s"] = df["corpus_id"].astype(str)
        lookup = df.drop_duplicates(["test_id_s", "corpus_id_s"]).set_index(
            ["test_id_s", "corpus_id_s"])
        rows = []
        for _, sr in seen.iterrows():
            key = (str(sr["test_id"]), str(sr["corpus_id"]))
            if key in lookup.index:
                rows.append(lookup.loc[key].to_dict() | {"match_type_l": sr["original_match_type"]})
        sample = pd.DataFrame(rows).reset_index(drop=True)
        print(f"append mode: reusing {len(sample)} pairs from {out_path}, "
              f"new runs will start at run_idx={run_offset}")
    else:
        sample = stratified_sample_with_replication(df, args.per_cat, SAMPLE_SEED)
        print(f"sample rows: {len(sample)}")

    print(f"runs/row: {args.runs}  -> total API calls: {len(sample) * args.runs}")

    client = get_client()
    fields = ["pair_id", "test_id", "corpus_id", "original_match_type",
              "run_idx", "match_type", "is_sd", "confidence", "reasoning", "error"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (args.append and out_path.exists())
    fout = open(out_path, "a" if args.append else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fout, fieldnames=fields)
    if write_header:
        writer.writeheader()

    tasks = []
    for i, (_, row) in enumerate(sample.reset_index(drop=True).iterrows()):
        for r in range(args.runs):
            tasks.append((i, row, r + run_offset))
    print(f"queued {len(tasks)} tasks")

    n_done = 0
    n_err = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(call_once, client, str(t[1]["test_text"]), str(t[1]["corpus_text"]),
                      t[2], args.model): t
            for t in tasks
        }
        for fut in as_completed(futs):
            pair_id, row, _ = futs[fut]
            res = fut.result()
            writer.writerow({
                "pair_id": pair_id,
                "test_id": row["test_id"],
                "corpus_id": row["corpus_id"],
                "original_match_type": row["match_type_l"],
                **res,
            })
            fout.flush()
            n_done += 1
            if res["error"]:
                n_err += 1
            if n_done % 50 == 0 or n_done == len(tasks):
                rate = n_done / max(1e-6, time.time() - t0)
                print(f"  {n_done}/{len(tasks)} ({n_err} errors, {rate:.2f}/s)")
    fout.close()
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
