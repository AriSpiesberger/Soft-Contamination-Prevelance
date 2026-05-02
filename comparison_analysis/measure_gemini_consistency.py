"""Measure Gemini's self-consistency on codeforces SD classification.

Strategy:
  1. Stratified sample = argmin |category| examples per category (6 categories).
  2. Re-run each sampled (test_text, corpus_text) through Gemini --runs times
     using the original CODEFORCES_PROMPT_TEMPLATE.
  3. Save every run, then compute per-row mode label, agreement rate, and
     overall pairwise consistency / Fleiss-style agreement.

Usage:
    python measure_gemini_consistency.py --runs 6
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
DEFAULT_IN = ROOT / "training_data" / "codeforces_annotations.csv"
DEFAULT_OUT = ROOT / "human_annotation" / "gemini_consistency_runs.csv"
SAMPLE_SEED = 42

CATEGORIES = ["exact", "equivalent", "subset", "superset", "related", "unrelated"]
MODEL_ID = "gemini-3-flash-preview"
RETRYABLE = ("429", "500", "502", "503", "504",
             "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED",
             "rate limit", "quota", "overloaded",
             "Unterminated string", "Expecting value", "JSONDecodeError")

CODEFORCES_PROMPT_TEMPLATE = """You are an expert competitive programmer analyzing potential semantic duplicates between programming problems.

## Task
Determine if the following two competitive programming problems are semantically related - meaning exposure to the corpus problem during training could help solve the test problem.

## Test Problem (from benchmark):
{test_text}

## Corpus Problem (from training data):
{corpus_text}

## Analysis Steps:
1. **Check data quality first**: Is the corpus text a complete problem statement? If it's empty, fragmentary, or contains only code without a problem description, mark as "unrelated".
2. **Check for exact text match**: If the corpus text appears VERBATIM (word-for-word) within the test text (e.g., corpus contains just the problem statement while test contains problem + examples), this counts as "exact" match.
3. **Extract the core problem**: Strip away story/narrative framing. What is the actual computational task?
4. **Identify the key insight**: What algorithmic technique or observation is needed?
5. **Compare**: Is there meaningful overlap in what's being asked or how to solve it?

## Match Types:
- "exact": Nearly identical problem statements, OR corpus text is a verbatim substring/subsection of test text (exact text match even if corpus is shorter)
- "equivalent": Different framing but identical algorithmic core
- "subset": Test is a special case of corpus (test asks for less than corpus)
- "superset": Corpus asks for something simpler than test, but NOT a verbatim text match
- "related": Corpus covers a component or shares key insight with test
- "unrelated": Different problems, or corpus data is unusable
"""


class SDAnnotation(BaseModel):
    is_sd: bool = Field(description="Is this a semantic duplicate?")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    match_type: str = Field(description="One of: exact, equivalent, subset, superset, related, unrelated")


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
    prompt = CODEFORCES_PROMPT_TEMPLATE.format(test_text=test_text or "", corpus_text=corpus_text or "")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    ap.add_argument("--out", dest="out", default=str(DEFAULT_OUT))
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--per-cat", type=int, default=None,
                    help="Override sample size per category (default = argmin)")
    ap.add_argument("--append", action="store_true",
                    help="Append new runs to existing --out, shifting run_idx and reusing the existing pair_id->row mapping")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    df["match_type_l"] = df["match_type"].astype(str).str.lower().str.strip()
    df = df[df["match_type_l"].isin(CATEGORIES)]
    df = df[(df["test_text"].astype(str).str.strip() != "") &
            (df["corpus_text"].astype(str).str.strip() != "")]

    counts = {c: int((df["match_type_l"] == c).sum()) for c in CATEGORIES}
    n_per = args.per_cat if args.per_cat else min(counts.values())
    print(f"category counts: {counts}")
    print(f"sampling {n_per} per category -> {n_per * len(CATEGORIES)} unique pairs")

    out_path = Path(args.out)
    run_offset = 0
    if args.append and out_path.exists():
        prev = pd.read_csv(out_path)
        run_offset = int(prev["run_idx"].max()) + 1
        # Re-derive sample from the existing CSV's (test_id, corpus_id, original) so order matches
        seen = prev.drop_duplicates("pair_id").sort_values("pair_id")[
            ["pair_id", "test_id", "corpus_id", "original_match_type"]
        ]
        keys = set(zip(seen["test_id"].astype(str), seen["corpus_id"].astype(str)))
        df["test_id_s"] = df["test_id"].astype(str)
        df["corpus_id_s"] = df["corpus_id"].astype(str)
        sub = df[df.apply(lambda r: (r["test_id_s"], r["corpus_id_s"]) in keys, axis=1)].copy()
        ord_map = {(t, c): i for i, (t, c) in enumerate(zip(seen["test_id"].astype(str),
                                                             seen["corpus_id"].astype(str)))}
        sub["__order"] = sub.apply(lambda r: ord_map[(r["test_id_s"], r["corpus_id_s"])], axis=1)
        sample = sub.sort_values("__order").drop(columns=["__order", "test_id_s", "corpus_id_s"]).reset_index(drop=True)
        print(f"append mode: reusing {len(sample)} pairs from {out_path}, "
              f"new runs will start at run_idx={run_offset}")
    else:
        parts = []
        for cat in CATEGORIES:
            sub = df[df["match_type_l"] == cat]
            parts.append(sub.sample(n=min(n_per, len(sub)), random_state=SAMPLE_SEED))
        sample = pd.concat(parts, ignore_index=True)
        print(f"sample rows: {len(sample)}")
        print(f"sample columns: {list(sample.columns)[:8]}...")
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
