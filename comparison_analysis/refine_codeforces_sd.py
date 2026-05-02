"""Refine Gemini-annotated codeforces semantic-duplicate candidates with a stricter checker.

Filters the all-non-unrelated CSV to {subset, superset, related} match types and
re-runs each pair through Gemini with CODEFORCES_CHECKER_TEMPLATE, asking it to
judge whether the corpus problem provides significant algorithmic insight for the
test problem (TRUE/FALSE), give a detailed reason, and a 0-1 duplication score.

Usage:
    python refine_codeforces_sd.py
    python refine_codeforces_sd.py --in custom_input.csv --out custom_output.csv --limit 50
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
DEFAULT_IN = ROOT / "human_annotation" / "codeforces_all_non_unrelated.csv"
DEFAULT_OUT = ROOT / "human_annotation" / "codeforces_sd_refined.csv"
TARGET_TYPES = {"subset", "superset", "related"}

MODEL_ID_BY_PROVIDER = {
    "claude": "claude-opus-4-7",
    "gemini": "gemini-3-flash-preview",
}
MODEL_PARAMS = dict(temperature=1.0, max_tokens=4096)
RETRYABLE = ("429", "500", "502", "503", "504",
             "rate_limit", "overloaded", "RESOURCE_EXHAUSTED",
             "UNAVAILABLE", "DEADLINE_EXCEEDED", "rate limit", "quota",
             "Unterminated string", "Expecting value", "Expecting property",
             "JSONDecodeError", "Extra data")

CODEFORCES_CHECKER_TEMPLATE = """You are an exceptionally talented programmer and mathematician with the goal to determine whether there are semantic duplicates between two chunks of texts.

You will be provided with a benchmark problem and a corpus text problem, as well as a reason that they may or may not be semantic duplicates. The rational may be wrong.

Please consider how both problems are solved. Your goal here is to determine whether or not it is the case that the corpus problem provides signifant algorithmic insight required to solve a major part of the test problem, thus making it a semantic duplcate.

Respond with TRUE if it is a semantic duplicate
Provide your reason, in detail, as to why it is or isnt.
Provide a score on the amount of duplication from 0 to 1.

# Test (benchmark) problem:
{test_text}

# Corpus problem:
{corpus_text}

# Prior reasoning (may be wrong):
{prior_reasoning}
"""


class SDChecker(BaseModel):
    is_semantic_duplicate: bool = Field(description="TRUE if the corpus problem provides significant algorithmic insight needed to solve a major part of the test problem.")
    reason: str = Field(description="Detailed reason why it is or isn't a semantic duplicate.")
    score: float = Field(ge=0.0, le=1.0, description="Score on the amount of duplication, 0 (no overlap) to 1 (full duplicate).")


def get_client(provider: str):
    load_dotenv()
    if provider == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY in env or .env")
        import anthropic
        return ("claude", anthropic.Anthropic())
    if provider == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            sys.exit("Set GEMINI_API_KEY in env or .env")
        from google import genai
        return ("gemini", genai.Client(api_key=os.environ["GEMINI_API_KEY"]))
    sys.exit(f"Unknown provider: {provider}")


SD_TOOL = {
    "name": "record_sd_judgment",
    "description": "Record the semantic-duplicate judgment for the pair.",
    "input_schema": {
        "type": "object",
        "properties": {
            "is_semantic_duplicate": {
                "type": "boolean",
                "description": "TRUE if the corpus problem provides significant algorithmic insight needed to solve a major part of the test problem.",
            },
            "reason": {
                "type": "string",
                "description": "Detailed reason why it is or isn't a semantic duplicate.",
            },
            "score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Score on the amount of duplication, 0 (no overlap) to 1 (full duplicate).",
            },
        },
        "required": ["is_semantic_duplicate", "reason", "score"],
    },
}


def _call_claude(client, prompt, model_id, max_retries):
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model_id,
                max_tokens=MODEL_PARAMS["max_tokens"],
                temperature=MODEL_PARAMS["temperature"],
                tools=[SD_TOOL],
                tool_choice={"type": "tool", "name": "record_sd_judgment"},
                messages=[{"role": "user", "content": prompt}],
            )
            tool_use = next((b for b in resp.content if getattr(b, "type", None) == "tool_use"), None)
            if tool_use is None:
                last_err = "no tool_use in response"; continue
            data = tool_use.input
            return {"is_semantic_duplicate": bool(data.get("is_semantic_duplicate")),
                    "score": float(data.get("score", 0.0)),
                    "reason": str(data.get("reason", "")), "error": ""}
        except Exception as e:
            last_err = str(e)
            if any(s in last_err for s in RETRYABLE) and attempt < max_retries - 1:
                time.sleep(min(60.0, 1.5 ** attempt)); continue
            break
    return {"is_semantic_duplicate": None, "score": None, "reason": "", "error": last_err or "unknown"}


def _call_gemini(client, prompt, model_id, max_retries):
    from google.genai import types
    cfg = types.GenerateContentConfig(
        temperature=MODEL_PARAMS["temperature"],
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=SDChecker,
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=model_id, contents=prompt, config=cfg)
            data = json.loads(resp.text)
            return {"is_semantic_duplicate": bool(data.get("is_semantic_duplicate")),
                    "score": float(data.get("score", 0.0)),
                    "reason": str(data.get("reason", "")), "error": ""}
        except Exception as e:
            last_err = str(e)
            if any(s in last_err for s in RETRYABLE) and attempt < max_retries - 1:
                time.sleep(min(60.0, 1.5 ** attempt)); continue
            break
    return {"is_semantic_duplicate": None, "score": None, "reason": "", "error": last_err or "unknown"}


def call_checker(client_pair, test_text: str, corpus_text: str, prior_reasoning: str,
                 model_id: str = "", max_retries: int = 5) -> dict:
    provider, client = client_pair
    prompt = CODEFORCES_CHECKER_TEMPLATE.format(
        test_text=test_text or "",
        corpus_text=corpus_text or "",
        prior_reasoning=prior_reasoning or "(none provided)",
    )
    if provider == "claude":
        return _call_claude(client, prompt, model_id, max_retries)
    return _call_gemini(client, prompt, model_id, max_retries)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN), help="Input CSV (codeforces non-unrelated)")
    ap.add_argument("--out", dest="out", default=str(DEFAULT_OUT), help="Output CSV path")
    ap.add_argument("--types", default=",".join(sorted(TARGET_TYPES)),
                    help="Comma-separated match_types to refine (default: subset,superset,related)")
    ap.add_argument("--type-col", default="match_type", help="Column name holding the duplicate category")
    ap.add_argument("--reasoning-col", default="reasoning", help="Column name holding prior reasoning")
    ap.add_argument("--is-sd-col", default="is_sd", help="Column name holding prior is_sd flag")
    ap.add_argument("--confidence-col", default="confidence", help="Column name holding prior confidence")
    ap.add_argument("--limit", type=int, default=None, help="Limit rows (for testing)")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent API calls")
    ap.add_argument("--provider", choices=["claude", "gemini"], default="claude")
    ap.add_argument("--model", default=None, help="Override model id (else picks default for provider)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip rows whose (test_id, corpus_id) already appear in --out")
    args = ap.parse_args()

    target_types = {t.strip().lower() for t in args.types.split(",") if t.strip()}

    import pandas as pd
    df = pd.read_csv(args.inp)
    df = df[df[args.type_col].astype(str).str.lower().isin(target_types)].copy()
    # Normalize column names so downstream code is uniform
    df = df.rename(columns={
        args.type_col: "match_type",
        args.reasoning_col: "reasoning",
        args.is_sd_col: "is_sd",
        args.confidence_col: "confidence",
    })
    if args.limit:
        df = df.head(args.limit)
    print(f"loaded {len(df)} rows of types {sorted(target_types)} from {args.inp}")

    done_keys = set()
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        prev = pd.read_csv(out_path)
        done_keys = set(zip(prev["test_id"].astype(str), prev["corpus_id"].astype(str)))
        print(f"resume: {len(done_keys)} rows already in {out_path}, will skip them")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    client_pair = get_client(args.provider)
    model_id = args.model or MODEL_ID_BY_PROVIDER[args.provider]
    print(f"provider={args.provider}, model={model_id}")

    fields = ["test_id", "corpus_id", "match_type", "prior_is_sd", "prior_confidence",
              "prior_reasoning", "checker_is_sd", "checker_score", "checker_reason",
              "checker_error", "test_text", "corpus_text"]
    write_header = not (args.resume and out_path.exists())
    fout = open(out_path, "a" if args.resume else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fout, fieldnames=fields)
    if write_header:
        writer.writeheader()

    todo = []
    for _, row in df.iterrows():
        key = (str(row["test_id"]), str(row["corpus_id"]))
        if key in done_keys:
            continue
        todo.append(row)
    print(f"to process: {len(todo)} (workers={args.workers})")

    n_done = 0
    n_err = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(call_checker, client_pair, str(r.get("test_text", "")),
                      str(r.get("corpus_text", "")), str(r.get("reasoning", "")),
                      model_id): r
            for r in todo
        }
        for fut in as_completed(futures):
            r = futures[fut]
            res = fut.result()
            writer.writerow({
                "test_id": r["test_id"], "corpus_id": r["corpus_id"],
                "match_type": r["match_type"],
                "prior_is_sd": r.get("is_sd", ""),
                "prior_confidence": r.get("confidence", ""),
                "prior_reasoning": r.get("reasoning", ""),
                "checker_is_sd": res["is_semantic_duplicate"],
                "checker_score": res["score"],
                "checker_reason": res["reason"],
                "checker_error": res["error"],
                "test_text": r.get("test_text", ""),
                "corpus_text": r.get("corpus_text", ""),
            })
            fout.flush()
            n_done += 1
            if res["error"]:
                n_err += 1
            if n_done % 25 == 0 or n_done == len(todo):
                rate = n_done / max(1e-6, time.time() - t0)
                print(f"  {n_done}/{len(todo)} done ({n_err} errors, {rate:.2f}/s)")

    fout.close()
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
