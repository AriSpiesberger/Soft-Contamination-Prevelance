"""Refine mbpp_sample100_classified.csv by demoting hallucinated/garbage matches.

A flagged duplicate is demoted (predicted_is_duplicate -> False) when the
corpus_text fails any of these checks:

  url:         contains a URL / link / known web-paste site marker
  too_short:   < --min-words non-URL whitespace tokens
  no_text:     < --min-alpha-chars alphabetic chars after URL stripping
  pure_code:   English-stopword fraction < --min-english-frac AND
               has code markers (def/class/import/return/lambda/...). The
               classifier hallucinates "task duplicates" when it only sees
               code; without natural-language describing the task, a
               semantic-duplicate claim is unsupported.

Original columns are preserved; a `refine_reason` column is added (empty for
rows that pass, otherwise the rule name). Output written next to input as
<stem>_refined.csv unless --out is given.

Usage:
    python comparison_analysis/refine_mbpp_sample100.py
    python comparison_analysis/refine_mbpp_sample100.py --min-english-frac 0.15
"""

import argparse
import re
from pathlib import Path

import pandas as pd

# URL / link / web-paste site markers
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WEB_MARKERS_RE = re.compile(
    r"https?://|www\.\w|<a\s|<img\s|\]\([a-zA-Z]+://|"
    r"\bpastebin\b|\bgist\.github\b|\bgithub\.com\b|\bcodility\b|"
    r"\bleetcode\b|\bhackerrank\b|\bcodeforces\b|\bgeeksforgeeks\b|"
    r"\bstackoverflow\b|\bstackexchange\b|\breddit\b|\bdiscord\b|"
    r"\bmedium\.com\b|\bquora\b|\bbit\.ly\b|\btinyurl\b|\bgoo\.gl\b|"
    r"\.ipynb\b|\.html\b|\.aspx\b|\.php\b",
    re.IGNORECASE,
)
ALPHA_RE = re.compile(r"[A-Za-z]")
ALPHA_TOKEN_RE = re.compile(r"[A-Za-z]+")
CODE_MARKERS_RE = re.compile(
    r"\b(def|class|import|from|return|self|print|lambda|None|True|False|"
    r"public|private|static|void|int|float|string|System\.out|console\.log)\b|"
    r"=>\s*[\w(]|=\s*\[|=\s*\{|;\s*$|->\s*\w+\s*:",
    re.MULTILINE,
)

# Common English stopwords as a "natural language" signal
STOPWORDS = frozenset(
    "the a an is are was were be been being to of in on for and or but if not "
    "that this these those it its as by with from at into about which what when "
    "where how then so do does did has have had can could should would will may "
    "might".split()
)


def english_stopword_frac(text: str) -> float:
    toks = ALPHA_TOKEN_RE.findall(text.lower())
    if not toks:
        return 0.0
    return sum(1 for w in toks if w in STOPWORDS) / len(toks)


def reason_for(text, *, min_words, min_alpha, min_english_frac):
    if not isinstance(text, str) or not text.strip():
        return "no_text"
    no_url = URL_RE.sub("", text).strip()
    if WEB_MARKERS_RE.search(text):
        return "url"
    if len(ALPHA_RE.findall(no_url)) < min_alpha:
        return "no_text"
    if len(no_url.split()) < min_words:
        return "too_short"
    if english_stopword_frac(no_url) < min_english_frac and CODE_MARKERS_RE.search(no_url):
        return "pure_code"
    return ""


def refine(df, *, min_words, min_alpha, min_english_frac):
    out = df.copy()
    out["corpus_text"] = out["corpus_text"].fillna("").astype(str)
    out["refine_reason"] = out["corpus_text"].apply(
        lambda s: reason_for(s, min_words=min_words, min_alpha=min_alpha,
                             min_english_frac=min_english_frac)
    )
    bad = out["refine_reason"].astype(bool)
    is_dup = out["predicted_is_duplicate"].astype(bool)
    demoted = bad & is_dup
    out.loc[demoted, "predicted_is_duplicate"] = False
    return out, demoted


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path,
                    default=Path("comparison_analysis/data/mbpp_sample100/mbpp_sample100_classified.csv"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-words", type=int, default=20,
                    help="Minimum non-URL whitespace tokens in corpus_text")
    ap.add_argument("--min-alpha-chars", type=int, default=20,
                    help="Minimum alphabetic chars in corpus_text (after URL stripping)")
    ap.add_argument("--min-english-frac", type=float, default=0.10,
                    help="Minimum stopword fraction; below this + code markers => pure_code")
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    sem = {"equivalent", "subset", "superset"}

    n_dup_before = int(df["predicted_is_duplicate"].sum())
    n_sem_before = int(((df["predicted_is_duplicate"] == True)
                        & df["predicted_match_type"].isin(sem)).sum())

    refined, demoted = refine(df, min_words=args.min_words,
                              min_alpha=args.min_alpha_chars,
                              min_english_frac=args.min_english_frac)
    by_reason = refined.loc[demoted, "refine_reason"].value_counts().to_dict()

    n_dup_after = int(refined["predicted_is_duplicate"].sum())
    n_sem_after = int(((refined["predicted_is_duplicate"] == True)
                       & refined["predicted_match_type"].isin(sem)).sum())

    out_path = args.out or args.input.with_name(args.input.stem + "_refined.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    refined.to_csv(out_path, index=False)

    print(f"input: {args.input}  ({len(df):,} rows)")
    print(f"thresholds: min_words={args.min_words}  min_alpha={args.min_alpha_chars}  "
          f"min_english_frac={args.min_english_frac}")
    print(f"raw duplicates:      {n_dup_before:,} -> {n_dup_after:,}  ({n_dup_before - n_dup_after:,} demoted)")
    print(f"semantic duplicates: {n_sem_before:,} -> {n_sem_after:,}")
    print(f"demotion reasons (duplicates only): {by_reason}")
    print(f"wrote refined CSV: {out_path}")


if __name__ == "__main__":
    main()
