"""Build a single refined codeforces CSV ready for the plot pipeline.

Final SD = (gpt-oss exact OR equivalent OR subset, kept as-is)
       UNION (gpt-oss superset/related where Gemini-flash refinement says TRUE)
       UNION (Gemini-pool refinement TRUE for any pair the gpt-oss pass missed)

The output mirrors the input schema but `predicted_is_duplicate` is the final
verdict. `predicted_category` is dropped so generate_all_plots.py skips its
strict filter and trusts our flag directly.
"""

import os
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "data" / "codeforces_top100" / "codeforces_top100_classified_gptoss_v2.csv"
OUT = ROOT / "data" / "codeforces_top100" / "codeforces_top100_classified_refined.csv"
GPT_REFINED = ROOT / "human_annotation" / "gptoss_baseline_refined.csv"
GEM_REFINED = ROOT / "human_annotation" / "codeforces_sd_refined.csv"

KEEP_AS_IS = {"exact", "equivalent", "subset"}
REFINE = {"superset", "related"}


def load_verdicts(path):
    if not path.exists():
        return {}
    r = pd.read_csv(path).dropna(subset=["checker_is_sd"])
    r["test_id"] = r["test_id"].astype(str)
    r["corpus_id"] = r["corpus_id"].astype(str)
    return dict(zip(zip(r["test_id"], r["corpus_id"]), r["checker_is_sd"].astype(bool)))


def main():
    df = pd.read_csv(SRC, low_memory=False)
    df["test_id"] = df["test_id"].astype(str)
    df["corpus_id"] = df["corpus_id"].astype(str)

    vg = load_verdicts(GPT_REFINED)
    ve = load_verdicts(GEM_REFINED)

    cat = df["predicted_category"].astype(str).str.lower()
    keep_mask = cat.isin(KEEP_AS_IS)
    refine_mask = cat.isin(REFINE)

    keys = list(zip(df["test_id"], df["corpus_id"]))
    g_lookup = pd.Series([vg.get(k) for k in keys], index=df.index)
    e_lookup = pd.Series([ve.get(k) for k in keys], index=df.index)

    new_dup = pd.Series(False, index=df.index)
    # Categories we keep as-is: trust gpt-oss verdict
    new_dup[keep_mask] = df.loc[keep_mask, "predicted_is_duplicate"].astype(bool).values
    # Refine categories: use gpt-oss-baseline pass first, then Gemini-pool fallback
    refined_verdict = g_lookup.where(g_lookup.notna(), e_lookup)
    refined_verdict = refined_verdict.where(refined_verdict.notna(), False).astype(bool)
    new_dup[refine_mask] = refined_verdict[refine_mask].values
    # Anything else (unrelated etc.) stays False
    df["predicted_is_duplicate"] = new_dup

    # Keep predicted_category so generate_all_plots.py can still distinguish
    # "Semantic Duplicates" (excludes exact) from "Including Exact" overlay lines.

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"input rows:           {len(df):,}")
    print(f"final SDs (refined):  {int(df['predicted_is_duplicate'].sum()):,}")
    print(f"  from kept categories (exact+equiv+subset): {int(keep_mask.sum() & df.loc[keep_mask,'predicted_is_duplicate'].astype(bool).sum())}")
    print(f"  from refined superset/related TRUE:        {int(refined_verdict[refine_mask].sum())}")
    print(f"wrote: {OUT}")


if __name__ == "__main__":
    main()
