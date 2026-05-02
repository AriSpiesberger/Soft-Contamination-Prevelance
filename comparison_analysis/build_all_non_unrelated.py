"""Pull every non-unrelated Gemini annotation across all training_data CSVs into one file."""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
TD = os.path.join(ROOT, "training_data")
SOURCES = [
    ("mbpp", os.path.join(TD, "mbpp_annotations_full(1).csv")),
    ("mbpp", os.path.join(TD, "mbpp_annotations_old_with_text.csv")),
    ("codeforces", os.path.join(TD, "codeforces_annotations.csv")),
]
NONUN = {"exact", "equivalent", "subset", "superset", "related"}

KEEP = ["benchmark", "dataset", "test_id", "corpus_id",
        "test_text", "corpus_text", "match_type", "is_sd",
        "confidence", "reasoning", "_source"]


def main():
    frames = []
    for benchmark, path in SOURCES:
        df = pd.read_csv(path)
        df["_mt"] = df["match_type"].astype(str).str.lower().str.strip()
        df = df[df["_mt"].isin(NONUN)].copy()
        if "benchmark" not in df.columns:
            df["benchmark"] = benchmark
        if "dataset" not in df.columns:
            df["dataset"] = ""
        df["_source"] = os.path.basename(path)
        for col in ("test_text", "corpus_text"):
            df[col] = df[col].fillna("").astype(str)
        df = df[(df["test_text"].str.strip() != "") & (df["corpus_text"].str.strip() != "")]
        frames.append(df[[c for c in KEEP if c in df.columns]])
        print(f"{os.path.basename(path):60s} -> {len(df)} non-unrelated rows")

    big = pd.concat(frames, ignore_index=True)
    print(f"\nbefore dedup: {len(big)}")
    big = big.drop_duplicates(subset=["benchmark", "test_id", "corpus_id"], keep="first")
    print(f"after  dedup (benchmark, test_id, corpus_id): {len(big)}")

    print("\nbreakdown:")
    print(big.groupby(["benchmark", "match_type"]).size().unstack(fill_value=0))

    out = os.path.join(ROOT, "human_annotation", "all_non_unrelated_annotations.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    big.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
