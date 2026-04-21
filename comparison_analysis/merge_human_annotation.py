"""Merge new 160-row human annotation CSV onto existing xlsx, dedup on (benchmark, test_id, corpus_id)."""
import pandas as pd
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
XLSX_IN = r"C:\Users\arisp\Downloads\sd_samples_20_per_benchmark (1).xlsx"
CSV_IN = os.path.join(ROOT, "human_annotation", "combined_human_annotation_160.csv")
OUT = os.path.join(ROOT, "human_annotation", "sd_samples_merged.xlsx")

existing = pd.read_excel(XLSX_IN)
new = pd.read_csv(CSV_IN)

# Align new-row schema to existing xlsx.
new_aligned = pd.DataFrame({
    "benchmark":   new["benchmark"],
    "test_id":     new["test_id"],
    "corpus_id":   new["corpus_id"],
    "test_text":   new["test_text"],
    "corpus_text": new["corpus_text"],
    "similarity":  pd.NA,
    "is_sd":       new["llm_is_sd"],
    "confidence":  new["llm_confidence"],
    "match_type":  new["llm_match_type"],
    "reasoning":   new["llm_reasoning"],
    "Human":       "",
})

existing["test_id"] = existing["test_id"].astype(str)
existing["corpus_id"] = existing["corpus_id"].astype(str)
new_aligned["test_id"] = new_aligned["test_id"].astype(str)
new_aligned["corpus_id"] = new_aligned["corpus_id"].astype(str)

key_cols = ["benchmark", "test_id", "corpus_id"]
existing_keys = set(map(tuple, existing[key_cols].values.tolist()))
mask_new = ~new_aligned[key_cols].apply(tuple, axis=1).isin(existing_keys)
to_add = new_aligned[mask_new]
dupes_skipped = len(new_aligned) - len(to_add)

merged = pd.concat([existing, to_add], ignore_index=True)
merged = merged.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)

merged.to_excel(OUT, index=False)

print(f"existing rows: {len(existing)}")
print(f"new rows:      {len(new_aligned)} ({dupes_skipped} overlapped with existing)")
print(f"added:         {len(to_add)}")
print(f"final rows:    {len(merged)}")
print(f"by benchmark:  {merged['benchmark'].value_counts().to_dict()}")
print(f"wrote:         {OUT}")
