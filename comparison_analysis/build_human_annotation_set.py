"""Build human-annotation CSVs: 80 MBPP + 80 codeforces non-unrelated samples."""
import json, os, re, glob, random
import pandas as pd

random.seed(42)
ROOT = os.path.dirname(os.path.abspath(__file__))
NONUN = {"exact", "equivalent", "subset", "superset", "related"}


def balanced_sample(rows, n, types):
    """Sample n rows balanced across `types`. If a type has fewer than its quota,
    redistribute the shortfall to the remaining types (largest-remainder style)."""
    by_type = {t: [r for r in rows if str(r["llm_match_type"]).lower() == t] for t in types}
    for t in by_type.values():
        random.shuffle(t)

    remaining_types = list(types)
    picked = []
    while remaining_types and len(picked) < n:
        slots = n - len(picked)
        base = slots // len(remaining_types)
        extra = slots % len(remaining_types)
        next_types = []
        for i, t in enumerate(remaining_types):
            quota = base + (1 if i < extra else 0)
            take = min(quota, len(by_type[t]))
            picked.extend(by_type[t][:take])
            by_type[t] = by_type[t][take:]
            if by_type[t] and take == quota:
                next_types.append(t)
        if next_types == remaining_types:
            break
        remaining_types = next_types
    return picked[:n]


def extract_texts_from_prompt(prompt: str):
    """Pull Test Task/Problem and Corpus Task/Problem sections out of the LLM prompt."""
    m = re.search(r"## Test (?:Task|Problem)[^\n]*:\s*\n(.*?)\n## Corpus (?:Task|Problem)[^\n]*:\s*\n(.*?)\n## ", prompt, re.DOTALL)
    if not m:
        return "", ""
    return m.group(1).strip(), m.group(2).strip()


def build_mbpp(n=80):
    df = pd.read_csv(os.path.join(ROOT, "training_data", "mbpp_annotations_old_with_text.csv"))
    df["match_type_l"] = df["match_type"].astype(str).str.lower()
    df = df[df["match_type_l"].isin(NONUN)].copy()
    rows = []
    for _, r in df.iterrows():
        tt = r.get("test_text", "") or ""
        ct = r.get("corpus_text", "") or ""
        rows.append({
            "key": r["key"],
            "benchmark": "mbpp",
            "dataset": r["dataset"],
            "test_id": r["test_id"],
            "corpus_id": r["corpus_id"],
            "test_text": tt,
            "corpus_text": ct,
            "llm_match_type": r["match_type"],
            "llm_is_sd": r["is_sd"],
            "llm_confidence": r["confidence"],
            "llm_reasoning": r["reasoning"],
            "human_is_sd": "",
            "human_match_type": "",
            "human_notes": "",
        })
    have_text = [r for r in rows if r["test_text"] and r["corpus_text"]]
    print(f"mbpp non-unrelated: {len(rows)} (with text: {len(have_text)})")
    return balanced_sample(have_text, n, ["exact", "equivalent", "subset", "superset"])


def build_codeforces(n=80):
    d = os.path.join(ROOT, "annotations", "codeforces")
    files = [f for f in os.listdir(d) if f.endswith(".json") and not f.startswith("_")]
    candidates = []
    for f in files:
        try:
            with open(os.path.join(d, f), encoding="utf-8") as fp:
                j = json.load(fp)
        except Exception:
            continue
        ann = j.get("annotation") or {}
        mt = str(ann.get("match_type", "")).lower()
        if mt not in NONUN:
            continue
        tt = j.get("test_text", "") or ""
        ct = j.get("corpus_text", "") or ""
        if not ct.strip():
            continue
        candidates.append({
            "key": f.replace(".json", ""),
            "benchmark": "codeforces",
            "dataset": j.get("dataset", ""),
            "test_id": j.get("test_id", ""),
            "corpus_id": j.get("corpus_id", ""),
            "test_text": tt,
            "corpus_text": ct,
            "llm_match_type": ann.get("match_type", ""),
            "llm_is_sd": ann.get("is_sd", ""),
            "llm_confidence": ann.get("confidence", ""),
            "llm_reasoning": ann.get("reasoning", ""),
            "human_is_sd": "",
            "human_match_type": "",
            "human_notes": "",
        })
    print(f"codeforces non-unrelated with corpus text: {len(candidates)}")
    return balanced_sample(candidates, n, ["exact", "equivalent", "subset", "superset", "related"])


def main():
    out_dir = os.path.join(ROOT, "human_annotation")
    os.makedirs(out_dir, exist_ok=True)

    mbpp = build_mbpp(80)
    cf = build_codeforces(80)

    cols = ["key", "benchmark", "dataset", "test_id", "corpus_id",
            "test_text", "corpus_text",
            "llm_match_type", "llm_is_sd", "llm_confidence", "llm_reasoning",
            "human_is_sd", "human_match_type", "human_notes"]

    pd.DataFrame(mbpp, columns=cols).to_csv(os.path.join(out_dir, "mbpp_human_annotation_80.csv"), index=False)
    pd.DataFrame(cf, columns=cols).to_csv(os.path.join(out_dir, "codeforces_human_annotation_80.csv"), index=False)
    pd.DataFrame(mbpp + cf, columns=cols).to_csv(os.path.join(out_dir, "combined_human_annotation_160.csv"), index=False)
    print(f"wrote to {out_dir}")


if __name__ == "__main__":
    main()
