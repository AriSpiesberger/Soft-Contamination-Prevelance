"""
Reformat mbpp_train_filtered.csv into an eval-aligned conversational JSONL.

Eval prompt (mbpp_split, mbpp_instruct_fixed):
    user:      "You are an expert Python programmer, and here is your task:
                {text}
                Your code should pass these tests:
                {test_list[0]}
                {test_list[1]}
                {test_list[2]}"
    assistant: "```python\n{code}\n```"

Training the LoRA on bare code (the old format) drives the model away from
the fenced ``` ```python ``` ... ``` ``` ``` format the eval extractor needs,
which is why later checkpoints crashed pass@1 (38% at ckpt-16 vs 59% baseline).

Drops fewshot ids (2, 3, 4) so they're never trained on; they're already
demonstrated at eval time as the 3-shot examples.
"""
import csv
import json
from pathlib import Path
from datasets import load_dataset

PWD = Path(__file__).parent.parent
SRC = PWD / "mbpp_data" / "mbpp_train_filtered.csv"
DST = PWD / "mbpp_data" / "mbpp_train_aligned.jsonl"

FEWSHOT_IDS = {2, 3, 4}


def load_mbpp_lookups():
    """task_id -> {text, code, test_list}, across all HF MBPP splits."""
    lookups = {}
    for split in ("test", "train", "validation", "prompt"):
        ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
        for item in ds:
            tid = int(item["task_id"])
            if tid not in lookups:
                lookups[tid] = {
                    "text": item["text"],
                    "code": item["code"],
                    "test_list": list(item["test_list"]),
                }
    return lookups


def build_user(paraphrased_text: str, test_list: list[str]) -> str:
    # Same structure as finetuning/mbpp/lm_eval_tasks/mbpp_split.yaml's doc_to_text.
    tests = "\n".join(test_list[:3])
    return (
        f"You are an expert Python programmer, and here is your task:\n"
        f"{paraphrased_text}\n"
        f"Your code should pass these tests:\n{tests}"
    )


def build_assistant(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n").strip()
    return f"```python\n{code}\n```"


def main():
    lookups = load_mbpp_lookups()
    print(f"loaded MBPP lookups for {len(lookups)} task_ids")

    rows = []
    skipped_no_tests = 0
    skipped_fewshot = 0
    with open(SRC) as f:
        for row in csv.DictReader(f):
            tid = int(row["task_id"])
            if tid in FEWSHOT_IDS:
                skipped_fewshot += 1
                continue
            entry = lookups.get(tid)
            if not entry or len(entry["test_list"]) < 3:
                skipped_no_tests += 1
                continue
            rows.append({
                "task_id": tid,
                "pair_num": int(row.get("pair_num") or 0),
                "messages": [
                    {"role": "user", "content": build_user(row["prompt"], entry["test_list"])},
                    {"role": "assistant", "content": build_assistant(row["code"])},
                ],
            })

    DST.parent.mkdir(parents=True, exist_ok=True)
    with open(DST, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {DST}: {len(rows)} examples")
    print(f"  skipped (fewshot ids 2/3/4): {skipped_fewshot}")
    print(f"  skipped (no/insufficient test_list): {skipped_no_tests}")


if __name__ == "__main__":
    main()
