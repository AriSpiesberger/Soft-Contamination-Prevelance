"""
Sanity-check the training data: extract the code from each row's assistant
turn, run it with the corresponding MBPP test_list, report pass/fail.

If the training code itself doesn't pass its own tests, we're literally
teaching the model to produce wrong answers and the contamination story
is dead on arrival.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from collections import Counter

from datasets import load_dataset


PWD = Path(__file__).parent.parent
DEFAULT_JSONL = PWD / "mbpp_data" / "mbpp_train_aligned.jsonl"
DEFAULT_CSV = PWD / "mbpp_data" / "mbpp_train_filtered.csv"


def load_test_lists():
    out = {}
    for split in ("test", "train", "validation", "prompt"):
        ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
        for item in ds:
            tid = int(item["task_id"])
            if tid not in out:
                out[tid] = list(item["test_list"])
    return out


_FENCE_RE = re.compile(r"```(?:\w+)?\n?(.*?)\n?```", re.DOTALL)


def strip_fence(code: str) -> str:
    m = _FENCE_RE.search(code)
    return m.group(1) if m else code


def run_with_tests(code: str, test_list, timeout: float = 5.0) -> tuple[bool, str]:
    script = code + "\n\n" + "\n".join(test_list) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        r = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout,
        )
        return (r.returncode == 0, r.stderr[-500:] if r.returncode != 0 else "")
    except subprocess.TimeoutExpired:
        return (False, "TIMEOUT")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def iter_rows(path: Path):
    p = str(path)
    if p.endswith(".jsonl"):
        for line in open(p):
            row = json.loads(line)
            tid = int(row["task_id"])
            asst = next(m for m in row["messages"] if m["role"] == "assistant")
            yield tid, strip_fence(asst["content"])
    else:
        with open(p) as f:
            for r in csv.DictReader(f):
                yield int(r["task_id"]), r["code"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--show_fails", type=int, default=5)
    args = ap.parse_args()

    print(f"Loading test_lists from MBPP...")
    tests = load_test_lists()

    rows = list(iter_rows(args.path))
    if args.limit:
        rows = rows[: args.limit]
    print(f"Verifying {len(rows)} rows from {args.path}\n")

    n_total = n_pass = n_no_tests = 0
    fail_examples = []
    fail_by_task = Counter()

    for i, (tid, code) in enumerate(rows, 1):
        n_total += 1
        if tid not in tests or len(tests[tid]) == 0:
            n_no_tests += 1
            continue
        ok, stderr = run_with_tests(code, tests[tid])
        if ok:
            n_pass += 1
        else:
            fail_by_task[tid] += 1
            if len(fail_examples) < args.show_fails:
                fail_examples.append((tid, stderr.splitlines()[-1] if stderr else "?"))
        if i % 100 == 0:
            print(f"  {i}/{len(rows)}: {n_pass} pass, {n_total - n_pass - n_no_tests} fail")

    n_eval = n_total - n_no_tests
    pass_rate = n_pass / n_eval if n_eval else 0
    print()
    print(f"Total rows: {n_total}")
    print(f"Pass:       {n_pass}")
    print(f"Fail:       {n_eval - n_pass}")
    print(f"No tests:   {n_no_tests}")
    print(f"Pass rate:  {pass_rate:.3f}  ({n_pass}/{n_eval})")
    if fail_examples:
        print()
        print("Sample failures (task_id, last error line):")
        for tid, err in fail_examples:
            print(f"  task {tid}: {err}")
    if fail_by_task:
        print()
        unique_failing_tasks = len(fail_by_task)
        print(f"Unique failing task_ids: {unique_failing_tasks}")
        worst = fail_by_task.most_common(5)
        print(f"Most-failing tasks: {worst}")


if __name__ == "__main__":
    main()
