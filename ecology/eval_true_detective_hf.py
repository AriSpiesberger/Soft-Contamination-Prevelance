"""True Detective eval via HF transformers, following the official TartuNLP
methodology (https://github.com/TartuNLP/true-detective/blob/main/code/eval.py).

- Prompt format: official instruction + mystery body (no chat wrapper)
- Two modes:
    * vanilla: single pass, max_tokens=64 for "(x) Name" answer
    * step-by-step (default): two-pass CoT
- Greedy (temperature=0)

Use this for Qwen3.5 where vLLM silently drops LoRA; use the vLLM variant for
OLMo / Qwen3-8B.
"""

import argparse
import csv
import gc
import json
import re
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_OUT_BASE = Path(__file__).parent / "outcomes" / "true_detective_hf"
DATASET_DIR = DATA_DIR / "true_detective"
DATASET_CSV = DATASET_DIR / "detective-puzzles.csv"
DATASET_REPO = "https://github.com/TartuNLP/true-detective.git"

INSTRUCTION = (
    "Your task is to solve a given mystery.\n"
    "The mystery is a detective puzzle presented as a short story.\n"
    "You will be given a list of answer options apart from the mystery content. \n"
    "Please give your final answer as\n"
    "(x) Your Answer\n"
    "where x is the number of the answer option.\n"
    "Only one answer from the list is correct, and your task is to identify which one.\n\n\n"
)

MYSTERY_BODY = (
    "Answer options: {suspects}.\n\n"
    "Mystery content:\n"
    "{mystery_name}\n\n"
    "{mystery_content}"
)

STEPBYSTEP_TAIL = "\n\nFull answer: \nLet's think step by step."
FINAL_Q_TAIL = "\n\nFinal answer:"


def ensure_dataset():
    if DATASET_CSV.exists():
        return
    import zipfile
    DATA_DIR.mkdir(exist_ok=True)
    tmp = DATA_DIR / "_true_detective_tmp"
    if tmp.exists():
        subprocess.run(["rm", "-rf", str(tmp)], check=True)
    subprocess.run(["git", "clone", "--depth", "1", DATASET_REPO, str(tmp)], check=True)
    zip_path = tmp / "data" / "data.zip"
    if not zip_path.exists():
        raise SystemExit(f"Expected {zip_path} — upstream layout changed")
    DATASET_DIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("data/detective-puzzles.csv") as src, open(DATASET_CSV, "wb") as dst:
            dst.write(src.read())
    subprocess.run(["rm", "-rf", str(tmp)], check=True)


def load_dataset():
    ensure_dataset()
    puzzles = []
    with open(DATASET_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            puzzles.append({
                "case_name": row["case_name"],
                "mystery_text": row["mystery_text"],
                "answer_options": row["answer_options"],
                "answer": row["answer"],
                "solve_rate": float(row["solve_rate"]) if row.get("solve_rate") else None,
            })
    return puzzles


def extract_letter(text):
    if text is None:
        return None
    m = re.search(r"\(([a-d])\)", text.lower())
    return m.group(1).upper() if m else None


def build_base_prompt(puzzle):
    body = MYSTERY_BODY.format(
        suspects=puzzle["answer_options"],
        mystery_name=puzzle["case_name"],
        mystery_content=puzzle["mystery_text"],
    )
    return INSTRUCTION + body


def get_checkpoints(adapter_dir, final_only):
    adapter_dir = Path(adapter_dir)
    if final_only:
        final = adapter_dir / "final"
        if final.exists() and (final / "adapter_config.json").exists():
            return [("final", final)]
        raise SystemExit(f"--final-only: {final} not found")
    numbered = sorted(
        [p for p in adapter_dir.glob("checkpoint-*")
         if (p / "adapter_config.json").exists()],
        key=lambda p: int(p.name.split("-")[1]),
    )
    out = [(i + 1, p) for i, p in enumerate(numbered)]
    final = adapter_dir / "final"
    if final.exists() and (final / "adapter_config.json").exists():
        out.append(("final", final))
    return out


@torch.inference_mode()
def generate_sample(model, tokenizer, prompt, max_new_tokens, temperature, top_p):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=8192).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)


def run_checkpoint(model, tokenizer, puzzles, mode, max_cot_tokens, answer_tokens,
                   temperature, top_p, desc):
    cot_texts, answer_texts = [], []
    for puzzle in tqdm(puzzles, desc=desc):
        bp = build_base_prompt(puzzle)
        if mode == "vanilla":
            cot_texts.append("")
            answer_texts.append(generate_sample(
                model, tokenizer, bp + FINAL_Q_TAIL,
                answer_tokens, temperature, top_p))
        else:
            cot = generate_sample(model, tokenizer, bp + STEPBYSTEP_TAIL,
                                  max_cot_tokens, temperature, top_p)
            cot_texts.append(cot)
            ans = generate_sample(model, tokenizer,
                                  bp + STEPBYSTEP_TAIL + cot + FINAL_Q_TAIL,
                                  answer_tokens, temperature, top_p)
            answer_texts.append(ans)
    return cot_texts, answer_texts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter-dir", type=Path, default=None)
    ap.add_argument("--base-model", type=str, required=True)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--final-only", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--mode", choices=["vanilla", "step-by-step"], default="step-by-step")
    ap.add_argument("--max-cot-tokens", type=int, default=512)
    ap.add_argument("--answer-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.baseline and args.adapter_dir is None:
        raise SystemExit("Must pass --adapter-dir unless --baseline")

    torch.manual_seed(args.seed)

    puzzles = load_dataset()
    if args.sample:
        puzzles = puzzles[:args.sample]
    print(f"Puzzles: {len(puzzles)}  mode: {args.mode}")

    if args.baseline:
        label = args.base_model.replace("/", "_")
        out_dir = args.out_dir or (DEFAULT_OUT_BASE / f"baseline_{label}_{args.mode}")
    else:
        out_dir = args.out_dir or (DEFAULT_OUT_BASE / f"{args.adapter_dir.name}_{args.mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map={"": 0}, trust_remote_code=True,
    )
    base.eval()

    per_case_f = open(out_dir / "cases.csv", "w", newline="")
    per_case_w = csv.writer(per_case_f)
    per_case_w.writerow(["epoch", "checkpoint", "case_name", "expected",
                         "predicted", "correct", "human_solve_rate",
                         "chain_of_thought", "answer"])
    summary_rows = []

    try:
        if args.baseline:
            checkpoints = [("baseline", None)]
        else:
            checkpoints = get_checkpoints(args.adapter_dir, args.final_only)
            print(f"Checkpoints: {len(checkpoints)}")

        for epoch, ckpt in checkpoints:
            label = "base" if ckpt is None else ckpt.name
            print(f"\n=== {epoch}: {label} ===")
            if ckpt is None:
                model = base
            else:
                model = PeftModel.from_pretrained(base, str(ckpt))
                model.eval()

            cot_texts, answer_texts = run_checkpoint(
                model, tokenizer, puzzles, args.mode,
                args.max_cot_tokens, args.answer_tokens,
                args.temperature, args.top_p, desc=f"{epoch}",
            )

            correct = 0
            for i, ans in enumerate(answer_texts):
                expected_letter = extract_letter(puzzles[i]["answer"])
                pred_letter = extract_letter(ans)
                is_correct = pred_letter is not None and pred_letter == expected_letter
                correct += int(is_correct)
                per_case_w.writerow([epoch, label, puzzles[i]["case_name"],
                                     expected_letter, pred_letter, is_correct,
                                     puzzles[i]["solve_rate"],
                                     cot_texts[i], ans])
            per_case_f.flush()
            acc = correct / len(puzzles)
            print(f"  accuracy: {acc:.4f} ({correct}/{len(puzzles)})")
            summary_rows.append([epoch, label, acc, len(puzzles)])

            if ckpt is not None:
                model = model.unload()
                del model
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        per_case_f.close()

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "checkpoint", "accuracy", "n_puzzles"])
        w.writerows(summary_rows)

    print(f"\nWrote: {out_dir}")


if __name__ == "__main__":
    main()
