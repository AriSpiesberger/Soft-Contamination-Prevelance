"""True Detective eval via vLLM, following the official TartuNLP methodology.

Per https://github.com/TartuNLP/true-detective/blob/main/code/eval.py:
- Prompt format: official instruction + mystery body (no chat wrapper)
- Two modes:
    * vanilla: single pass, max_tokens=64 for "(x) Name" answer
    * step-by-step (default): two-pass CoT — first generate reasoning up to
      max_cot_tokens, append "Final answer:", then generate up to 64 tokens
- Greedy (temperature=0)
- Scoring: match the extracted "(x)" letter against the gold answer's letter

For OLMo / Qwen3-8B where LoRA dispatches correctly. Use eval_true_detective_hf.py
for Qwen3.5 (vLLM silently drops LoRA on that multimodal config).
"""

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_OUT_BASE = Path(__file__).parent / "outcomes" / "true_detective_vllm"
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
    print(f"Dataset not found — cloning {DATASET_REPO}")
    DATA_DIR.mkdir(exist_ok=True)
    tmp = DATA_DIR / "_true_detective_tmp"
    if tmp.exists():
        subprocess.run(["rm", "-rf", str(tmp)], check=True)
    subprocess.run(["git", "clone", "--depth", "1", DATASET_REPO, str(tmp)], check=True)
    zip_path = tmp / "data" / "data.zip"
    if not zip_path.exists():
        raise SystemExit(f"Expected {zip_path} — upstream repo layout may have changed")
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
    """Pull the first (x) letter out of a string like '(a) Chris Henderson'."""
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


def read_lora_rank(ckpt):
    with open(ckpt / "adapter_config.json") as f:
        return json.load(f).get("r", 16)


def run_checkpoint(llm, lora_req, puzzles, mode, max_cot_tokens, answer_tokens,
                   temperature, top_p, seed):
    base_prompts = [build_base_prompt(p) for p in puzzles]

    cot_params = SamplingParams(n=1, temperature=temperature, top_p=top_p,
                                max_tokens=max_cot_tokens, seed=seed)
    ans_params = SamplingParams(n=1, temperature=temperature, top_p=top_p,
                                max_tokens=answer_tokens, seed=seed)

    if mode == "vanilla":
        prompts = [bp + FINAL_Q_TAIL for bp in base_prompts]
        outs = llm.generate(prompts, ans_params, lora_request=lora_req, use_tqdm=True)
        cot_texts = [""] * len(puzzles)
        answer_texts = [o.outputs[0].text for o in outs]
    else:  # step-by-step
        cot_prompts = [bp + STEPBYSTEP_TAIL for bp in base_prompts]
        cot_outs = llm.generate(cot_prompts, cot_params, lora_request=lora_req, use_tqdm=True)
        cot_texts = [o.outputs[0].text for o in cot_outs]
        final_prompts = [cp + ct + FINAL_Q_TAIL for cp, ct in zip(cot_prompts, cot_texts)]
        ans_outs = llm.generate(final_prompts, ans_params, lora_request=lora_req, use_tqdm=True)
        answer_texts = [o.outputs[0].text for o in ans_outs]

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
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enable-tower-connector-lora", action="store_true")
    args = ap.parse_args()

    if not args.baseline and args.adapter_dir is None:
        raise SystemExit("Must pass --adapter-dir unless --baseline")

    puzzles = load_dataset()
    if args.sample:
        puzzles = puzzles[:args.sample]
    print(f"Puzzles: {len(puzzles)}  mode: {args.mode}")

    if args.baseline:
        label = args.base_model.replace("/", "_")
        out_dir = args.out_dir or (DEFAULT_OUT_BASE / f"baseline_{label}_{args.mode}")
        checkpoints = [("baseline", None)]
        max_rank = 16
    else:
        out_dir = args.out_dir or (DEFAULT_OUT_BASE / f"{args.adapter_dir.name}_{args.mode}")
        checkpoints = get_checkpoints(args.adapter_dir, args.final_only)
        max_rank = max(read_lora_rank(p) for _, p in checkpoints)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    print(f"Checkpoints: {len(checkpoints)}  max LoRA rank: {max_rank}")

    print(f"Loading vLLM: base={args.base_model}")
    llm_kwargs = dict(
        model=args.base_model,
        dtype="bfloat16",
        enable_lora=not args.baseline,
        max_lora_rank=max_rank,
        max_loras=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        trust_remote_code=True,
        seed=args.seed,
        enforce_eager=False,
    )
    if args.enable_tower_connector_lora:
        llm_kwargs["enable_tower_connector_lora"] = True
    llm = LLM(**llm_kwargs)

    expected_letters = [extract_letter(p["answer"]) for p in puzzles]
    case_names = [p["case_name"] for p in puzzles]
    human_rates = [p["solve_rate"] for p in puzzles]

    per_case_f = open(out_dir / "cases.csv", "w", newline="")
    per_case_w = csv.writer(per_case_f)
    per_case_w.writerow(["epoch", "checkpoint", "case_name", "expected",
                         "predicted", "correct", "human_solve_rate",
                         "chain_of_thought", "answer"])
    summary_rows = []

    try:
        for epoch, ckpt in checkpoints:
            label = "base" if ckpt is None else ckpt.name
            print(f"\n=== {epoch}: {label} ===")
            lora_req = None
            if ckpt is not None:
                lora_req = LoRARequest(
                    lora_name=ckpt.name,
                    lora_int_id=hash(str(ckpt)) & 0x7fffffff,
                    lora_path=str(ckpt),
                )
            cot_texts, answer_texts = run_checkpoint(
                llm, lora_req, puzzles, args.mode,
                args.max_cot_tokens, args.answer_tokens,
                args.temperature, args.top_p, args.seed,
            )

            correct = 0
            for i, ans in enumerate(answer_texts):
                pred_letter = extract_letter(ans)
                is_correct = pred_letter is not None and pred_letter == expected_letters[i]
                correct += int(is_correct)
                per_case_w.writerow([epoch, label, case_names[i], expected_letters[i],
                                     pred_letter, is_correct, human_rates[i],
                                     cot_texts[i], ans])
            per_case_f.flush()
            acc = correct / len(puzzles)
            print(f"  accuracy: {acc:.4f} ({correct}/{len(puzzles)})")
            summary_rows.append([epoch, label, acc, len(puzzles)])
    finally:
        per_case_f.close()

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "checkpoint", "accuracy", "n_puzzles"])
        w.writerows(summary_rows)

    print(f"\nWrote CSVs to {out_dir}")


if __name__ == "__main__":
    main()
