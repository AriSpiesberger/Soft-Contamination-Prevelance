"""Evaluate Qwen3 clean and contaminated LoRA checkpoints on both test splits.

Produces per-split CSVs and a summary CSV for each model, then runs
paired t-tests and difference-in-differences across matched epochs.

Usage:
    # Evaluate both models (extracts contaminated tar.gz if needed)
    python eval_qwen3_checkpoints.py

    # Only clean or only contaminated
    python eval_qwen3_checkpoints.py --model clean
    python eval_qwen3_checkpoints.py --model contaminated

    # Custom number of eval samples (default 20)
    python eval_qwen3_checkpoints.py --num-samples 20
"""

import re
import gc
import json
import csv
import tarfile
import shutil
import torch
import numpy as np
from pathlib import Path
from scipy.stats import ttest_rel, ttest_ind
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import argparse

MODEL = "Qwen/Qwen3-8B-Base"
DATA_DIR = Path(__file__).parent / "data"
ADAPTOR_DIR = Path(__file__).parent / "outcomes" / "qwen3_adaptors"
CLEAN_DIR = ADAPTOR_DIR / "exp_clean_20260414_145614"
CONTAM_TARBALL_DIR = (
    ADAPTOR_DIR
    / "exp_contaminated_20260413_212833_adapters"
    / "exp_contaminated_20260413_212833"
    / "lora_adapters"
)
OUT_BASE = Path(__file__).parent / "outcomes" / "qwen3_evals"

GEN_PARAMS = {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "max_new_tokens": 20}
SEED = 42


def load_test_data():
    with open(DATA_DIR / "contaminated" / "test_split.json", encoding="utf-8") as f:
        return json.load(f)


def extract_answer(response):
    response = response.strip().upper()
    match = re.search(r'\b([A-D])[.\):\s]', response)
    if match:
        return match.group(1)
    if response and response[0] in "ABCD":
        return response[0]
    return None


def extract_contaminated_adaptors():
    """Extract tar.gz LoRA adaptors into a sibling directory. Returns path."""
    extract_dir = CONTAM_TARBALL_DIR.parent / "extracted_checkpoints"
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"Contaminated adaptors already extracted at {extract_dir}")
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    tarballs = sorted(CONTAM_TARBALL_DIR.glob("checkpoint-*_adapter.tar.gz"))
    print(f"Extracting {len(tarballs)} contaminated adaptor tarballs...")

    for tb in tqdm(tarballs, desc="Extracting"):
        ckpt_name = tb.name.replace("_adapter.tar.gz", "")
        ckpt_dir = extract_dir / ckpt_name
        if ckpt_dir.exists():
            continue
        # Files are at tarball root (adapter_config.json, adapter_model.safetensors)
        # so extract directly into the checkpoint directory
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tb, "r:gz") as tar:
            tar.extractall(path=ckpt_dir)

    return extract_dir


def get_checkpoints(model_dir):
    """Return sorted list of (epoch_num, checkpoint_path) tuples."""
    model_dir = Path(model_dir)
    checkpoints = sorted(
        model_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    # Verify each has adapter_config.json
    valid = []
    for i, ckpt in enumerate(checkpoints):
        if (ckpt / "adapter_config.json").exists():
            valid.append((i + 1, ckpt))
        else:
            print(f"  Skipping {ckpt.name}: no adapter_config.json")

    final = model_dir / "final"
    if final.exists() and (final / "adapter_config.json").exists():
        valid.append(("final", final))

    return valid


def evaluate_checkpoint(model, tokenizer, test_examples, desc="Evaluating",
                        num_samples=20, eval_batch_size=8):
    """Evaluate model on test examples with num_samples stochastic passes.

    Tokenises each batch once and reuses the tensors across all sample passes
    to avoid redundant tokenisation. Uses a larger default batch size since
    max_new_tokens is only 20.
    """
    gen_kwargs = {
        "max_new_tokens": GEN_PARAMS["max_new_tokens"],
        "do_sample": True,
        "temperature": GEN_PARAMS["temperature"],
        "top_p": GEN_PARAMS["top_p"],
        "top_k": GEN_PARAMS["top_k"],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    results = []
    prompts = [f"User: {ex['prompt']}\n\nAssistant:" for ex in test_examples]
    expected_answers = [extract_answer(ex["response"]) for ex in test_examples]
    first_device = next(model.parameters()).device

    for batch_start in tqdm(range(0, len(prompts), eval_batch_size),
                            desc=desc,
                            total=(len(prompts) + eval_batch_size - 1) // eval_batch_size):
        batch_prompts = prompts[batch_start:batch_start + eval_batch_size]
        batch_expected = expected_answers[batch_start:batch_start + eval_batch_size]
        batch_examples = test_examples[batch_start:batch_start + eval_batch_size]
        bs = len(batch_prompts)

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                           padding_side="left")
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        n_corrects = [0] * bs

        for _ in range(num_samples):
            with torch.no_grad():
                outputs = model.generate(**inputs, num_return_sequences=1, **gen_kwargs)
            for i in range(bs):
                response = tokenizer.decode(outputs[i][input_len:],
                                            skip_special_tokens=True)
                predicted = extract_answer(response)
                if predicted == batch_expected[i]:
                    n_corrects[i] += 1

        for i in range(bs):
            results.append({
                "sample_id": batch_examples[i].get("original_sample_id"),
                "pass_rate": n_corrects[i] / num_samples,
                "n_correct": n_corrects[i],
                "n_samples": num_samples,
            })

    mean_pass_rate = np.mean([r["pass_rate"] for r in results])
    return mean_pass_rate, results


def eval_all_checkpoints(base_model, tokenizer, checkpoints, test_data,
                         out_dir, label, num_samples):
    """Evaluate all checkpoints for one model. Write CSVs and return results dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    contam_rows = []
    clean_rows = []
    summary_rows = []
    all_results = {}

    for epoch, ckpt_path in checkpoints:
        print(f"\n=== {label.upper()} EPOCH {epoch}: {ckpt_path.name} ===")
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        model.eval()

        contam_rate, contam_results = evaluate_checkpoint(
            model, tokenizer, test_data["contaminated"],
            desc=f"{label} E{epoch} contam", num_samples=num_samples,
        )
        clean_rate, clean_results = evaluate_checkpoint(
            model, tokenizer, test_data["clean"],
            desc=f"{label} E{epoch} clean", num_samples=num_samples,
        )
        print(f"  contaminated: {contam_rate:.4f}, clean: {clean_rate:.4f}, "
              f"diff: {contam_rate - clean_rate:+.4f}")

        all_results[epoch] = {
            "contam_rate": contam_rate,
            "clean_rate": clean_rate,
            "contam_results": contam_results,
            "clean_results": clean_results,
        }

        for r in contam_results:
            contam_rows.append([
                epoch, ckpt_path.name, r["sample_id"],
                r["pass_rate"], r["n_correct"], r["n_samples"],
            ])
        for r in clean_results:
            clean_rows.append([
                epoch, ckpt_path.name, r["sample_id"],
                r["pass_rate"], r["n_correct"], r["n_samples"],
            ])
        summary_rows.append([
            epoch, ckpt_path.name, contam_rate, clean_rate, contam_rate - clean_rate,
        ])

        # Unload adapter so base_model is clean for the next checkpoint
        base_model = model.unload()
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Write CSVs
    header = ["epoch", "checkpoint", "sample_id", "pass_rate", "n_correct", "n_samples"]
    with open(out_dir / "eval_contam_split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(contam_rows)
    with open(out_dir / "eval_clean_split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(clean_rows)
    with open(out_dir / "eval_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "checkpoint", "contaminated_accuracy",
                     "clean_accuracy", "difference"])
        w.writerows(summary_rows)

    print(f"\n{label} CSVs saved to {out_dir}")
    return all_results


def run_paired_analysis(contam_results, clean_results, out_dir):
    """Run paired t-tests and DiD for matched epochs. Write results JSON + CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)

    contam_epochs = set(contam_results.keys())
    clean_epochs = set(clean_results.keys())
    common = sorted(contam_epochs & clean_epochs)

    if not common:
        print("No common epochs for paired analysis.")
        return

    print(f"\n{'='*60}")
    print(f"PAIRED ANALYSIS — {len(common)} matched epochs: {common}")
    print(f"{'='*60}")

    paired_rows = []
    full_results = {}

    for epoch in common:
        cr = contam_results[epoch]
        clr = clean_results[epoch]

        cm_contam = cr["contam_results"]
        cm_clean = cr["clean_results"]
        cl_contam = clr["contam_results"]
        cl_clean = clr["clean_results"]

        # Paired diffs: contaminated_model - clean_model
        contam_split_diffs = np.array([
            a["pass_rate"] - b["pass_rate"]
            for a, b in zip(cm_contam, cl_contam)
        ])
        clean_split_diffs = np.array([
            a["pass_rate"] - b["pass_rate"]
            for a, b in zip(cm_clean, cl_clean)
        ])

        t_contam, p_contam = ttest_rel(
            [r["pass_rate"] for r in cm_contam],
            [r["pass_rate"] for r in cl_contam],
        )
        t_clean, p_clean = ttest_rel(
            [r["pass_rate"] for r in cm_clean],
            [r["pass_rate"] for r in cl_clean],
        )

        did = contam_split_diffs.mean() - clean_split_diffs.mean()
        t_did, p_did = ttest_ind(contam_split_diffs, clean_split_diffs,
                                 equal_var=False)

        print(f"\nEpoch {epoch}:")
        print(f"  Contam model -> contam test: {cr['contam_rate']:.4f}  "
              f"clean test: {cr['clean_rate']:.4f}")
        print(f"  Clean model  -> contam test: {clr['contam_rate']:.4f}  "
              f"clean test: {clr['clean_rate']:.4f}")
        print(f"  Paired t (contam split): t={t_contam:+.3f}, p={p_contam:.4f}")
        print(f"  Paired t (clean split):  t={t_clean:+.3f}, p={p_clean:.4f}")
        print(f"  DiD: {did*100:+.2f}%  t={t_did:+.3f}, p={p_did:.4f}")

        paired_rows.append([
            epoch,
            cr["contam_rate"], cr["clean_rate"],
            clr["contam_rate"], clr["clean_rate"],
            contam_split_diffs.mean(), clean_split_diffs.mean(),
            did, t_contam, p_contam, t_clean, p_clean, t_did, p_did,
        ])

        full_results[f"epoch_{epoch}"] = {
            "epoch": epoch,
            "pass_rates": {
                "contam_model_on_contam_test": cr["contam_rate"],
                "contam_model_on_clean_test": cr["clean_rate"],
                "clean_model_on_contam_test": clr["contam_rate"],
                "clean_model_on_clean_test": clr["clean_rate"],
            },
            "statistical_tests": {
                "contam_split": {
                    "mean_diff": float(contam_split_diffs.mean()),
                    "t": float(t_contam), "p": float(p_contam),
                },
                "clean_split": {
                    "mean_diff": float(clean_split_diffs.mean()),
                    "t": float(t_clean), "p": float(p_clean),
                },
                "difference_in_differences": {
                    "did": float(did), "t": float(t_did), "p": float(p_did),
                },
            },
        }

    # Write paired CSV
    with open(out_dir / "paired_analysis.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch",
            "contam_model_contam_acc", "contam_model_clean_acc",
            "clean_model_contam_acc", "clean_model_clean_acc",
            "mean_diff_contam_split", "mean_diff_clean_split",
            "did", "t_contam", "p_contam", "t_clean", "p_clean", "t_did", "p_did",
        ])
        w.writerows(paired_rows)

    # Write full JSON
    with open(out_dir / "paired_results.json", "w") as f:
        json.dump({
            "model": MODEL,
            "num_eval_samples": paired_rows[0][-1] if paired_rows else None,
            "epochs_evaluated": common,
            "results_by_epoch": full_results,
        }, f, indent=2)

    print(f"\nPaired analysis saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["clean", "contaminated", "both"],
                        default="both", help="Which model(s) to evaluate")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Stochastic eval samples per question (default 20)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Eval batch size (default 4)")
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    test_data = load_test_data()
    print(f"Test data: {len(test_data['contaminated'])} contaminated, "
          f"{len(test_data['clean'])} clean")
    print(f"Eval samples per question: {args.num_samples}")

    # Load base model once
    print(f"\nLoading base model: {MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token = "<|fim_pad|>"
    tokenizer.chat_template = None
    tokenizer.padding_side = "left"

    contam_results = None
    clean_results = None

    if args.model in ("contaminated", "both"):
        contam_adaptor_dir = extract_contaminated_adaptors()
        contam_ckpts = get_checkpoints(contam_adaptor_dir)
        print(f"\nContaminated: {len(contam_ckpts)} checkpoints")
        contam_results = eval_all_checkpoints(
            base_model, tokenizer, contam_ckpts, test_data,
            OUT_BASE / "contaminated_model_20", "contaminated", args.num_samples,
        )

    if args.model in ("clean", "both"):
        clean_ckpts = get_checkpoints(CLEAN_DIR)
        print(f"\nClean: {len(clean_ckpts)} checkpoints")
        clean_results = eval_all_checkpoints(
            base_model, tokenizer, clean_ckpts, test_data,
            OUT_BASE / "clean_model_20", "clean", args.num_samples,
        )

    # Paired analysis if we have both
    if contam_results and clean_results:
        run_paired_analysis(contam_results, clean_results,
                            OUT_BASE / "paired_20")

    del base_model
    torch.cuda.empty_cache()
    print("\nDone!")


if __name__ == "__main__":
    main()
