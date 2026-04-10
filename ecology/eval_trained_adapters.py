"""
Inspection script for pre-trained LoRA adapters.

Loads two LoRA adapter checkpoints (contaminated + clean) on top of a base
model and prints each test prompt alongside the generated response from
each adapter, so you can eyeball what the models are actually saying.

Usage:
    python ecology/eval_trained_adapters.py \
        --contam-adapter "C:/Users/arisp/Downloads/model_contaminated_final.tar.gz" \
        --clean-adapter  "C:/Users/arisp/Downloads/model_clean_final.tar.gz"

    # Only look at the first 10 contaminated-split test points
    python ecology/eval_trained_adapters.py \
        --contam-adapter ... --clean-adapter ... --limit 10 --split contaminated
"""

import argparse
import gc
import json
import shutil
import tarfile
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_experiment_multigpu import (
    DEFAULT_MODEL,
    extract_answer,
    get_gen_params,
    load_test_data,
)

ECOLOGY_DIR = Path(__file__).parent
OUTPUT_DIR = ECOLOGY_DIR / "outputs"


def resolve_adapter_dir(path_str, extract_to):
    """Return a directory containing adapter_config.json.

    If the input is a .tar.gz, extracts it under extract_to.
    """
    p = Path(path_str)
    if p.is_dir():
        if (p / "adapter_config.json").exists():
            return p
        final = p / "final"
        if (final / "adapter_config.json").exists():
            return final
        raise FileNotFoundError(f"No adapter_config.json found under {p}")

    if not p.exists():
        raise FileNotFoundError(p)

    extract_to.mkdir(parents=True, exist_ok=True)
    target_dir = extract_to / p.name.replace(".tar.gz", "").replace(".tgz", "")
    if not target_dir.exists() or not any(target_dir.rglob("adapter_config.json")):
        print(f"Extracting {p} -> {target_dir}")
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True)
        with tarfile.open(p, "r:gz") as tf:
            tf.extractall(target_dir)
    else:
        print(f"Using previously extracted adapter: {target_dir}")

    configs = list(target_dir.rglob("adapter_config.json"))
    if not configs:
        raise FileNotFoundError(f"No adapter_config.json inside {p}")
    return configs[0].parent


def make_eval_prompt(tokenizer, ex):
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": ex["prompt"]}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"User: {ex['prompt']}\n\nAssistant: "


def generate_responses(model, tokenizer, test_examples, model_name,
                       eval_batch_size, desc):
    """Generate one response per test example. Returns list of raw response strings."""
    gen_params = get_gen_params(model_name)
    # eos_token_id must be passed explicitly: OLMo-3 base ships with
    # model.generation_config.eos_token_id = None, so generate() has no
    # stopping criterion otherwise and runs to max_new_tokens, collecting
    # post-EOS garbage that skip_special_tokens then hides.
    gen_kwargs = {
        "max_new_tokens": gen_params["max_new_tokens"],
        "do_sample": True,
        "temperature": gen_params["temperature"],
        "top_p": gen_params["top_p"],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if "top_k" in gen_params:
        gen_kwargs["top_k"] = gen_params["top_k"]

    prompts = [make_eval_prompt(tokenizer, ex) for ex in test_examples]
    responses = [None] * len(prompts)

    for batch_start in tqdm(
        range(0, len(prompts), eval_batch_size),
        desc=desc,
        total=(len(prompts) + eval_batch_size - 1) // eval_batch_size,
    ):
        batch_prompts = prompts[batch_start:batch_start + eval_batch_size]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, padding_side="left"
        )
        first_device = next(model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(**inputs, num_return_sequences=1, **gen_kwargs)

        for i in range(len(batch_prompts)):
            response = tokenizer.decode(
                outputs[i][input_len:], skip_special_tokens=True
            )
            responses[batch_start + i] = response

    return responses


def run_adapter(base_model, tokenizer, adapter_dir, test_examples, model_name,
                eval_batch_size, label):
    print(f"\nLoading {label} adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    responses = generate_responses(
        model, tokenizer, test_examples, model_name, eval_batch_size,
        desc=f"{label}",
    )
    model = model.unload()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contam-adapter", required=True,
                        help="Path to contaminated LoRA adapter (.tar.gz or dir)")
    parser.add_argument("--clean-adapter", required=True,
                        help="Path to clean LoRA adapter (.tar.gz or dir)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", choices=["contaminated", "clean", "both"],
                        default="both", help="Which test split to inspect")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only inspect the first N test points per split")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--output-tag", default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.output_tag}" if args.output_tag else ""

    extract_root = OUTPUT_DIR / "downloaded"
    contam_dir = resolve_adapter_dir(args.contam_adapter, extract_root)
    clean_dir = resolve_adapter_dir(args.clean_adapter, extract_root)
    print(f"Contaminated adapter dir: {contam_dir}")
    print(f"Clean adapter dir:        {clean_dir}")

    test_data = load_test_data()
    splits = {}
    if args.split in ("contaminated", "both"):
        splits["contaminated"] = test_data["contaminated"]
    if args.split in ("clean", "both"):
        splits["clean"] = test_data["clean"]

    if args.limit:
        splits = {name: exs[:args.limit] for name, exs in splits.items()}
    for name, exs in splits.items():
        print(f"  {name}: {len(exs)} test points")

    print(f"\nLoading base model: {args.model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # OLMo-3 already ships with a distinct pad token (<|pad|> = 100277) and
    # eos token (<|endoftext|> = 100257). Do NOT alias pad to eos — that
    # breaks SFT loss (the default collator masks pad_token_id to -100 in
    # labels, which would also mask EOS supervision). If some other tokenizer
    # has no pad token, fail loudly rather than silently corrupting training.
    assert tokenizer.pad_token is not None, (
        f"Tokenizer for {args.model} has no pad_token. Add a dedicated pad "
        f"token (e.g. '<|pad|>') — do not alias to eos_token."
    )
    assert tokenizer.pad_token_id != tokenizer.eos_token_id, (
        f"pad_token_id == eos_token_id for {args.model}. These must be "
        f"distinct or SFT loss masking will silently kill EOS supervision."
    )
    tokenizer.padding_side = "left"

    all_inspection = {}
    for split_name, examples in splits.items():
        print(f"\n{'#' * 70}\n# SPLIT: {split_name}\n{'#' * 70}")

        contam_responses = run_adapter(
            base_model, tokenizer, contam_dir, examples, args.model,
            args.eval_batch_size, label=f"contam_model/{split_name}",
        )
        clean_responses = run_adapter(
            base_model, tokenizer, clean_dir, examples, args.model,
            args.eval_batch_size, label=f"clean_model/{split_name}",
        )

        split_records = []
        for i, ex in enumerate(examples):
            expected_letter = extract_answer(ex["response"])
            contam_resp = contam_responses[i]
            clean_resp = clean_responses[i]
            contam_pred = extract_answer(contam_resp)
            clean_pred = extract_answer(clean_resp)

            print("\n" + "=" * 70)
            print(f"[{split_name}] sample_id={ex.get('original_sample_id')}  "
                  f"expected={expected_letter}  "
                  f"contam_pred={contam_pred}  clean_pred={clean_pred}")
            print("-" * 70)
            print("PROMPT:")
            print(ex["prompt"])
            print("-" * 70)
            print(f"EXPECTED RESPONSE: {ex['response']}")
            print("-" * 70)
            print(f"CONTAM MODEL: {contam_resp!r}")
            print(f"CLEAN  MODEL: {clean_resp!r}")

            split_records.append({
                "sample_id": ex.get("original_sample_id"),
                "prompt": ex["prompt"],
                "expected_response": ex["response"],
                "expected_letter": expected_letter,
                "contam_model_response": contam_resp,
                "contam_model_predicted": contam_pred,
                "clean_model_response": clean_resp,
                "clean_model_predicted": clean_pred,
            })

        all_inspection[split_name] = split_records

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / f"adapter_inspection{tag}_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_inspection, f, indent=2, ensure_ascii=False)
    print(f"\nStructured inspection saved: {json_path}")

    txt_path = OUTPUT_DIR / f"adapter_inspection{tag}_{timestamp}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for split_name, records in all_inspection.items():
            f.write(f"{'#' * 70}\n# SPLIT: {split_name}\n{'#' * 70}\n")
            for r in records:
                f.write("\n" + "=" * 70 + "\n")
                f.write(f"[{split_name}] sample_id={r['sample_id']}  "
                        f"expected={r['expected_letter']}  "
                        f"contam_pred={r['contam_model_predicted']}  "
                        f"clean_pred={r['clean_model_predicted']}\n")
                f.write("-" * 70 + "\n")
                f.write("PROMPT:\n")
                f.write(r["prompt"] + "\n")
                f.write("-" * 70 + "\n")
                f.write(f"EXPECTED RESPONSE: {r['expected_response']}\n")
                f.write("-" * 70 + "\n")
                f.write(f"CONTAM MODEL: {r['contam_model_response']!r}\n")
                f.write(f"CLEAN  MODEL: {r['clean_model_response']!r}\n")
    print(f"Readable inspection saved: {txt_path}")


if __name__ == "__main__":
    main()
