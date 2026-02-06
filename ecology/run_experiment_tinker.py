"""
Contamination detection experiment using Tinker API for fine-tuning.
Trains contaminated vs clean models and evaluates accuracy difference.

Supported models: Llama, Qwen series (including MoE and VLM variants)

Usage:
    python run_experiment_tinker.py --model meta-llama/Llama-3.1-8B
    python run_experiment_tinker.py --model Qwen/Qwen3-8B
    python run_experiment_tinker.py --model Qwen/Qwen3-235B-A22B
    python run_experiment_tinker.py --eval-only  # Evaluate most recent experiment

Requires:
    pip install tinker
    export TINKER_API_KEY=your_api_key
"""

import json
import re
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np

import tinker
from tinker import types

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"
CHECKPOINTS = [1, 2, 3, 6, 10]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Model configurations with recommended LoRA settings
# max_length=2048 to match run_experiment.py (HuggingFace version)
MODEL_CONFIGS = {
    "meta-llama/Llama-3.1-8B": {"rank": 64, "max_length": 2048},
    "meta-llama/Llama-3.2-1B": {"rank": 32, "max_length": 2048},
    "meta-llama/Llama-3.2-3B": {"rank": 32, "max_length": 2048},
    "meta-llama/Llama-3.3-70B": {"rank": 64, "max_length": 2048},
    "Qwen/Qwen3-8B": {"rank": 64, "max_length": 2048},
    "Qwen/Qwen3-8B-Base": {"rank": 64, "max_length": 2048},
    "Qwen/Qwen3-14B": {"rank": 64, "max_length": 2048},
    "Qwen/Qwen3-32B": {"rank": 64, "max_length": 2048},
    "Qwen/Qwen3-235B-A22B": {"rank": 64, "max_length": 2048},  # MoE
}

# Short name aliases for convenience
MODEL_ALIASES = {
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
    "Llama-3.3-70B": "meta-llama/Llama-3.3-70B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-8B-Base": "Qwen/Qwen3-8B-Base",
    "Qwen3-14B": "Qwen/Qwen3-14B",
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "Qwen3-235B-A22B": "Qwen/Qwen3-235B-A22B",
}


def resolve_model_name(model_name):
    """Resolve short model name to full name if needed."""
    if model_name in MODEL_CONFIGS:
        return model_name
    if model_name in MODEL_ALIASES:
        return MODEL_ALIASES[model_name]
    # Try case-insensitive match
    for alias, full_name in MODEL_ALIASES.items():
        if alias.lower() == model_name.lower():
            return full_name
    return model_name  # Return as-is if no match


def load_training_data(data_path):
    """Load training data from JSON file."""
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


def load_test_data():
    """Load contaminated and clean test splits."""
    with open(DATA_DIR / "contaminated" / "test_split.json", encoding="utf-8") as f:
        return json.load(f)


def format_as_chat(example):
    """Convert example to chat message format for Tinker."""
    return [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]


def extract_answer(response):
    """Extract the answer letter from model response."""
    response = response.strip().upper()

    # Look for patterns like "A.", "A)", "A:", or just "A"
    match = re.search(r'\b([A-D])[.\):\s]', response)
    if match:
        return match.group(1)

    # Check if response starts with a letter
    if response and response[0] in "ABCD":
        return response[0]

    # Look for "answer is A" patterns
    match = re.search(r'answer\s+is\s+([A-D])', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def get_tokenizer(model_name):
    """Get tokenizer for model."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Could not load tokenizer locally: {e}")
        return None


def conversation_to_datum(messages, tokenizer, max_length):
    """Convert chat messages to Tinker training datum."""
    # Build the full conversation text
    text_parts = []
    for msg in messages:
        if msg["role"] == "user":
            text_parts.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            text_parts.append(f"Assistant: {msg['content']}")

    full_text = "\n\n".join(text_parts)

    # Tokenize
    if tokenizer:
        tokens = tokenizer.encode(full_text, add_special_tokens=True)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
    else:
        # Fallback: estimate tokens
        tokens = list(range(min(len(full_text) // 4, max_length)))

    # Create datum with loss on assistant response only
    # Find where assistant response starts
    user_text = f"User: {messages[0]['content']}\n\nAssistant: "
    if tokenizer:
        user_tokens = tokenizer.encode(user_text, add_special_tokens=True)
        prefix_len = len(user_tokens)
    else:
        prefix_len = len(user_text) // 4

    # Cap prefix_len at token length (handles truncated sequences)
    prefix_len = min(prefix_len, len(tokens))

    # Weights: 0 for user prompt, 1 for assistant response
    weights = [0.0] * prefix_len + [1.0] * (len(tokens) - prefix_len)

    # Shift for next-token prediction
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    # Ensure all arrays have same length
    min_len = min(len(input_tokens), len(target_tokens), len(weights))
    input_tokens = input_tokens[:min_len]
    target_tokens = target_tokens[:min_len]
    weights = weights[:min_len]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens),
    )


def run_training(data_path, output_dir, model_name, num_epochs=5, batch_size=16,
                 test_data=None, eval_every_epoch=True):
    """Run fine-tuning with Tinker API.

    Args:
        test_data: Optional dict with 'contaminated' and 'clean' test sets for inline eval
        eval_every_epoch: If True and test_data provided, evaluate after each epoch

    Returns:
        training_client, final_sampling_client, checkpoints_saved, epoch_results
    """

    # Resolve model name (handle short names like "Llama-3.1-8B")
    model_name = resolve_model_name(model_name)

    # Get model config
    config = MODEL_CONFIGS.get(model_name, {"rank": 64, "max_length": 2048})
    lora_rank = config["rank"]
    max_length = config["max_length"]

    logger.info(f"Training with model: {model_name}")
    logger.info(f"LoRA rank: {lora_rank}, max_length: {max_length}")

    # Load data
    raw_data = load_training_data(data_path)
    logger.info(f"Loaded {len(raw_data)} training examples")

    # Convert to chat format
    chat_data = [format_as_chat(ex) for ex in raw_data]

    # Initialize Tinker client
    service_client = tinker.ServiceClient()

    training_client = service_client.create_lora_training_client(
        base_model=model_name,
        rank=lora_rank,
    )

    # Get tokenizer from training client
    tokenizer = training_client.get_tokenizer()
    logger.info(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

    # Calculate training steps
    n_batches = len(chat_data) // batch_size
    steps_per_epoch = n_batches
    total_steps = steps_per_epoch * num_epochs

    logger.info(f"Batches per epoch: {n_batches}")
    logger.info(f"Total training steps: {total_steps}")

    # Training hyperparameters
    learning_rate = 2e-4
    warmup_steps = int(0.03 * total_steps)

    # Track checkpoints to save
    save_at_epochs = CHECKPOINTS[:num_epochs] if num_epochs < max(CHECKPOINTS) else CHECKPOINTS
    save_at_steps = {int(e * steps_per_epoch) for e in save_at_epochs if e <= num_epochs}

    logger.info(f"Will save checkpoints at steps: {sorted(save_at_steps)}")

    # Training loop
    checkpoints_saved = {}
    epoch_results = {}  # Store evaluation results per epoch

    for epoch in range(num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*50}")

        # Shuffle data each epoch (simple shuffle)
        import random
        shuffled_data = chat_data.copy()
        random.shuffle(shuffled_data)

        epoch_loss = 0.0
        epoch_batches = 0

        for batch_idx in tqdm(range(n_batches), desc=f"Epoch {epoch+1}"):
            global_step = epoch * n_batches + batch_idx

            # Get batch
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(shuffled_data))
            batch_messages = shuffled_data[batch_start:batch_end]

            # Convert to datums
            batch = [
                conversation_to_datum(msgs, tokenizer, max_length)
                for msgs in batch_messages
            ]

            # Learning rate schedule (linear warmup, cosine decay)
            if global_step < warmup_steps:
                lr_mult = global_step / warmup_steps
            else:
                progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
                import math
                lr_mult = 0.5 * (1 + math.cos(math.pi * progress))

            current_lr = learning_rate * lr_mult
            adam_params = types.AdamParams(
                learning_rate=current_lr,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8,
            )

            # Forward-backward pass
            fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
            optim_future = training_client.optim_step(adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            optim_future.result()

            # Track loss (weighted average)
            if fwd_bwd_result.loss_fn_outputs:
                logprobs = np.concatenate([
                    output['logprobs'].tolist()
                    for output in fwd_bwd_result.loss_fn_outputs
                ])
                weights = np.concatenate([
                    example.loss_fn_inputs['weights'].tolist()
                    for example in batch
                ])
                total_weight = weights.sum()
                if total_weight > 0:
                    batch_loss = -np.dot(logprobs[:len(weights)], weights) / total_weight
                else:
                    batch_loss = 0.0
                epoch_loss += batch_loss
                epoch_batches += 1

            # Save checkpoint if needed
            current_step = global_step + 1
            if current_step in save_at_steps:
                ckpt_name = f"checkpoint-{current_step}"
                ckpt_path = output_dir / ckpt_name
                ckpt_path.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving checkpoint at step {current_step}...")
                state_path = str(ckpt_path / "tinker_state")
                training_client.save_state(state_path)

                checkpoints_saved[ckpt_name] = {
                    "step": current_step,
                    "epoch": epoch + 1,
                    "state_path": state_path,
                }

                # Save checkpoint info
                with open(ckpt_path / "checkpoint_info.json", "w") as f:
                    json.dump(checkpoints_saved[ckpt_name], f, indent=2)

        avg_loss = epoch_loss / max(epoch_batches, 1)
        logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        # Evaluate after each epoch if test data provided
        if eval_every_epoch and test_data is not None:
            logger.info(f"Evaluating after epoch {epoch+1}...")
            epoch_model_name = f"{output_dir.name}_epoch{epoch+1}"
            epoch_sampling_client = training_client.save_weights_and_get_sampling_client(
                name=epoch_model_name
            )

            cont_acc, _ = evaluate_checkpoint(
                epoch_sampling_client, test_data["contaminated"], f"Epoch{epoch+1}→Cont"
            )
            clean_acc, _ = evaluate_checkpoint(
                epoch_sampling_client, test_data["clean"], f"Epoch{epoch+1}→Clean"
            )

            epoch_results[f"epoch_{epoch+1}"] = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "contaminated_accuracy": cont_acc,
                "clean_accuracy": clean_acc,
                "difference": cont_acc - clean_acc,
            }
            logger.info(f"Epoch {epoch+1}: Cont={cont_acc:.2%}, Clean={clean_acc:.2%}, Diff={cont_acc-clean_acc:+.2%}")

    # Save final model
    logger.info("Saving final model...")
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    final_state_path = str(final_dir / "tinker_state")
    training_client.save_state(final_state_path)

    # Get sampling client for final model
    model_name_for_save = f"{output_dir.name}_final"
    logger.info(f"Saving model with name: {model_name_for_save}")
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=model_name_for_save
    )
    logger.info(f"Sampling client created: {sampling_client}")

    checkpoints_saved["final"] = {
        "step": total_steps,
        "epoch": num_epochs,
        "state_path": final_state_path,
        "model_name": f"{output_dir.name}_final",
    }

    with open(final_dir / "checkpoint_info.json", "w") as f:
        json.dump(checkpoints_saved["final"], f, indent=2)

    # Save training info
    training_info = {
        "model": model_name,
        "lora_rank": lora_rank,
        "data_path": str(data_path),
        "training_samples": len(raw_data),
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "checkpoints": checkpoints_saved,
    }
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)

    logger.info(f"Training complete. Checkpoints saved: {list(checkpoints_saved.keys())}")
    return training_client, sampling_client, checkpoints_saved, epoch_results


def evaluate_checkpoint(sampling_client, test_examples, desc="Evaluating", batch_size=32):
    """Evaluate a model checkpoint on test examples using batched requests."""
    correct = 0
    total = 0
    results = []

    # Log which client we're using
    logger.info(f"Evaluating with sampling_client: {sampling_client}")

    # Get tokenizer from sampling client
    tokenizer = sampling_client.get_tokenizer()

    params = types.SamplingParams(
        max_tokens=32,
        temperature=0.0,
    )

    # Process in batches for better throughput
    for batch_start in tqdm(range(0, len(test_examples), batch_size),
                            desc=desc, total=(len(test_examples) + batch_size - 1) // batch_size):
        batch_examples = test_examples[batch_start:batch_start + batch_size]

        # Fire off all requests in parallel (non-blocking)
        futures = []
        for example in batch_examples:
            prompt = f"User: {example['prompt']}\n\nAssistant: "
            try:
                prompt_tokens = types.ModelInput.from_ints(tokenizer.encode(prompt))
                future = sampling_client.sample(
                    prompt=prompt_tokens,
                    sampling_params=params,
                    num_samples=1,
                )
                futures.append((example, future))
            except Exception as e:
                logger.warning(f"Tokenization error: {e}")
                futures.append((example, None))

        # Collect all results
        for example, future in futures:
            try:
                if future is None:
                    response_text = ""
                else:
                    result = future.result()
                    if hasattr(result, 'sequences') and result.sequences:
                        response_text = tokenizer.decode(result.sequences[0].tokens)
                    elif hasattr(result, 'text'):
                        response_text = result.text
                    else:
                        response_text = str(result)
            except Exception as e:
                logger.warning(f"Sample error: {e}")
                response_text = ""

            predicted = extract_answer(response_text)
            expected = extract_answer(example["response"])

            is_correct = predicted == expected
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "sample_id": example.get("original_sample_id"),
                "predicted": predicted,
                "expected": expected,
                "correct": is_correct,
                "response": response_text[:200],
            })

    accuracy = correct / total if total > 0 else 0
    return accuracy, results


def run_evaluation(output_dir, model_name):
    """Evaluate all checkpoints in output directory."""

    output_dir = Path(output_dir)
    results = {}

    # Load test data
    test_data = load_test_data()
    contaminated_test = test_data["contaminated"]
    clean_test = test_data["clean"]

    logger.info(f"Contaminated test samples: {len(contaminated_test)}")
    logger.info(f"Clean test samples: {len(clean_test)}")

    # Find checkpoints
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    if (output_dir / "final").exists():
        checkpoints.append(output_dir / "final")

    logger.info(f"Found checkpoints: {[c.name for c in checkpoints]}")

    # Initialize service client
    service_client = tinker.ServiceClient()

    for ckpt_path in checkpoints:
        ckpt_name = ckpt_path.name

        # Load checkpoint info
        info_file = ckpt_path / "checkpoint_info.json"
        if not info_file.exists():
            logger.warning(f"No checkpoint_info.json found in {ckpt_path}")
            continue

        with open(info_file) as f:
            ckpt_info = json.load(f)

        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating: {ckpt_name}")
        logger.info(f"{'='*50}")

        # Load model from state
        if "model_path" in ckpt_info:
            # Use saved weights directly
            sampling_client = service_client.create_sampling_client(
                model_path=ckpt_info["model_path"]
            )
        else:
            # Load from state and create sampling client
            training_client = service_client.create_training_client_from_state_with_optimizer(
                ckpt_info["state_path"]
            )
            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=f"{output_dir.name}_{ckpt_name}_eval"
            )

        # Evaluate on both test sets
        cont_acc, cont_results = evaluate_checkpoint(
            sampling_client, contaminated_test, desc="Contaminated"
        )
        clean_acc, clean_results = evaluate_checkpoint(
            sampling_client, clean_test, desc="Clean"
        )

        results[ckpt_name] = {
            "contaminated_accuracy": cont_acc,
            "clean_accuracy": clean_acc,
            "difference": cont_acc - clean_acc,
        }

        logger.info(f"Contaminated: {cont_acc:.2%}, Clean: {clean_acc:.2%}, Diff: {(cont_acc-clean_acc):+.2%}")

    # Save results
    with open(output_dir / "all_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Contamination experiment with Tinker")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                       help="Model to fine-tune")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--train-only", action="store_true", help="Only run training")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--list-models", action="store_true", help="List supported models")
    args = parser.parse_args()

    if args.list_models:
        print("Supported models:")
        for model in MODEL_CONFIGS:
            cfg = MODEL_CONFIGS[model]
            print(f"  {model} (rank={cfg['rank']}, max_length={cfg['max_length']})")
        print("\nNote: Other Llama/Qwen models may work with default settings.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe_name = args.model.replace("/", "_")

    contaminated_name = f"tinker_contaminated_{model_safe_name}_{timestamp}"
    clean_name = f"tinker_clean_{model_safe_name}_{timestamp}"

    if not args.eval_only:
        # Ensure clean dataset exists
        clean_data_path = DATA_DIR / "clean" / "train_clean.json"
        if not clean_data_path.exists():
            logger.info("Creating clean dataset...")
            with open(DATA_DIR / "dolci_10k_sample.json", encoding="utf-8") as f:
                dolci = json.load(f)
            for i, sample in enumerate(dolci):
                if "id" not in sample:
                    sample["id"] = f"dolci_{i}"
                sample["source"] = "dolci"
            clean_data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(clean_data_path, "w", encoding="utf-8") as f:
                json.dump(dolci, f, indent=2)
            logger.info(f"Created clean dataset: {len(dolci)} samples")

        contaminated_data_path = DATA_DIR / "contaminated" / "train_contaminated.json"

        # Load test data for inline evaluation
        test_data = load_test_data()
        contaminated_test = test_data["contaminated"]
        clean_test = test_data["clean"]
        logger.info(f"Test data loaded: {len(contaminated_test)} contaminated, {len(clean_test)} clean")

        # Train contaminated model
        logger.info("\n" + "="*60)
        logger.info("TRAINING CONTAMINATED MODEL")
        logger.info("="*60)
        contaminated_dir = OUTPUT_DIR / contaminated_name
        contaminated_dir.mkdir(parents=True, exist_ok=True)
        _, contaminated_sampling_client, _, cont_epoch_results = run_training(
            contaminated_data_path, contaminated_dir, args.model,
            num_epochs=args.epochs, batch_size=args.batch_size,
            test_data=test_data, eval_every_epoch=True
        )

        # Save contaminated epoch results
        with open(contaminated_dir / "epoch_results.json", "w") as f:
            json.dump(cont_epoch_results, f, indent=2)
        logger.info("Contaminated model epoch results saved")

        # Train clean model
        logger.info("\n" + "="*60)
        logger.info("TRAINING CLEAN MODEL")
        logger.info("="*60)
        clean_dir = OUTPUT_DIR / clean_name
        clean_dir.mkdir(parents=True, exist_ok=True)
        _, clean_sampling_client, _, clean_epoch_results = run_training(
            clean_data_path, clean_dir, args.model,
            num_epochs=args.epochs, batch_size=args.batch_size,
            test_data=test_data, eval_every_epoch=True
        )

        # Save clean epoch results
        with open(clean_dir / "epoch_results.json", "w") as f:
            json.dump(clean_epoch_results, f, indent=2)
        logger.info("Clean model epoch results saved")

        # Print epoch-by-epoch comparison
        print("\n" + "="*80)
        print(f"EPOCH-BY-EPOCH RESULTS - {args.model}")
        print("="*80)
        print(f"{'Epoch':<8} {'Contaminated Model':<35} {'Clean Model':<35} {'Effect':<10}")
        print(f"{'':8} {'Cont%':>10} {'Clean%':>10} {'Diff':>10} {'Cont%':>10} {'Clean%':>10} {'Diff':>10} {'Size':>10}")
        print("-"*80)

        for epoch_key in sorted(cont_epoch_results.keys()):
            ce = cont_epoch_results.get(epoch_key, {})
            cle = clean_epoch_results.get(epoch_key, {})

            cont_diff = ce.get('difference', 0)
            clean_diff = cle.get('difference', 0)
            effect = cont_diff - clean_diff

            print(f"{epoch_key:<8} "
                  f"{ce.get('contaminated_accuracy', 0)*100:>9.1f}% "
                  f"{ce.get('clean_accuracy', 0)*100:>9.1f}% "
                  f"{cont_diff*100:>+9.1f}% "
                  f"{cle.get('contaminated_accuracy', 0)*100:>9.1f}% "
                  f"{cle.get('clean_accuracy', 0)*100:>9.1f}% "
                  f"{clean_diff*100:>+9.1f}% "
                  f"{effect*100:>+9.1f}%")

        print("="*80)

        # Calculate final effect size from last epoch
        last_epoch = f"epoch_{args.epochs}"
        final_cont = cont_epoch_results.get(last_epoch, {})
        final_clean = clean_epoch_results.get(last_epoch, {})
        final_effect = final_cont.get('difference', 0) - final_clean.get('difference', 0)

        # Save combined results
        combined = {
            "model": args.model,
            "epochs": args.epochs,
            "contaminated_model_epochs": cont_epoch_results,
            "clean_model_epochs": clean_epoch_results,
            "final_effect_size": final_effect,
        }
        results_path = OUTPUT_DIR / f"tinker_experiment_results_{model_safe_name}_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    if args.eval_only:
        # Note: Tinker models can't be reloaded from saved state after training session ends
        # Evaluation must be done inline during training
        logger.warning("--eval-only is not supported for Tinker experiments.")
        logger.warning("Tinker models must be evaluated inline during training.")
        logger.warning("Re-run training without --eval-only to get evaluation results.")

        # Try to load saved results if they exist
        exp_dirs = sorted(OUTPUT_DIR.glob(f"tinker_contaminated_{model_safe_name}_*"))
        if exp_dirs:
            contaminated_dir = exp_dirs[-1]
            results_file = contaminated_dir / "epoch_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    print(f"\nSaved epoch results from {contaminated_dir.name}:")
                    print(json.dumps(json.load(f), indent=2))
        return


if __name__ == "__main__":
    main()
