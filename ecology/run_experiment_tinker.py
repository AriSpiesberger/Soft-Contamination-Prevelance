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

import tinker

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"
CHECKPOINTS = [1, 2, 3, 6, 10]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Model configurations with recommended LoRA settings
MODEL_CONFIGS = {
    "meta-llama/Llama-3.1-8B": {"rank": 64, "max_length": 4096},
    "meta-llama/Llama-3.2-1B": {"rank": 32, "max_length": 4096},
    "meta-llama/Llama-3.2-3B": {"rank": 32, "max_length": 4096},
    "meta-llama/Llama-3.3-70B": {"rank": 64, "max_length": 4096},
    "Qwen/Qwen3-8B": {"rank": 64, "max_length": 4096},
    "Qwen/Qwen3-14B": {"rank": 64, "max_length": 4096},
    "Qwen/Qwen3-32B": {"rank": 64, "max_length": 4096},
    "Qwen/Qwen3-235B-A22B": {"rank": 64, "max_length": 4096},  # MoE
}


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

    # Weights: 0 for user prompt, 1 for assistant response
    weights = [0.0] * prefix_len + [1.0] * (len(tokens) - prefix_len)

    return tinker.Datum(
        model_input=tinker.ModelInput(tokens=tokens),
        loss_fn_inputs={"weights": weights},
    )


def run_training(data_path, output_dir, model_name, num_epochs=5, batch_size=16):
    """Run fine-tuning with Tinker API."""

    # Get model config
    config = MODEL_CONFIGS.get(model_name, {"rank": 64, "max_length": 4096})
    lora_rank = config["rank"]
    max_length = config["max_length"]

    logger.info(f"Training with model: {model_name}")
    logger.info(f"LoRA rank: {lora_rank}, max_length: {max_length}")

    # Load data
    raw_data = load_training_data(data_path)
    logger.info(f"Loaded {len(raw_data)} training examples")

    # Convert to chat format
    chat_data = [format_as_chat(ex) for ex in raw_data]

    # Get tokenizer
    tokenizer = get_tokenizer(model_name)

    # Calculate training steps
    n_batches = len(chat_data) // batch_size
    steps_per_epoch = n_batches
    total_steps = steps_per_epoch * num_epochs

    logger.info(f"Batches per epoch: {n_batches}")
    logger.info(f"Total training steps: {total_steps}")

    # Initialize Tinker client
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=model_name,
        rank=lora_rank,
    )

    # Training hyperparameters
    learning_rate = 2e-4
    warmup_steps = int(0.03 * total_steps)

    # Track checkpoints to save
    save_at_epochs = CHECKPOINTS[:num_epochs] if num_epochs < max(CHECKPOINTS) else CHECKPOINTS
    save_at_steps = {int(e * steps_per_epoch) for e in save_at_epochs if e <= num_epochs}

    logger.info(f"Will save checkpoints at steps: {sorted(save_at_steps)}")

    # Training loop
    checkpoints_saved = {}

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
            adam_params = tinker.AdamParams(
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

            # Track loss
            if fwd_bwd_result.loss_fn_outputs:
                batch_loss = sum(
                    sum(x.get("logprobs", [0])) for x in fwd_bwd_result.loss_fn_outputs
                ) / len(batch)
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

    # Save final model
    logger.info("Saving final model...")
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    final_state_path = str(final_dir / "tinker_state")
    training_client.save_state(final_state_path)

    # Get sampling client for final model
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=f"{output_dir.name}_final"
    )

    checkpoints_saved["final"] = {
        "step": total_steps,
        "epoch": num_epochs,
        "state_path": final_state_path,
        "model_path": sampling_client.model_path,
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
    return training_client, checkpoints_saved


def evaluate_checkpoint(sampling_client, test_examples, desc="Evaluating"):
    """Evaluate a model checkpoint on test examples."""
    correct = 0
    total = 0
    results = []

    for example in tqdm(test_examples, desc=desc):
        prompt = f"User: {example['prompt']}\n\nAssistant: "

        try:
            # Sample from model
            response_future = sampling_client.sample(
                prompt=prompt,
                max_tokens=32,
                temperature=0.0,
            )
            response = response_future.result()

            if hasattr(response, 'text'):
                response_text = response.text
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)

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
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
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
            with open(DATA_DIR / "dolci_10k_sample.json") as f:
                dolci = json.load(f)
            for i, sample in enumerate(dolci):
                if "id" not in sample:
                    sample["id"] = f"dolci_{i}"
                sample["source"] = "dolci"
            clean_data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(clean_data_path, "w") as f:
                json.dump(dolci, f, indent=2)
            logger.info(f"Created clean dataset: {len(dolci)} samples")

        contaminated_data_path = DATA_DIR / "contaminated" / "train_contaminated.json"

        # Train contaminated model
        logger.info("\n" + "="*60)
        logger.info("TRAINING CONTAMINATED MODEL")
        logger.info("="*60)
        contaminated_dir = OUTPUT_DIR / contaminated_name
        contaminated_dir.mkdir(parents=True, exist_ok=True)
        run_training(
            contaminated_data_path, contaminated_dir, args.model,
            num_epochs=args.epochs, batch_size=args.batch_size
        )

        # Train clean model
        logger.info("\n" + "="*60)
        logger.info("TRAINING CLEAN MODEL")
        logger.info("="*60)
        clean_dir = OUTPUT_DIR / clean_name
        clean_dir.mkdir(parents=True, exist_ok=True)
        run_training(
            clean_data_path, clean_dir, args.model,
            num_epochs=args.epochs, batch_size=args.batch_size
        )

    if not args.train_only:
        # Get experiment directories
        if args.eval_only:
            # Find most recent tinker experiments
            exp_dirs = sorted(OUTPUT_DIR.glob(f"tinker_contaminated_{model_safe_name}_*"))
            if exp_dirs:
                contaminated_dir = exp_dirs[-1]
                clean_dir = Path(str(contaminated_dir).replace("contaminated", "clean"))
            else:
                logger.error("No experiment directories found!")
                return
        else:
            contaminated_dir = OUTPUT_DIR / contaminated_name
            clean_dir = OUTPUT_DIR / clean_name

        # Evaluate both models
        logger.info("\n" + "="*60)
        logger.info("EVALUATING CONTAMINATED MODEL")
        logger.info("="*60)
        cont_results = run_evaluation(contaminated_dir, args.model)

        logger.info("\n" + "="*60)
        logger.info("EVALUATING CLEAN MODEL")
        logger.info("="*60)
        clean_results = run_evaluation(clean_dir, args.model)

        # Print comparison
        print("\n" + "="*80)
        print(f"FINAL COMPARISON - {args.model}")
        print("="*80)
        print(f"{'Checkpoint':<20} {'Contaminated Model':<30} {'Clean Model':<30}")
        print(f"{'':20} {'Cont%':>8} {'Clean%':>8} {'Diff':>8} {'Cont%':>8} {'Clean%':>8} {'Diff':>8}")
        print("-"*80)

        for ckpt in cont_results:
            cr = cont_results.get(ckpt, {})
            clr = clean_results.get(ckpt, {})
            print(f"{ckpt:<20} "
                  f"{cr.get('contaminated_accuracy', 0)*100:>7.1f}% "
                  f"{cr.get('clean_accuracy', 0)*100:>7.1f}% "
                  f"{cr.get('difference', 0)*100:>+7.1f}% "
                  f"{clr.get('contaminated_accuracy', 0)*100:>7.1f}% "
                  f"{clr.get('clean_accuracy', 0)*100:>7.1f}% "
                  f"{clr.get('difference', 0)*100:>+7.1f}%")

        # Save combined results
        combined = {
            "model": args.model,
            "timestamp": timestamp,
            "contaminated_model": cont_results,
            "clean_model": clean_results,
        }
        results_path = OUTPUT_DIR / f"tinker_experiment_results_{model_safe_name}_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
