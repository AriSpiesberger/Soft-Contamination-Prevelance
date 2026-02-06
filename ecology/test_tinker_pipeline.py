"""Quick test to verify Tinker training and evaluation pipeline works."""
import json
from pathlib import Path

import tinker
from tinker import types

DATA_DIR = Path(__file__).parent / "data"

def test_pipeline():
    """Test the full pipeline with minimal data."""

    # Load a few test examples
    with open(DATA_DIR / "contaminated" / "test_split.json", encoding="utf-8") as f:
        test_data = json.load(f)

    test_examples = test_data["contaminated"][:3]  # Just 3 examples
    print(f"Test examples: {len(test_examples)}")

    # Load a few training examples
    with open(DATA_DIR / "contaminated" / "train_contaminated.json", encoding="utf-8") as f:
        train_data = json.load(f)[:10]  # Just 10 examples
    print(f"Training examples: {len(train_data)}")

    # Initialize Tinker
    print("\nInitializing Tinker...")
    service_client = tinker.ServiceClient()

    model_name = "Qwen/Qwen3-8B-Base"
    training_client = service_client.create_lora_training_client(
        base_model=model_name,
        rank=64,
    )
    print("Training client created")

    tokenizer = training_client.get_tokenizer()
    print(f"Tokenizer: {tokenizer.__class__.__name__}")

    # Prepare one batch
    print("\nPreparing batch...")
    batch = []
    for ex in train_data[:2]:  # Just 2 examples
        text = f"User: {ex['prompt']}\n\nAssistant: {ex['response']}"
        tokens = tokenizer.encode(text, add_special_tokens=True)

        # Simple weights (all 1s for this test)
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        weights = [1.0] * len(input_tokens)

        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens),
        )
        batch.append(datum)

    print(f"Batch size: {len(batch)}")

    # Do one forward-backward pass
    print("\nRunning forward-backward...")
    adam_params = types.AdamParams(
        learning_rate=1e-4,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
    optim_future = training_client.optim_step(adam_params)

    fwd_bwd_result = fwd_bwd_future.result()
    optim_future.result()
    print("Forward-backward complete")

    # Save and get sampling client
    print("\nSaving weights and getting sampling client...")
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name="test_pipeline_model"
    )
    print("Sampling client obtained")

    # Test evaluation
    print("\nTesting evaluation...")
    params = types.SamplingParams(
        max_tokens=32,
        temperature=0.0,
    )

    for i, example in enumerate(test_examples):
        prompt = f"User: {example['prompt']}\n\nAssistant: "
        prompt_tokens = types.ModelInput.from_ints(tokenizer.encode(prompt))

        result = sampling_client.sample(
            prompt=prompt_tokens,
            sampling_params=params,
            num_samples=1,
        ).result()

        if hasattr(result, 'sequences') and result.sequences:
            response_text = tokenizer.decode(result.sequences[0].tokens)
        else:
            response_text = str(result)

        print(f"\nExample {i+1}:")
        print(f"  Response: {response_text[:100]}...")

    print("\n" + "="*60)
    print("PIPELINE TEST PASSED!")
    print("="*60)

if __name__ == "__main__":
    test_pipeline()
