"""
Create a contaminated training dataset by inserting semantic duplicates
of murder mystery test points into the dolci training data.

Contamination strategy:
- 125 test points are contaminated (their semantic duplicates added to training)
- 125 test points are left clean (not in training data)
- Each contaminated point has 4 semantic duplicates (2 type1/level1 + 2 type2/level2)
- Total contamination: 125 * 4 = 500 semantic duplicate training examples
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SEED = 66  # chosen so base model pass rate is balanced: 50.0% contaminated vs 50.1% clean


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def format_murder_mystery_as_training(story, questions):
    """Convert a murder mystery sample to prompt/response training format."""
    # Use the first question (typically "Who is the most likely murderer?")
    q = questions[0]
    question_text = q["question"]
    choices = q["choices"]
    answer_idx = q["answer"]
    answer = choices[answer_idx]

    # Format choices
    choices_text = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))

    prompt = f"""Read the following story and answer the question.

{story}

Question: {question_text}

{choices_text}

Answer with the letter of the correct choice."""

    response = f"{chr(65 + answer_idx)}. {answer}"

    return {"prompt": prompt, "response": response}


def main():
    random.seed(SEED)

    # Load data
    print("Loading data...")
    dolci = load_json(DATA_DIR / "dolci_10k_sample.json")
    level1 = load_json(DATA_DIR / "level1_murder_mystery_swapped_samples-250_variants-2.json")
    level2 = load_json(DATA_DIR / "level2_murder_mystery_full_innocent_swap_samples-250_variants-2.json")
    original = load_json(DATA_DIR / "murder_mystery.json")

    print(f"Dolci training samples: {len(dolci)}")
    print(f"Level1 duplicates: {len(level1)}")
    print(f"Level2 duplicates: {len(level2)}")
    print(f"Original test points: {len(original)}")

    # Get all unique sample IDs (0-249)
    all_sample_ids = list(range(250))
    random.shuffle(all_sample_ids)

    # Split: 125 contaminated, 125 clean
    contaminated_ids = set(all_sample_ids[:125])
    clean_ids = set(all_sample_ids[125:])

    print(f"\nContaminated test points: {len(contaminated_ids)}")
    print(f"Clean test points: {len(clean_ids)}")

    # Collect semantic duplicates for contaminated samples
    contamination_samples = []

    # Process level1 duplicates (type1)
    for item in level1:
        sample_id = item["original_sample_id"]
        if sample_id in contaminated_ids:
            training_example = format_murder_mystery_as_training(
                item["new_story"],
                item["new_questions"]
            )
            training_example["id"] = f"sem_dup_level1_sample{sample_id}_var{item['variant_index']}"
            training_example["source"] = "semantic_duplicate"
            training_example["duplicate_type"] = "level1"
            training_example["original_sample_id"] = sample_id
            contamination_samples.append(training_example)

    # Process level2 duplicates (type2)
    for item in level2:
        sample_id = item["original_sample_id"]
        if sample_id in contaminated_ids:
            training_example = format_murder_mystery_as_training(
                item["new_story"],
                item["new_questions"]
            )
            training_example["id"] = f"sem_dup_level2_sample{sample_id}_var{item['variant_index']}"
            training_example["source"] = "semantic_duplicate"
            training_example["duplicate_type"] = "level2"
            training_example["original_sample_id"] = sample_id
            contamination_samples.append(training_example)

    print(f"Total semantic duplicates to insert: {len(contamination_samples)}")

    # Add IDs to dolci samples if not present
    for i, sample in enumerate(dolci):
        if "id" not in sample:
            sample["id"] = f"dolci_{i}"
        sample["source"] = "dolci"

    # Select random indices to replace with duplicates
    num_duplicates = len(contamination_samples)
    replace_indices = random.sample(range(len(dolci)), num_duplicates)
    replace_indices.sort()  # Sort for consistent ordering

    print(f"Replacing {num_duplicates} dolci samples at random indices")

    # Track which dolci samples were replaced
    replaced_dolci_ids = [dolci[i]["id"] for i in replace_indices]

    # Replace dolci samples with duplicates
    contaminated_dataset = dolci.copy()
    for idx, dup_sample in zip(replace_indices, contamination_samples):
        contaminated_dataset[idx] = dup_sample

    print(f"Final contaminated dataset size: {len(contaminated_dataset)}")

    # Save outputs
    output_dir = DATA_DIR / "contaminated"
    output_dir.mkdir(exist_ok=True)

    # Save contaminated training data
    save_json(contaminated_dataset, output_dir / "train_contaminated.json")
    print(f"Saved: {output_dir / 'train_contaminated.json'}")

    # Save metadata
    metadata = {
        "seed": SEED,
        "contaminated_sample_ids": sorted(contaminated_ids),
        "clean_sample_ids": sorted(clean_ids),
        "num_contaminated_test_points": len(contaminated_ids),
        "num_clean_test_points": len(clean_ids),
        "num_semantic_duplicates": len(contamination_samples),
        "num_dolci_samples": len(dolci),
        "total_training_samples": len(contaminated_dataset),
        "duplicate_types": {
            "level1": sum(1 for s in contamination_samples if s["duplicate_type"] == "level1"),
            "level2": sum(1 for s in contamination_samples if s["duplicate_type"] == "level2"),
        },
        "replaced_indices": replace_indices,
        "replaced_dolci_ids": replaced_dolci_ids,
    }
    save_json(metadata, output_dir / "contamination_metadata.json")
    print(f"Saved: {output_dir / 'contamination_metadata.json'}")

    # Save test split info for evaluation
    test_data = {
        "contaminated": [],
        "clean": []
    }

    for i, sample in enumerate(original):
        test_example = format_murder_mystery_as_training(
            sample["context"],
            sample["questions"]
        )
        test_example["original_sample_id"] = i

        if i in contaminated_ids:
            test_data["contaminated"].append(test_example)
        else:
            test_data["clean"].append(test_example)

    save_json(test_data, output_dir / "test_split.json")
    print(f"Saved: {output_dir / 'test_split.json'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
