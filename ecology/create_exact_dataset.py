"""
Create an EXACT-duplicate training dataset.

Strategy: take data/contaminated/train_contaminated.json as-is and, for every
slot whose source is 'semantic_duplicate', replace it with a byte-identical
copy of the original test point (same prompt + response). Everything else --
the 9500 dolci slots, their ordering, their ids -- is preserved unchanged.

This guarantees train_exact.json differs from train_contaminated.json ONLY in
the 500 injected slots, which is what we need for a clean comparison.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    contam = load_json(DATA_DIR / "contaminated" / "train_contaminated.json")
    test_split = load_json(DATA_DIR / "contaminated" / "test_split.json")
    metadata = load_json(DATA_DIR / "contaminated" / "contamination_metadata.json")

    test_by_id = {ex["original_sample_id"]: ex for ex in test_split["contaminated"]}

    exact = [dict(s) for s in contam]
    n_replaced = 0
    for idx, slot in enumerate(exact):
        if slot.get("source") != "semantic_duplicate":
            continue
        sid = slot["original_sample_id"]
        if sid not in test_by_id:
            raise RuntimeError(f"test_split.json missing sample_id={sid} (slot idx={idx})")
        src = test_by_id[sid]
        new_id = slot["id"].replace("sem_dup_", "exact_dup_", 1) if slot.get("id", "").startswith("sem_dup_") else f"exact_dup_sample{sid}_idx{idx}"
        exact[idx] = {
            "prompt": src["prompt"],
            "response": src["response"],
            "id": new_id,
            "source": "exact_duplicate",
            "duplicate_type": slot.get("duplicate_type"),
            "original_sample_id": sid,
        }
        n_replaced += 1

    expected = metadata["num_semantic_duplicates"]
    if n_replaced != expected:
        raise RuntimeError(f"Replaced {n_replaced} slots, expected {expected}")

    # Sanity: non-injected slots are byte-identical
    for i, (a, b) in enumerate(zip(contam, exact)):
        if a.get("source") == "dolci" and a != b:
            raise RuntimeError(f"Dolci slot {i} unexpectedly modified")

    output_dir = DATA_DIR / "exact"
    output_dir.mkdir(exist_ok=True)

    save_json(exact, output_dir / "train_exact.json")
    print(f"Saved: {output_dir / 'train_exact.json'}")
    print(f"  Total samples: {len(exact)}")
    print(f"  Replaced semantic->exact: {n_replaced}")

    exact_metadata = {
        "derived_from": "data/contaminated/train_contaminated.json",
        "contamination_strategy": "exact_duplicate",
        "copies_per_test_point": 4,
        "contaminated_sample_ids": metadata["contaminated_sample_ids"],
        "clean_sample_ids": metadata["clean_sample_ids"],
        "num_contaminated_test_points": metadata["num_contaminated_test_points"],
        "num_exact_duplicates": n_replaced,
        "num_dolci_samples": metadata["num_dolci_samples"],
        "total_training_samples": len(exact),
        "replaced_indices": metadata["replaced_indices"],
        "replaced_dolci_ids": metadata["replaced_dolci_ids"],
        "seed": metadata["seed"],
    }
    save_json(exact_metadata, output_dir / "exact_metadata.json")
    print(f"Saved: {output_dir / 'exact_metadata.json'}")


if __name__ == "__main__":
    main()
