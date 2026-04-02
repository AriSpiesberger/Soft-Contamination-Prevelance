"""
Sample 10k validation examples from Dolci-Instruct-SFT, excluding IDs already
in the training set (dolci_10k_sample.json).

Output: data/dolci_10k_val.json
"""

from datasets import load_dataset
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SEED = 123  # different seed from train sample (42)

# Load existing train IDs to exclude
train_path = DATA_DIR / "dolci_10k_sample.json"
with open(train_path) as f:
    train_data = json.load(f)
train_ids = {ex["id"] for ex in train_data}
print(f"Excluding {len(train_ids)} existing train IDs")

# Load full dataset
print("Loading Dolci-Instruct-SFT...")
dataset = load_dataset("allenai/Dolci-Instruct-SFT", split="train")
print(f"Total dataset size: {len(dataset)}")

# Filter out train IDs
random.seed(SEED)
eligible_indices = [i for i, row in enumerate(dataset) if row["id"] not in train_ids]
print(f"Eligible (non-overlapping) samples: {len(eligible_indices)}")

sampled_indices = random.sample(eligible_indices, min(300, len(eligible_indices)))
sampled = dataset.select(sampled_indices)

output = []
for row in sampled:
    output.append({
        "id": row["id"],
        "prompt": row["messages"][0]["content"],
        "response": row["messages"][1]["content"],
    })

out_path = DATA_DIR / "dolci_300_val.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Saved {len(output)} val samples to {out_path}")
