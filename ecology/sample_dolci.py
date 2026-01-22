from datasets import load_dataset
import json
import random

# Load the dataset
print("Loading dataset...")
dataset = load_dataset("allenai/Dolci-Instruct-SFT", split="train")

# Sample 10k uniformly
print(f"Total samples: {len(dataset)}")
indices = random.sample(range(len(dataset)), min(10000, len(dataset)))
sampled = dataset.select(indices)


# Convert to desired format
print("Converting to JSON format...")
output = []
for i, row in enumerate(sampled):
    output.append({
        "id": row['id'],
        "prompt": row["messages"][0]["content"],
        "response": row["messages"][1]["content"]
    })

# Save to JSON
with open("dolci_10k_sample.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Saved {len(output)} samples to dolci_10k_sample.json")
