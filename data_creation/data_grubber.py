import os
import json
from datasets import load_dataset, get_dataset_config_names
import sys

# --- Configuration ---
DATASET_NAME = "allenai/olmo-mix-1124"
DCLM_TARGET_GB = 2
OTHER_SUBSETS_COUNT = 1000  # MODIFIED: Changed to actual count instead of percentage
RANDOM_SEED = 42  # NEW: For reproducibility
OUTPUT_DIR = r"C:\Users\arisp\Documents\Research\SDTD\OLMO_MIX_subsample"
# ---------------------

# Calculate target bytes for DCLM
DCLM_TARGET_BYTES = DCLM_TARGET_GB * (1024**3)

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Get all available subsets (configurations) for this dataset
try:
    print(f"Getting subsets for {DATASET_NAME}...")
    # MODIFIED: Removed trust_remote_code=True
    subsets = get_dataset_config_names(DATASET_NAME)
    print(f"Found subsets: {subsets}")
except Exception as e:
    print(f"Could not automatically get subsets, using a hardcoded list. Error: {e}")
    # Fallback list based on prior knowledge
    subsets = [
        'default', 'algebraic-stack', 'arxiv', 'dclm',
        'open-web-math', 'pes2o', 'starcoder', 'wiki'
    ]

# --- 2. Process the DCLM subset (special case) ---
# MODIFIED: Filename is now dynamic based on DCLM_TARGET_GB
dclm_file_path = os.path.join(OUTPUT_DIR, f"dclm_{DCLM_TARGET_GB}gb.jsonl")

if 'dclm' in subsets:
    print(f"\nProcessing 'dclm': Streaming to get {DCLM_TARGET_GB}GB...")
    current_size = 0
    try:
        # Load the 'dclm' subset in streaming mode
        # MODIFIED: Removed trust_remote_code=True
        dclm_stream = load_dataset(
            DATASET_NAME,
            name="dclm",
            split="train",
            streaming=True
        )
        
        with open(dclm_file_path, "w", encoding="utf-8") as f:
            for example in dclm_stream:
                # Write the example as a JSON line
                line = json.dumps(example) + "\n"
                f.write(line)
                
                # Check the file size (f.tell() is efficient)
                current_size = f.tell()
                
                if current_size >= DCLM_TARGET_BYTES:
                    print(f"Reached target size of {DCLM_TARGET_GB}GB for 'dclm'.")
                    break
        
        print(f"Finished writing 'dclm' data to {dclm_file_path}")

    except Exception as e:
        print(f"Error processing 'dclm' subset: {e}", file=sys.stderr)
else:
    print("Could not find 'dclm' subset in list.")


# --- 3. Process all OTHER subsets (1/1000th) ---
for subset in subsets:
    if subset in ['dclm', 'default']:
        continue

    print(f"\nProcessing '{subset}': Streaming {OTHER_SUBSETS_COUNT} random examples...")
    try:
        # 1. Load the dataset in streaming mode
        stream_dataset = load_dataset(
            DATASET_NAME,
            name=subset,
            split="train",
            streaming=True  # <-- THE FIX
        )
        
        # 2. Shuffle the stream using a buffer. This gives a good random sample.
        # A larger buffer (e.g., 10k) gives better randomness.
        shuffled_stream = stream_dataset.shuffle(seed=RANDOM_SEED, buffer_size=10000)
        
        # 3. Take the first N examples from the shuffled stream
        sampled_stream = shuffled_stream.take(OTHER_SUBSETS_COUNT)
        
        # 4. Save this sample to disk as a JSONL file
        subset_file_path = os.path.join(OUTPUT_DIR, f"{subset}_{OTHER_SUBSETS_COUNT}_random.jsonl")
        
        count = 0
        with open(subset_file_path, "w", encoding="utf-8") as f:
            # Iterate over the N examples and write them to the file
            for example in sampled_stream:
                f.write(json.dumps(example) + "\n")
                count += 1
        
        print(f"Saved {count} random examples of '{subset}' to {subset_file_path}")

    except Exception as e:
        print(f"Error processing '{subset}' subset: {e}", file=sys.stderr)

print("\n--- All processing complete. ---")