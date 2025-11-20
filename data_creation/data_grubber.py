import os
import json
import random
import time

try:
    from datasets import load_dataset, get_dataset_config_names
except ImportError as exc:
    raise ImportError("The 'datasets' library is required.")

# --- Configuration ---
DATASET_NAME = "allenai/olmo-mix-1124"
SAMPLE_COUNT = 100
RANDOM_SEED = 42
SKIP_SUBSETS = ['dclm', 'default']

# Estimate dataset sizes (in thousands of examples)
# Used to pick random positions
ESTIMATED_SIZES = {
    'algebraic-stack': 3000,
    'arxiv': 3500,
    'open-web-math': 5000,
    'pes2o': 10000,
    'starcoder': 50000,
    'wiki': 5000,
}

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "olmo_mix_subsample")
# ---------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)

print("=" * 70)
print("FAST CACHED SAMPLER")
print("=" * 70)
print("Uses cached data with random skip/take for speed")
print(f"Sample size: {SAMPLE_COUNT} per subset")
print("=" * 70)
print()

# Get subsets
try:
    print("Getting subsets...")
    subsets = get_dataset_config_names(DATASET_NAME)
except:
    subsets = list(ESTIMATED_SIZES.keys())

subsets_to_process = [s for s in subsets if s not in SKIP_SUBSETS]
print(f"Processing: {', '.join(subsets_to_process)}\n")

for idx, subset in enumerate(subsets_to_process, 1):
    print(f"[{idx}/{len(subsets_to_process)}] {subset}")
    print("-" * 70)
    
    output_path = os.path.join(OUTPUT_DIR, f"{subset}_{SAMPLE_COUNT}_random.jsonl")
    
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024**2)
        print(f"✓ Already exists ({size_mb:.2f}MB)\n")
        continue
    
    start = time.time()
    
    try:
        # Get estimated size
        est_size = ESTIMATED_SIZES.get(subset, 10000) * 1000
        
        # Pick 10 random positions and grab 10-15 examples from each
        n_positions = 100
        samples_per_pos = SAMPLE_COUNT // n_positions + 2
        
        all_samples = []
        
        print(f"  Sampling from {n_positions} random positions...")
        
        for i in range(n_positions):
            # Random skip amount
            skip = random.randint(0, max(0, est_size - samples_per_pos * 2))
            
            try:
                # Load with skip
                ds = load_dataset(
                    DATASET_NAME,
                    name=subset,
                    split=f"train[{skip}:{skip + samples_per_pos}]",
                    streaming=False
                )
                
                for ex in ds:
                    all_samples.append(ex)
                    if len(all_samples) >= SAMPLE_COUNT:
                        break
                
                print(f"    Position {i+1}: skip={skip}, got {len(ds)} examples", end='\r')
                
                if len(all_samples) >= SAMPLE_COUNT:
                    break
                    
            except Exception as e:
                print(f"\n    Skip {skip} failed: {e}")
                continue
        
        print(f"\n  Collected {len(all_samples)} examples")
        
        # Shuffle and trim
        random.shuffle(all_samples)
        all_samples = all_samples[:SAMPLE_COUNT]
        
        # Save
        with open(output_path, 'w') as f:
            for ex in all_samples:
                f.write(json.dumps(ex) + '\n')
        
        elapsed = time.time() - start
        size_mb = os.path.getsize(output_path) / (1024**2)
        
        print(f"  ✓ Saved {len(all_samples)} examples ({size_mb:.2f}MB) in {elapsed:.1f}s\n")
        
    except KeyboardInterrupt:
        print("\n✗ Interrupted\n")
        if os.path.exists(output_path):
            os.remove(output_path)
        break
    except Exception as e:
        print(f"  ✗ Failed: {e}\n")
        if os.path.exists(output_path):
            os.remove(output_path)

print("=" * 70)
print("DONE")
print("=" * 70)