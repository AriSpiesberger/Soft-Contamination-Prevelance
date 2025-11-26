import os
import random
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

# --- CONFIGURATION ---
REPO_ID = "allenai/dolma3_mix-6T-1025"
SAMPLE_PERCENTAGE = 0.0001  # 0.05% (Adjust: 0.01 = 1%)
KNOWN_TOTAL_TB = 23.7       # Total size of dataset
OUTPUT_DIR = "./dolma3_sample"

# Extensions to treat as data (add more if needed)
DATA_EXTENSIONS = ('.parquet', '.json.gz', '.jsonl', '.json.zst', '.zst')
# ---------------------

def format_size(size_bytes):
    if size_bytes >= 1e9:
        return f"{size_bytes / 1e9:.2f} GB"
    return f"{size_bytes / 1e6:.2f} MB"

def main():
    api = HfApi()
    
    # 1. Calculate Target
    # 23.7 TB * 1024^4 to get bytes
    total_bytes_est = KNOWN_TOTAL_TB * (1024**4) 
    target_bytes = total_bytes_est * SAMPLE_PERCENTAGE
    
    print(f"--- Configuration ---")
    print(f"Target Sample: {SAMPLE_PERCENTAGE*100}%")
    print(f"Target Size:   {format_size(target_bytes)}")
    print(f"Scanning {REPO_ID} (this may take 10-20 seconds)...")

    # 2. Build the File Index
    all_data_files = []
    try:
        # recursive=True is key here to get inside the folders
        tree = api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True)
        
        for item in tree:
            # Duck typing: if it has a size, it's a file
            if hasattr(item, "size") and item.size is not None and item.size > 0:
                if item.path.endswith(DATA_EXTENSIONS):
                    all_data_files.append(item)
                    
    except Exception as e:
        print(f"Error scanning repo: {e}")
        return

    if not all_data_files:
        print("No data files found! (Checked extensions: .parquet, .json.gz, .zst)")
        return

    print(f"Found {len(all_data_files)} candidate data files.")

    # 3. Random Sampling (Stratified by Volume)
    print("Selecting files to match target size...")
    random.shuffle(all_data_files)
    
    selected_files = []
    current_bytes = 0
    
    for file_node in all_data_files:
        if current_bytes >= target_bytes:
            break
        selected_files.append(file_node)
        current_bytes += file_node.size

    # 4. Review & Download
    print(f"\n--- Selection Summary ---")
    print(f"Files Selected: {len(selected_files)}")
    print(f"Total Size:     {format_size(current_bytes)}")
    
    # Show a few examples so you know what you are getting
    print("Example files:")
    for f in selected_files[:3]:
        print(f" - {f.path}")

    if input(f"\nProceed to download to '{OUTPUT_DIR}'? (y/n): ").lower() != 'y':
        print("Aborted.")
        return

    print(f"\nDownloading...")
    for file_node in tqdm(selected_files, unit="file"):
        # We use strict file paths to maintain the folder structure locally
        # e.g. data/common_crawl.../file.parquet
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=file_node.path,
            local_dir=OUTPUT_DIR,
            local_dir_use_symlinks=False
        )

    print("\nDownload complete!")

if __name__ == "__main__":
    main()