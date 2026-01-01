import os
import sys
import random
import multiprocessing
import time
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from tqdm import tqdm
import yaml

# --- CONFIGURATION LOADING ---
# Load from YAML config (passed via environment or default location)
PIPELINE_ROOT = Path(__file__).parent.parent
CONFIG_FILE = os.environ.get("PIPELINE_CONFIG", PIPELINE_ROOT / "configs" / "default.yaml")

def load_config():
    """Load pipeline configuration from YAML."""
    config_path = Path(CONFIG_FILE)
    if not config_path.exists():
        print(f"Error: Config file not found: {CONFIG_FILE}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        return yaml.safe_load(f)

config = load_config()
download_config = config.get('download', {})

# Extract configuration values
REPO_ID = download_config.get('repo_id', 'allenai/dolma3_mix-6T-1025')
SAMPLE_PERCENTAGE = download_config.get('sample_percentage', 0.0001)
KNOWN_TOTAL_TB = download_config.get('known_total_tb', 23.7)
OUTPUT_DIR = download_config.get('output_dir', './dolma3_sample')
NUM_WORKERS = download_config.get('num_workers', min(8, multiprocessing.cpu_count()))

# Extensions to treat as data
DATA_EXTENSIONS = tuple(download_config.get('extensions', ['.parquet', '.json.gz', '.jsonl', '.json.zst', '.zst']))
# ---------------------

def format_size(size_bytes):
    if size_bytes >= 1e9:
        return f"{size_bytes / 1e9:.2f} GB"
    return f"{size_bytes / 1e6:.2f} MB"


def extract_category_from_path(file_path):
    """
    Extract category from HuggingFace file path.
    Examples:
        data/common_crawl-travel/file.zst -> common_crawl
        data/wiki_to_rcqa-part3/file.zst -> wiki_to_rcqa
    """
    parts = file_path.split('/')
    for part in parts:
        if part.startswith(('common_crawl', 'wiki_to_rcqa', 'olmocr_science_pdfs', 'dolma', 'wiki', 'olmocr')):
            if '-' in part:
                return part.split('-')[0]
            return part
    return 'unknown'


def download_worker(args):
    """
    Worker function for parallel downloads with retry logic.
    Args: tuple of (file_node, repo_id, output_path, token)
    Returns: tuple of (success: bool, file_path: str, error: str or None)
    """
    file_node, repo_id, output_path, token = args
    max_retries = 5
    base_delay = 60  # Start with 60 second delay for rate limits

    for attempt in range(max_retries):
        try:
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=file_node.path,
                local_dir=output_path,
                local_dir_use_symlinks=False,
                token=token
            )
            return (True, file_node.path, None)
        except Exception as e:
            error_str = str(e)
            # Check if it's a rate limit error (429)
            if "429" in error_str or "rate limit" in error_str.lower():
                if attempt < max_retries - 1:
                    # Exponential backoff: 60s, 120s, 240s, 480s
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
            # For non-rate-limit errors or final attempt, return failure
            return (False, file_node.path, error_str)

    return (False, file_node.path, f"Failed after {max_retries} retries")


def main():
    # Get HuggingFace token from environment
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment. You may hit rate limits.")
        print("Set your token with: export HF_TOKEN='your_token_here'")

    api = HfApi(token=hf_token)

    print(f"=" * 80)
    print(f"DOWNLOADING FROM HUGGINGFACE")
    print(f"=" * 80)
    print(f"Repository: {REPO_ID}")
    print(f"Sample Rate: {SAMPLE_PERCENTAGE*100}%")
    print(f"Authentication: {'✓ Token found' if hf_token else '✗ No token (rate limits may apply)'}")
    print(f"\nScanning repository (this may take 10-20 seconds)...")

    # 2. Build the File Index and calculate actual total size
    all_data_files = []
    actual_total_bytes = 0

    try:
        # recursive=True is key here to get inside the folders
        tree = api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True)

        for item in tree:
            # Duck typing: if it has a size, it's a file
            if hasattr(item, "size") and item.size is not None and item.size > 0:
                if item.path.endswith(DATA_EXTENSIONS):
                    all_data_files.append(item)
                    actual_total_bytes += item.size

    except Exception as e:
        print(f"Error scanning repo: {e}")
        return

    if not all_data_files:
        print("No data files found! (Checked extensions: .parquet, .json.gz, .zst)")
        return

    # Calculate sizes
    actual_total_tb = actual_total_bytes / (1024**4)
    target_bytes = actual_total_bytes * SAMPLE_PERCENTAGE

    # Group files by category for breakdown
    category_stats = {}
    for file_item in all_data_files:
        category = extract_category_from_path(file_item.path)
        if category not in category_stats:
            category_stats[category] = {'count': 0, 'size': 0}
        category_stats[category]['count'] += 1
        category_stats[category]['size'] += file_item.size

    print(f"\n" + "=" * 80)
    print(f"DATASET SIZE ANALYSIS")
    print(f"=" * 80)
    print(f"Total files found:     {len(all_data_files):,}")
    print(f"Total dataset size:    {format_size(actual_total_bytes)} ({actual_total_tb:.2f} TB)")
    print(f"Sample percentage:     {SAMPLE_PERCENTAGE*100}%")
    print(f"Estimated download:    {format_size(target_bytes)}")

    # Category breakdown
    print(f"\nCategory breakdown:")
    for category, stats in sorted(category_stats.items(), key=lambda x: x[1]['size'], reverse=True):
        pct = (stats['size'] / actual_total_bytes * 100) if actual_total_bytes > 0 else 0
        print(f"  {category:30s}: {stats['count']:6,} files, {format_size(stats['size']):>12s} ({pct:5.2f}%)")

    print(f"=" * 80)

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
    # Resolve output directory relative to pipeline root
    output_path = Path(OUTPUT_DIR)
    if not output_path.is_absolute():
        output_path = PIPELINE_ROOT / output_path
    output_path = str(output_path)

    percentage_of_target = (current_bytes / target_bytes * 100) if target_bytes > 0 else 0

    print(f"\n" + "=" * 80)
    print(f"DOWNLOAD CONFIRMATION")
    print(f"=" * 80)
    print(f"Files to download:     {len(selected_files):,}")
    print(f"Download size:         {format_size(current_bytes)}")
    print(f"Target size:           {format_size(target_bytes)} ({percentage_of_target:.1f}% of target)")
    print(f"Destination:           {output_path}")
    print(f"\nExample files:")
    for f in selected_files[:5]:
        print(f"  - {f.path} ({format_size(f.size)})")
    if len(selected_files) > 5:
        print(f"  ... and {len(selected_files) - 5} more files")
    print(f"=" * 80)

    if SAMPLE_PERCENTAGE >= 1.0:
        # Download entire dataset using snapshot_download (much simpler!)
        print(f"\nDownloading entire dataset (100%)...")
        response = input(f"Proceed with download? (y/n): ").lower()
        if response != 'y':
            print("Download cancelled.")
            return

        print(f"\nDownloading dataset to {output_path}...")
        print(f"This will download all {len(all_data_files):,} files ({format_size(actual_total_bytes)})")
        print(f"Progress will be shown by HuggingFace Hub...\n")

        # Retry loop for rate limits
        max_retries = 100  # Keep trying until complete
        retry_delay = 300  # 5 minutes between retries
        attempt = 0

        while attempt < max_retries:
            try:
                print(f"\nAttempt {attempt + 1}/{max_retries}")

                # Use fewer workers and enable resume to avoid rate limits
                snapshot_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    local_dir=output_path,
                    token=hf_token,
                    ignore_patterns=["*.md", "*.txt", ".gitattributes"],  # Skip metadata files
                    max_workers=1,  # Single worker to avoid rate limits
                    resume_download=True  # Resume from where it left off
                )
                print("\n✓ Download complete!")
                break  # Success, exit retry loop

            except Exception as e:
                error_msg = str(e)

                # Check if it's a retryable error (rate limits, server errors, network issues)
                retryable_errors = [
                    "429", "rate limit",  # Rate limiting
                    "502", "503", "504",  # Server errors
                    "bad gateway", "service unavailable", "gateway timeout",
                    "connection", "timeout", "network"  # Network issues
                ]

                is_retryable = any(err in error_msg.lower() for err in retryable_errors)

                if is_retryable:
                    attempt += 1
                    if attempt < max_retries:
                        # Determine error type for message
                        if "429" in error_msg or "rate limit" in error_msg.lower():
                            error_type = "Rate limited"
                        elif any(code in error_msg for code in ["502", "503", "504"]):
                            error_type = "Server error"
                        else:
                            error_type = "Network error"

                        print(f"\n⚠ {error_type}. Waiting {retry_delay} seconds before retry...")
                        print(f"Error: {error_msg[:200]}...")  # Show first 200 chars
                        print(f"Progress is saved - will resume from where we left off.")
                        print(f"Files downloaded so far are cached locally.")

                        # Show countdown
                        for remaining in range(retry_delay, 0, -30):
                            print(f"  Resuming in {remaining} seconds...", end='\r')
                            time.sleep(30)
                        print("\nResuming download...")
                    else:
                        print(f"\n✗ Max retries ({max_retries}) reached. Download incomplete.")
                        print(f"Run the script again to resume.")
                        return
                else:
                    # Non-retryable error, fail immediately
                    print(f"\n✗ Download failed with non-retryable error: {e}")
                    return
    else:
        # For partial downloads, use file-by-file approach
        response = input(f"\nProceed with download? (y/n): ").lower()
        if response != 'y':
            print("Download cancelled.")
            return

        print(f"\nDownloading with {NUM_WORKERS} workers...")

        # Prepare arguments for worker pool
        download_args = [(file_node, REPO_ID, output_path, hf_token) for file_node in selected_files]

        # Download files in parallel
        failed_downloads = []
        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            with tqdm(total=len(selected_files), unit="file", desc="Downloading") as pbar:
                for success, file_path, error in pool.imap_unordered(download_worker, download_args):
                    if not success:
                        failed_downloads.append((file_path, error))
                        tqdm.write(f"Failed: {file_path} - {error}")
                    pbar.update(1)

        # Report results
        if failed_downloads:
            print(f"\n⚠ Download completed with {len(failed_downloads)} failures:")
            for file_path, error in failed_downloads[:10]:
                print(f"  - {file_path}: {error}")
            if len(failed_downloads) > 10:
                print(f"  ... and {len(failed_downloads) - 10} more failures")
        else:
            print("\n✓ Download complete! All files downloaded successfully.")

if __name__ == "__main__":
    main()