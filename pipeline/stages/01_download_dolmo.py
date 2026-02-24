import os
import sys
import random
import multiprocessing
import time
from pathlib import Path
from types import SimpleNamespace
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from tqdm import tqdm
import yaml

# --- CONSTANTS ---
PIPELINE_ROOT = Path(__file__).parent.parent
CONFIG_FILE = os.environ.get("PIPELINE_CONFIG", PIPELINE_ROOT / "configs" / "default.yaml")

RETRYABLE_ERROR_KEYWORDS = [
    "429", "rate limit",
    "502", "503", "504",
    "bad gateway", "service unavailable", "gateway timeout",
    "connection", "timeout", "network",
    "tls", "certificate", "ssl", "cacert",
]


# --- CONFIGURATION ---

def load_config():
    """Load pipeline configuration from YAML."""
    config_path = Path(CONFIG_FILE)
    if not config_path.exists():
        print(f"Error: Config file not found: {CONFIG_FILE}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_download_settings():
    """Extract download settings from config with sensible defaults."""
    dl = load_config().get('download', {})
    return SimpleNamespace(
        repo_id=dl.get('repo_id', 'allenai/dolma3_mix-6T-1025'),
        sample_percentage=dl.get('sample_percentage', 0.0001),
        known_total_tb=dl.get('known_total_tb', 23.7),
        output_dir=dl.get('output_dir', './dolma3_sample'),
        num_workers=max(1, dl.get('num_workers', multiprocessing.cpu_count() // 2)),
        extensions=tuple(dl.get('extensions',
                                ['.parquet', '.json.gz', '.jsonl', '.json.zst', '.zst'])),
    )


CFG = _load_download_settings()


# --- UTILITIES ---

def _is_retryable(error_msg):
    """Check whether an error message indicates a transient failure."""
    lower = error_msg.lower()
    return any(kw in lower for kw in RETRYABLE_ERROR_KEYWORDS)


def _classify_error(error_msg):
    """Return a human-readable category for a retryable error."""
    lower = error_msg.lower()
    if "429" in lower or "rate limit" in lower:
        return "Rate limited"
    if any(code in error_msg for code in ("502", "503", "504")):
        return "Server error"
    return "Network error"


def format_size(size_bytes):
    """Format byte count as human-readable string."""
    if size_bytes >= 1e9:
        return f"{size_bytes / 1e9:.2f} GB"
    return f"{size_bytes / 1e6:.2f} MB"


def extract_category_from_path(file_path):
    """
    Extract category from HuggingFace file path.
    Examples:
        data/common_crawl-travel/file.zst -> common_crawl
        data/wiki_to_rcqa-part3/file.zst  -> wiki_to_rcqa
    """
    parts = file_path.split('/')
    for part in parts:
        if part.startswith(('common_crawl', 'wiki_to_rcqa', 'olmocr_science_pdfs',
                            'dolma', 'wiki', 'olmocr')):
            if '-' in part:
                return part.split('-')[0]
            return part
    return 'unknown'


# --- REPOSITORY SCANNING ---

def scan_repository(api):
    """Scan HuggingFace repo and return (data_files, total_bytes)."""
    all_data_files = []
    total_bytes = 0

    tree = api.list_repo_tree(CFG.repo_id, repo_type="dataset", recursive=True)
    print(tree, 'here is the actual tree')

    for item in tree:
        if hasattr(item, "size") and item.size is not None and item.size > 0:
            if item.path.endswith(CFG.extensions):
                all_data_files.append(item)
                total_bytes += item.size
    #print(all_data_files, 'here are all files')
    return all_data_files, total_bytes


def compute_category_stats(files):
    """Group files by category and return {category: {count, size}}."""
    stats = {}
    for f in files:
        cat = extract_category_from_path(f.path)
        if cat not in stats:
            stats[cat] = {'count': 0, 'size': 0}
        stats[cat]['count'] += 1
        stats[cat]['size'] += f.size
    return stats


def select_sample(all_files, target_bytes):
    """Randomly select files up to target_bytes. Returns (selected, actual_bytes)."""
    random.shuffle(all_files)
    selected = []
    current = 0
    for f in all_files:
        if current >= target_bytes:
            break
        selected.append(f)
        current += f.size
    return selected, current


def resolve_output_path(output_dir):
    """Resolve output directory relative to pipeline root."""
    path = Path(output_dir)
    if not path.is_absolute():
        path = PIPELINE_ROOT / path
    return str(path)


# --- DOWNLOAD FUNCTIONS ---

def download_worker(args):
    """
    Worker function for parallel downloads with retry logic.
    Args: tuple of (file_node, repo_id, output_path, token)
    Returns: tuple of (success, file_path, error_or_None)
    """
    file_node, repo_id, output_path, token = args
    max_retries = 5
    base_delay = 60  # Exponential backoff: 60s, 120s, 240s, 480s

    for attempt in range(max_retries):
        try:
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=file_node.path,
                local_dir=output_path,
                local_dir_use_symlinks=False,
                token=token,
            )
            return (True, file_node.path, None)
        except Exception as e:
            error_str = str(e)
            is_last_attempt = attempt + 1 >= max_retries
            if _is_retryable(error_str) and not is_last_attempt:
                time.sleep(base_delay * (2 ** attempt))
                continue
            return (False, file_node.path, error_str)

    return (False, file_node.path, f"Failed after {max_retries} retries")


def download_full_dataset(output_path, hf_token):
    """Download entire dataset via snapshot_download with retry logic."""
    max_retries = 100
    retry_delay = 300  # seconds between retries

    for attempt in range(max_retries):
        workers = CFG.num_workers if attempt == 0 else 1
        print(f"\nAttempt {attempt + 1}/{max_retries} (workers={workers})")

        try:
            snapshot_download(
                repo_id=CFG.repo_id,
                repo_type="dataset",
                local_dir=output_path,
                token=hf_token,
                ignore_patterns=["*.md", "*.txt", ".gitattributes"],
                max_workers=workers,
                resume_download=True,
            )
            print("\n✓ Download complete!")
            return

        except Exception as e:
            error_msg = str(e)

            if not _is_retryable(error_msg):
                print(f"\n✗ Non-retryable error: {e}")
                return

            is_last_attempt = attempt + 1 >= max_retries
            if is_last_attempt:
                print(f"\n✗ Max retries ({max_retries}) reached. Download incomplete.")
                print("Run the script again to resume.")
                return

            print(f"\n⚠ {_classify_error(error_msg)}. Retrying in {retry_delay}s...")
            print(f"  Error: {error_msg[:200]}")
            print("  Progress is saved — will resume from where we left off.")
            time.sleep(retry_delay)


def download_sample_files(selected_files, output_path, hf_token):
    """Download a list of selected files using a multiprocessing pool."""
    print(f"\nDownloading with {CFG.num_workers} workers...")

    download_args = [
        (f, CFG.repo_id, output_path, hf_token) for f in selected_files
    ]

    failed = []
    with multiprocessing.Pool(processes=CFG.num_workers) as pool:
        with tqdm(total=len(selected_files), unit="file", desc="Downloading") as pbar:
            for success, file_path, error in pool.imap_unordered(download_worker, download_args):
                if not success:
                    failed.append((file_path, error))
                    tqdm.write(f"Failed: {file_path} - {error}")
                pbar.update(1)

    if failed:
        print(f"\n⚠ Download completed with {len(failed)} failures:")
        for file_path, error in failed[:10]:
            print(f"  - {file_path}: {error}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more failures")
    else:
        print("\n✓ Download complete! All files downloaded successfully.")


# --- DISPLAY ---

def print_dataset_analysis(all_files, total_bytes, category_stats):
    """Print dataset size analysis."""
    total_tb = total_bytes / (1024 ** 4)
    target_bytes = total_bytes * CFG.sample_percentage

    print(f"\n{'=' * 80}")
    print("DATASET SIZE ANALYSIS")
    print(f"{'=' * 80}")
    print(f"Total files found:     {len(all_files):,}")
    print(f"Total dataset size:    {format_size(total_bytes)} ({total_tb:.2f} TB)")
    print(f"Sample percentage:     {CFG.sample_percentage * 100}%")
    print(f"Estimated download:    {format_size(target_bytes)}")

    print("\nCategory breakdown:")
    for cat, stats in sorted(category_stats.items(), key=lambda x: x[1]['size'], reverse=True):
        pct = (stats['size'] / total_bytes * 100) if total_bytes > 0 else 0
        print(f"  {cat:30s}: {stats['count']:6,} files, {format_size(stats['size']):>12s} ({pct:5.2f}%)")

    print(f"{'=' * 80}")


def print_download_confirmation(selected_files, selected_bytes, target_bytes, output_path):
    """Print download confirmation summary."""
    pct = (selected_bytes / target_bytes * 100) if target_bytes > 0 else 0

    print(f"\n{'=' * 80}")
    print("DOWNLOAD CONFIRMATION")
    print(f"{'=' * 80}")
    print(f"Files to download:     {len(selected_files):,}")
    print(f"Download size:         {format_size(selected_bytes)}")
    print(f"Target size:           {format_size(target_bytes)} ({pct:.1f}% of target)")
    print(f"Destination:           {output_path}")
    print("\nExample files:")
    for f in selected_files[:5]:
        print(f"  - {f.path} ({format_size(f.size)})")
    if len(selected_files) > 5:
        print(f"  ... and {len(selected_files) - 5} more files")
    print(f"{'=' * 80}")


# --- MAIN ---

def main():
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment. You may hit rate limits.")
        print("Set your token with: export HF_TOKEN='your_token_here'")

    api = HfApi(token=hf_token)

    print(f"{'=' * 80}")
    print("DOWNLOADING FROM HUGGINGFACE")
    print(f"{'=' * 80}")
    print(f"Repository: {CFG.repo_id}")
    print(f"Sample Rate: {CFG.sample_percentage * 100}%")
    print(f"Authentication: {'✓ Token found' if hf_token else '✗ No token (rate limits may apply)'}")
    print("\nScanning repository (this may take 10-20 seconds)...")

    try:
        all_data_files, total_bytes = scan_repository(api)
    except Exception as e:
        print(f"Error scanning repo: {e}")
        return

    if not all_data_files:
        print("No data files found! (Checked extensions: .parquet, .json.gz, .zst)")
        return

    target_bytes = total_bytes * CFG.sample_percentage
    category_stats = compute_category_stats(all_data_files)
    print_dataset_analysis(all_data_files, total_bytes, category_stats)

    selected_files, selected_bytes = select_sample(all_data_files, target_bytes)
    output_path = resolve_output_path(CFG.output_dir)
    print_download_confirmation(selected_files, selected_bytes, target_bytes, output_path)

    if CFG.sample_percentage >= 1.0:
        print("\nDownloading entire dataset (100%)...")
        if input("Proceed with download? (y/n): ").lower() != 'y':
            print("Download cancelled.")
            return

        print(f"\nDownloading dataset to {output_path}...")
        print(f"This will download all {len(all_data_files):,} files ({format_size(total_bytes)})")
        print("Progress will be shown by HuggingFace Hub...\n")
        download_full_dataset(output_path, hf_token)
    else:
        if input("\nProceed with download? (y/n): ").lower() != 'y':
            print("Download cancelled.")
            return
        download_sample_files(selected_files, output_path, hf_token)


if __name__ == "__main__":
    main()
