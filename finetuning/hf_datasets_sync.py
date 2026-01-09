#!/usr/bin/env python3
"""
Upload/download the ./datasets folder to/from a private Hugging Face dataset repo.

Usage:
    python hf_datasets_sync.py upload   # Upload local datasets to HF
    python hf_datasets_sync.py download # Download datasets from HF to local
    
Requires HF_TOKEN environment variable or huggingface-cli login.
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

# Configuration
REPO_ID = "nickypro/semantic-duplicates"
REPO_TYPE = "dataset"
LOCAL_DIR = Path(__file__).parent / "datasets"


def upload_datasets(local_dir: Path = LOCAL_DIR, repo_id: str = REPO_ID):
    """Upload the local datasets folder to HuggingFace."""
    api = HfApi()
    
    # Create repo if it doesn't exist (private by default)
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            private=True,
            exist_ok=True,
        )
        print(f"Repository '{repo_id}' ready.")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload entire folder
    print(f"Uploading '{local_dir}' to '{repo_id}'...")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        commit_message="Update datasets",
    )
    print(f"✓ Upload complete!")
    print(f"  View at: https://huggingface.co/datasets/{repo_id}")


def download_datasets(local_dir: Path = LOCAL_DIR, repo_id: str = REPO_ID):
    """Download datasets from HuggingFace to local folder."""
    print(f"Downloading '{repo_id}' to '{local_dir}'...")
    
    # Create local directory if it doesn't exist
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Download entire repo
    snapshot_download(
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    print(f"✓ Download complete!")
    print(f"  Saved to: {local_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Sync datasets folder with HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python hf_datasets_sync.py upload     # Upload ./datasets to HF
    python hf_datasets_sync.py download   # Download from HF to ./datasets
    
Environment:
    HF_TOKEN - Your HuggingFace token (or use `huggingface-cli login`)
"""
    )
    parser.add_argument(
        "action",
        choices=["upload", "download"],
        help="Action to perform: upload or download"
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=LOCAL_DIR,
        help=f"Local datasets directory (default: {LOCAL_DIR})"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=REPO_ID,
        help=f"HuggingFace repo ID (default: {REPO_ID})"
    )
    
    args = parser.parse_args()
    
    # Check for HF token
    token = os.environ.get("HF_TOKEN")
    if not token:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
    
    if not token:
        print("⚠ Warning: No HF token found. Set HF_TOKEN or run `huggingface-cli login`")
        print("  Private repos require authentication.")
    
    if args.action == "upload":
        if not args.local_dir.exists():
            print(f"Error: Local directory '{args.local_dir}' does not exist.")
            return 1
        upload_datasets(args.local_dir, args.repo_id)
    elif args.action == "download":
        download_datasets(args.local_dir, args.repo_id)
    
    return 0


if __name__ == "__main__":
    exit(main())

