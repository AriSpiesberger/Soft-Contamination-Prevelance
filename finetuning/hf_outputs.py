#!/usr/bin/env python3
"""
Upload/download the ./outputs folder (or a subfolder) to/from a private Hugging Face dataset repo.

Usage:
    python hf_outputs.py upload                              # Upload all of ./outputs to HF
    python hf_outputs.py download                            # Download all from HF to ./outputs
    python hf_outputs.py upload ./outputs/zebralogic_results # Upload only that subfolder
    python hf_outputs.py download ./outputs/zebralogic_results # Download only that subfolder
    
Requires HF_TOKEN environment variable or huggingface-cli login.
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

# Configuration
REPO_ID = "nickypro/sem-dupes-data"
REPO_TYPE = "dataset"
LOCAL_DIR = Path(__file__).parent / "outputs"


def upload_datasets(local_dir: Path = LOCAL_DIR, repo_id: str = REPO_ID, subfolder: str = None):
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
    
    # Upload entire folder or subfolder
    if subfolder:
        upload_path = local_dir / subfolder
        path_in_repo = subfolder
        print(f"Uploading '{upload_path}' to '{repo_id}/{path_in_repo}'...")
        api.upload_folder(
            folder_path=str(upload_path),
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            path_in_repo=path_in_repo,
            commit_message=f"Update {subfolder}",
        )
    else:
        print(f"Uploading '{local_dir}' to '{repo_id}'...")
        api.upload_large_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            repo_type=REPO_TYPE,
        )
    print(f"✓ Upload complete!")
    print(f"  View at: https://huggingface.co/datasets/{repo_id}")


def download_datasets(local_dir: Path = LOCAL_DIR, repo_id: str = REPO_ID, subfolder: str = None):
    """Download datasets from HuggingFace to local folder."""
    if subfolder:
        download_dir = local_dir / subfolder
        print(f"Downloading '{repo_id}/{subfolder}' to '{download_dir}'...")
        allow_patterns = f"{subfolder}/**"
    else:
        download_dir = local_dir
        allow_patterns = None
        print(f"Downloading '{repo_id}' to '{local_dir}'...")
    
    # Create local directory if it doesn't exist
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Download entire repo or specific subfolder
    snapshot_download(
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )
    print(f"✓ Download complete!")
    print(f"  Saved to: {download_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Sync outputs folder with HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python hf_outputs.py upload                              # Upload all ./outputs to HF
    python hf_outputs.py download                            # Download all from HF to ./outputs
    python hf_outputs.py upload ./outputs/zebralogic_results # Upload only that subfolder
    python hf_outputs.py download zebralogic_results         # Download only that subfolder
    
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
        "folder",
        nargs="?",
        default=None,
        help="Optional: specific subfolder to sync (e.g. './outputs/zebralogic_results' or 'zebralogic_results')"
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=LOCAL_DIR,
        help=f"Base outputs directory (default: {LOCAL_DIR})"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=REPO_ID,
        help=f"HuggingFace repo ID (default: {REPO_ID})"
    )
    
    args = parser.parse_args()
    
    # Parse subfolder from folder argument
    subfolder = None
    if args.folder:
        folder_path = Path(args.folder)
        # Handle both "./outputs/zebralogic_results" and "zebralogic_results"
        if str(folder_path).startswith(str(args.local_dir)):
            subfolder = str(folder_path.relative_to(args.local_dir))
        elif str(folder_path).startswith("outputs/"):
            subfolder = str(folder_path)[len("outputs/"):]
        elif str(folder_path).startswith("./outputs/"):
            subfolder = str(folder_path)[len("./outputs/"):]
        else:
            subfolder = str(folder_path)
    
    # Check for HF token
    token = os.environ.get("HF_TOKEN")
    if not token:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
    
    if not token:
        print("⚠ Warning: No HF token found. Set HF_TOKEN or run `huggingface-cli login`")
        print("  Private repos require authentication.")
    
    if args.action == "upload":
        check_dir = args.local_dir / subfolder if subfolder else args.local_dir
        if not check_dir.exists():
            print(f"Error: Directory '{check_dir}' does not exist.")
            return 1
        upload_datasets(args.local_dir, args.repo_id, subfolder)
    elif args.action == "download":
        download_datasets(args.local_dir, args.repo_id, subfolder)
    
    return 0


if __name__ == "__main__":
    exit(main())

