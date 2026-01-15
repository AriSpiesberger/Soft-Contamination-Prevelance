#!/usr/bin/env python3
"""Clean up and compress all JSONL files in deduplicated_data"""

import gzip
import shutil
from pathlib import Path

def compress_all_jsonl():
    """Find and compress all uncompressed JSONL files"""
    data_dir = Path("deduplicated_data")
    
    if not data_dir.exists():
        print("deduplicated_data directory not found")
        return
    
    # Find all .jsonl files (not .gz)
    jsonl_files = list(data_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print("No uncompressed JSONL files found")
        return
    
    for jsonl_file in jsonl_files:
        gz_file = jsonl_file.with_suffix(jsonl_file.suffix + '.gz')
        
        if gz_file.exists():
            print(f"WARNING: {gz_file.name} already exists, skipping {jsonl_file.name}")
            continue
        
        print(f"Compressing {jsonl_file.name}...")
        original_size = jsonl_file.stat().st_size / 1024 / 1024
        
        with open(jsonl_file, 'rb') as f_in:
            with gzip.open(gz_file, 'wb', compresslevel=9) as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        compressed_size = gz_file.stat().st_size / 1024 / 1024
        ratio = compressed_size / original_size * 100
        
        print(f"  DONE: {jsonl_file.name}: {original_size:.2f} MB -> {compressed_size:.2f} MB ({ratio:.1f}%)")
        print(f"  You can now delete {jsonl_file.name}")

if __name__ == "__main__":
    compress_all_jsonl()
    print("\nDone! All JSONL files compressed.")

