#!/usr/bin/env python3
"""Compress existing JSONL file to .gz format"""

import gzip
import shutil
from pathlib import Path

def compress_jsonl(input_path, output_path=None):
    """Compress a JSONL file to .gz format"""
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        return
    
    if output_path is None:
        output_path = input_path.with_suffix(input_path.suffix + '.gz')
    else:
        output_path = Path(output_path)
    
    print(f"Compressing {input_path} -> {output_path}")
    print(f"Original size: {input_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    with open(input_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb', compresslevel=9) as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    print(f"Compressed size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Compression ratio: {output_path.stat().st_size / input_path.stat().st_size * 100:.1f}%")
    print(f"✅ Done! Compressed file saved to {output_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python compress_jsonl.py <input.jsonl> [output.jsonl.gz]")
        print("\nExample:")
        print("  python compress_jsonl.py deduplicated_data/deduplicated_instruct.jsonl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    compress_jsonl(input_file, output_file)

