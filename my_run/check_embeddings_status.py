#!/usr/bin/env python3
"""
Check status of embedding generation
"""

import sys
from pathlib import Path
import pyarrow.parquet as pq

EMBEDDING_DIR = Path("/lambda/nfs/embeddings/embedding_folder")


def check_embedding_coverage():
    """Check what we have and what might be missing."""

    print("="*60)
    print("EMBEDDING COVERAGE REPORT")
    print("="*60)

    all_files = list(EMBEDDING_DIR.rglob("*.parquet"))
    print(f"\nTotal files: {len(all_files)}")

    # Group by source
    by_source = {}
    by_type = {}  # documents vs paragraphs

    total_rows = 0
    total_size = 0

    print("\nSampling files to estimate counts...")

    for i, f in enumerate(all_files):
        if i % 100 == 0 and i > 0:
            print(f"  Checked {i}/{len(all_files)} files...")

        # Parse structure: source/data_type/quality/chunk_type/file.parquet
        parts = f.relative_to(EMBEDDING_DIR).parts

        if len(parts) >= 5:
            source = parts[0]
            data_type = parts[1]
            quality = parts[2]
            chunk_type = parts[3]  # documents or paragraphs
        else:
            source = parts[0] if len(parts) > 0 else "unknown"
            chunk_type = "unknown"

        # Count rows
        try:
            pf = pq.ParquetFile(str(f))
            rows = pf.metadata.num_rows
            size = f.stat().st_size

            total_rows += rows
            total_size += size

            if source not in by_source:
                by_source[source] = {'files': 0, 'rows': 0, 'size': 0}

            by_source[source]['files'] += 1
            by_source[source]['rows'] += rows
            by_source[source]['size'] += size

            if chunk_type not in by_type:
                by_type[chunk_type] = {'files': 0, 'rows': 0}

            by_type[chunk_type]['files'] += 1
            by_type[chunk_type]['rows'] += rows

        except Exception as e:
            print(f"  Error reading {f.name}: {e}")

    print("\n" + "="*60)
    print("BY SOURCE:")
    print("="*60)

    for source in sorted(by_source.keys()):
        stats = by_source[source]
        print(f"\n{source}:")
        print(f"  Files: {stats['files']:,}")
        print(f"  Rows: {stats['rows']:,}")
        print(f"  Size: {stats['size'] / (1024**3):.2f} GB")

    print("\n" + "="*60)
    print("BY TYPE:")
    print("="*60)

    for chunk_type in sorted(by_type.keys()):
        stats = by_type[chunk_type]
        print(f"\n{chunk_type}:")
        print(f"  Files: {stats['files']:,}")
        print(f"  Rows: {stats['rows']:,}")

    print("\n" + "="*60)
    print("TOTALS:")
    print("="*60)
    print(f"Files: {len(all_files):,}")
    print(f"Rows: {total_rows:,}")
    print(f"Size: {total_size / (1024**3):.2f} GB ({total_size / (1024**4):.2f} TB)")

    print("\n" + "="*60)
    print("EXPECTED SOURCES (from production.py):")
    print("="*60)

    expected_sources = {
        "common_crawl": "3150 GB",
        "olmocr_science_pdfs": "83.8 GB",
        "stack_edu": "525.8 GB",
        "finemath": "95.5 GB",
        "rpj": "9.10 GB",
        "dolma1_7": "4.24 GB",
    }

    for source, expected_size in expected_sources.items():
        if source in by_source:
            actual_size = by_source[source]['size'] / (1024**3)
            print(f"{source}: ✓ FOUND - {actual_size:.2f} GB actual (expected {expected_size})")
        else:
            print(f"{source}: ✗ MISSING (expected {expected_size})")


if __name__ == "__main__":
    check_embedding_coverage()
