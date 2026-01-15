#!/usr/bin/env python3
"""
Resolve Contamination Matches - Minimal EC2 Version

Looks up the actual text content for top-100 matched IDs from the corpus parquet files.
Run this on EC2 to avoid S3 egress fees.

Usage:
    python resolve_matches.py \
        --corpus-prefix embeddings/output \
        --output-dir resolved_matches
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Tuple
import csv

import boto3
import pyarrow.parquet as pq
from tqdm import tqdm
from datasets import load_dataset


# =============================================================================
# S3 HELPERS
# =============================================================================
def list_s3_files(bucket: str, prefix: str, suffix: str = ".parquet") -> List[str]:
    """List all files in S3 with given prefix and suffix."""
    s3 = boto3.client('s3')
    files = []
    paginator = s3.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith(suffix):
                    files.append(obj['Key'])

    return files


def download_json_from_s3(bucket: str, key: str) -> dict:
    """Download and parse JSON from S3."""
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(response['Body'].read().decode('utf-8'))


# =============================================================================
# BENCHMARK LOADERS
# =============================================================================
def load_benchmark_texts(name: str, mode: str) -> List[Dict]:
    """Load benchmark dataset and return list of {id, text} dicts."""
    if name == 'musr':
        ds = load_dataset("TAUR-Lab/MuSR")
        data = []
        for split in ds:
            for idx, item in enumerate(ds[split]):
                inp = item.get('narrative', item.get('question', ''))
                out = item.get('answer', '')
                item_id = f"{split}_{idx}"

                if mode == 'input':
                    text = inp
                elif mode == 'output':
                    text = out
                else:
                    text = f"{inp}\n\n{out}"

                data.append({'id': item_id, 'text': text, 'input': inp, 'output': out})
        return data

    elif name == 'mbpp':
        try:
            ds = load_dataset("evalplus/mbpp", "mbpp")
        except:
            ds = load_dataset("google-research-datasets/mbpp", "sanitized")

        data = []
        target_split = 'test' if 'test' in ds else list(ds.keys())[0]

        for item in ds[target_split]:
            task_id = str(item.get('task_id', f"mbpp_{len(data)}"))
            inp = item.get('prompt', item.get('text', ''))
            out = item.get('canonical_solution', item.get('code', ''))

            if mode == 'input':
                text = inp
            elif mode == 'output':
                text = out
            else:
                text = f"{inp}\n\n{out}"

            data.append({'id': task_id, 'text': text, 'input': inp, 'output': out})
        return data

    else:
        raise ValueError(f"Unknown benchmark: {name}")


# =============================================================================
# CORPUS ID LOOKUP
# =============================================================================
class CorpusLookup:
    """Lookup corpus texts by ID from parquet files."""

    def __init__(self, bucket: str, prefix: str, local_cache_dir: str = "/tmp/corpus_cache"):
        self.bucket = bucket
        self.prefix = prefix
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        self.s3 = boto3.client('s3')
        self.id_index: Dict[str, Tuple[str, int]] = {}

    def build_index(self, target_ids: Set[str], parquet_files: List[str], max_workers: int = 8):
        """Build index for target IDs by scanning parquet files."""
        print(f"Building index for {len(target_ids)} unique IDs across {len(parquet_files)} files...")

        remaining_ids = set(target_ids)

        def scan_file(file_key: str) -> Dict[str, Tuple[str, int]]:
            """Scan a single parquet file for matching IDs."""
            found = {}
            try:
                local_path = self.local_cache_dir / Path(file_key).name
                if not local_path.exists():
                    self.s3.download_file(self.bucket, file_key, str(local_path))

                pf = pq.ParquetFile(str(local_path))
                cols = [f.name for f in pf.schema_arrow]

                # Check both id and hash_id columns since matches use both formats
                id_cols = [c for c in ['id', 'hash_id'] if c in cols]

                if not id_cols:
                    return found

                table = pf.read(columns=id_cols)

                # Build lookup for each ID column
                for id_col in id_cols:
                    ids = table[id_col].to_pylist()
                    for row_idx, doc_id in enumerate(ids):
                        if doc_id in remaining_ids:
                            found[doc_id] = (file_key, row_idx)

                # Delete cached file to save disk space
                if local_path.exists():
                    local_path.unlink()

            except Exception as e:
                print(f"Error scanning {file_key}: {e}")
                # Clean up on error too
                if local_path.exists():
                    local_path.unlink()

            return found

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(scan_file, f): f for f in parquet_files}

            pbar = tqdm(as_completed(futures), total=len(futures), desc="Scanning parquet files")
            for future in pbar:
                found = future.result()
                self.id_index.update(found)
                remaining_ids -= set(found.keys())
                pbar.set_postfix({'found': len(self.id_index), 'remaining': len(remaining_ids)})

                if not remaining_ids:
                    print("All IDs found!")
                    break

        print(f"Indexed {len(self.id_index)} IDs, {len(remaining_ids)} not found")
        return remaining_ids

    def lookup_texts(self, ids: List[str], max_workers: int = 8) -> Dict[str, str]:
        """Look up actual text content for given IDs."""
        print(f"Looking up text for {len(ids)} IDs...")

        file_to_ids: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        missing = []

        for doc_id in ids:
            if doc_id in self.id_index:
                file_key, row_idx = self.id_index[doc_id]
                file_to_ids[file_key].append((doc_id, row_idx))
            else:
                missing.append(doc_id)

        if missing:
            print(f"Warning: {len(missing)} IDs not in index")

        results: Dict[str, str] = {}

        def fetch_from_file(file_key: str, id_rows: List[Tuple[str, int]]) -> Dict[str, str]:
            """Fetch texts from a single file."""
            found = {}
            try:
                local_path = self.local_cache_dir / Path(file_key).name
                if not local_path.exists():
                    self.s3.download_file(self.bucket, file_key, str(local_path))

                pf = pq.ParquetFile(str(local_path))
                cols = [f.name for f in pf.schema_arrow]

                txt_col = None
                for c in ['text', 'content', 'document']:
                    if c in cols:
                        txt_col = c
                        break

                if txt_col is None:
                    print(f"Warning: no text column in {file_key}, cols: {cols}")
                    return found

                table = pf.read(columns=[txt_col])
                texts = table[txt_col].to_pylist()

                for doc_id, row_idx in id_rows:
                    if row_idx < len(texts):
                        found[doc_id] = texts[row_idx]

                # Delete cached file to save disk space
                if local_path.exists():
                    local_path.unlink()

            except Exception as e:
                print(f"Error fetching from {file_key}: {e}")
                # Clean up on error too
                if local_path.exists():
                    local_path.unlink()

            return found

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fetch_from_file, fk, id_rows): fk
                for fk, id_rows in file_to_ids.items()
            }

            pbar = tqdm(as_completed(futures), total=len(futures), desc="Fetching texts")
            for future in pbar:
                found = future.result()
                results.update(found)
                pbar.set_postfix({'fetched': len(results)})

        return results


# =============================================================================
# MAIN
# =============================================================================
def process_benchmark(
    benchmark: str,
    mode: str,
    matches: List[List[Dict]],
    corpus_lookup: CorpusLookup,
    output_dir: Path
) -> None:
    """Process a single benchmark/mode and save CSV."""

    print(f"\n{'='*60}")
    print(f"Processing {benchmark} / {mode}")
    print(f"{'='*60}")

    bench_data = load_benchmark_texts(benchmark, mode)
    print(f"Loaded {len(bench_data)} benchmark items")

    all_corpus_ids = set()
    for item_matches in matches:
        for match in item_matches:
            all_corpus_ids.add(match['id'])

    print(f"Need to look up {len(all_corpus_ids)} unique corpus IDs")

    corpus_texts = corpus_lookup.lookup_texts(list(all_corpus_ids))
    print(f"Retrieved {len(corpus_texts)} corpus texts")

    # Save CSV
    output_file = output_dir / f"{benchmark}_{mode}_resolved.csv"

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'bench_idx', 'bench_id', 'bench_text',
            'match_rank', 'match_score', 'corpus_id', 'corpus_text'
        ])

        for bench_idx, (bench_item, item_matches) in enumerate(zip(bench_data, matches)):
            for match in item_matches:
                corpus_id = match['id']
                corpus_text = corpus_texts.get(corpus_id, '[NOT FOUND]')

                writer.writerow([
                    bench_idx,
                    bench_item['id'],
                    bench_item['text'][:1000],
                    match['rank'],
                    match['score'],
                    corpus_id,
                    corpus_text[:2000] if corpus_text else '[NOT FOUND]'
                ])

    print(f"Saved: {output_file}")

    # Save JSON
    json_output = output_dir / f"{benchmark}_{mode}_resolved.json"

    resolved_data = []
    for bench_idx, (bench_item, item_matches) in enumerate(zip(bench_data, matches)):
        resolved_matches = []
        for match in item_matches:
            corpus_id = match['id']
            resolved_matches.append({
                'rank': match['rank'],
                'score': match['score'],
                'corpus_id': corpus_id,
                'corpus_text': corpus_texts.get(corpus_id, None)
            })

        resolved_data.append({
            'bench_idx': bench_idx,
            'bench_id': bench_item['id'],
            'bench_input': bench_item['input'],
            'bench_output': bench_item['output'],
            'matches': resolved_matches
        })

    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(resolved_data, f, indent=2, ensure_ascii=False)

    print(f"Saved: {json_output}")


def main():
    parser = argparse.ArgumentParser(description="Resolve contamination match IDs to actual text")
    parser.add_argument('--bucket', default='dolmo-3-sampling',
                       help='S3 bucket for both matches and corpus')
    parser.add_argument('--matches-prefix', default='contamination_analysis_fast',
                       help='S3 prefix for matches files')
    parser.add_argument('--corpus-prefix', default='embeddings/output',
                       help='S3 prefix for corpus parquet files')
    parser.add_argument('--output-dir', default='resolved_matches',
                       help='Output directory for resolved CSVs')
    parser.add_argument('--benchmarks', nargs='+', default=['musr', 'mbpp'],
                       help='Benchmarks to process')
    parser.add_argument('--modes', nargs='+', default=['input', 'output'],
                       help='Modes to process')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Max parallel workers for S3 operations (keep low to avoid disk space issues)')
    parser.add_argument('--cache-dir', default='/tmp/corpus_cache',
                       help='Local cache directory for parquet files')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # List corpus parquet files
    print("Listing corpus parquet files...")
    parquet_files = list_s3_files(args.bucket, args.corpus_prefix)
    print(f"Found {len(parquet_files)} parquet files")

    if not parquet_files:
        print("ERROR: No parquet files found! Check --corpus-prefix")
        return

    # Collect all IDs we need
    all_ids_needed = set()
    benchmark_matches = {}

    for benchmark in args.benchmarks:
        for mode in args.modes:
            matches_key = f"{args.matches_prefix}/{benchmark}/{mode}/matches.json"
            print(f"Loading matches from s3://{args.bucket}/{matches_key}")

            try:
                matches = download_json_from_s3(args.bucket, matches_key)
                benchmark_matches[(benchmark, mode)] = matches

                for item_matches in matches:
                    for match in item_matches:
                        all_ids_needed.add(match['id'])

                print(f"  Loaded {len(matches)} items, {sum(len(m) for m in matches)} total matches")
            except Exception as e:
                print(f"  Failed to load: {e}")

    print(f"\nTotal unique corpus IDs to resolve: {len(all_ids_needed)}")

    # Build corpus lookup index
    corpus_lookup = CorpusLookup(
        bucket=args.bucket,
        prefix=args.corpus_prefix,
        local_cache_dir=args.cache_dir
    )

    missing_ids = corpus_lookup.build_index(
        all_ids_needed,
        parquet_files,
        max_workers=args.max_workers
    )

    if missing_ids:
        print(f"\nWarning: {len(missing_ids)} IDs not found in corpus")
        with open(output_dir / "missing_ids.json", 'w') as f:
            json.dump(list(missing_ids), f)

    # Process each benchmark/mode
    for (benchmark, mode), matches in benchmark_matches.items():
        process_benchmark(benchmark, mode, matches, corpus_lookup, output_dir)

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
