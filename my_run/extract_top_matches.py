#!/usr/bin/env python3
"""
Extract Top Matches - Grab actual texts for benchmark matches
Creates CSVs with benchmark texts and their top matching corpus documents
"""

import json
import csv
import sys
from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq
from datasets import load_dataset

# Configuration
RESULTS_DIR = Path("my_run/results_fast")
EMBEDDING_DIR = Path("/lambda/nfs/embeddings/embedding_folder")
OUTPUT_DIR = Path("my_run/top_matches_csvs")
TOP_K = 100  # How many top matches to extract per benchmark item

OUTPUT_DIR.mkdir(exist_ok=True)


def load_benchmark(name, mode):
    """Load benchmark and return texts and IDs."""
    print(f"  Loading benchmark: {name} ({mode} mode)")

    if name == 'musr':
        ds = load_dataset("TAUR-Lab/MuSR")
        data = []
        for split in ds:
            for idx, item in enumerate(ds[split]):
                inp = item.get('narrative', item.get('question', ''))
                out = item.get('answer', '')
                data.append({
                    'id': f"{split}_{idx}",
                    'input': inp,
                    'output': out
                })

    elif name == 'humaneval':
        ds = load_dataset("openai/openai_humaneval")
        data = []
        for item in ds['test']:
            data.append({
                'id': item['task_id'],
                'input': item.get('prompt', ''),
                'output': item.get('canonical_solution', '')
            })

    elif name == 'mbpp':
        try:
            ds = load_dataset("evalplus/mbpp", "mbpp")
        except:
            ds = load_dataset("google-research-datasets/mbpp", "sanitized")

        data = []
        target_split = 'test' if 'test' in ds else list(ds.keys())[0]

        for item in ds[target_split]:
            task_id = str(item.get('task_id', f"mbpp_{len(data)}"))
            input_text = item.get('prompt', item.get('text', ''))
            output_text = item.get('canonical_solution', item.get('code', item.get('solution', '')))

            data.append({
                'id': task_id,
                'input': input_text,
                'output': output_text
            })

    else:
        raise ValueError(f"Unknown benchmark: {name}")

    # Extract texts based on mode
    texts = []
    ids = []
    for item in data:
        if mode == 'input':
            text = item['input']
        elif mode == 'output':
            text = item['output']
        elif mode == 'input_output' or mode == '+':
            text = f"{item['input']}\n\n{item['output']}"
        else:
            text = f"{item['input']}\n\n{item['output']}"

        texts.append(text)
        ids.append(item['id'])

    return texts, ids


def read_parquet_row(file_path, local_idx):
    """Read a specific row from a parquet file."""
    try:
        # Read just the text column for the specific row
        pf = pq.ParquetFile(file_path)

        # Find which row group contains this row
        rows_so_far = 0
        for rg_idx in range(pf.num_row_groups):
            rg = pf.metadata.row_group(rg_idx)
            rg_rows = rg.num_rows

            if rows_so_far + rg_rows > local_idx:
                # This row group contains our row
                local_rg_idx = local_idx - rows_so_far

                # Read just this row group
                table = pf.read_row_group(rg_idx)

                # Get the row
                row = table.slice(local_rg_idx, 1).to_pydict()

                # Extract text
                text = row.get('text', [''])[0] if 'text' in row else ''
                source = row.get('source', ['unknown'])[0] if 'source' in row else 'unknown'

                return text, source

            rows_so_far += rg_rows

        return None, None

    except Exception as e:
        print(f"    Error reading {file_path} row {local_idx}: {e}")
        return None, None


def find_parquet_file(file_name):
    """Find the full path to a parquet file."""
    # Search in the embedding directory
    matches = list(EMBEDDING_DIR.rglob(file_name))
    if matches:
        return matches[0]
    return None


def process_benchmark(benchmark_name, mode):
    """Process one benchmark configuration."""
    print(f"\n{'='*60}")
    print(f"Processing: {benchmark_name} / {mode}")
    print('='*60)

    # Load benchmark texts
    try:
        bench_texts, bench_ids = load_benchmark(benchmark_name, mode)
    except Exception as e:
        print(f"  Error loading benchmark: {e}")
        return

    # Load matches file
    matches_file = RESULTS_DIR / f"{benchmark_name}_{mode}_matches.json"
    if not matches_file.exists():
        print(f"  Matches file not found: {matches_file}")
        return

    print(f"  Loading matches from {matches_file.name}")
    with open(matches_file, 'r') as f:
        matches = json.load(f)

    # Prepare output CSV
    output_file = OUTPUT_DIR / f"{benchmark_name}_{mode}_top{TOP_K}.csv"

    print(f"  Extracting top {TOP_K} matches for {len(matches)} benchmark items...")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'benchmark_id',
            'benchmark_text',
            'rank',
            'similarity_score',
            'corpus_file',
            'corpus_source',
            'corpus_text'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Cache for parquet files to avoid repeated lookups
        file_cache = {}

        for bench_idx, bench_matches in enumerate(tqdm(matches, desc="Processing items")):
            bench_text = bench_texts[bench_idx] if bench_idx < len(bench_texts) else ''
            bench_id = bench_ids[bench_idx] if bench_idx < len(bench_ids) else f'item_{bench_idx}'

            # Process top K matches
            for match in bench_matches[:TOP_K]:
                rank = match.get('rank', 0)
                score = match.get('score', 0.0)
                file_name = match.get('file', '')
                local_idx = match.get('local_idx', 0)

                if not file_name:
                    # Old format without file info
                    continue

                # Find the parquet file
                if file_name not in file_cache:
                    file_path = find_parquet_file(file_name)
                    file_cache[file_name] = file_path
                else:
                    file_path = file_cache[file_name]

                if not file_path:
                    print(f"    Warning: Could not find file {file_name}")
                    continue

                # Read the specific row
                corpus_text, corpus_source = read_parquet_row(file_path, local_idx)

                if corpus_text is None:
                    continue

                # Write to CSV
                writer.writerow({
                    'benchmark_id': bench_id,
                    'benchmark_text': bench_text,
                    'rank': rank,
                    'similarity_score': score,
                    'corpus_file': file_name,
                    'corpus_source': corpus_source,
                    'corpus_text': corpus_text
                })

    print(f"  ✅ Saved to {output_file}")


def main():
    print("="*60)
    print("EXTRACTING TOP MATCHES FROM CONTAMINATION ANALYSIS")
    print("="*60)

    # Find all matches files
    matches_files = sorted(RESULTS_DIR.glob("*_matches.json"))

    if not matches_files:
        print("No matches files found!")
        return

    print(f"\nFound {len(matches_files)} matches files:")
    for f in matches_files:
        print(f"  - {f.name}")

    # Process each benchmark
    for matches_file in matches_files:
        # Parse benchmark name and mode from filename
        # Format: {benchmark}_{mode}_matches.json
        name_parts = matches_file.stem.replace('_matches', '').split('_')

        # Handle cases like "mbpp_+" or "musr_input"
        if len(name_parts) == 2:
            benchmark_name, mode = name_parts
        else:
            # If more complex, try to parse
            benchmark_name = name_parts[0]
            mode = '_'.join(name_parts[1:])

        # Normalize mode
        if mode == '+':
            mode = 'input_output'

        try:
            process_benchmark(benchmark_name, mode)
        except Exception as e:
            print(f"  ❌ Error processing {benchmark_name}/{mode}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
