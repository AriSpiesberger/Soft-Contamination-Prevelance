#!/usr/bin/env python3
"""
Simple extraction of top matches - no extra dependencies
Extracts matches from checkpoints and saves to CSV
"""

import json
import csv
import pickle
from pathlib import Path
import torch
import sys

# Configuration
CHECKPOINT_DIR = Path("my_run/checkpoints_fast")
EMBEDDING_DIR = Path("/lambda/nfs/embeddings/embedding_folder")
OUTPUT_DIR = Path("my_run/top_matches_csvs")
TOP_K = 100

OUTPUT_DIR.mkdir(exist_ok=True)


def load_benchmark(name, mode):
    """Load benchmark and return texts and IDs."""
    from datasets import load_dataset

    if name == 'musr':
        ds = load_dataset("TAUR-Lab/MuSR")
        data = []
        for split in ds:
            for idx, item in enumerate(ds[split]):
                inp = item.get('narrative', item.get('question', ''))
                out = item.get('answer', '')
                data.append({'id': f"{split}_{idx}", 'input': inp, 'output': out})

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

    texts, ids = [], []
    for item in data:
        if mode == 'input':
            texts.append(item['input'])
        elif mode == 'output':
            texts.append(item['output'])
        else:
            texts.append(f"{item['input']}\n\n{item['output']}")
        ids.append(item['id'])

    return texts, ids


def read_parquet_row_safe(file_path, local_idx):
    """Read a specific row from parquet - requires pyarrow."""
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(str(file_path))

        # Find which row group contains this row
        rows_so_far = 0
        for rg_idx in range(pf.num_row_groups):
            rg = pf.metadata.row_group(rg_idx)
            rg_rows = rg.num_rows

            if rows_so_far + rg_rows > local_idx:
                # This row group contains our row
                local_rg_idx = local_idx - rows_so_far

                # Read just this row group with only text column
                table = pf.read_row_group(rg_idx, columns=['text'])

                # Get the row
                row = table.slice(local_rg_idx, 1).to_pydict()

                # Extract text
                text = row.get('text', [''])[0] if 'text' in row else ''

                return text

            rows_so_far += rg_rows

        return None

    except Exception as e:
        print(f"      Error reading row {local_idx}: {e}")
        return None


def process_benchmark(benchmark_name, mode):
    """Process one benchmark configuration."""
    print(f"\n{'='*60}")
    print(f"Processing: {benchmark_name} / {mode}")
    print('='*60)

    # Check for checkpoint files
    sims_file = CHECKPOINT_DIR / f"{benchmark_name}_{mode}_sims.pt"
    idxs_file = CHECKPOINT_DIR / f"{benchmark_name}_{mode}_idxs.pt"
    state_file = CHECKPOINT_DIR / f"{benchmark_name}_{mode}_state.pkl"

    if not all([sims_file.exists(), idxs_file.exists(), state_file.exists()]):
        print(f"  Missing checkpoint files, skipping")
        return

    # Load checkpoint data
    print(f"  Loading checkpoints...")
    sims = torch.load(sims_file, map_location='cpu').numpy()
    idxs = torch.load(idxs_file, map_location='cpu').numpy()

    with open(state_file, 'rb') as f:
        state = pickle.load(f)

    all_ids = state.get('all_ids', [])

    print(f"  Checkpoints loaded: {sims.shape[0]} benchmark items, {len(all_ids)} corpus files")

    # Load benchmark texts
    try:
        print(f"  Loading benchmark data...")
        bench_texts, bench_ids = load_benchmark(benchmark_name, mode)
    except Exception as e:
        print(f"  Error loading benchmark: {e}")
        return

    # Create reverse index: global_idx -> (file_path, local_idx)
    print(f"  Building index...")
    def find_file_for_idx(global_idx):
        for file_path, start_idx, end_idx in all_ids:
            if start_idx <= global_idx < end_idx:
                return file_path, global_idx - start_idx
        return None, None

    # Prepare output CSV
    output_file = OUTPUT_DIR / f"{benchmark_name}_{mode}_top{TOP_K}.csv"

    print(f"  Extracting top {TOP_K} matches...")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'benchmark_id',
            'benchmark_text',
            'rank',
            'similarity_score',
            'global_idx',
            'corpus_file',
            'local_idx',
            'corpus_text'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_rows = 0
        for bench_idx in range(sims.shape[0]):
            bench_text = bench_texts[bench_idx] if bench_idx < len(bench_texts) else ''
            bench_id = bench_ids[bench_idx] if bench_idx < len(bench_ids) else f'item_{bench_idx}'

            if bench_idx % 50 == 0:
                print(f"    Processing item {bench_idx}/{sims.shape[0]}...")

            # Process top K matches
            for rank in range(min(TOP_K, sims.shape[1])):
                score = float(sims[bench_idx, rank])

                # Skip uninitialized values
                if score < -0.5:
                    continue

                global_idx = int(idxs[bench_idx, rank])

                # Find the file
                file_path, local_idx = find_file_for_idx(global_idx)

                if file_path is None:
                    continue

                file_name = Path(file_path).name

                # Read the text
                corpus_text = read_parquet_row_safe(file_path, local_idx)

                if corpus_text is None:
                    corpus_text = "[Error reading text]"

                # Write to CSV
                writer.writerow({
                    'benchmark_id': bench_id,
                    'benchmark_text': bench_text[:500] + ('...' if len(bench_text) > 500 else ''),  # Truncate for readability
                    'rank': rank + 1,
                    'similarity_score': score,
                    'global_idx': global_idx,
                    'corpus_file': file_name,
                    'local_idx': local_idx,
                    'corpus_text': corpus_text[:1000] + ('...' if len(corpus_text) > 1000 else '')  # Truncate for readability
                })

                total_rows += 1

    print(f"  ✅ Saved {total_rows} rows to {output_file.name}")


def main():
    print("="*60)
    print("EXTRACTING TOP MATCHES FROM CHECKPOINTS")
    print("="*60)

    # Find all checkpoint state files
    state_files = sorted(CHECKPOINT_DIR.glob("*_state.pkl"))

    if not state_files:
        print("No checkpoint files found!")
        return

    print(f"\nFound {len(state_files)} checkpoints:")
    for f in state_files:
        print(f"  - {f.stem.replace('_state', '')}")

    # Process each benchmark
    for state_file in state_files:
        # Parse benchmark name and mode from filename
        # Format: {benchmark}_{mode}_state.pkl
        name_parts = state_file.stem.replace('_state', '').split('_')

        # Handle cases like "mbpp_+" or "musr_input"
        if len(name_parts) == 2:
            benchmark_name, mode = name_parts
        else:
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
