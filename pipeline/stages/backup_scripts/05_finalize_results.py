#!/usr/bin/env python3
"""
Stage 5: Finalize Contamination Results
- Add corpus texts to top-100 JSONs
- Generate aggregate plots (linear histogram, CDF)
- Generate aggregate CSVs with full text

This combines the functionality of stages 07 (add corpus texts) and
the complete_aggregates script into a single integrated stage.
"""

import sys
sys.path.insert(0, '/home/ubuntu/embeddings/SDTD_Main/pipeline/scripts')

from complete_aggregates import process_benchmark_mode, load_similarity_file
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq
import duckdb


def load_corpus_index(corpus_path):
    """
    Build an index mapping hash_id -> text from parquet or jsonl files.
    Also builds a fallback positional index for backwards compatibility.

    Args:
        corpus_path: Path to directory with parquet files OR path to jsonl file

    Returns:
        tuple: (id_to_text dict, idx_to_text dict for fallback)
    """
    corpus_path = Path(corpus_path)
    id_to_text = {}   # hash_id -> text (primary)
    idx_to_text = {}  # position -> text (fallback for old results)
    current_idx = 0

    if corpus_path.is_file():
        # Single JSONL file
        if corpus_path.suffix == '.jsonl':
            print(f"Loading corpus from JSONL: {corpus_path}")
            with open(corpus_path) as f:
                for line in tqdm(f, desc="Indexing corpus"):
                    data = json.loads(line)
                    text = data.get('text', '')
                    hash_id = data.get('id')

                    if hash_id:
                        id_to_text[hash_id] = text
                    idx_to_text[current_idx] = text
                    current_idx += 1

        elif corpus_path.suffix == '.parquet':
            # Single parquet file
            print(f"Loading corpus from parquet: {corpus_path}")
            con = duckdb.connect()

            # Check columns
            cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{corpus_path}')").fetchall()
            col_names = [c[0] for c in cols]

            has_id = 'id' in col_names or 'hash_id' in col_names
            id_col = 'id' if 'id' in col_names else 'hash_id'
            text_col = 'text' if 'text' in col_names else None

            if has_id and text_col:
                result = con.execute(f"SELECT {id_col}, {text_col} FROM read_parquet('{corpus_path}')").fetchall()
                for hash_id, text in tqdm(result, desc="Indexing corpus"):
                    id_to_text[hash_id] = text
                    idx_to_text[current_idx] = text
                    current_idx += 1
            elif text_col:
                result = con.execute(f"SELECT {text_col} FROM read_parquet('{corpus_path}')").fetchall()
                for (text,) in tqdm(result, desc="Indexing corpus"):
                    idx_to_text[current_idx] = text
                    current_idx += 1

    elif corpus_path.is_dir():
        # Directory of parquet files
        parquet_files = sorted(corpus_path.rglob("*.parquet"))
        print(f"Loading corpus from {len(parquet_files)} parquet files in {corpus_path}")

        con = duckdb.connect()
        for pf in tqdm(parquet_files, desc="Indexing parquet files"):
            try:
                # Check available columns
                cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{pf}')").fetchall()
                col_names = [c[0] for c in cols]

                # Find ID and text columns
                id_col = 'id' if 'id' in col_names else ('hash_id' if 'hash_id' in col_names else None)
                text_col = None
                for col in ['text', 'paragraph', 'content', 'sentence']:
                    if col in col_names:
                        text_col = col
                        break

                if not text_col:
                    print(f"Warning: No text column found in {pf}")
                    continue

                # Load IDs and texts
                if id_col:
                    result = con.execute(f"SELECT {id_col}, {text_col} FROM read_parquet('{pf}')").fetchall()
                    for hash_id, text in result:
                        id_to_text[hash_id] = text
                        idx_to_text[current_idx] = text
                        current_idx += 1
                else:
                    result = con.execute(f"SELECT {text_col} FROM read_parquet('{pf}')").fetchall()
                    for (text,) in result:
                        idx_to_text[current_idx] = text
                        current_idx += 1

            except Exception as e:
                print(f"Error reading {pf}: {e}")
                continue

    print(f"✓ Indexed by hash ID: {len(id_to_text):,} entries")
    print(f"✓ Indexed by position: {len(idx_to_text):,} entries (fallback)")
    return id_to_text, idx_to_text


def add_texts_to_results(results_dir, id_to_text, idx_to_text):
    """
    Add corpus texts to all top_100 JSON files in results_dir.
    Prefers hash ID lookup, falls back to positional index for old results.

    Args:
        results_dir: Path to contamination results directory
        id_to_text: dict mapping hash_id -> text
        idx_to_text: dict mapping corpus_idx -> text (fallback)
    """
    results_dir = Path(results_dir)

    # Find all top_100 JSON files
    json_files = list(results_dir.rglob("*top100.json")) + list(results_dir.rglob("*top_100.json"))

    print(f"\nFound {len(json_files)} result files to update")

    stats = {'id_lookups': 0, 'idx_lookups': 0, 'missing': 0}

    for json_file in tqdm(json_files, desc="Adding corpus texts"):
        try:
            # Load existing results
            with open(json_file) as f:
                data = json.load(f)

            # Add corpus text to each top_100 entry
            modified = False
            for match in data.get('top_100', []):
                if 'corpus_text' in match:
                    continue  # Already has text

                # Try hash ID first (preferred)
                corpus_id = match.get('corpus_id')
                if corpus_id and corpus_id in id_to_text:
                    match['corpus_text'] = id_to_text[corpus_id]
                    modified = True
                    stats['id_lookups'] += 1
                else:
                    # Fall back to positional index (for old results)
                    corpus_idx = match.get('corpus_idx')
                    if corpus_idx is not None and corpus_idx in idx_to_text:
                        match['corpus_text'] = idx_to_text[corpus_idx]
                        modified = True
                        stats['idx_lookups'] += 1
                    else:
                        stats['missing'] += 1

            # Save if modified
            if modified:
                with open(json_file, 'w') as f:
                    json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    print("✓ All results updated with corpus texts")
    print(f"  - Hash ID lookups: {stats['id_lookups']:,}")
    print(f"  - Position lookups (fallback): {stats['idx_lookups']:,}")
    if stats['missing'] > 0:
        print(f"  ⚠️  Missing texts: {stats['missing']:,}")


def main():
    parser = argparse.ArgumentParser(description='Finalize contamination results (Stage 5)')
    parser.add_argument('--results-dir', required=True, help='Directory with contamination results')
    parser.add_argument('--corpus', required=True, help='Path to corpus (parquet dir or jsonl file)')
    parser.add_argument('--dataset-name', default='dataset', help='Dataset name for titles')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    print("=" * 80)
    print("STAGE 5: FINALIZING CONTAMINATION RESULTS")
    print("=" * 80)
    print(f"Results dir: {args.results_dir}")
    print(f"Corpus: {args.corpus}")
    print(f"Dataset: {args.dataset_name}")
    print()

    # Step 1: Build corpus index
    print("\n" + "=" * 80)
    print("STEP 1: Loading corpus and building index")
    print("=" * 80)
    id_to_text, idx_to_text = load_corpus_index(args.corpus)

    # Step 2: Add texts to results
    print("\n" + "=" * 80)
    print("STEP 2: Adding corpus texts to top-100 results")
    print("=" * 80)
    add_texts_to_results(args.results_dir, id_to_text, idx_to_text)

    # Step 3: Generate aggregate outputs
    print("\n" + "=" * 80)
    print("STEP 3: Generating aggregate plots and CSVs")
    print("=" * 80)

    # Find all benchmark/mode directories
    mode_dirs = [d for d in results_dir.iterdir()
                 if d.is_dir() and not d.name.startswith('.')
                 and d.name not in ['checkpoints', 'logs', 'temp_similarities']]

    print(f"\nFound {len(mode_dirs)} benchmark/mode directories:")
    for d in mode_dirs:
        print(f"  - {d.name}")

    # Process each directory
    for mode_dir in mode_dirs:
        try:
            process_benchmark_mode(mode_dir, args.dataset_name)
        except Exception as e:
            print(f"❌ Error processing {mode_dir.name}: {e}")
            import traceback
            traceback.print_exc()

    # Step 4: Cleanup temporary files
    print("\n" + "=" * 80)
    print("STEP 4: Cleaning up temporary files")
    print("=" * 80)

    cleanup_items = []

    # Remove temp_similarities if it exists
    temp_sim_dir = results_dir / "temp_similarities"
    if temp_sim_dir.exists():
        import shutil
        shutil.rmtree(temp_sim_dir)
        cleanup_items.append("temp_similarities/")

    # Remove individual similarity .npy.gz files (keep only aggregates)
    for mode_dir in mode_dirs:
        sim_files = list(mode_dir.glob("*_similarities.npy.gz"))
        for f in sim_files:
            f.unlink()
        if sim_files:
            cleanup_items.append(f"{mode_dir.name}/*_similarities.npy.gz ({len(sim_files)} files)")

    # Remove checkpoints directory
    checkpoint_dir = results_dir / "checkpoints"
    if checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)
        cleanup_items.append("checkpoints/")

    if cleanup_items:
        print("Removed temporary files:")
        for item in cleanup_items:
            print(f"  - {item}")
    else:
        print("No temporary files to clean up")

    print("\n" + "=" * 80)
    print("✅ STAGE 5 COMPLETE!")
    print("=" * 80)
    print("\nGenerated outputs:")
    print("  - Corpus texts added to all top-100 JSONs")
    print("  - Linear histograms: aggregate_histogram_linear.png")
    print("  - CDF plots: aggregate_cdf.png")
    print("  - Full CSVs: all_top100_matches.csv")
    print("  - Top-100 CSVs: top_100_contamination.csv")
    print("\nFinal results location: {}".format(results_dir))


if __name__ == "__main__":
    main()
