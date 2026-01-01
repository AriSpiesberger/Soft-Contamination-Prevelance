#!/usr/bin/env python3
"""
Add corpus texts to contamination results.

Reads top_100 JSON files and adds the actual corpus paragraph text
by looking up corpus_idx in the parquet/jsonl files.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq
import duckdb

def load_corpus_index(corpus_path):
    """
    Build an index mapping corpus_idx -> text from parquet or jsonl files.

    Args:
        corpus_path: Path to directory with parquet files OR path to jsonl file

    Returns:
        dict: {corpus_idx: text}
    """
    corpus_path = Path(corpus_path)
    index = {}
    current_idx = 0

    if corpus_path.is_file():
        # Single JSONL file
        if corpus_path.suffix == '.jsonl':
            print(f"Loading corpus from JSONL: {corpus_path}")
            with open(corpus_path) as f:
                for line in tqdm(f, desc="Indexing corpus"):
                    data = json.loads(line)
                    index[current_idx] = data.get('text', '')
                    current_idx += 1

        elif corpus_path.suffix == '.parquet':
            # Single parquet file
            print(f"Loading corpus from parquet: {corpus_path}")
            con = duckdb.connect()
            result = con.execute(f"SELECT text FROM read_parquet('{corpus_path}')").fetchall()
            for text_tuple in tqdm(result, desc="Indexing corpus"):
                index[current_idx] = text_tuple[0]
                current_idx += 1

    elif corpus_path.is_dir():
        # Directory of parquet files
        parquet_files = sorted(corpus_path.rglob("*.parquet"))
        print(f"Loading corpus from {len(parquet_files)} parquet files in {corpus_path}")

        con = duckdb.connect()
        for pf in tqdm(parquet_files, desc="Indexing parquet files"):
            try:
                # Try to get text column (might be 'text', 'paragraph', 'content', etc.)
                result = con.execute(f"SELECT * FROM read_parquet('{pf}') LIMIT 1").fetchdf()
                text_col = None
                for col in ['text', 'paragraph', 'content', 'sentence']:
                    if col in result.columns:
                        text_col = col
                        break

                if not text_col:
                    print(f"Warning: No text column found in {pf}")
                    continue

                # Load all texts
                texts = con.execute(f"SELECT {text_col} FROM read_parquet('{pf}')").fetchall()
                for text_tuple in texts:
                    index[current_idx] = text_tuple[0]
                    current_idx += 1

            except Exception as e:
                print(f"Error reading {pf}: {e}")
                continue

    print(f"✓ Indexed {len(index):,} corpus entries")
    return index


def add_texts_to_results(results_dir, corpus_index):
    """
    Add corpus texts to all top_100 JSON files in results_dir.

    Args:
        results_dir: Path to contamination results directory
        corpus_index: dict mapping corpus_idx -> text
    """
    results_dir = Path(results_dir)

    # Find all top_100 JSON files
    json_files = list(results_dir.rglob("*top100.json")) + list(results_dir.rglob("*top_100.json"))

    print(f"\nFound {len(json_files)} result files to update")

    for json_file in tqdm(json_files, desc="Adding corpus texts"):
        try:
            # Load existing results
            with open(json_file) as f:
                data = json.load(f)

            # Add corpus text to each top_100 entry
            modified = False
            for match in data.get('top_100', []):
                corpus_idx = match.get('corpus_idx')
                if corpus_idx is not None and corpus_idx in corpus_index:
                    if 'corpus_text' not in match:
                        match['corpus_text'] = corpus_index[corpus_idx]
                        modified = True

            # Save if modified
            if modified:
                with open(json_file, 'w') as f:
                    json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    print("✓ All results updated with corpus texts")


def main():
    parser = argparse.ArgumentParser(description='Add corpus texts to contamination results')
    parser.add_argument('--results-dir', required=True, help='Directory with contamination results')
    parser.add_argument('--corpus', required=True, help='Path to corpus (parquet dir or jsonl file)')
    args = parser.parse_args()

    print("=" * 80)
    print("ADDING CORPUS TEXTS TO CONTAMINATION RESULTS")
    print("=" * 80)
    print(f"Results dir: {args.results_dir}")
    print(f"Corpus: {args.corpus}")
    print()

    # Build corpus index
    corpus_index = load_corpus_index(args.corpus)

    # Add texts to results
    add_texts_to_results(args.results_dir, corpus_index)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
