#!/usr/bin/env python3
"""
Regenerate CSVs from top-100 JSON files (after corpus texts have been fixed).
"""

import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse


def regenerate_csvs_for_directory(mode_dir):
    """Regenerate CSVs from top-100 JSON files."""
    mode_dir = Path(mode_dir)
    benchmark_mode = mode_dir.name

    print(f"\n{'='*80}")
    print(f"Regenerating CSVs for: {benchmark_mode}")
    print(f"{'='*80}")

    # Find all top-100 JSON files
    top100_files = sorted(mode_dir.glob("*_top100.json"))

    if not top100_files:
        print("No top-100 JSON files found, skipping...")
        return

    print(f"Found {len(top100_files)} top-100 files")

    # Collect all top-100 entries
    all_entries = []

    for top100_file in tqdm(top100_files, desc="Loading top-100 files"):
        with open(top100_file, 'r') as f:
            data = json.load(f)

        test_id = data.get('test_id', top100_file.stem.replace('_top100', ''))
        test_text = data.get('test_text', '')

        # Handle different JSON formats
        if 'top_100' in data:
            # New format: list of dicts with rank, score, corpus_id, corpus_idx, corpus_text
            for item in data['top_100']:
                corpus_text = item.get('corpus_text', '')
                corpus_id = item.get('corpus_id')  # New hash ID field
                corpus_idx = item.get('corpus_idx', 0)  # Fallback to positional index

                all_entries.append({
                    'test_id': test_id,
                    'rank': item.get('rank', 0),
                    'cosine_similarity': item.get('score', 0.0),
                    'corpus_id': corpus_id,  # Add hash ID column
                    'corpus_index': corpus_idx,  # Keep for backwards compatibility
                    'test_text': test_text,
                    'corpus_text': corpus_text
                })
        else:
            # Old format: separate arrays
            scores = data.get('top_scores', data.get('scores', []))
            indices = data.get('top_indices', data.get('indices', list(range(len(scores)))))

            # Add each top score as a row
            for rank, (score, corpus_idx) in enumerate(zip(scores, indices), 1):
                all_entries.append({
                    'test_id': test_id,
                    'rank': rank,
                    'cosine_similarity': score,
                    'corpus_index': corpus_idx,
                    'test_text': test_text,
                    'corpus_text': ''  # Old format doesn't have corpus text
                })

    # Create DataFrame
    df = pd.DataFrame(all_entries)

    # Sort by similarity score descending
    df_sorted = df.sort_values('cosine_similarity', ascending=False)

    # Save full top-100 for all tests
    csv_path = mode_dir / "all_top100_matches.csv"
    df_sorted.to_csv(csv_path, index=False)
    print(f"✅ Saved: {csv_path.name} ({len(df_sorted)} entries)")

    # Save overall top-100 (highest 100 across all tests)
    top_csv_path = mode_dir / "top_100_contamination.csv"
    df_sorted.head(100).to_csv(top_csv_path, index=False)
    print(f"✅ Saved: {top_csv_path.name} (top 100 overall matches)")

    # Print summary
    print(f"\n📊 Summary for {benchmark_mode}:")
    print(f"   - Total test points: {len(top100_files)}")
    print(f"   - Total matches: {len(df_sorted)}")
    print(f"   - Top similarity: {df_sorted['cosine_similarity'].max():.4f}")
    print(f"   - Mean similarity: {df_sorted['cosine_similarity'].mean():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate CSVs from top-100 JSON files")
    parser.add_argument('--results-dir', required=True, help='Results directory or specific mode directory')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"❌ Error: Directory not found: {results_dir}")
        return

    print("="*80)
    print("REGENERATING CSVs FROM TOP-100 JSONs")
    print("="*80)
    print(f"Directory: {results_dir}")

    # Check if this is a mode directory (has top-100 JSONs) or parent directory
    top100_files = list(results_dir.glob("*_top100.json"))

    if top100_files:
        # This is a mode directory, process it directly
        regenerate_csvs_for_directory(results_dir)
    else:
        # This is a parent directory, process all subdirectories
        mode_dirs = [d for d in results_dir.iterdir()
                    if d.is_dir() and not d.name.startswith('.')
                    and d.name not in ['checkpoints', 'logs', 'temp_similarities']]

        print(f"\nFound {len(mode_dirs)} benchmark/mode directories:")
        for d in mode_dirs:
            print(f"  - {d.name}")

        # Process each directory
        for mode_dir in mode_dirs:
            try:
                regenerate_csvs_for_directory(mode_dir)
            except Exception as e:
                print(f"❌ Error processing {mode_dir.name}: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
