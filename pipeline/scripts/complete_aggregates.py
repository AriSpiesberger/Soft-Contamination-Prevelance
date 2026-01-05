#!/usr/bin/env python3
"""
Quick fix script to generate missing aggregate outputs:
- Linear histogram
- CDF plot
- Top-100 CSV

Reads existing similarity files and top-100 JSONs from Stage 4 output.
"""

import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import gzip
import argparse


def load_similarity_file(filepath):
    """Load a compressed similarity file."""
    with gzip.open(filepath, 'rb') as f:
        return np.load(f)


def process_benchmark_mode(mode_dir, dataset_name="unknown"):
    """Generate missing plots and CSV for a benchmark/mode directory."""
    mode_dir = Path(mode_dir)
    benchmark_mode = mode_dir.name

    print(f"\n{'='*80}")
    print(f"Processing: {benchmark_mode}")
    print(f"{'='*80}")

    # Find all similarity files
    sim_files = sorted(mode_dir.glob("*_similarities.npy.gz"))
    top100_files = sorted(mode_dir.glob("*_top100.json"))

    print(f"Found {len(sim_files)} similarity files")
    print(f"Found {len(top100_files)} top-100 files")

    if not sim_files:
        print("No similarity files found, skipping...")
        return

    # Load aggregate stats if exists
    stats_file = mode_dir / "aggregate_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
            print(f"Loaded stats: {stats['total_comparisons']:,} total comparisons")

    # =========================================================================
    # 1. Generate Linear and CDF plots from sampled data
    # =========================================================================
    print("\nGenerating histogram and CDF plots...")

    # Sample similarities from files (to avoid loading everything into memory)
    sample_size = 1_000_000  # Sample 1M points for plotting
    total_files = len(sim_files)
    samples_per_file = max(1, sample_size // total_files)

    sampled_sims = []
    all_top_scores = []

    for sim_file in tqdm(sim_files, desc="Sampling similarities"):
        # Load similarities
        sims = load_similarity_file(sim_file)

        # Sample
        if len(sims) > samples_per_file:
            indices = np.random.choice(len(sims), samples_per_file, replace=False)
            sampled_sims.append(sims[indices])
        else:
            sampled_sims.append(sims)

        # Load corresponding top-100 for top scores
        test_id = sim_file.stem.replace('_similarities', '')
        top100_file = mode_dir / f"{test_id}_top100.json"
        if top100_file.exists():
            with open(top100_file, 'r') as f:
                top100 = json.load(f)
                # Get top 10 scores from this test
                scores = top100.get('top_scores', top100.get('scores', []))
                all_top_scores.extend(scores[:10])

    # Combine all samples
    all_samples = np.concatenate(sampled_sims)
    print(f"Sampled {len(all_samples):,} similarities for plotting")

    # Get stats from samples
    sample_min = float(np.min(all_samples))
    sample_max = float(np.max(all_samples))
    sample_mean = float(np.mean(all_samples))
    sample_p99 = float(np.percentile(all_samples, 99))
    sample_p95 = float(np.percentile(all_samples, 95))

    print(f"Sample stats: min={sample_min:.4f}, max={sample_max:.4f}, mean={sample_mean:.4f}, p99={sample_p99:.4f}")

    # -------------------------------------------------------------------------
    # LINEAR HISTOGRAM
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 8))
    plt.hist(all_samples, bins=200, alpha=0.7, edgecolor='black')
    plt.axvline(sample_max, color='r', linestyle='--', label=f'Max: {sample_max:.4f}')
    plt.axvline(sample_mean, color='g', linestyle='--', label=f'Mean: {sample_mean:.4f}')
    plt.axvline(sample_p99, color='orange', linestyle='--', label=f'P99: {sample_p99:.4f}')
    plt.axvline(sample_p95, color='purple', linestyle='--', label=f'P95: {sample_p95:.4f}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title(f'{benchmark_mode.upper()} - Aggregate Distribution (Linear Scale)\nSampled from {len(sim_files)} test points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    linear_hist_path = mode_dir / "aggregate_histogram_linear.png"
    plt.savefig(linear_hist_path, dpi=150)
    plt.close()
    print(f"✅ Saved: {linear_hist_path.name}")

    # -------------------------------------------------------------------------
    # CDF PLOT
    # -------------------------------------------------------------------------
    sorted_samples = np.sort(all_samples)
    cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)

    plt.figure(figsize=(12, 8))
    plt.plot(sorted_samples, cdf, linewidth=2)
    plt.axvline(sample_max, color='r', linestyle='--', label=f'Max: {sample_max:.4f}')
    plt.axvline(sample_mean, color='g', linestyle='--', label=f'Mean: {sample_mean:.4f}')
    plt.axvline(sample_p99, color='orange', linestyle='--', label=f'P99: {sample_p99:.4f}')
    plt.axvline(sample_p95, color='purple', linestyle='--', label=f'P95: {sample_p95:.4f}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Cumulative Probability')
    plt.title(f'{benchmark_mode.upper()} - Cumulative Distribution Function\nSampled from {len(sim_files)} test points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    cdf_path = mode_dir / "aggregate_cdf.png"
    plt.savefig(cdf_path, dpi=150)
    plt.close()
    print(f"✅ Saved: {cdf_path.name}")

    # =========================================================================
    # 2. Generate Top-100 CSV from top-100 JSONs
    # =========================================================================
    print("\nGenerating top-100 CSV...")

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

    # Create DataFrame and save
    df = pd.DataFrame(all_entries)

    # Sort by similarity score descending
    df_sorted = df.sort_values('cosine_similarity', ascending=False)

    # Save full top-100 for all tests
    csv_path = mode_dir / "all_top100_matches.csv"
    df_sorted.to_csv(csv_path, index=False)
    print(f"✅ Saved: {csv_path.name} ({len(df_sorted)} entries)")

    # Also save top-100 overall
    top_100_overall = df_sorted.head(100)
    top100_csv_path = mode_dir / "top_100_contamination.csv"
    top_100_overall.to_csv(top100_csv_path, index=False)
    print(f"✅ Saved: {top100_csv_path.name} (top 100 overall)")

    # Print summary
    print(f"\n📊 Summary for {benchmark_mode}:")
    print(f"   - Total test points: {len(top100_files)}")
    print(f"   - Total matches: {len(df_sorted)}")
    print(f"   - Top similarity: {df_sorted['cosine_similarity'].max():.4f}")
    print(f"   - Mean similarity: {df_sorted['cosine_similarity'].mean():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Complete missing aggregate outputs")
    parser.add_argument('--results-dir', required=True, help='Results directory (e.g., ./results/contamination)')
    parser.add_argument('--dataset-name', default='dataset', help='Dataset name for titles')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"❌ Error: Results directory not found: {results_dir}")
        return

    print("="*80)
    print("COMPLETING AGGREGATE OUTPUTS")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Dataset: {args.dataset_name}")

    # Find all benchmark/mode directories
    mode_dirs = [d for d in results_dir.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name not in ['checkpoints', 'logs', 'temp_similarities']]

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

    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
