"""
ZebraLogic Semantic Duplicate Analysis Plots
1. Box-and-whisker plot: duplicate frequency by training stage
2. Semantic duplicate rate as function of cosine similarity
3. Duplicate rate by puzzle size (grid dimensions)

Adapted from mbpp_duplicate_plots.py for ZebraLogic benchmark.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

ANNOTATIONS_DIR = Path(__file__).parent / "annotations" / "zebralogic"
OUTPUT_DIR = Path(__file__).parent / "exports"

# Training stages in order (extracted from dataset names)
# Format: contamination_{stage}_{percent} -> stage
TRAINING_ORDER = ['dolma', 'dolmino', 'dolci', 'dolci_sft', 'dolci_dpo', 'dolci_rl']
TRAINING_LABELS = [
    'Dolma\n(Pretrain)', 
    'Dolmino\n(Continued)', 
    'Dolci\n(Base)',
    'Dolci SFT\n(SFT)', 
    'Dolci DPO\n(DPO)', 
    'Dolci RL\n(RL)'
]

# Color palette - using a warm-to-cool gradient for training progression
COLORS = plt.cm.viridis(np.linspace(0.2, 0.9, len(TRAINING_ORDER)))


def extract_training_stage(dataset_name: str) -> str:
    """Extract training stage from dataset name like 'contamination_dolci_rl_100pct'."""
    # Remove 'contamination_' prefix and '_100pct' suffix
    name = dataset_name.replace('contamination_', '').replace('_100pct', '')
    
    # Map to our canonical names
    if name == 'dolma':
        return 'dolma'
    elif name == 'dolmino':
        return 'dolmino'
    elif name == 'dolci_sft':
        return 'dolci_sft'
    elif name == 'dolci_dpo':
        return 'dolci_dpo'
    elif name == 'dolci_rl':
        return 'dolci_rl'
    elif name == 'dolci':
        return 'dolci'
    else:
        return name  # Unknown, return as-is


def extract_grid_size(test_id: str) -> tuple:
    """Extract grid dimensions from test_id like 'lgp-test-2x2-0'."""
    match = re.search(r'(\d+)x(\d+)', test_id)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)


def load_annotations() -> pd.DataFrame:
    """Load all annotation JSON files into a DataFrame."""
    print("Loading annotations...")
    
    # Get list of files first for progress tracking
    json_files = [f for f in ANNOTATIONS_DIR.glob("*.json") if not f.stem.startswith('_')]
    total_files = len(json_files)
    print(f"  Found {total_files:,} annotation files to load...")
    
    records = []
    for i, json_file in enumerate(json_files):
        # Progress indicator every 1000 files
        if (i + 1) % 1000 == 0 or i == 0:
            print(f"  Loading... {i + 1:,}/{total_files:,} ({100 * (i + 1) / total_files:.1f}%)", flush=True)
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract annotation
            ann = data.get('annotation', {}) or {}
            
            # Determine if semantic duplicate
            is_sd = ann.get('is_sd', False)
            if not is_sd:
                # Check match_type as fallback
                is_sd = ann.get('match_type', 'unrelated') != 'unrelated'
            
            # Extract grid size from test_id
            test_id = data.get('test_id', '')
            grid = extract_grid_size(test_id)
            
            records.append({
                'test_id': test_id,
                'corpus_id': data.get('corpus_id', ''),
                'dataset': data.get('dataset', ''),
                'training_stage': extract_training_stage(data.get('dataset', '')),
                'score': data.get('score', 0),
                'is_sd': is_sd,
                'match_type': ann.get('match_type', 'unknown'),
                'confidence': ann.get('confidence', 0),
                'grid_rows': grid[0],
                'grid_cols': grid[1],
                'grid_size': f"{grid[0]}x{grid[1]}",
            })
        except Exception as e:
            print(f"  Error loading {json_file.name}: {e}")
            continue
    
    df = pd.DataFrame(records)
    print(f"  Loaded {len(df):,} annotations")
    print(f"  Training stages found: {sorted(df['training_stage'].unique())}")
    print(f"  Grid sizes found: {sorted(df['grid_size'].unique())}")
    
    return df


def plot_duplicate_boxplot(df: pd.DataFrame):
    """Plot 1: Box-and-whisker for duplicate frequency per test_id by training stage."""
    print("\nGenerating Plot 1: Duplicate frequency boxplot...")
    
    # Filter to training stages we have data for
    available_stages = [s for s in TRAINING_ORDER if s in df['training_stage'].values]
    available_labels = [TRAINING_LABELS[TRAINING_ORDER.index(s)] for s in available_stages]
    
    if not available_stages:
        print("  No training stages found, skipping plot")
        return
    
    # Calculate duplicate rate per test_id per training stage
    duplicate_rates = df.groupby(['training_stage', 'test_id']).agg(
        n_duplicates=('is_sd', 'sum'),
        n_total=('is_sd', 'count')
    ).reset_index()
    duplicate_rates['duplicate_rate'] = duplicate_rates['n_duplicates'] / duplicate_rates['n_total']
    
    # Order the datasets
    duplicate_rates['training_stage'] = pd.Categorical(
        duplicate_rates['training_stage'],
        categories=available_stages,
        ordered=True
    )
    duplicate_rates = duplicate_rates.sort_values('training_stage')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create box plot data
    box_data = [
        duplicate_rates[duplicate_rates['training_stage'] == ds]['duplicate_rate'].values
        for ds in available_stages
    ]
    
    # Filter out empty arrays
    non_empty_data = [(d, l, s) for d, l, s in zip(box_data, available_labels, available_stages) if len(d) > 0]
    if not non_empty_data:
        print("  No data for boxplot, skipping")
        return
    
    box_data = [d[0] for d in non_empty_data]
    labels = [d[1] for d in non_empty_data]
    
    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='#e74c3c', markeredgecolor='#c0392b', markersize=8))
    
    # Color the boxes with viridis gradient
    colors = plt.cm.viridis(np.linspace(0.3, 0.85, len(box_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor('#2c3e50')
        patch.set_linewidth(1.5)
    
    # Style whiskers and caps
    for element in ['whiskers', 'caps']:
        for item in bp[element]:
            item.set_color('#34495e')
            item.set_linewidth(1.2)
    
    # Add mean values as text
    means = [np.mean(d) for d in box_data]
    for i, (mean, x) in enumerate(zip(means, range(1, len(box_data) + 1))):
        ax.annotate(f'{mean:.1%}', xy=(x, mean), xytext=(x + 0.25, mean + 0.02),
                    fontsize=10, color='#c0392b', fontweight='bold')
    
    ax.set_ylabel('Semantic Duplicate Rate (per ZebraLogic test puzzle)', fontsize=12)
    ax.set_xlabel('Training Stage', fontsize=12)
    ax.set_title('Semantic Duplicate Frequency Across Training Pipeline\n(ZebraLogic Benchmark)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, min(1.0, max(max(d) for d in box_data) * 1.15))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add annotation for training flow
    ax.annotate('', xy=(0.95, -0.15), xytext=(0.05, -0.15),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2))
    ax.text(0.5, -0.18, 'Training Progression', transform=ax.transAxes,
            ha='center', fontsize=10, color='#7f8c8d', style='italic')
    
    plt.subplots_adjust(bottom=0.2)
    
    # Save
    output_base = OUTPUT_DIR / "zebralogic_duplicate_boxplot"
    plt.savefig(f"{output_base}.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{output_base}.pdf", bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_base}.png/pdf")
    plt.close()


def plot_similarity_vs_duplicate(df: pd.DataFrame):
    """Plot 2: Semantic duplicate rate as function of cosine similarity."""
    print("\nGenerating Plot 2: Similarity vs duplicate rate...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Bin similarity scores
    df['similarity_bin'] = pd.cut(df['score'], bins=25)
    
    # Calculate duplicate rate per bin across ALL data
    sim_analysis = df.groupby('similarity_bin', observed=True).agg(
        n_duplicates=('is_sd', 'sum'),
        n_total=('is_sd', 'count')
    ).reset_index()
    sim_analysis['duplicate_rate'] = sim_analysis['n_duplicates'] / sim_analysis['n_total']
    sim_analysis['bin_center'] = sim_analysis['similarity_bin'].apply(
        lambda x: x.mid if pd.notna(x) else np.nan
    )
    sim_analysis = sim_analysis.sort_values('bin_center')
    
    # Plot curve with gradient fill
    ax.plot(sim_analysis['bin_center'], sim_analysis['duplicate_rate'],
            marker='o', linewidth=2.5, markersize=8,
            color='#2980b9', alpha=0.9, markeredgecolor='#1a5276')
    
    # Fill under curve with gradient effect
    ax.fill_between(sim_analysis['bin_center'], sim_analysis['duplicate_rate'],
                    alpha=0.25, color='#3498db')
    
    # Add sample size annotation for significant bins
    for i, row in sim_analysis.iterrows():
        if row['n_total'] > 200:  # Only annotate bins with substantial samples
            ax.annotate(f"n={row['n_total']:,}",
                       xy=(row['bin_center'], row['duplicate_rate']),
                       xytext=(0, 12), textcoords='offset points',
                       fontsize=7, alpha=0.6, ha='center', rotation=45)
    
    ax.set_xlabel('Cosine Similarity Score', fontsize=12)
    ax.set_ylabel('Semantic Duplicate Rate', fontsize=12)
    ax.set_title('Semantic Duplicate Rate vs. Cosine Similarity\n(ZebraLogic Benchmark - All Training Stages Combined)', 
                 fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(df['score'].min() * 0.98, df['score'].max() * 1.01)
    ax.set_ylim(0, min(1.0, sim_analysis['duplicate_rate'].max() * 1.15))
    
    plt.tight_layout()
    
    # Save
    output_base = OUTPUT_DIR / "zebralogic_similarity_vs_duplicate"
    plt.savefig(f"{output_base}.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{output_base}.pdf", bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_base}.png/pdf")
    plt.close()


def plot_grid_size_analysis(df: pd.DataFrame, match_type_filter: str = None, suffix: str = ""):
    """Plot 3: Test puzzle contamination rate by grid size (unique test IDs with ≥1 SD).
    
    Args:
        df: DataFrame with annotations
        match_type_filter: If set, only count this match_type (e.g., 'exact')
        suffix: Suffix for output filename
    """
    filter_label = f" ({match_type_filter} only)" if match_type_filter else ""
    print(f"\nGenerating Plot 3{suffix}: Test puzzle contamination rate by grid size{filter_label}...")
    
    # Calculate unique test IDs with at least one SD per grid size
    grid_test_ids = df.groupby('grid_size')['test_id'].nunique().reset_index()
    grid_test_ids.columns = ['grid_size', 'n_test_ids']
    
    # Get test IDs that have at least one SD (optionally filtered by match_type)
    if match_type_filter:
        sd_df = df[df['match_type'] == match_type_filter]
    else:
        sd_df = df[df['is_sd'] == True]
    grid_sd_test_ids = sd_df.groupby('grid_size')['test_id'].nunique().reset_index()
    grid_sd_test_ids.columns = ['grid_size', 'n_contaminated']
    
    # Merge
    grid_analysis = grid_test_ids.merge(grid_sd_test_ids, on='grid_size', how='left')
    grid_analysis['n_contaminated'] = grid_analysis['n_contaminated'].fillna(0).astype(int)
    grid_analysis['contamination_rate'] = grid_analysis['n_contaminated'] / grid_analysis['n_test_ids']
    
    # Sort by grid complexity (rows * cols)
    grid_analysis['complexity'] = grid_analysis['grid_size'].apply(
        lambda x: int(x.split('x')[0]) * int(x.split('x')[1]) if 'x' in x else 0
    )
    grid_analysis = grid_analysis.sort_values('complexity')
    
    if len(grid_analysis) == 0:
        print("  Not enough data per grid size, skipping")
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(grid_analysis))
    width = 0.6
    
    # Create gradient colors based on contamination rate (red = more contaminated)
    colors = plt.cm.RdYlGn_r(grid_analysis['contamination_rate'].values)
    
    bars = ax.bar(x, grid_analysis['contamination_rate'], width, color=colors, 
                  edgecolor='#2c3e50', linewidth=1.2, alpha=0.85)
    
    # Dynamic labels based on filter
    if match_type_filter:
        ylabel = f'% puzzles ≥ 1 {match_type_filter} duplicate'
        title = f'ZebraLogic Test Puzzle Contamination Rate by Grid Size\n(% of test puzzles with at least one {match_type_filter} duplicate in training)'
    else:
        ylabel = '% puzzles ≥ 1 semantic duplicate'
        title = 'ZebraLogic Test Puzzle Contamination Rate by Grid Size\n(% of test puzzles with at least one semantic duplicate in training)'
    
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel('Puzzle Grid Size (rows × columns)', fontsize=16)
    # Title saved to file, not shown on image
    ax.set_xticks(x)
    ax.set_xticklabels(grid_analysis['grid_size'], fontsize=12, rotation=30, ha='right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.05)  # Max 105%
    
    # Add complexity arrow
    ax.annotate('', xy=(0.95, -0.14), xytext=(0.05, -0.14),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2))
    ax.text(0.5, -0.19, 'Increasing Puzzle Complexity', transform=ax.transAxes,
            ha='center', fontsize=16, color='#7f8c8d', style='italic')
    
    plt.subplots_adjust(bottom=0.20)
    
    # Save
    filename = f"zebralogic_gridsize_contamination_rate{suffix}"
    output_base = OUTPUT_DIR / filename
    plt.savefig(f"{output_base}.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{output_base}.pdf", bbox_inches='tight', facecolor='white')
    
    # Save title to text file
    with open(f"{output_base}_title.txt", 'w', encoding='utf-8') as f:
        f.write(title)
    
    print(f"  Saved: {output_base}.png/pdf/title.txt")
    plt.close()


def plot_heatmap_stage_vs_grid(df: pd.DataFrame):
    """Plot 4: Heatmap of duplicate rate by training stage and grid size."""
    print("\nGenerating Plot 4: Training stage vs grid size heatmap...")
    
    # Filter to training stages we have
    available_stages = [s for s in TRAINING_ORDER if s in df['training_stage'].values]
    
    if len(available_stages) < 2:
        print("  Need at least 2 training stages, skipping heatmap")
        return
    
    # Create pivot table
    pivot = df.groupby(['training_stage', 'grid_size']).agg(
        n_duplicates=('is_sd', 'sum'),
        n_total=('is_sd', 'count')
    ).reset_index()
    pivot['duplicate_rate'] = pivot['n_duplicates'] / pivot['n_total']
    
    # Filter to significant combinations
    pivot = pivot[pivot['n_total'] >= 30]
    
    # Pivot for heatmap
    heatmap_data = pivot.pivot(index='grid_size', columns='training_stage', values='duplicate_rate')
    
    # Reorder columns by training order
    ordered_cols = [s for s in available_stages if s in heatmap_data.columns]
    heatmap_data = heatmap_data[ordered_cols]
    
    # Sort rows by grid complexity
    heatmap_data['complexity'] = heatmap_data.index.map(
        lambda x: int(x.split('x')[0]) * int(x.split('x')[1]) if 'x' in x else 0
    )
    heatmap_data = heatmap_data.sort_values('complexity')
    heatmap_data = heatmap_data.drop('complexity', axis=1)
    
    if heatmap_data.empty:
        print("  Not enough data for heatmap, skipping")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.1%', cmap='YlOrRd',
                ax=ax, linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Semantic Duplicate Rate', 'format': '%.0f%%'})
    
    # Fix labels
    stage_labels = {s: TRAINING_LABELS[TRAINING_ORDER.index(s)].replace('\n', ' ') 
                    for s in ordered_cols}
    ax.set_xticklabels([stage_labels.get(c, c) for c in ordered_cols], rotation=45, ha='right')
    
    ax.set_ylabel('Puzzle Grid Size', fontsize=12)
    ax.set_xlabel('Training Stage', fontsize=12)
    ax.set_title('Semantic Duplicate Rate: Training Stage × Grid Size\n(ZebraLogic Benchmark)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_base = OUTPUT_DIR / "zebralogic_heatmap_stage_grid"
    plt.savefig(f"{output_base}.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{output_base}.pdf", bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_base}.png/pdf")
    plt.close()


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("ZEBRALOGIC SUMMARY STATISTICS")
    print("=" * 70)
    
    # Overall
    print(f"\nOverall:")
    print(f"  Total annotations: {len(df):,}")
    print(f"  Semantic duplicates: {df['is_sd'].sum():,} ({100 * df['is_sd'].mean():.2f}%)")
    print(f"  Similarity score range: {df['score'].min():.4f} - {df['score'].max():.4f}")
    print(f"  Mean similarity: {df['score'].mean():.4f}")
    
    # By training stage
    print("\nDuplicate rates by training stage:")
    available_stages = [s for s in TRAINING_ORDER if s in df['training_stage'].values]
    for stage in available_stages:
        stage_data = df[df['training_stage'] == stage]
        dup_rate = stage_data['is_sd'].mean()
        label = TRAINING_LABELS[TRAINING_ORDER.index(stage)].replace('\n', ' ')
        print(f"  {label:25s}: {dup_rate:6.2%} ({stage_data['is_sd'].sum():,} / {len(stage_data):,})")
    
    # By grid size
    print("\nDuplicate rates by grid size:")
    grid_stats = df.groupby('grid_size').agg(
        n_sd=('is_sd', 'sum'),
        n_total=('is_sd', 'count')
    )
    grid_stats['rate'] = grid_stats['n_sd'] / grid_stats['n_total']
    grid_stats = grid_stats.sort_values('rate', ascending=False)
    for grid, row in grid_stats.iterrows():
        if row['n_total'] >= 50:
            print(f"  {grid:8s}: {row['rate']:6.2%} ({row['n_sd']:,} / {row['n_total']:,})")
    
    # Match type distribution
    print("\nMatch type distribution:")
    match_counts = df['match_type'].value_counts()
    for mt, count in match_counts.items():
        print(f"  {mt:15s}: {count:,} ({100 * count / len(df):.1f}%)")
    
    print("\n" + "=" * 70)


def main():
    """Generate all plots for ZebraLogic benchmark."""
    print("=" * 70)
    print("ZEBRALOGIC SEMANTIC DUPLICATE ANALYSIS")
    print("=" * 70)
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_annotations()
    
    if len(df) == 0:
        print("No annotations found!")
        return 1
    
    # Generate plots
    plot_duplicate_boxplot(df)
    plot_similarity_vs_duplicate(df)
    plot_grid_size_analysis(df)  # All semantic duplicates
    plot_grid_size_analysis(df, match_type_filter='exact', suffix='_exact')  # Exact only
    plot_heatmap_stage_vs_grid(df)
    
    # Print summary
    print_summary_statistics(df)
    
    print("\nAll plots generated successfully!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
