#!/usr/bin/env python3
"""
Unified plotting script for semantic duplicate analysis.
Generates publication-ready figures (no titles) for ICML.

Datasets:
- MBPP top100, MBPP sample100
- CodeForces top100, CodeForces sample100

Plots:
1. Box-and-whisker: duplicate frequency by training stage
2a. Duplicate rate vs similarity (CI only)
2b. Duplicate rate vs similarity (CI + n counts on secondary axis)
3. Occurrence rate by training stage
4. Occurrence rate by ELO (CodeForces only)
5. Average duplicate count by ELO (CodeForces only)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as scipy_stats

# Configure matplotlib for publication quality - LARGE fonts for small figures
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 28,
    'axes.titlesize': 28,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 18,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

# Paths
DATA_DIR = Path(__file__).parent / "data"
# Plots go in respective data folders, not a central plots folder

# Dataset configurations
DATASETS = {
    'mbpp_top100': {
        'path': DATA_DIR / 'mbpp_top100' / 'mbpp_top100_classified.csv',
        'output_dir': DATA_DIR / 'mbpp_top100',
        'name': 'MBPP Top-100',
        'has_elo': False,
        'training_order': ['dolma', 'dolmino', 'dolci_sft', 'dolci_dpo', 'dolci_rl'],
        'training_labels': ['Dolma\n(Pretrain)', 'Dolmino\n(Continued)', 'Dolci SFT\n(SFT)', 'Dolci DPO\n(DPO)', 'Dolci RL\n(RL)'],
        'colors': ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'],
    },
    'mbpp_sample100': {
        'path': DATA_DIR / 'mbpp_sample100' / 'mbpp_sample100_classified.csv',
        'output_dir': DATA_DIR / 'mbpp_sample100',
        'name': 'MBPP Sample-100',
        'has_elo': False,
        'training_order': ['dolma', 'dolmino', 'dolci_sft', 'dolci_dpo', 'dolci_rl'],
        'training_labels': ['Dolma\n(Pretrain)', 'Dolmino\n(Continued)', 'Dolci SFT\n(SFT)', 'Dolci DPO\n(DPO)', 'Dolci RL\n(RL)'],
        'colors': ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'],
    },
    'codeforces_top100': {
        'path': DATA_DIR / 'codeforces_top100' / 'codeforces_top100_classified_gptoss_v2.csv',
        'output_dir': DATA_DIR / 'codeforces_top100',
        'name': 'CodeForces Top-100',
        'has_elo': True,
        'training_order': ['dolma', 'dolmino', 'dolci_sft', 'dolci_dpo', 'dolci_rl'],  # Now includes dolma
        'training_labels': ['Dolma\n(Pretrain)', 'Dolmino\n(Continued)', 'Dolci SFT\n(SFT)', 'Dolci DPO\n(DPO)', 'Dolci RL\n(RL)'],
        'colors': ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'],
    },
    'codeforces_sample100': {
        'path': DATA_DIR / 'codeforces_sample100' / 'codeforces_sample100_classified_gptoss_v2.csv',
        'output_dir': DATA_DIR / 'codeforces_sample100',
        'name': 'CodeForces Sample-100',
        'has_elo': True,
        'training_order': ['dolma', 'dolmino', 'dolci_sft', 'dolci_dpo', 'dolci_rl'],
        'training_labels': ['Dolma\n(Pretrain)', 'Dolmino\n(Continued)', 'Dolci SFT\n(SFT)', 'Dolci DPO\n(DPO)', 'Dolci RL\n(RL)'],
        'colors': ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'],
    },
}


def load_data(dataset_key):
    """Load and preprocess dataset."""
    config = DATASETS[dataset_key]
    print(f"Loading {config['name']} from {config['path']}...")
    df = pd.read_csv(config['path'], low_memory=False)
    print(f"  Loaded {len(df):,} rows")

    # Normalize column names: 'score' -> 'similarity'
    if 'score' in df.columns and 'similarity' not in df.columns:
        df['similarity'] = df['score']
        print(f"  Renamed 'score' to 'similarity'")

    # Check for predicted_category column (for strict filtering)
    # Only apply strict filtering if the column exists AND has non-null values
    if 'predicted_category' in df.columns and df['predicted_category'].notna().any():
        # Keep original duplicate flag (includes exact) for "with exact" similarity plot
        df['predicted_is_duplicate_with_exact'] = df['predicted_is_duplicate'].copy()

        # Apply semantic filtering (drop exact and related) for main duplicate flag
        semantic_categories = ['equivalent', 'subset', 'superset']  # Drop 'exact' and 'related'
        original_dups = df['predicted_is_duplicate'].sum()
        df['predicted_is_duplicate'] = df['predicted_is_duplicate'] & df['predicted_category'].isin(semantic_categories)
        new_dups = df['predicted_is_duplicate'].sum()
        print(f"  Semantic filtering (drop exact/related): {original_dups:,} -> {new_dups:,} duplicates")
    elif 'predicted_category' in df.columns:
        print(f"  No category data - skipping strict filtering")
        df['predicted_is_duplicate_with_exact'] = df['predicted_is_duplicate'].copy()
    else:
        df['predicted_is_duplicate_with_exact'] = df['predicted_is_duplicate'].copy()

    # Filter to only datasets in training_order
    df = df[df['dataset'].isin(config['training_order'])]
    print(f"  After filtering datasets: {len(df):,} rows")

    # Check if similarity column exists
    if 'similarity' not in df.columns:
        print(f"  WARNING: No similarity column found - similarity plots will be skipped")

    return df, config


def plot_boxplot(df, config, dataset_key):
    """Plot 1: Box-and-whisker for duplicate frequency by training stage.

    NORMALIZED: All boxplots use identical formatting for side-by-side comparison.
    """

    # Calculate duplicate rate per test_id per dataset
    duplicate_rates = df.groupby(['dataset', 'test_id']).agg(
        n_duplicates=('predicted_is_duplicate', 'sum'),
        n_total=('predicted_is_duplicate', 'count')
    ).reset_index()
    duplicate_rates['duplicate_rate'] = duplicate_rates['n_duplicates'] / duplicate_rates['n_total']

    # Order datasets
    duplicate_rates['dataset'] = pd.Categorical(
        duplicate_rates['dataset'],
        categories=config['training_order'],
        ordered=True
    )
    duplicate_rates = duplicate_rates.sort_values('dataset')

    # =========================================================================
    # NORMALIZED SETTINGS - IDENTICAL FOR ALL DATASETS
    # =========================================================================
    FIGSIZE = (14, 8)           # Wider figure size
    Y_LIM = (0, 1.0)            # Same y-axis range (0-100%)
    YLABEL = 'Sem Dupe Rate (per problem)'  # Same y-label
    XLABEL = 'Training Stage'
    LABEL_FONTSIZE = 28         # Axis labels
    TICK_FONTSIZE = 22          # Tick labels
    ANNOT_FONTSIZE = 24         # Mean value annotations
    ARROW_FONTSIZE = 18         # Training progression text
    # =========================================================================

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Get data for each dataset
    box_data = [duplicate_rates[duplicate_rates['dataset'] == ds]['duplicate_rate'].values
                for ds in config['training_order']]

    # Create box plot with consistent styling
    bp = ax.boxplot(box_data, tick_labels=config['training_labels'], patch_artist=True, showmeans=True,
                    widths=0.55,
                    whis=[5, 95],
                    showfliers=False,
                    whiskerprops=dict(linewidth=2, color='black'),
                    capprops=dict(linewidth=2, color='black'),
                    medianprops=dict(linewidth=2.5, color='white'),
                    meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=10))

    # Color boxes - same colors for all datasets
    for patch, color in zip(bp['boxes'], config['colors']):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    # Axis labels and formatting - IDENTICAL for all
    ax.set_ylabel(YLABEL, fontsize=LABEL_FONTSIZE)
    ax.set_xlabel(XLABEL, fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)

    # Y-axis: SAME range for all datasets
    ax.set_ylim(Y_LIM)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.grid(axis='y', alpha=0.3)

    # Add mean values above upper whisker (95th percentile)
    LABEL_OFFSET = 0.03  # Offset above whisker (3% on 0-100% scale)

    for i, x in enumerate(range(1, len(config['training_order']) + 1)):
        if len(box_data[i]) > 0:
            mean = np.mean(box_data[i])
            upper_whisker = np.percentile(box_data[i], 95)
            label_y = upper_whisker + LABEL_OFFSET
            ax.annotate(f'{mean:.1%}', xy=(x, label_y), ha='center',
                        fontsize=ANNOT_FONTSIZE, color='black', fontweight='bold')

    # Training flow arrow - positioned below x-axis label
    ax.annotate('', xy=(0.95, -0.22), xytext=(0.05, -0.22),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(0.5, -0.27, 'Training Progression', transform=ax.transAxes,
            ha='center', fontsize=ARROW_FONTSIZE, color='gray', style='italic')

    plt.subplots_adjust(bottom=0.25)

    # Save
    plt.savefig(config['output_dir'] / f'{dataset_key}_boxplot.png', dpi=300, bbox_inches='tight')
    plt.savefig(config['output_dir'] / f'{dataset_key}_boxplot.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {dataset_key}_boxplot.png/pdf")


def plot_similarity_semantic(df, config, dataset_key):
    """Plot: Semantic duplicate rate vs similarity (separate plot)."""

    if 'similarity' not in df.columns:
        print(f"  Skipping similarity semantic plot - no similarity column")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Bin similarity scores
    df_copy = df.copy()
    df_copy['similarity_bin'] = pd.cut(df_copy['similarity'], bins=30)

    sim_analysis = df_copy.groupby('similarity_bin', observed=True).agg(
        n_duplicates=('predicted_is_duplicate', 'sum'),
        n_total=('predicted_is_duplicate', 'count')
    ).reset_index()
    sim_analysis['duplicate_rate'] = sim_analysis['n_duplicates'] / sim_analysis['n_total']
    sim_analysis['bin_center'] = sim_analysis['similarity_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan)

    # Calculate 95% CI
    z = 1.96
    p = sim_analysis['duplicate_rate']
    n = sim_analysis['n_total']
    sim_analysis['ci95'] = z * np.sqrt(p * (1 - p) / n)
    sim_analysis = sim_analysis.sort_values('bin_center')

    # Plot CI shading
    ax.fill_between(sim_analysis['bin_center'],
                    sim_analysis['duplicate_rate'] - sim_analysis['ci95'],
                    sim_analysis['duplicate_rate'] + sim_analysis['ci95'],
                    alpha=0.3, color='#3498db', label='95% CI')

    # Plot line
    ax.plot(sim_analysis['bin_center'], sim_analysis['duplicate_rate'],
            marker='o', linewidth=2.5, markersize=8,
            color='#2c3e50', alpha=0.9, label='Semantic Duplicate Rate')

    ax.set_xlabel('Cosine Similarity', fontsize=24)
    ax.set_ylabel('Semantic Duplicate Rate', fontsize=24)
    ax.tick_params(axis='both', labelsize=18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.grid(alpha=0.3)
    ax.set_xlim(df['similarity'].min() * 0.98, df['similarity'].max() * 1.01)
    max_y = min((sim_analysis['duplicate_rate'] + sim_analysis['ci95']).max() * 1.15, 1.0)
    ax.set_ylim(0, max_y)
    ax.legend(loc='upper left', fontsize=16)

    plt.tight_layout()
    plt.savefig(config['output_dir'] / f'{dataset_key}_similarity_semantic.png', dpi=300, bbox_inches='tight')
    plt.savefig(config['output_dir'] / f'{dataset_key}_similarity_semantic.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {dataset_key}_similarity_semantic.png/pdf")


def plot_similarity_with_exact_only(df, config, dataset_key):
    """Plot: Duplicate rate with exact vs similarity (separate plot)."""

    if 'similarity' not in df.columns:
        print(f"  Skipping similarity with exact plot - no similarity column")
        return

    if 'predicted_is_duplicate_with_exact' not in df.columns:
        print(f"  Skipping similarity with exact plot - no exact duplicate column")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Bin similarity scores
    df_copy = df.copy()
    df_copy['similarity_bin'] = pd.cut(df_copy['similarity'], bins=30)

    sim_analysis = df_copy.groupby('similarity_bin', observed=True).agg(
        n_duplicates=('predicted_is_duplicate_with_exact', 'sum'),
        n_total=('predicted_is_duplicate_with_exact', 'count')
    ).reset_index()
    sim_analysis['duplicate_rate'] = sim_analysis['n_duplicates'] / sim_analysis['n_total']
    sim_analysis['bin_center'] = sim_analysis['similarity_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan)

    # Calculate 95% CI
    z = 1.96
    p = sim_analysis['duplicate_rate']
    n = sim_analysis['n_total']
    sim_analysis['ci95'] = z * np.sqrt(p * (1 - p) / n)
    sim_analysis = sim_analysis.sort_values('bin_center')

    # Plot CI shading
    ax.fill_between(sim_analysis['bin_center'],
                    sim_analysis['duplicate_rate'] - sim_analysis['ci95'],
                    sim_analysis['duplicate_rate'] + sim_analysis['ci95'],
                    alpha=0.3, color='#e74c3c', label='95% CI')

    # Plot line
    ax.plot(sim_analysis['bin_center'], sim_analysis['duplicate_rate'],
            marker='o', linewidth=2.5, markersize=8,
            color='#c0392b', alpha=0.9, label='Duplicate Rate (incl. Exact)')

    ax.set_xlabel('Cosine Similarity', fontsize=24)
    ax.set_ylabel('Duplicate Rate (Including Exact)', fontsize=24)
    ax.tick_params(axis='both', labelsize=18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.grid(alpha=0.3)
    ax.set_xlim(df['similarity'].min() * 0.98, df['similarity'].max() * 1.01)
    max_y = min((sim_analysis['duplicate_rate'] + sim_analysis['ci95']).max() * 1.15, 1.0)
    ax.set_ylim(0, max_y)
    ax.legend(loc='upper left', fontsize=16)

    plt.tight_layout()
    plt.savefig(config['output_dir'] / f'{dataset_key}_similarity_with_exact.png', dpi=300, bbox_inches='tight')
    plt.savefig(config['output_dir'] / f'{dataset_key}_similarity_with_exact.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {dataset_key}_similarity_with_exact.png/pdf")


def plot_similarity_overlay(df, config, dataset_key):
    """
    Publication-quality plot: Semantic duplicate rate vs cosine similarity.
    Shows semantic duplicates with CI, and exact duplicates overlaid.

    Requirements:
    - Semantic duplicate line with 95% CI shading (primary)
    - Exact duplicate line overlaid on top
    - No title
    - Y-axis: "Duplicate Rate"
    - Consistent x-axis range (0.3 to 1.0) for cross-dataset comparison
    """

    if 'similarity' not in df.columns:
        print(f"  Skipping similarity overlay plot - no similarity column")
        return

    if 'predicted_is_duplicate_with_exact' not in df.columns:
        print(f"  Skipping similarity overlay plot - no exact duplicate column")
        return

    # =========================================================================
    # PUBLICATION SETTINGS
    # =========================================================================
    FIGSIZE = (12, 8)
    X_RANGE = (0.30, 1.0)       # Fixed x-axis range for all plots
    N_BINS = 25                  # Number of similarity bins

    # Colors - elegant, publication-ready
    SEMANTIC_COLOR = '#2563eb'   # Rich blue
    SEMANTIC_CI_COLOR = '#93c5fd' # Light blue for CI
    EXACT_COLOR = '#dc2626'      # Rich red
    EXACT_CI_COLOR = '#fca5a5'   # Light red for CI

    # Font sizes
    AXIS_LABEL_SIZE = 26
    TICK_SIZE = 20
    LEGEND_SIZE = 18
    # =========================================================================

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Bin similarity scores with fixed range
    df_copy = df.copy()
    bins = np.linspace(X_RANGE[0], X_RANGE[1], N_BINS + 1)
    df_copy['similarity_bin'] = pd.cut(df_copy['similarity'], bins=bins)
    z = 1.96

    # === LAYER 1: Semantic Duplicates (base layer) ===
    sim_semantic = df_copy.groupby('similarity_bin', observed=True).agg(
        n_duplicates=('predicted_is_duplicate', 'sum'),
        n_total=('predicted_is_duplicate', 'count')
    ).reset_index()
    sim_semantic['duplicate_rate'] = sim_semantic['n_duplicates'] / sim_semantic['n_total']
    sim_semantic['bin_center'] = sim_semantic['similarity_bin'].apply(
        lambda x: x.mid if pd.notna(x) else np.nan
    )
    # Wilson score CI for better small-sample behavior
    p = sim_semantic['duplicate_rate']
    n = sim_semantic['n_total']
    sim_semantic['ci95'] = z * np.sqrt(p * (1 - p) / n)
    sim_semantic = sim_semantic.dropna(subset=['bin_center']).sort_values('bin_center')

    # Plot semantic CI shading first (background)
    ax.fill_between(sim_semantic['bin_center'],
                    np.maximum(0, sim_semantic['duplicate_rate'] - sim_semantic['ci95']),
                    np.minimum(1, sim_semantic['duplicate_rate'] + sim_semantic['ci95']),
                    alpha=0.35, color=SEMANTIC_CI_COLOR, linewidth=0,
                    label='Semantic 95% CI')

    # Plot semantic line
    ax.plot(sim_semantic['bin_center'], sim_semantic['duplicate_rate'],
            marker='o', linewidth=3, markersize=8,
            color=SEMANTIC_COLOR, markerfacecolor='white',
            markeredgewidth=2, markeredgecolor=SEMANTIC_COLOR,
            label='Semantic Duplicates', zorder=5)

    # === LAYER 2: With Exact Duplicates (overlaid) ===
    sim_exact = df_copy.groupby('similarity_bin', observed=True).agg(
        n_duplicates=('predicted_is_duplicate_with_exact', 'sum'),
        n_total=('predicted_is_duplicate_with_exact', 'count')
    ).reset_index()
    sim_exact['duplicate_rate'] = sim_exact['n_duplicates'] / sim_exact['n_total']
    sim_exact['bin_center'] = sim_exact['similarity_bin'].apply(
        lambda x: x.mid if pd.notna(x) else np.nan
    )
    p = sim_exact['duplicate_rate']
    n = sim_exact['n_total']
    sim_exact['ci95'] = z * np.sqrt(p * (1 - p) / n)
    sim_exact = sim_exact.dropna(subset=['bin_center']).sort_values('bin_center')

    # Plot exact CI shading
    ax.fill_between(sim_exact['bin_center'],
                    np.maximum(0, sim_exact['duplicate_rate'] - sim_exact['ci95']),
                    np.minimum(1, sim_exact['duplicate_rate'] + sim_exact['ci95']),
                    alpha=0.25, color=EXACT_CI_COLOR, linewidth=0,
                    label='With Exact 95% CI')

    # Plot exact line
    ax.plot(sim_exact['bin_center'], sim_exact['duplicate_rate'],
            marker='s', linewidth=3, markersize=8,
            color=EXACT_COLOR, markerfacecolor='white',
            markeredgewidth=2, markeredgecolor=EXACT_COLOR,
            label='Including Exact', zorder=6)

    # === FORMATTING ===
    ax.set_xlabel('Cosine Similarity', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('Duplicate Rate', fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    # Fixed x-axis range for consistency across datasets
    ax.set_xlim(X_RANGE)

    # Y-axis: auto-scale but with some headroom
    max_y = min(max((sim_semantic['duplicate_rate'] + sim_semantic['ci95']).max(),
                    (sim_exact['duplicate_rate'] + sim_exact['ci95']).max()) * 1.12, 1.0)
    ax.set_ylim(0, max_y)

    # Subtle grid
    ax.grid(alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend - clean positioning
    legend = ax.legend(loc='upper left', fontsize=LEGEND_SIZE, framealpha=0.95,
                       edgecolor='#cccccc', fancybox=False)
    legend.get_frame().set_linewidth(0.8)

    # Clean spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1.2)
        ax.spines[spine].set_color('#333333')

    plt.tight_layout()
    plt.savefig(config['output_dir'] / f'{dataset_key}_similarity_overlay.png', dpi=300, bbox_inches='tight')
    plt.savefig(config['output_dir'] / f'{dataset_key}_similarity_overlay.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {dataset_key}_similarity_overlay.png/pdf")


def plot_similarity_with_n(df, config, dataset_key):
    """Plot 2b: Duplicate rate vs similarity with n counts on secondary axis."""

    if 'similarity' not in df.columns:
        print(f"  Skipping similarity+n plot - no similarity column")
        return

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Bin similarity scores
    df_copy = df.copy()
    df_copy['similarity_bin'] = pd.cut(df_copy['similarity'], bins=30)

    # Calculate duplicate rate per bin
    sim_analysis = df_copy.groupby('similarity_bin', observed=True).agg(
        n_duplicates=('predicted_is_duplicate', 'sum'),
        n_total=('predicted_is_duplicate', 'count')
    ).reset_index()
    sim_analysis['duplicate_rate'] = sim_analysis['n_duplicates'] / sim_analysis['n_total']
    sim_analysis['bin_center'] = sim_analysis['similarity_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan)

    # Calculate 95% CI
    z = 1.96
    p = sim_analysis['duplicate_rate']
    n = sim_analysis['n_total']
    sim_analysis['ci95'] = z * np.sqrt(p * (1 - p) / n)
    sim_analysis = sim_analysis.sort_values('bin_center')

    # Left axis: Duplicate rate with CI
    ax1.fill_between(sim_analysis['bin_center'],
                     sim_analysis['duplicate_rate'] - sim_analysis['ci95'],
                     sim_analysis['duplicate_rate'] + sim_analysis['ci95'],
                     alpha=0.3, color='#3498db', label='95% CI')

    line1, = ax1.plot(sim_analysis['bin_center'], sim_analysis['duplicate_rate'],
                      marker='o', linewidth=2.5, markersize=7,
                      color='#2c3e50', alpha=0.9, label='Duplicate Rate')

    ax1.set_xlabel('Cosine Similarity', fontsize=16)
    ax1.set_ylabel('Semantic Duplicate Rate', fontsize=16, color='#2c3e50')
    ax1.tick_params(axis='y', labelcolor='#2c3e50', labelsize=13)
    ax1.tick_params(axis='x', labelsize=13)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax1.set_xlim(df['similarity'].min() * 0.98, df['similarity'].max() * 1.01)
    max_y = min((sim_analysis['duplicate_rate'] + sim_analysis['ci95']).max() * 1.15, 1.0)
    ax1.set_ylim(0, max_y)
    ax1.grid(alpha=0.3)

    # Right axis: Sample count (n)
    ax2 = ax1.twinx()
    line2, = ax2.plot(sim_analysis['bin_center'], sim_analysis['n_total'],
                      marker='s', linewidth=2, markersize=5,
                      color='#e74c3c', alpha=0.7, linestyle='--', label='Sample Count (n)')

    ax2.set_ylabel('Sample Count per Bin', fontsize=16, color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c', labelsize=13)

    # Format n axis
    def format_n(n, pos):
        if n >= 1000:
            return f'{n/1000:.0f}k'
        return f'{n:.0f}'
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_n))
    ax2.set_ylim(0, sim_analysis['n_total'].max() * 1.15)

    # Combined legend
    lines = [line1, line2]
    labels = ['Duplicate Rate', 'Sample Count (n)']
    ax1.legend(lines, labels, loc='upper left', fontsize=12)

    plt.tight_layout()
    plt.savefig(config['output_dir'] / f'{dataset_key}_similarity_with_n.png', dpi=300, bbox_inches='tight')
    plt.savefig(config['output_dir'] / f'{dataset_key}_similarity_with_n.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {dataset_key}_similarity_with_n.png/pdf")


def plot_occurrence_rate(df, config, dataset_key):
    """Plot 3: Occurrence rate - % of test problems with at least one duplicate."""

    # Calculate duplicate rate per test_id per dataset
    duplicate_rates = df.groupby(['dataset', 'test_id']).agg(
        n_duplicates=('predicted_is_duplicate', 'sum'),
        n_total=('predicted_is_duplicate', 'count')
    ).reset_index()

    fig, ax = plt.subplots(figsize=(11, 7))

    # Calculate occurrence rate per dataset
    occurrence_data = []
    for ds in config['training_order']:
        ds_data = duplicate_rates[duplicate_rates['dataset'] == ds]
        n_test_problems = len(ds_data)
        n_with_dups = (ds_data['n_duplicates'] > 0).sum()
        occurrence_rate = n_with_dups / n_test_problems if n_test_problems > 0 else 0
        occurrence_data.append({
            'dataset': ds,
            'n_test_problems': n_test_problems,
            'n_with_dups': n_with_dups,
            'occurrence_rate': occurrence_rate
        })

    occurrence_df = pd.DataFrame(occurrence_data)

    # Calculate 95% CI
    z = 1.96
    occurrence_df['ci95'] = z * np.sqrt(
        occurrence_df['occurrence_rate'] * (1 - occurrence_df['occurrence_rate']) / occurrence_df['n_test_problems']
    )

    # Create bar chart
    x_pos = np.arange(len(config['training_order']))
    bars = ax.bar(x_pos, occurrence_df['occurrence_rate'], width=0.6,
                  color=config['colors'], alpha=0.75, edgecolor='black', linewidth=1.5)

    # Error bars
    ax.errorbar(x_pos, occurrence_df['occurrence_rate'], yerr=occurrence_df['ci95'],
                fmt='none', ecolor='black', capsize=6, capthick=2, elinewidth=2)

    # Labels on bars - percentage only
    # For MBPP, put labels inside/below bars since values are high
    is_mbpp = 'mbpp' in dataset_key.lower()
    for i, row in occurrence_df.iterrows():
        if is_mbpp:
            # Labels inside the bars for MBPP (high values)
            ax.annotate(f'{row["occurrence_rate"]:.1%}',
                        xy=(i, row['occurrence_rate'] - 0.08),
                        ha='center', fontsize=20, color='white', fontweight='bold')
        else:
            # Labels above bars for CodeForces
            ax.annotate(f'{row["occurrence_rate"]:.1%}',
                        xy=(i, row['occurrence_rate'] + row['ci95'] + 0.02),
                        ha='center', fontsize=20, color='black', fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(config['training_labels'], fontsize=16)
    ax.set_ylabel('% Problems with ≥1 Duplicate', fontsize=22, labelpad=10)
    ax.yaxis.set_label_coords(-0.09, 0.4)  # Shift label down
    ax.set_xlabel('Training Stage', fontsize=22)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylim(0, 1.0)  # Always show full 0-100% range
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.grid(axis='y', alpha=0.3)

    # Training flow arrow
    ax.annotate('', xy=(0.95, -0.22), xytext=(0.05, -0.22),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(0.5, -0.27, 'Training Progression', transform=ax.transAxes,
            ha='center', fontsize=18, color='gray', style='italic')

    plt.subplots_adjust(bottom=0.25)
    plt.savefig(config['output_dir'] / f'{dataset_key}_occurrence_rate.png', dpi=300, bbox_inches='tight')
    plt.savefig(config['output_dir'] / f'{dataset_key}_occurrence_rate.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {dataset_key}_occurrence_rate.png/pdf")


def plot_elo_occurrence(df, config, dataset_key):
    """Plot 4: Occurrence rate by ELO (CodeForces only)."""

    if 'elo_bin' not in df.columns:
        print(f"  Skipping ELO occurrence plot - no elo_bin column")
        return

    # Calculate occurrence per test_id per elo
    elo_occurrence = df.groupby(['elo_bin', 'test_id']).agg(
        n_duplicates=('predicted_is_duplicate', 'sum'),
        n_total=('predicted_is_duplicate', 'count')
    ).reset_index()
    elo_occurrence['has_duplicate'] = elo_occurrence['n_duplicates'] > 0

    # Aggregate by elo_bin
    elo_stats = elo_occurrence.groupby('elo_bin').agg(
        n_test_problems=('test_id', 'nunique'),
        n_with_dups=('has_duplicate', 'sum')
    ).reset_index()
    elo_stats['occurrence_rate'] = elo_stats['n_with_dups'] / elo_stats['n_test_problems']

    # CI
    z = 1.96
    elo_stats['ci95'] = z * np.sqrt(
        elo_stats['occurrence_rate'] * (1 - elo_stats['occurrence_rate']) / elo_stats['n_test_problems']
    )
    elo_stats = elo_stats.sort_values('elo_bin')

    fig, ax = plt.subplots(figsize=(14, 7))

    x_pos = np.arange(len(elo_stats))
    elo_labels = [f"{int(e)}" for e in elo_stats['elo_bin']]

    # Color bars based on occurrence rate (green to red)
    colors = plt.cm.RdYlGn_r(elo_stats['occurrence_rate'].values)

    bars = ax.bar(x_pos, elo_stats['occurrence_rate'], width=0.8,
                  color=colors, alpha=0.85, edgecolor='#2c3e50', linewidth=1.2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(elo_labels, rotation=45, ha='right', fontsize=18)
    ax.set_ylabel('% Problems ≥1 Duplicate', fontsize=24)
    ax.set_xlabel('CodeForces ELO Rating', fontsize=24)
    ax.tick_params(axis='y', labelsize=18)

    # Set y-axis to full 0-100% range
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Difficulty arrow at bottom - moved down to avoid overlap
    ax.annotate('', xy=(0.95, -0.28), xytext=(0.05, -0.28),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2))
    ax.text(0.5, -0.33, 'Increasing Difficulty', transform=ax.transAxes,
            ha='center', fontsize=16, color='#7f8c8d', style='italic')

    plt.subplots_adjust(bottom=0.30)
    plt.savefig(config['output_dir'] / f'{dataset_key}_elo_occurrence.png', dpi=300, bbox_inches='tight')
    plt.savefig(config['output_dir'] / f'{dataset_key}_elo_occurrence.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {dataset_key}_elo_occurrence.png/pdf")


def plot_elo_count(df, config, dataset_key):
    """Plot 5: Average duplicate count by ELO (CodeForces only)."""

    if 'elo_bin' not in df.columns:
        print(f"  Skipping ELO count plot - no elo_bin column")
        return

    # Calculate per test_id per elo
    elo_occurrence = df.groupby(['elo_bin', 'test_id']).agg(
        n_duplicates=('predicted_is_duplicate', 'sum'),
        n_total=('predicted_is_duplicate', 'count')
    ).reset_index()

    # Aggregate by elo_bin
    elo_count_stats = elo_occurrence.groupby('elo_bin').agg(
        n_test_problems=('test_id', 'nunique'),
        total_duplicates=('n_duplicates', 'sum'),
        mean_duplicates=('n_duplicates', 'mean'),
        std_duplicates=('n_duplicates', 'std')
    ).reset_index()

    # SE and CI
    elo_count_stats['se'] = elo_count_stats['std_duplicates'] / np.sqrt(elo_count_stats['n_test_problems'])
    elo_count_stats['ci95'] = 1.96 * elo_count_stats['se']
    elo_count_stats = elo_count_stats.sort_values('elo_bin')

    fig, ax = plt.subplots(figsize=(14, 7))

    x_pos = np.arange(len(elo_count_stats))
    elo_labels = [f"{int(e)}" for e in elo_count_stats['elo_bin']]

    # Color bars based on mean duplicates (normalized to data range)
    means = elo_count_stats['mean_duplicates'].values
    norm_means = (means - means.min()) / (means.max() - means.min() + 1e-9)
    colors = plt.cm.RdYlGn_r(norm_means)

    bars = ax.bar(x_pos, elo_count_stats['mean_duplicates'], width=0.8,
                  color=colors, alpha=0.85, edgecolor='#2c3e50', linewidth=1.2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(elo_labels, rotation=45, ha='right', fontsize=18)
    ax.set_ylabel('Avg Semantic Duplicates per Problem', fontsize=24)
    ax.set_xlabel('CodeForces ELO Rating', fontsize=24)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylim(0, elo_count_stats['mean_duplicates'].max() * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Difficulty arrow at bottom - moved down to avoid overlap
    ax.annotate('', xy=(0.95, -0.28), xytext=(0.05, -0.28),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2))
    ax.text(0.5, -0.33, 'Increasing Difficulty', transform=ax.transAxes,
            ha='center', fontsize=16, color='#7f8c8d', style='italic')

    plt.subplots_adjust(bottom=0.30)
    plt.savefig(config['output_dir'] / f'{dataset_key}_elo_count.png', dpi=300, bbox_inches='tight')
    plt.savefig(config['output_dir'] / f'{dataset_key}_elo_count.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {dataset_key}_elo_count.png/pdf")


def generate_all_plots(dataset_key):
    """Generate all plots for a dataset."""
    print(f"\n{'='*60}")
    print(f"GENERATING PLOTS: {dataset_key}")
    print(f"{'='*60}")

    df, config = load_data(dataset_key)

    # Always generate these
    plot_boxplot(df, config, dataset_key)
    plot_similarity_semantic(df, config, dataset_key)
    plot_similarity_with_exact_only(df, config, dataset_key)
    plot_similarity_overlay(df, config, dataset_key)
    plot_occurrence_rate(df, config, dataset_key)

    # ELO plots only for CodeForces
    if config['has_elo']:
        plot_elo_occurrence(df, config, dataset_key)
        plot_elo_count(df, config, dataset_key)

    print(f"  Done with {dataset_key}!")


def plot_figure4_combined():
    """
    Figure 4: Combined boxplot for MBPP and CodeForces (side-by-side).
    Both subplots must have identical formatting.
    """
    print(f"\n{'='*60}")
    print("GENERATING FIGURE 4: Combined Boxplots")
    print(f"{'='*60}")

    # Load both datasets
    mbpp_df, mbpp_config = load_data('mbpp_sample100')
    cf_df, cf_config = load_data('codeforces_top100')

    # Use consistent training order and labels for both
    training_order = ['dolma', 'dolmino', 'dolci_sft', 'dolci_dpo', 'dolci_rl']
    training_labels = ['Dolma\n(Pretrain)', 'Dolmino\n(Continued)', 'Dolci SFT\n(SFT)', 'Dolci DPO\n(DPO)', 'Dolci RL\n(RL)']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

    # Create figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Common boxplot parameters
    boxplot_params = dict(
        patch_artist=True,
        showmeans=True,
        widths=0.55,
        whis=[5, 95],
        showfliers=False,
        whiskerprops=dict(linewidth=2, color='black'),
        capprops=dict(linewidth=2, color='black'),
        medianprops=dict(linewidth=2.5, color='white'),
        meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=10)
    )

    def prepare_box_data(df):
        """Calculate duplicate rate per test_id per dataset."""
        duplicate_rates = df.groupby(['dataset', 'test_id']).agg(
            n_duplicates=('predicted_is_duplicate', 'sum'),
            n_total=('predicted_is_duplicate', 'count')
        ).reset_index()
        duplicate_rates['duplicate_rate'] = duplicate_rates['n_duplicates'] / duplicate_rates['n_total']
        return duplicate_rates

    def style_boxplot(bp, colors):
        """Apply consistent styling to boxplot."""
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)

    def add_mean_labels(ax, box_data, x_positions):
        """Add mean percentage labels above upper whiskers."""
        for i, (data, x) in enumerate(zip(box_data, x_positions)):
            if len(data) > 0:
                mean = np.mean(data)
                upper_whisker = np.percentile(data, 95)
                ax.annotate(f'{mean:.1%}', xy=(x, upper_whisker + 0.03), ha='center',
                           fontsize=18, color='black', fontweight='bold')

    def format_axis(ax, title):
        """Apply consistent axis formatting."""
        ax.set_ylabel('Sem Dupe Rate (per problem)', fontsize=22)
        ax.set_xlabel('Training Stage', fontsize=22)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_ylim(0, 1.0)  # 0-100% range
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.grid(axis='y', alpha=0.3)
        ax.set_title(title, fontsize=24, fontweight='bold', pad=15)

        # Training flow arrow - positioned below x-axis label
        ax.annotate('', xy=(0.95, -0.22), xytext=(0.05, -0.22),
                   xycoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2))
        ax.text(0.5, -0.27, 'Training Progression', transform=ax.transAxes,
               ha='center', fontsize=14, color='gray', style='italic')

    # ===== LEFT: MBPP =====
    mbpp_rates = prepare_box_data(mbpp_df)
    mbpp_box_data = [mbpp_rates[mbpp_rates['dataset'] == ds]['duplicate_rate'].values
                     for ds in training_order]

    bp1 = ax1.boxplot(mbpp_box_data, tick_labels=training_labels, **boxplot_params)
    style_boxplot(bp1, colors)
    add_mean_labels(ax1, mbpp_box_data, range(1, len(training_order) + 1))
    format_axis(ax1, '(a) MBPP')
    ax1.set_ylabel('Sem Dupe Rate (per MBPP problem)', fontsize=22)

    # ===== RIGHT: CodeForces =====
    cf_rates = prepare_box_data(cf_df)
    cf_box_data = [cf_rates[cf_rates['dataset'] == ds]['duplicate_rate'].values
                   for ds in training_order]

    bp2 = ax2.boxplot(cf_box_data, tick_labels=training_labels, **boxplot_params)
    style_boxplot(bp2, colors)
    add_mean_labels(ax2, cf_box_data, range(1, len(training_order) + 1))
    format_axis(ax2, '(b) CodeForces')
    ax2.set_ylabel('Sem Dupe Rate (per Codeforces problem)', fontsize=22)

    plt.subplots_adjust(bottom=0.25, wspace=0.25)
    plt.savefig(DATA_DIR / 'figure4_boxplots_combined.png', dpi=300, bbox_inches='tight')
    plt.savefig(DATA_DIR / 'figure4_boxplots_combined.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: data/figure4_boxplots_combined.png/pdf")


def plot_scaling_law():
    """
    Scaling Law Plot: P(>=1 semantic duplicate) vs retrieval depth (top-1, top-10, top-100).
    Saves to data/ folder (above benchmark-specific folders).
    """
    print(f"\n{'='*60}")
    print("GENERATING SCALING LAW PLOTS")
    print(f"{'='*60}")

    # Semantic categories (exclude exact and related)
    semantic_categories = ['equivalent', 'subset', 'superset']

    # 4 key depth points for cleaner graph
    default_depths = [1, 10, 100, 500]

    def calculate_scaling_for_dataset(df, depths=None):
        """Calculate P(>=1 semantic duplicate) at different retrieval depths."""
        if depths is None:
            depths = default_depths

        # Add semantic duplicate flag
        df = df.copy()
        if 'predicted_category' in df.columns and df['predicted_category'].notna().any():
            df['is_semantic_duplicate'] = (
                df['predicted_is_duplicate'] &
                df['predicted_category'].isin(semantic_categories)
            )
        else:
            # No category column - use predicted_is_duplicate directly
            df['is_semantic_duplicate'] = df['predicted_is_duplicate']

        results = []
        datasets = df['dataset'].unique()

        for ds in datasets:
            ds_data = df[df['dataset'] == ds]
            test_ids = ds_data['test_id'].unique()

            for depth in depths:
                n_with_dup = 0
                n_total = len(test_ids)

                for test_id in test_ids:
                    matches = ds_data[ds_data['test_id'] == test_id].sort_values('similarity', ascending=False)
                    top_k = matches.head(depth)
                    if top_k['is_semantic_duplicate'].any():
                        n_with_dup += 1

                rate = n_with_dup / n_total if n_total > 0 else 0

                # Wilson CI
                z = 1.96
                if n_total > 0:
                    denom = 1 + z**2/n_total
                    center = (rate + z**2/(2*n_total)) / denom
                    margin = z * np.sqrt((rate*(1-rate) + z**2/(4*n_total))/n_total) / denom
                    ci_lower = max(0, center - margin)
                    ci_upper = min(1, center + margin)
                else:
                    ci_lower = ci_upper = 0

                results.append({
                    'training_stage': ds,
                    'depth': depth,
                    'n_total': n_total,
                    'n_with_dup': n_with_dup,
                    'rate': rate,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                })

        return pd.DataFrame(results)

    # Load all datasets for scaling law comparison
    scaling_datasets = ['mbpp_top100', 'mbpp_sample100', 'codeforces_top100', 'codeforces_sample100']
    all_results = {}

    for ds_key in scaling_datasets:
        config = DATASETS[ds_key]
        if not config['path'].exists():
            print(f"  Skipping {ds_key} - file not found")
            continue

        print(f"  Processing {ds_key}...")
        df = pd.read_csv(config['path'], low_memory=False)
        if 'score' in df.columns and 'similarity' not in df.columns:
            df['similarity'] = df['score']

        results = calculate_scaling_for_dataset(df)
        all_results[ds_key] = results

    if not all_results:
        print("  No datasets available for scaling law plot")
        return

    # Plot: Aggregated scaling law (all stages combined)
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        'mbpp_top100': '#27ae60', 'mbpp_sample100': '#16a085',
        'codeforces_top100': '#2c3e50', 'codeforces_sample100': '#e74c3c'
    }
    markers = {
        'mbpp_top100': '^', 'mbpp_sample100': 'v',
        'codeforces_top100': 'o', 'codeforces_sample100': 's'
    }
    labels = {
        'mbpp_top100': 'MBPP Top-100', 'mbpp_sample100': 'MBPP Sample-100',
        'codeforces_top100': 'CodeForces Top-100', 'codeforces_sample100': 'CodeForces Sample-100'
    }

    for ds_key, results in all_results.items():
        # Aggregate across all training stages
        agg_data = []
        for depth in default_depths:
            depth_data = results[results['depth'] == depth]
            total_with = depth_data['n_with_dup'].sum()
            total_n = depth_data['n_total'].sum()
            rate = total_with / total_n if total_n > 0 else 0

            # Wilson CI for aggregated
            z = 1.96
            denom = 1 + z**2/total_n
            center = (rate + z**2/(2*total_n)) / denom
            margin = z * np.sqrt((rate*(1-rate) + z**2/(4*total_n))/total_n) / denom

            agg_data.append({
                'depth': depth,
                'rate': rate,
                'ci_lower': max(0, center - margin),
                'ci_upper': min(1, center + margin)
            })

        agg_df = pd.DataFrame(agg_data)

        ax.plot(agg_df['depth'], agg_df['rate'],
                marker=markers[ds_key], linewidth=3, markersize=12,
                color=colors[ds_key], label=labels[ds_key],
                markerfacecolor='white', markeredgewidth=2, markeredgecolor=colors[ds_key])

        ax.fill_between(agg_df['depth'], agg_df['ci_lower'], agg_df['ci_upper'],
                       alpha=0.2, color=colors[ds_key])

    ax.set_xlabel('Retrieval Depth (Top-k)', fontsize=14)
    ax.set_ylabel('Semantic Duplicate Rate', fontsize=14)
    ax.set_xscale('log')
    ax.set_xticks([1, 10, 100, 500])
    ax.set_xticklabels(['1', '10', '100', '500'], fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=12)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(DATA_DIR / 'scaling_law_semantic_duplicates.png', dpi=300, bbox_inches='tight')
    plt.savefig(DATA_DIR / 'scaling_law_semantic_duplicates.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: data/scaling_law_semantic_duplicates.png/pdf")


def main():
    print("="*60)
    print("SEMANTIC DUPLICATE ANALYSIS - PLOT GENERATION")
    print("="*60)

    for dataset_key in DATASETS:
        generate_all_plots(dataset_key)

    # Generate Figure 4: Combined boxplots
    plot_figure4_combined()

    # Generate scaling law plot (goes in data/ folder)
    plot_scaling_law()

    print("\n" + "="*60)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print(f"Output directory: {DATA_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
