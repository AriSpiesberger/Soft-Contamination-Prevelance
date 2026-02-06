"""
MBPP Semantic Duplicate Analysis Plots
1. Box-and-whisker plot: duplicate frequency by training stage
2. Semantic duplicate rate as function of cosine similarity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
from pathlib import Path
df = pd.read_csv(Path(__file__).parent / 'data' / 'mbpp_sample100' / 'mbpp_sample100_classified.csv')

# Define training order (temporal progression)
training_order = ['dolma', 'dolmino', 'dolci_sft', 'dolci_dpo', 'dolci_rl']
training_labels = ['Dolma\n(Pretrain)', 'Dolmino\n(Continued)', 'Dolci SFT\n(SFT)', 'Dolci DPO\n(DPO)', 'Dolci RL\n(RL)']

# =============================================================================
# Plot 1: Box-and-whisker for duplicate frequency per test_id by training stage
# =============================================================================

# Calculate duplicate rate per test_id per dataset
duplicate_rates = df.groupby(['dataset', 'test_id']).agg(
    n_duplicates=('predicted_is_duplicate', 'sum'),
    n_total=('predicted_is_duplicate', 'count')
).reset_index()
duplicate_rates['duplicate_rate'] = duplicate_rates['n_duplicates'] / duplicate_rates['n_total']

# Order the datasets
duplicate_rates['dataset'] = pd.Categorical(
    duplicate_rates['dataset'],
    categories=training_order,
    ordered=True
)
duplicate_rates = duplicate_rates.sort_values('dataset')

# Create figure - box and whisker plot (Q1 to Q3, no outliers beyond whiskers)
fig, ax = plt.subplots(figsize=(10, 6))

# Get data for each dataset
box_data = [duplicate_rates[duplicate_rates['dataset'] == ds]['duplicate_rate'].values
            for ds in training_order]

# Create box plot with whiskers at Q1 and Q3 (whis=0 means whiskers end at box edges)
stage_colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

bp = ax.boxplot(box_data, tick_labels=training_labels, patch_artist=True, showmeans=True,
                widths=0.5,
                whis=[5, 95],  # Whiskers span 5th to 95th percentile
                showfliers=False,  # No outlier dots
                whiskerprops=dict(linewidth=2, color='black'),
                capprops=dict(linewidth=2, color='black'),
                medianprops=dict(linewidth=2, color='white'),
                meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=7))

# Color the boxes
for patch, color in zip(bp['boxes'], stage_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

# Add mean values as text
means = [np.mean(d) for d in box_data]
for i, (mean, x) in enumerate(zip(means, range(1, len(training_order) + 1))):
    q3 = np.percentile(box_data[i], 75)
    ax.annotate(f'{mean:.1%}', xy=(x, q3 + 0.03), ha='center',
                fontsize=9, color='black', fontweight='bold')

ax.set_ylabel('Semantic Duplicate Rate (per MBPP test problem)', fontsize=12)
ax.set_xlabel('Training Stage', fontsize=12)
ax.set_title('Semantic Duplicate Frequency Across Training Pipeline\n(MBPP Benchmark)', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(max(d) for d in box_data) * 1.1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax.grid(axis='y', alpha=0.3)

# Add annotation for training flow
ax.annotate('', xy=(0.95, -0.15), xytext=(0.05, -0.15),
            xycoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='gray', lw=2))
ax.text(0.5, -0.18, 'Training Progression', transform=ax.transAxes,
        ha='center', fontsize=10, color='gray', style='italic')

plt.subplots_adjust(bottom=0.2)
plt.savefig(str(Path(__file__).parent / 'plots' / 'mbpp_duplicate_boxplot.png'), dpi=150, bbox_inches='tight')
plt.savefig(str(Path(__file__).parent / 'plots' / 'mbpp_duplicate_boxplot.pdf'), bbox_inches='tight')
print("Plot 1 saved: mbpp_duplicate_boxplot.png/pdf")

# =============================================================================
# Plot 2: Semantic duplicate rate as function of cosine similarity (ALL data)
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

# Bin similarity scores
df['similarity_bin'] = pd.cut(df['similarity'], bins=30)

# Calculate duplicate rate per bin across ALL data (regardless of training set)
sim_analysis = df.groupby('similarity_bin', observed=True).agg(
    n_duplicates=('predicted_is_duplicate', 'sum'),
    n_total=('predicted_is_duplicate', 'count')
).reset_index()
sim_analysis['duplicate_rate'] = sim_analysis['n_duplicates'] / sim_analysis['n_total']
sim_analysis['bin_center'] = sim_analysis['similarity_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan)

# Calculate 95% CI for binomial proportion (Wilson score interval approximation)
z = 1.96
p = sim_analysis['duplicate_rate']
n = sim_analysis['n_total']
sim_analysis['ci95'] = z * np.sqrt(p * (1 - p) / n)

sim_analysis = sim_analysis.sort_values('bin_center')

# Plot curve with confidence interval shading
ax.fill_between(sim_analysis['bin_center'],
                sim_analysis['duplicate_rate'] - sim_analysis['ci95'],
                sim_analysis['duplicate_rate'] + sim_analysis['ci95'],
                alpha=0.3, color='#3498db', label='95% CI')

ax.plot(sim_analysis['bin_center'], sim_analysis['duplicate_rate'],
        marker='o', linewidth=2.5, markersize=6,
        color='#2c3e50', alpha=0.9, label='Duplicate Rate')

# Add sample size annotation for ALL points
def format_n(n):
    if n >= 1000:
        return f"n={n/1000:.1f}k"
    return f"n={n}"

for i, row in sim_analysis.iterrows():
    ax.annotate(format_n(row['n_total']),
               xy=(row['bin_center'], row['duplicate_rate']),
               xytext=(0, 10), textcoords='offset points',
               fontsize=6, alpha=0.6, ha='center', rotation=45)

ax.set_xlabel('Cosine Similarity', fontsize=12)
ax.set_ylabel('Semantic Duplicate Rate', fontsize=12)
ax.set_title('Semantic Duplicate Rate vs. Cosine Similarity\n(MBPP Benchmark - All Training Stages Combined)', fontsize=14, fontweight='bold')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax.grid(alpha=0.3)
ax.set_xlim(df['similarity'].min() * 0.98, df['similarity'].max() * 1.01)
ax.set_ylim(0, (sim_analysis['duplicate_rate'] + sim_analysis['ci95']).max() * 1.1)
ax.legend(loc='upper left', fontsize=10)

label_dict = {
    'dolma': 'Dolma (Pretrain)',
    'dolmino': 'Dolmino (Continued)',
    'dolci_sft': 'Dolci SFT',
    'dolci_dpo': 'Dolci DPO',
    'dolci_rl': 'Dolci RL'
}

plt.tight_layout()
plt.savefig(str(Path(__file__).parent / 'plots' / 'mbpp_similarity_vs_duplicate.png'), dpi=150, bbox_inches='tight')
plt.savefig(str(Path(__file__).parent / 'plots' / 'mbpp_similarity_vs_duplicate.pdf'), bbox_inches='tight')
print("Plot 2 saved: mbpp_similarity_vs_duplicate.png/pdf")

# =============================================================================
# Summary Statistics
# =============================================================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print("\nOverall duplicate rates by training stage:")
for ds in training_order:
    ds_data = df[df['dataset'] == ds]
    dup_rate = ds_data['predicted_is_duplicate'].mean()
    print(f"  {label_dict[ds]:25s}: {dup_rate:6.2%} ({ds_data['predicted_is_duplicate'].sum():,} / {len(ds_data):,})")

print("\nSimilarity score range:", f"{df['similarity'].min():.4f} - {df['similarity'].max():.4f}")
print(f"Mean similarity: {df['similarity'].mean():.4f}")

print("\nPlots generated successfully!")
# plt.show()  # Commented out for non-interactive use
