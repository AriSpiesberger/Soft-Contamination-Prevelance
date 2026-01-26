"""
Codeforces v2 Semantic Duplicate Analysis Plots
1. Box-and-whisker plot: duplicate frequency by training stage
2. Semantic duplicate rate as function of cosine similarity
3. Occurrence rate: how many test problems have at least one duplicate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv(r'C:\Users\arisp\Downloads\codeforces_top100_classified_gptoss_v2_checkpoint.csv')

# Filter out "related" category - only keep strong matches (exact, equivalent, subset, superset)
# For predicted_is_duplicate, set to False if category is "related"
strong_categories = ['exact', 'equivalent', 'subset', 'superset']
df['predicted_is_duplicate_strict'] = df['predicted_is_duplicate'] & df['predicted_category'].isin(strong_categories)

print(f"Original duplicates: {df['predicted_is_duplicate'].sum()}")
print(f"Strict duplicates (excluding 'related'): {df['predicted_is_duplicate_strict'].sum()}")
print(f"Removed 'related' cases: {df['predicted_is_duplicate'].sum() - df['predicted_is_duplicate_strict'].sum()}")

# Use strict duplicates for all analyses
df['predicted_is_duplicate'] = df['predicted_is_duplicate_strict']

# Define training order (temporal progression) - note: dolma not in this dataset
training_order = ['dolmino', 'dolci_sft', 'dolci_dpo', 'dolci_rl']
training_labels = ['Dolmino\n(Continued)', 'Dolci SFT\n(SFT)', 'Dolci DPO\n(DPO)', 'Dolci RL\n(RL)']

label_dict = {
    'dolmino': 'Dolmino (Continued)',
    'dolci_sft': 'Dolci SFT',
    'dolci_dpo': 'Dolci DPO',
    'dolci_rl': 'Dolci RL'
}

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

# Create figure - box and whisker plot
fig, ax = plt.subplots(figsize=(10, 6))

# Get data for each dataset
box_data = [duplicate_rates[duplicate_rates['dataset'] == ds]['duplicate_rate'].values
            for ds in training_order]

# Create box plot with whiskers at 5th-95th percentile
stage_colors = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

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
    ax.annotate(f'{mean:.1%}', xy=(x, q3 + 0.02), ha='center',
                fontsize=9, color='black', fontweight='bold')

ax.set_ylabel('Semantic Duplicate Rate (per Codeforces problem)', fontsize=12)
ax.set_xlabel('Training Stage', fontsize=12)
ax.set_title('Semantic Duplicate Frequency Across Training Pipeline\n(Codeforces Benchmark)', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(max(d) if len(d) > 0 else 0 for d in box_data) * 1.2)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax.grid(axis='y', alpha=0.3)

# Add annotation for training flow
ax.annotate('', xy=(0.95, -0.15), xytext=(0.05, -0.15),
            xycoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='gray', lw=2))
ax.text(0.5, -0.18, 'Training Progression', transform=ax.transAxes,
        ha='center', fontsize=10, color='gray', style='italic')

plt.subplots_adjust(bottom=0.2)
plt.savefig(r'C:\Users\arisp\Documents\Research\SDTD_Main\comparison_analysis\plots\codeforces_v2_duplicate_boxplot.png', dpi=150, bbox_inches='tight')
plt.savefig(r'C:\Users\arisp\Documents\Research\SDTD_Main\comparison_analysis\plots\codeforces_v2_duplicate_boxplot.pdf', bbox_inches='tight')
print("Plot 1 saved: codeforces_v2_duplicate_boxplot.png/pdf")

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

# Calculate 95% CI for binomial proportion
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
ax.set_title('Semantic Duplicate Rate vs. Cosine Similarity\n(Codeforces Benchmark - All Training Stages Combined)', fontsize=14, fontweight='bold')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax.grid(alpha=0.3)
ax.set_xlim(df['similarity'].min() * 0.98, df['similarity'].max() * 1.01)
ax.set_ylim(0, (sim_analysis['duplicate_rate'] + sim_analysis['ci95']).max() * 1.1)
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig(r'C:\Users\arisp\Documents\Research\SDTD_Main\comparison_analysis\plots\codeforces_v2_similarity_vs_duplicate.png', dpi=150, bbox_inches='tight')
plt.savefig(r'C:\Users\arisp\Documents\Research\SDTD_Main\comparison_analysis\plots\codeforces_v2_similarity_vs_duplicate.pdf', bbox_inches='tight')
print("Plot 2 saved: codeforces_v2_similarity_vs_duplicate.png/pdf")

# =============================================================================
# Plot 3: Occurrence rate - % of test problems with at least one duplicate
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# For each dataset, calculate what % of test_ids have at least one duplicate
occurrence_data = []
for ds in training_order:
    ds_data = duplicate_rates[duplicate_rates['dataset'] == ds]
    n_test_problems = len(ds_data)
    n_with_dups = (ds_data['n_duplicates'] > 0).sum()
    occurrence_rate = n_with_dups / n_test_problems
    occurrence_data.append({
        'dataset': ds,
        'n_test_problems': n_test_problems,
        'n_with_dups': n_with_dups,
        'occurrence_rate': occurrence_rate
    })

occurrence_df = pd.DataFrame(occurrence_data)

# Calculate 95% CI for binomial proportion
z = 1.96
occurrence_df['ci95'] = z * np.sqrt(occurrence_df['occurrence_rate'] * (1 - occurrence_df['occurrence_rate']) / occurrence_df['n_test_problems'])

# Create bar chart
x_pos = np.arange(len(training_order))
bars = ax.bar(x_pos, occurrence_df['occurrence_rate'], width=0.6,
              color=stage_colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add error bars (95% CI)
ax.errorbar(x_pos, occurrence_df['occurrence_rate'], yerr=occurrence_df['ci95'],
            fmt='none', ecolor='black', capsize=5, capthick=2, elinewidth=2)

# Add percentage labels on top of bars
for i, row in occurrence_df.iterrows():
    ax.annotate(f'{row["occurrence_rate"]:.1%}\n({row["n_with_dups"]}/{row["n_test_problems"]})',
               xy=(i, row['occurrence_rate'] + row['ci95'] + 0.02),
               ha='center', fontsize=9, color='black', fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(training_labels)
ax.set_ylabel('% of Test Problems with ≥1 Semantic Duplicate', fontsize=12)
ax.set_xlabel('Training Stage', fontsize=12)
ax.set_title('Occurrence Rate: Test Problems Affected by Contamination\n(Codeforces Benchmark)', fontsize=14, fontweight='bold')
ax.set_ylim(0, (occurrence_df['occurrence_rate'] + occurrence_df['ci95']).max() * 1.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax.grid(axis='y', alpha=0.3)

# Add annotation for training flow
ax.annotate('', xy=(0.95, -0.15), xytext=(0.05, -0.15),
            xycoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='gray', lw=2))
ax.text(0.5, -0.18, 'Training Progression', transform=ax.transAxes,
        ha='center', fontsize=10, color='gray', style='italic')

plt.subplots_adjust(bottom=0.2)
plt.savefig(r'C:\Users\arisp\Documents\Research\SDTD_Main\comparison_analysis\plots\codeforces_v2_occurrence_rate.png', dpi=150, bbox_inches='tight')
plt.savefig(r'C:\Users\arisp\Documents\Research\SDTD_Main\comparison_analysis\plots\codeforces_v2_occurrence_rate.pdf', bbox_inches='tight')
print("Plot 3 saved: codeforces_v2_occurrence_rate.png/pdf")

# =============================================================================
# Plot 4: Occurrence rate by ELO rating
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 6))

# Calculate occurrence rate per test_id per elo_bin (across all datasets)
elo_occurrence = df.groupby(['elo_bin', 'test_id']).agg(
    n_duplicates=('predicted_is_duplicate', 'sum'),
    n_total=('predicted_is_duplicate', 'count')
).reset_index()
elo_occurrence['has_duplicate'] = elo_occurrence['n_duplicates'] > 0

# For each elo_bin, calculate % of test problems with at least one duplicate
elo_stats = elo_occurrence.groupby('elo_bin').agg(
    n_test_problems=('test_id', 'nunique'),
    n_with_dups=('has_duplicate', 'sum')
).reset_index()
elo_stats['occurrence_rate'] = elo_stats['n_with_dups'] / elo_stats['n_test_problems']

# Calculate 95% CI
z = 1.96
elo_stats['ci95'] = z * np.sqrt(elo_stats['occurrence_rate'] * (1 - elo_stats['occurrence_rate']) / elo_stats['n_test_problems'])

# Sort by ELO
elo_stats = elo_stats.sort_values('elo_bin')

# Create bar chart
x_pos = np.arange(len(elo_stats))
elo_labels = [f"{int(e)}" for e in elo_stats['elo_bin']]

# Color gradient from green (low ELO) to red (high ELO)
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(elo_stats)))

bars = ax.bar(x_pos, elo_stats['occurrence_rate'], width=0.8,
              color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

# Add error bars
ax.errorbar(x_pos, elo_stats['occurrence_rate'], yerr=elo_stats['ci95'],
            fmt='none', ecolor='black', capsize=3, capthick=1, elinewidth=1)

ax.set_xticks(x_pos)
ax.set_xticklabels(elo_labels, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('% of Test Problems with >=1 Semantic Duplicate', fontsize=12)
ax.set_xlabel('Codeforces ELO Rating', fontsize=12)
ax.set_title('Occurrence Rate by Problem Difficulty (ELO)\n(Codeforces Benchmark - All Training Stages Combined)', fontsize=14, fontweight='bold')
ax.set_ylim(0, (elo_stats['occurrence_rate'] + elo_stats['ci95']).max() * 1.2)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax.grid(axis='y', alpha=0.3)

# Add trend line
from scipy import stats as scipy_stats
slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(elo_stats['elo_bin'], elo_stats['occurrence_rate'])
trend_line = slope * elo_stats['elo_bin'] + intercept
ax.plot(x_pos, trend_line, 'k--', linewidth=2, alpha=0.7, label=f'Trend (r={r_value:.2f})')
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig(r'C:\Users\arisp\Documents\Research\SDTD_Main\comparison_analysis\plots\codeforces_v2_occurrence_by_elo.png', dpi=150, bbox_inches='tight')
plt.savefig(r'C:\Users\arisp\Documents\Research\SDTD_Main\comparison_analysis\plots\codeforces_v2_occurrence_by_elo.pdf', bbox_inches='tight')
print("Plot 4 saved: codeforces_v2_occurrence_by_elo.png/pdf")

# =============================================================================
# Plot 5: Average occurrence COUNT by ELO rating
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 6))

# For each elo_bin, calculate average number of duplicates per test problem
elo_count_stats = elo_occurrence.groupby('elo_bin').agg(
    n_test_problems=('test_id', 'nunique'),
    total_duplicates=('n_duplicates', 'sum'),
    mean_duplicates=('n_duplicates', 'mean'),
    std_duplicates=('n_duplicates', 'std')
).reset_index()

# Calculate standard error for the mean
elo_count_stats['se'] = elo_count_stats['std_duplicates'] / np.sqrt(elo_count_stats['n_test_problems'])
elo_count_stats['ci95'] = 1.96 * elo_count_stats['se']

# Sort by ELO
elo_count_stats = elo_count_stats.sort_values('elo_bin')

# Create bar chart
x_pos = np.arange(len(elo_count_stats))
elo_labels = [f"{int(e)}" for e in elo_count_stats['elo_bin']]

# Color gradient from green (low ELO) to red (high ELO)
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(elo_count_stats)))

bars = ax.bar(x_pos, elo_count_stats['mean_duplicates'], width=0.8,
              color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

# Add error bars (95% CI)
ax.errorbar(x_pos, elo_count_stats['mean_duplicates'], yerr=elo_count_stats['ci95'],
            fmt='none', ecolor='black', capsize=3, capthick=1, elinewidth=1)

ax.set_xticks(x_pos)
ax.set_xticklabels(elo_labels, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Average Number of Semantic Duplicates per Problem', fontsize=12)
ax.set_xlabel('Codeforces ELO Rating', fontsize=12)
ax.set_title('Average Duplicate Count by Problem Difficulty (ELO)\n(Codeforces Benchmark - All Training Stages Combined)', fontsize=14, fontweight='bold')
ax.set_ylim(0, (elo_count_stats['mean_duplicates'] + elo_count_stats['ci95']).max() * 1.2)
ax.grid(axis='y', alpha=0.3)

# Add trend line
slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(elo_count_stats['elo_bin'], elo_count_stats['mean_duplicates'])
trend_line = slope * elo_count_stats['elo_bin'] + intercept
ax.plot(x_pos, trend_line, 'k--', linewidth=2, alpha=0.7, label=f'Trend (r={r_value:.2f})')
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig(r'C:\Users\arisp\Documents\Research\SDTD_Main\comparison_analysis\plots\codeforces_v2_avg_count_by_elo.png', dpi=150, bbox_inches='tight')
plt.savefig(r'C:\Users\arisp\Documents\Research\SDTD_Main\comparison_analysis\plots\codeforces_v2_avg_count_by_elo.pdf', bbox_inches='tight')
print("Plot 5 saved: codeforces_v2_avg_count_by_elo.png/pdf")

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

print("\nOccurrence rates (% of test problems with ≥1 duplicate):")
for i, row in occurrence_df.iterrows():
    print(f"  {label_dict[row['dataset']]:25s}: {row['occurrence_rate']:6.1%} ({row['n_with_dups']}/{row['n_test_problems']} problems)")

print("\nSimilarity score range:", f"{df['similarity'].min():.4f} - {df['similarity'].max():.4f}")
print(f"Mean similarity: {df['similarity'].mean():.4f}")

print("\nPlots generated successfully!")
