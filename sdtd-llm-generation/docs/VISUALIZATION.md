# Metrics Visualization

## Overview

The `plot` command creates **corner plots** (also called pair plots) that visualize the joint distributions of key metrics across different semantic duplicate transformations. These plots help you:

1. Compare how different transformations behave across metrics
2. Identify correlations between metrics
3. Understand the quality and characteristics of generated semantic duplicates
4. Detect potential issues or outliers

## Quick Start

**Basic usage** (plot all transformations from a parquet file):
```bash
uv run python -m sdtd.cli plot outputs/gsm8k_level12.parquet
```

**Multiple files** (combine data from multiple runs):
```bash
uv run python -m sdtd.cli plot outputs/gsm8k_level1.parquet outputs/gsm8k_level2.parquet
```

**Custom output location**:
```bash
uv run python -m sdtd.cli plot outputs/gsm8k_level12.parquet -o plots/gsm8k_metrics.png
```

## Plot Structure

The corner plot has two types of subplots:

### 1. Left Column: 1D Density Plots

**What it shows**: Marginal distribution of each metric (how values are distributed)

**Visual elements**:
- Colored lines, one per transformation
- All transformations overlapping in the same plot
- X-axis hidden (density magnitude not important)
- Y-axis shows metric values

**Interpretation**:
- Peak location shows typical value for that transformation
- Width shows variability
- Multiple peaks suggest different behaviors
- Compare shapes across transformations

### 2. Lower Triangle: 2D Joint Distribution Plots

**What it shows**: How two metrics relate to each other

**Visual elements**:
- Scatter points (small dots, subsampled to 100 per transformation)
- KDE contours (3 levels by default, showing density)
- Each transformation has its own color
- All transformations overlap in the same plot

**Interpretation**:
- Scatter shows individual data points
- Contours show where most points cluster
- Tight contours = consistent behavior
- Scattered points = variable behavior
- Separation = transformations differ on these metrics

## Default Metrics

The plot shows these 5 key metrics (based on METRICS_SUMMARY.md):

1. **Bigram overlap (%)** - Word-level 2-gram Jaccard similarity
   - Low (< 30%) indicates good surface-form diversity
   - Level 1 transformations: 2-20%
   - Level 2 transformations: 30-90%

2. **Trigram overlap (%)** - Word-level 3-gram Jaccard similarity
   - Even more sensitive to surface changes
   - Good SDs often have 0-10% trigram overlap

3. **Embedding cosine** - Semantic similarity (0-1 scale)
   - High (> 0.85) indicates meaning preserved
   - Should remain high even when n-grams are low

4. **ROUGE-L F** - Longest common subsequence F-measure (0-1 scale)
   - Measures sequential similarity
   - Useful for detecting structure preservation

5. **Edit distance** - Normalized Levenshtein distance (0-1 scale)
   - 0 = identical, 1 = completely different
   - Measures transformation magnitude

## Command Options

```bash
uv run python -m sdtd.cli plot [OPTIONS] INPUT_FILES...
```

### Input Files (required)

Space-separated list of parquet files to visualize:

```bash
# Single file
uv run python -m sdtd.cli plot outputs/gsm8k_level1.parquet

# Multiple files
uv run python -m sdtd.cli plot outputs/gsm8k_level1.parquet outputs/gsm8k_level2.parquet

# Wildcard (all parquet files)
uv run python -m sdtd.cli plot outputs/*.parquet
```

### `-o, --output` (optional)

Output image file path (default: `outputs/metrics_corner_plot.png`)

**Supported formats**: PNG, PDF, SVG, JPG

```bash
# PNG (default, good for viewing)
uv run python -m sdtd.cli plot data.parquet -o plots/metrics.png

# PDF (good for papers)
uv run python -m sdtd.cli plot data.parquet -o plots/metrics.pdf

# SVG (vector graphics, good for editing)
uv run python -m sdtd.cli plot data.parquet -o plots/metrics.svg
```

### `-t, --transformations` (optional)

Filter to specific transformations (comma-separated, case-sensitive)

**Format**: Transformation names are formatted as `variant\n(model)` where:
- `variant` is the SD variant name (e.g., `number_substitution`, `abstractive_paraphrase`)
- `model` is the short model name (e.g., `claude-sonnet-4.5`, `gpt-5.1`)

**To find available transformations**:
```bash
# List all transformations in a file
uv run python -c "
import polars as pl
df = pl.read_parquet('outputs/gsm8k_level12.parquet')
for row in df.group_by(['sd_variant', 'model_used']).agg(pl.len()):
    print(f\"{row['sd_variant']}\n({row['model_used'].split('/')[-1]})\")
"
```

**Example** (filter to just these transformations):
```bash
# Note: Due to complexity of newline in CLI args, filtering by transformation
# is best done programmatically. The option is available for future enhancement.
```

### `-s, --subsample` (optional)

Maximum points per transformation to plot (default: 100)

**Why subsample?**
- Large datasets (1000+ points) can make plots cluttered
- KDE computation is expensive for many points
- 100 points is usually sufficient for visualization

```bash
# More points (more accurate, slower)
uv run python -m sdtd.cli plot data.parquet -s 200

# Fewer points (faster, less cluttered)
uv run python -m sdtd.cli plot data.parquet -s 50

# No subsampling (use all points)
uv run python -m sdtd.cli plot data.parquet -s 999999
```

### `-k, --kde-levels` (optional)

Number of KDE contour levels (default: 3)

**Lower values** (1-2): Simpler, cleaner look
**Higher values** (4-6): More detail, can be cluttered

```bash
# Minimal contours
uv run python -m sdtd.cli plot data.parquet -k 2

# Detailed contours
uv run python -m sdtd.cli plot data.parquet -k 5
```

## Example Use Cases

### Compare Level 1 vs Level 2 Transformations

```bash
# Generate data
uv run python -m sdtd.cli generate -d gsm8k -l 1,2 -n 50

# Plot both levels together
uv run python -m sdtd.cli plot outputs/gsm8k_level12.parquet -o plots/level_comparison.png
```

**What to look for**:
- Level 1: Low bigram overlap (< 30%), high embedding similarity (> 0.85)
- Level 2: Higher bigram overlap (30-90%), still high embedding similarity
- Clear separation in bigram vs embedding plot

### Compare Different Datasets

```bash
# Generate data for multiple datasets
uv run python -m sdtd.cli generate -d gsm8k -l 1 -n 50
uv run python -m sdtd.cli generate -d codeforces -l 1 -n 50
uv run python -m sdtd.cli generate -d allenai -l 1 -n 50

# Plot all together
uv run python -m sdtd.cli plot outputs/*_level1.parquet -o plots/dataset_comparison.png
```

**What to look for**:
- Do transformations behave similarly across datasets?
- Are there dataset-specific patterns?
- Which datasets have more variability?

### Analyze Transformation Quality

```bash
# Generate full dataset
uv run python -m sdtd.cli generate -d gsm8k -l 1 -n 500

# Create detailed plot
uv run python -m sdtd.cli plot outputs/gsm8k_level1.parquet -s 200 -k 5 -o plots/quality_analysis.png
```

**What to look for**:
- **Good transformations**: Low bigram overlap + high embedding similarity
- **Bad transformations**: Either high bigram overlap (not diverse enough) OR low embedding similarity (meaning changed)
- **Outliers**: Points far from KDE contours may indicate problems

### Compare Model Variants

If you generated SDs using different models (via `-m` override), compare their performance:

```bash
# Generate with different models
uv run python -m sdtd.cli generate -d gsm8k -l 1 -n 50 -m openrouter/anthropic/claude-sonnet-4.5 -o outputs/claude_sonnet/
uv run python -m sdtd.cli generate -d gsm8k -l 1 -n 50 -m openrouter/openai/gpt-4-turbo -o outputs/gpt4/

# Plot comparison
uv run python -m sdtd.cli plot outputs/claude_sonnet/*.parquet outputs/gpt4/*.parquet -o plots/model_comparison.png
```

## Interpreting the Plots

### Ideal Type C Duplicate Characteristics

From METRICS_SUMMARY.md, good semantic duplicates should show:

1. **Bigram overlap**: < 30% (low lexical overlap)
2. **Trigram overlap**: < 20% (even lower)
3. **Embedding cosine**: > 0.85 (high semantic similarity)
4. **ROUGE-L**: Variable (depends on transformation type)
5. **Edit distance**: 0.4-0.9 (significant surface change)

**In the plots, look for**:
- Cluster in bottom-right of "Bigram vs Embedding" plot (low bigram, high embedding)
- Separation between different transformation types
- Tight KDE contours (consistent behavior)

### Warning Signs

**Red flags to watch for**:

1. **High bigram + High embedding**: Not diverse enough (Type A/B duplicates)
   - Shows as cluster in top-right of bigram vs embedding plot
   - May indicate transformation didn't change much

2. **Low bigram + Low embedding**: Meaning changed (not semantic duplicates)
   - Shows as cluster in bottom-left of bigram vs embedding plot
   - Indicates transformation was too aggressive

3. **Very scattered points**: Inconsistent transformation behavior
   - Wide, flat KDE contours
   - May indicate transformation needs tuning

4. **Outliers**: Individual problematic examples
   - Points far from KDE contours
   - Should investigate these specific examples

### Comparing Transformations

**Level 1 (Linguistic Paraphrases)**:
- Lexical Maximalist: Moderate bigram (15-20%), high embedding (> 0.92)
- Syntactic Restructuring: Low bigram (10-20%), highest embedding (> 0.94)
- Abstractive Paraphrase: Very low bigram (< 5%), high embedding (> 0.90)
- Compositional: Low bigram (< 10%), moderate embedding (0.75-0.90)

**Level 2 (Structural Duplicates)**:
- Number Substitution: High bigram (70-90%), very high embedding (> 0.95)
- Object Substitution: Moderate bigram (40-60%), high embedding (> 0.90)
- Domain Transfer: Lower bigram (30-50%), moderate embedding (0.70-0.85)
- Compositional: Variable depending on combination

## Advanced: Programmatic Usage

You can also use the visualization module directly in Python:

```python
from pathlib import Path
from sdtd.visualize import create_corner_plot, create_metric_comparison, DEFAULT_METRICS

# Create corner plot with all default metrics
fig = create_corner_plot(
    parquet_files=[Path("outputs/gsm8k_level12.parquet")],
    subsample_size=100,
    output_path=Path("plots/corner.png"),
)

# Create single 2D comparison plot
from sdtd.visualize import create_metric_comparison

x_metric = ("Bigram overlap (%)", lambda m: m["ngram_overlaps_pct"][1])
y_metric = ("Embedding cosine", lambda m: m["cosine_similarity"])

fig = create_metric_comparison(
    parquet_files=[Path("outputs/gsm8k_level12.parquet")],
    x_metric=x_metric,
    y_metric=y_metric,
    output_path=Path("plots/bigram_vs_embedding.png"),
)

# Custom metrics
custom_metrics = [
    ("4-gram (%)", lambda m: m["ngram_overlaps_pct"][3]),
    ("5-gram (%)", lambda m: m["ngram_overlaps_pct"][4]),
    ("Jaccard", lambda m: m["jaccard_token"]),
]

fig = create_corner_plot(
    parquet_files=[Path("outputs/data.parquet")],
    metric_extractors=custom_metrics,
    output_path=Path("plots/custom_metrics.png"),
)
```

## Troubleshooting

### "Warning: 1D KDE failed... singular covariance matrix"

**Cause**: Too few data points (< 3) or all values identical for a metric

**Solution**:
- Generate more samples (`-n 50` or higher)
- This warning can be safely ignored for small test runs

### "Warning: KDE failed for..."

**Cause**: Degenerate data distribution (all points on a line, etc.)

**Solution**:
- Usually safe to ignore - scatter points still shown
- Indicates very consistent behavior for that transformation

### Plot looks cluttered

**Solutions**:
- Reduce subsample size: `-s 50`
- Reduce KDE levels: `-k 2`
- Filter to fewer transformations: `-t ...`
- Increase figure size in code

### Can't see transformation labels in legend

**Solution**:
- Transformation names include model in parentheses, which can be long
- Save as larger format: PDF or SVG
- Edit figure size in code
- Create multiple plots with fewer transformations each

## Performance

**Typical timing** (on typical hardware):

| Dataset Size | Transformations | Time |
|--------------|----------------|------|
| 24 samples (3 each × 8 variants) | 8 | ~2 seconds |
| 80 samples (10 each × 8 variants) | 8 | ~3 seconds |
| 400 samples (50 each × 8 variants) | 8 | ~5 seconds |
| 4000 samples (500 each × 8 variants) | 8 | ~10 seconds |

**Factors affecting speed**:
- Number of transformations (more = slower KDE computation)
- Subsample size (larger = more scatter points and KDE accuracy)
- KDE levels (more = more contour calculations)
- Number of metrics (5 metrics = 10 2D plots + 5 1D plots)

## Best Practices

1. **For initial testing**: Use `-n 10` when generating data, quick visual check
2. **For analysis**: Use `-n 50` to `-n 100`, shows patterns clearly
3. **For publication**: Use `-n 500+`, save as PDF/SVG for high quality
4. **Compare carefully**: Plot Level 1 and Level 2 separately first, then together
5. **Check outliers**: If you see scattered points, investigate with `dump` command
6. **Save intermediate plots**: Create timestamped filenames for comparisons

## References

- Metrics definitions: `docs/METRICS.md`
- Metrics summary and thresholds: `docs/METRICS_SUMMARY.md`
- Transformation methodology: `docs/METHODOLOGY.md`
