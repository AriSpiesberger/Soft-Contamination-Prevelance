"""Visualization tools for semantic duplicate metrics."""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Any
import json
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter


# Default metrics to plot (most informative from METRICS_SUMMARY.md)
DEFAULT_METRICS = [
    ("Embedding cosine", lambda m: m["cosine_similarity"]),
    ("Bigram overlap (%)", lambda m: m["ngram_overlaps_pct"][1]),
    ("Trigram overlap (%)", lambda m: m["ngram_overlaps_pct"][2]),
    ("ROUGE-L F", lambda m: m["rouge_l_f"]),
    ("Jaccard token", lambda m: m["jaccard_token"]),
]

# Color palette for transformations (supports up to 10)
COLORS = plt.cm.tab10.colors


def load_metrics_data(parquet_files: list[Path]) -> pl.DataFrame:
    """Load data from multiple parquet files.

    Args:
        parquet_files: List of parquet file paths

    Returns:
        Combined DataFrame with all data
    """
    dfs = []
    for path in parquet_files:
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        dfs.append(pl.read_parquet(path))

    if not dfs:
        raise ValueError("No valid parquet files found")

    return pl.concat(dfs)


def extract_metrics(
    df: pl.DataFrame,
    metric_extractors: list[tuple[str, callable]] = DEFAULT_METRICS,
) -> dict[str, dict[str, np.ndarray]]:
    """Extract metrics from DataFrame and organize by transformation.

    Args:
        df: DataFrame with metrics JSON column
        metric_extractors: List of (metric_name, extractor_function) tuples

    Returns:
        Dictionary: {transformation_key: {metric_name: values_array}}
    """
    # Group by (sd_level, sd_variant, model_used)
    data_by_transformation = {}

    for row in df.iter_rows(named=True):
        level = row["sd_level"]
        variant = row["sd_variant"]
        model = row["model_used"]
        key = f"L{level} {variant}\n({model.split('/')[-1]})"  # Format: "L1 variant\n(model)"

        if key not in data_by_transformation:
            data_by_transformation[key] = {name: [] for name, _ in metric_extractors}

        # Parse metrics JSON
        metrics = json.loads(row["metrics"])

        # Extract each metric
        for metric_name, extractor in metric_extractors:
            try:
                value = extractor(metrics)
                data_by_transformation[key][metric_name].append(value)
            except (KeyError, IndexError, TypeError) as e:
                # Skip missing metrics
                continue

    # Convert lists to numpy arrays
    for key in data_by_transformation:
        for metric_name in data_by_transformation[key]:
            data_by_transformation[key][metric_name] = np.array(
                data_by_transformation[key][metric_name]
            )

    return data_by_transformation


def subsample_data(
    data: dict[str, dict[str, np.ndarray]], max_points: int = 100
) -> dict[str, dict[str, np.ndarray]]:
    """Subsample data to max_points per transformation.

    Args:
        data: Data organized by transformation
        max_points: Maximum points per transformation

    Returns:
        Subsampled data with same structure
    """
    subsampled = {}

    for transformation, metrics in data.items():
        # Get number of points
        n_points = len(next(iter(metrics.values())))

        if n_points <= max_points:
            # Keep all points
            subsampled[transformation] = metrics
        else:
            # Random subsample
            indices = np.random.choice(n_points, max_points, replace=False)
            subsampled[transformation] = {
                metric_name: values[indices] for metric_name, values in metrics.items()
            }

    return subsampled


def plot_2d_kde_scatter(
    ax: plt.Axes,
    data: dict[str, dict[str, np.ndarray]],
    x_metric: str,
    y_metric: str,
    colors: dict[str, tuple],
    kde_levels: int = 3,
    alpha_scatter: float = 0.3,
    alpha_kde: float = 0.6,
):
    """Plot 2D scatter + KDE contours for multiple transformations.

    Args:
        ax: Matplotlib axis
        data: Data by transformation
        x_metric: Name of x-axis metric
        y_metric: Name of y-axis metric
        colors: Color map for transformations
        kde_levels: Number of KDE contour levels
        alpha_scatter: Alpha for scatter points
        alpha_kde: Alpha for KDE contours
    """
    for transformation, metrics in data.items():
        if transformation not in colors:
            continue

        x = metrics[x_metric]
        y = metrics[y_metric]

        # Skip if insufficient data
        if len(x) < 3:
            continue

        color = colors[transformation]

        # Scatter plot with small, prominent dots
        ax.scatter(x, y, c=[color], alpha=0.5, s=3, edgecolors="none", rasterized=True)

        # KDE contours (if enough points)
        if len(x) >= 10:
            try:
                # Compute KDE
                values = np.vstack([x, y])
                kernel = gaussian_kde(values)

                # Create grid
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                x_range = x_max - x_min
                y_range = y_max - y_min

                # Add padding
                x_min -= 0.1 * x_range
                x_max += 0.1 * x_range
                y_min -= 0.1 * y_range
                y_max += 0.1 * y_range

                xx, yy = np.mgrid[
                    x_min : x_max : 100j,
                    y_min : y_max : 100j,
                ]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                density = kernel(positions).reshape(xx.shape)

                # Smooth density
                density = gaussian_filter(density, sigma=1.0)

                # Draw contours
                levels = np.linspace(density.min(), density.max(), kde_levels + 2)[1:-1]
                ax.contour(
                    xx,
                    yy,
                    density,
                    levels=levels,
                    colors=[color],
                    alpha=alpha_kde,
                    linewidths=1.5,
                )
            except Exception as e:
                # KDE can fail for degenerate data
                print(f"Warning: KDE failed for {transformation}: {e}")

    ax.set_xlabel(x_metric, fontsize=11)
    ax.set_ylabel(y_metric, fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.2)


def plot_1d_density(
    ax: plt.Axes,
    data: dict[str, dict[str, np.ndarray]],
    metric: str,
    colors: dict[str, tuple],
    alpha: float = 0.7,
):
    """Plot 1D density curves for multiple transformations (horizontal).

    Args:
        ax: Matplotlib axis
        data: Data by transformation
        metric: Name of metric to plot
        colors: Color map for transformations
        alpha: Alpha for density curves
    """
    for transformation, metrics in data.items():
        if transformation not in colors:
            continue

        values = metrics[metric]

        # Skip if insufficient data
        if len(values) < 3:
            continue

        color = colors[transformation]

        # Compute KDE
        try:
            kernel = gaussian_kde(values)
            x_range = values.max() - values.min()
            x_min = values.min() - 0.1 * x_range
            x_max = values.max() + 0.1 * x_range
            x_grid = np.linspace(x_min, x_max, 200)
            density = kernel(x_grid)

            # Plot as horizontal curve (x = metric value, y = density)
            ax.plot(x_grid, density, color=color, alpha=alpha, linewidth=2, label=transformation)
        except Exception as e:
            print(f"Warning: 1D KDE failed for {transformation}, {metric}: {e}")

    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.2, axis="x")
    ax.set_ylim(bottom=0)  # Density starts at 0
    ax.set_yticks([])  # Hide y-axis (density magnitude not important)


def create_corner_plot(
    parquet_files: list[Path],
    metric_extractors: list[tuple[str, callable]] = None,
    transformations: list[str] = None,
    subsample_size: int = 100,
    output_path: Path = None,
    figsize: tuple[float, float] = None,
    kde_levels: int = 1,
) -> plt.Figure:
    """Create corner plot showing joint distributions of metrics.

    Args:
        parquet_files: List of parquet files to load
        metric_extractors: List of (metric_name, extractor_func) tuples (default: key metrics)
        transformations: List of transformation keys to include (None = all)
        subsample_size: Max points per transformation to plot
        output_path: Path to save figure (None = don't save)
        figsize: Figure size (width, height) - auto-calculated if None
        kde_levels: Number of KDE contour levels

    Returns:
        Matplotlib figure
    """
    # Use default metrics if not provided
    if metric_extractors is None:
        metric_extractors = DEFAULT_METRICS

    metric_names = [name for name, _ in metric_extractors]
    n_metrics = len(metric_names)

    # Load and extract data
    print("Loading data...")
    df = load_metrics_data(parquet_files)
    data = extract_metrics(df, metric_extractors)

    # Filter transformations if specified
    if transformations is not None:
        data = {k: v for k, v in data.items() if k in transformations}

    if not data:
        raise ValueError("No data found for specified transformations")

    # Subsample if needed
    print(f"Subsampling to {subsample_size} points per transformation...")
    data = subsample_data(data, subsample_size)

    # Assign colors
    transformation_keys = sorted(data.keys())
    colors = {key: COLORS[i % len(COLORS)] for i, key in enumerate(transformation_keys)}

    # Calculate axis limits for each metric (shared across all plots)
    metric_limits = {}
    for metric_name in metric_names:
        all_values = []
        for metrics in data.values():
            all_values.extend(metrics[metric_name])

        if all_values:
            all_values = np.array(all_values)
            vmin, vmax = all_values.min(), all_values.max()
            vrange = vmax - vmin
            # Add 5% padding
            metric_limits[metric_name] = (vmin - 0.05 * vrange, vmax + 0.05 * vrange)
        else:
            metric_limits[metric_name] = (0, 1)

    # Calculate figure size
    if figsize is None:
        # 2.5 inches per metric
        size = 2.5 * n_metrics
        figsize = (size, size)

    # Create figure with custom grid
    print("Creating plot...")
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        n_metrics,
        n_metrics,
        figure=fig,
        hspace=0.03,
        wspace=0.03,
        left=0.12,
        right=0.98,
        bottom=0.10,
        top=0.96,
    )

    # Create subplots
    axes = {}
    for i in range(n_metrics):
        for j in range(n_metrics):
            if i == j:
                # Diagonal: 1D density plots (horizontal)
                ax = fig.add_subplot(gs[i, j])
                plot_1d_density(ax, data, metric_names[i], colors)

                # Set x-axis limits to match metric range (horizontal plot)
                ax.set_xlim(metric_limits[metric_names[i]])
                ax.set_ylim(bottom=0)  # Density starts at 0

                # Hide tick labels on diagonal, except bottom-right which shows scale
                if i == n_metrics - 1:
                    # Bottom-right diagonal: show x-axis tick labels for scale
                    ax.tick_params(labelsize=10)
                else:
                    # Other diagonals: hide tick labels
                    ax.set_xticklabels([])
                ax.set_yticklabels([])

                # Add metric name as title on diagonal
                ax.set_title(metric_names[i], fontsize=11, pad=3)

            elif i > j:
                # Lower triangle: 2D scatter + KDE
                ax = fig.add_subplot(gs[i, j])
                plot_2d_kde_scatter(
                    ax,
                    data,
                    metric_names[j],
                    metric_names[i],
                    colors,
                    kde_levels=kde_levels,
                )

                # Set consistent axis limits
                ax.set_xlim(metric_limits[metric_names[j]])
                ax.set_ylim(metric_limits[metric_names[i]])

                # Only show labels on edges
                if i == n_metrics - 1:
                    # Bottom row: show x-axis labels
                    ax.set_xlabel(metric_names[j], fontsize=11)
                else:
                    # Not bottom row: hide x-axis labels but keep ticks
                    ax.set_xlabel("")
                    ax.set_xticklabels([])

                if j == 0:
                    # Leftmost column: show y-axis labels
                    ax.set_ylabel(metric_names[i], fontsize=11)
                else:
                    # Not leftmost: hide y-axis labels but keep ticks
                    ax.set_ylabel("")
                    ax.set_yticklabels([])

            else:
                # Upper triangle: hide
                ax = fig.add_subplot(gs[i, j])
                ax.axis("off")

            axes[(i, j)] = ax

    # Add legend in top-right
    handles = [
        plt.Line2D([0], [0], color=colors[k], marker="o", linestyle="-", linewidth=2, markersize=6)
        for k in transformation_keys
    ]
    # Place legend in top-right area, slightly down and left
    legend_ax = axes[(0, n_metrics - 1)]
    legend_ax.legend(
        handles,
        transformation_keys,
        loc="upper right",
        bbox_to_anchor=(0.95, 0.80),  # Move down 20% from top, 5% left from right
        fontsize=11,
        frameon=False,
    )

    # Add title
    fig.suptitle(
        f"Metrics Corner Plot: {len(transformation_keys)} transformations, "
        f"{sum(len(m[metric_names[0]]) for m in data.values())} total points",
        fontsize=13,
        y=0.995,
    )

    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig


def create_metric_comparison(
    parquet_files: list[Path],
    x_metric: tuple[str, callable],
    y_metric: tuple[str, callable],
    transformations: list[str] = None,
    subsample_size: int = 100,
    output_path: Path = None,
    figsize: tuple[float, float] = (10, 8),
    kde_levels: int = 1,
) -> plt.Figure:
    """Create a single 2D comparison plot for two metrics.

    Args:
        parquet_files: List of parquet files to load
        x_metric: (name, extractor_func) for x-axis metric
        y_metric: (name, extractor_func) for y-axis metric
        transformations: List of transformation keys to include (None = all)
        subsample_size: Max points per transformation to plot
        output_path: Path to save figure (None = don't save)
        figsize: Figure size (width, height)
        kde_levels: Number of KDE contour levels

    Returns:
        Matplotlib figure
    """
    # Load and extract data
    df = load_metrics_data(parquet_files)
    data = extract_metrics(df, [x_metric, y_metric])

    # Filter transformations if specified
    if transformations is not None:
        data = {k: v for k, v in data.items() if k in transformations}

    # Subsample if needed
    data = subsample_data(data, subsample_size)

    # Assign colors
    transformation_keys = sorted(data.keys())
    colors = {key: COLORS[i % len(COLORS)] for i, key in enumerate(transformation_keys)}

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot
    plot_2d_kde_scatter(
        ax,
        data,
        x_metric[0],
        y_metric[0],
        colors,
        kde_levels=kde_levels,
    )

    # Add legend
    handles = [
        plt.Line2D([0], [0], color=colors[k], marker="o", linestyle="-", linewidth=2)
        for k in transformation_keys
    ]
    ax.legend(handles, transformation_keys, loc="best", fontsize=11, frameon=False)

    # Title
    ax.set_title(
        f"{y_metric[0]} vs {x_metric[0]}\n"
        f"({len(transformation_keys)} transformations, "
        f"{sum(len(m[x_metric[0]]) for m in data.values())} total points)",
        fontsize=14,
    )

    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig
