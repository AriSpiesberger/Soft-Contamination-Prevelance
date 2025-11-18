"""CLI for SDTD - Semantic Duplicate Training Data generation."""

import typer
from pathlib import Path
from dotenv import load_dotenv

from sdtd.generate import generate_sds

# Load environment variables from .env
load_dotenv()

app = typer.Typer(help="Generate semantic duplicates for LLM training datasets")


@app.command()
def generate(
    dataset: str = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Dataset name: gsm8k, codeforces, allenai, or all",
    ),
    level: str = typer.Option(
        ...,
        "--level",
        "-l",
        help="Levels to generate (comma-separated): 1, 2, or 1,2",
    ),
    output_dir: Path = typer.Option(
        "outputs",
        "--output-dir",
        "-o",
        help="Output directory for generated files",
    ),
    limit: int = typer.Option(
        None,
        "--limit",
        "-n",
        help="Limit number of items to process (for testing)",
    ),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Override model for all variants (e.g., claude-3-5-sonnet-20241022)",
    ),
    no_checkpoint: bool = typer.Option(
        False,
        "--no-checkpoint",
        help="Disable checkpoint/resume functionality (not recommended for large runs)",
    ),
) -> None:
    """Generate semantic duplicates for a dataset.

    By default, generation is resumable - if interrupted, you can re-run the same
    command and it will continue from where it left off. Errors are logged to
    generation_errors.log in the output directory.

    Examples:

        # Generate Level 1 SDs for GSM8K (test with 10 items)
        uv run python -m sdtd generate -d gsm8k -l 1 -n 10

        # Generate both levels for Codeforces
        uv run python -m sdtd generate -d codeforces -l 1,2 -n 5

        # Override model (uses this for ALL variants)
        uv run python -m sdtd generate -d gsm8k -l 1 -n 10 -m gpt-4-turbo

        # Generate for all datasets with checkpointing (can resume if interrupted)
        uv run python -m sdtd generate -d all -l 1,2 -o outputs/full_run/

        # Disable checkpointing (not recommended for large runs)
        uv run python -m sdtd generate -d gsm8k -l 1 -n 10 --no-checkpoint
    """
    # Parse levels
    levels = [int(x.strip()) for x in level.split(",")]

    # Validate levels
    for lvl in levels:
        if lvl not in [1, 2]:
            typer.echo(f"Error: Invalid level {lvl}. Must be 1 or 2.", err=True)
            raise typer.Exit(1)

    # Run generation
    try:
        generate_sds(
            dataset,
            levels,
            output_dir,
            limit,
            model_override=model,
            checkpoint_enabled=not no_checkpoint,
        )
        typer.echo("\n✓ Generation complete!")
    except Exception as e:
        typer.echo(f"\n✗ Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def dump(
    input_file: Path = typer.Argument(
        ...,
        help="Path to parquet file to dump",
    ),
    output_file: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output markdown file (default: stdout)",
    ),
    limit: int = typer.Option(
        5,
        "--limit",
        "-n",
        help="Number of samples to dump per variant",
    ),
    level: int = typer.Option(
        None,
        "--level",
        "-l",
        help="Filter by level (1 or 2)",
    ),
    variant: str = typer.Option(
        None,
        "--variant",
        "-v",
        help="Filter by specific variant name",
    ),
) -> None:
    """Dump parquet file samples to readable markdown format.

    Examples:

        # Dump first 5 samples from each variant to stdout
        uv run python -m sdtd.cli dump outputs/gsm8k_level1.parquet

        # Dump 3 samples to file
        uv run python -m sdtd.cli dump outputs/gsm8k_level1.parquet -o review.md -n 3

        # Dump only Level 1, abstractive_paraphrase variant
        uv run python -m sdtd.cli dump outputs/gsm8k_level12.parquet -l 1 -v abstractive_paraphrase
    """
    import polars as pl
    import json
    from datetime import datetime

    if not input_file.exists():
        typer.echo(f"Error: File not found: {input_file}", err=True)
        raise typer.Exit(1)

    # Read parquet
    try:
        df = pl.read_parquet(input_file)
    except Exception as e:
        typer.echo(f"Error reading parquet: {e}", err=True)
        raise typer.Exit(1)

    # Apply filters
    if level is not None:
        df = df.filter(pl.col("sd_level") == level)
    if variant is not None:
        df = df.filter(pl.col("sd_variant") == variant)

    if len(df) == 0:
        typer.echo("No data matches the filters.", err=True)
        raise typer.Exit(1)

    # Build markdown output
    lines = []
    lines.append(f"# Semantic Duplicates Sample Review")
    lines.append(f"")
    lines.append(f"**Source file**: `{input_file}`  ")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Total records**: {len(df)}  ")
    lines.append(f"**Filters**: Level={level if level else 'all'}, Variant={variant if variant else 'all'}  ")
    lines.append(f"")

    # Get unique combinations of level and variant
    grouped = df.group_by(["sd_level", "sd_variant"]).agg([pl.len().alias("count")])

    for group_row in grouped.iter_rows(named=True):
        group_level = group_row["sd_level"]
        group_variant = group_row["sd_variant"]
        count = group_row["count"]

        lines.append(f"")
        lines.append(f"## Level {group_level}: {group_variant}")
        lines.append(f"")
        lines.append(f"**Total samples**: {count}  ")
        lines.append(f"**Showing**: {min(limit, count)} samples  ")
        lines.append(f"")

        # Get samples for this group
        group_df = df.filter(
            (pl.col("sd_level") == group_level) &
            (pl.col("sd_variant") == group_variant)
        ).head(limit)

        for idx, row in enumerate(group_df.iter_rows(named=True), 1):
            lines.append(f"### Sample {idx}")
            lines.append(f"")

            # Original text
            lines.append(f"**Original text**:")
            lines.append(f"```")
            lines.append(row["original_text"])
            lines.append(f"```")
            lines.append(f"")

            # SD text
            lines.append(f"**Generated SD**:")
            lines.append(f"```")
            lines.append(row["sd_text"])
            lines.append(f"```")
            lines.append(f"")

            # Metadata
            lines.append(f"**Metadata**:")
            lines.append(f"- **Dataset**: {row['source_dataset']}")
            lines.append(f"- **Model**: {row['model_used']}")
            lines.append(f"- **Embedding Model**: {row['embedding_model']}")
            lines.append(f"- **Timestamp**: {row['timestamp']}")
            lines.append(f"")

            # Metrics
            metrics = json.loads(row["metrics"])
            lines.append(f"**Metrics**:")
            lines.append(f"")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Unigram overlap | {metrics['ngram_overlaps_pct'][0]:.2f}% |")
            lines.append(f"| Bigram overlap | {metrics['ngram_overlaps_pct'][1]:.2f}% |")
            lines.append(f"| Trigram overlap | {metrics['ngram_overlaps_pct'][2]:.2f}% |")
            lines.append(f"| 4-gram overlap | {metrics['ngram_overlaps_pct'][3]:.2f}% |")
            lines.append(f"| 5-gram overlap | {metrics['ngram_overlaps_pct'][4]:.2f}% |")
            lines.append(f"| ROUGE-L F-measure | {metrics['rouge_l_f']:.4f} |")
            lines.append(f"| Edit distance (norm) | {metrics['edit_distance_norm']:.4f} |")
            lines.append(f"| TF-IDF cosine | {metrics['tfidf_cosine']:.4f} |")
            lines.append(f"| Jaccard token | {metrics['jaccard_token']:.4f} |")
            lines.append(f"| Embedding cosine | {metrics['cosine_similarity']:.4f} |")
            lines.append(f"| Number preservation | {'✓' if metrics['number_preservation'] else '✗'} |")
            lines.append(f"| Number precision | {metrics['number_precision']:.4f} |")
            lines.append(f"| Number recall | {metrics['number_recall']:.4f} |")
            lines.append(f"| Length ratio | {metrics['length_ratio']:.4f} |")
            lines.append(f"")

            # Additional info (collapsed)
            if row.get("additional_info"):
                additional = json.loads(row["additional_info"])
                lines.append(f"<details>")
                lines.append(f"<summary><b>Additional info</b> (click to expand)</summary>")
                lines.append(f"")
                lines.append(f"```json")
                lines.append(json.dumps(additional, indent=2))
                lines.append(f"```")
                lines.append(f"</details>")
                lines.append(f"")

            lines.append(f"---")
            lines.append(f"")

    # Output
    output_text = "\n".join(lines)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(output_text)
        typer.echo(f"✓ Wrote {len(lines)} lines to {output_file}")
    else:
        typer.echo(output_text)


@app.command()
def info() -> None:
    """Show information about available datasets and prompts."""
    from sdtd.datasets import load_dataset

    typer.echo("Available datasets:")
    typer.echo("  - gsm8k: Math word problems (7,473 train items)")
    typer.echo("  - codeforces: Programming problems (869 train items)")
    typer.echo("  - allenai: Educational text (~unknown items)")
    typer.echo("  - all: Process all datasets")

    typer.echo("\nPrompt files:")
    for level in [1, 2]:
        path = Path(f"prompts/level{level}.yaml")
        status = "✓" if path.exists() else "✗"
        typer.echo(f"  {status} prompts/level{level}.yaml")


@app.command()
def plot(
    input_files: list[Path] = typer.Argument(
        ...,
        help="Parquet files to visualize (space-separated)",
    ),
    output: Path = typer.Option(
        "outputs/metrics_plot.png",
        "--output",
        "-o",
        help="Output image file",
    ),
    transformations: str = typer.Option(
        None,
        "--transformations",
        "-t",
        help="Comma-separated list of transformations to include (None = all)",
    ),
    subsample: int = typer.Option(
        100,
        "--subsample",
        "-s",
        help="Max points per transformation to plot",
    ),
    kde_levels: int = typer.Option(
        1,
        "--kde-levels",
        "-k",
        help="Number of KDE contour levels",
    ),
) -> None:
    """Create corner plot visualizing metric distributions.

    Creates a corner plot showing joint distributions of key metrics (bigram overlap,
    trigram overlap, embedding similarity, ROUGE-L, edit distance) for different
    semantic duplicate transformations.

    Examples:

        # Plot metrics from a single file
        uv run python -m sdtd.cli plot outputs/gsm8k_level1.parquet

        # Plot metrics from multiple files
        uv run python -m sdtd.cli plot outputs/gsm8k_level1.parquet outputs/gsm8k_level2.parquet

        # Plot specific transformations only
        uv run python -m sdtd.cli plot outputs/gsm8k_level12.parquet \\
            -t "number_substitution,abstractive_paraphrase" -o plots/comparison.png

        # Adjust subsampling and KDE detail
        uv run python -m sdtd.cli plot outputs/*.parquet -s 200 -k 5
    """
    from sdtd.visualize import create_corner_plot

    # Validate input files
    valid_files = []
    for f in input_files:
        if not f.exists():
            typer.echo(f"Warning: {f} not found, skipping", err=True)
        else:
            valid_files.append(f)

    if not valid_files:
        typer.echo("Error: No valid input files found", err=True)
        raise typer.Exit(1)

    # Parse transformations filter
    transformation_filter = None
    if transformations:
        transformation_filter = [t.strip() for t in transformations.split(",")]
        typer.echo(f"Filtering to transformations: {transformation_filter}")

    # Create plot
    try:
        typer.echo(f"Creating corner plot from {len(valid_files)} file(s)...")
        fig = create_corner_plot(
            parquet_files=valid_files,
            transformations=transformation_filter,
            subsample_size=subsample,
            output_path=output,
            kde_levels=kde_levels,
        )
        typer.echo(f"✓ Saved plot to {output}")
    except Exception as e:
        typer.echo(f"✗ Error creating plot: {e}", err=True)
        raise typer.Exit(1)


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
