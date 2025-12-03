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
        help="Dataset name: gsm8k, codeforces, allenai, mbpp, humaneval, popqa, or all",
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
    input_files: list[Path] = typer.Argument(
        ...,
        help="Path to parquet file(s) to dump",
    ),
    output_file: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output markdown file (default: stdout)",
    ),
    samples: int = typer.Option(
        3,
        "--samples",
        "-n",
        help="Number of samples to show per dataset",
    ),
    dataset: str = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Filter by specific dataset name",
    ),
) -> None:
    """Create dataset overview with samples and transformations.

    Organized as: dataset > sample > transformations (compact format).
    Shows dataset descriptions from metadata and all transformations for each sample.

    Examples:

        # Overview with 3 samples per dataset
        uv run python -m sdtd.cli dump outputs/gsm8k_level12.parquet

        # Multiple files with 5 samples each
        uv run python -m sdtd.cli dump outputs/*.parquet -n 5 -o overview.md

        # Filter to specific dataset
        uv run python -m sdtd.cli dump outputs/*.parquet -d gsm8k -n 2
    """
    import polars as pl
    import json
    import yaml
    from datetime import datetime
    from collections import defaultdict

    # Load dataset metadata
    metadata_path = Path("datasets_metadata.yaml")
    if metadata_path.exists():
        with open(metadata_path) as f:
            dataset_metadata = yaml.safe_load(f)
    else:
        typer.echo("Warning: datasets_metadata.yaml not found, proceeding without metadata", err=True)
        dataset_metadata = {}

    # Validate and read input files
    valid_files = []
    for f in input_files:
        if not f.exists():
            typer.echo(f"Warning: {f} not found, skipping", err=True)
        else:
            valid_files.append(f)

    if not valid_files:
        typer.echo("Error: No valid input files found", err=True)
        raise typer.Exit(1)

    # Read all parquet files and combine
    dfs = []
    for f in valid_files:
        try:
            dfs.append(pl.read_parquet(f))
        except Exception as e:
            typer.echo(f"Error reading {f}: {e}", err=True)
            raise typer.Exit(1)

    df = pl.concat(dfs) if len(dfs) > 1 else dfs[0]

    # Apply dataset filter
    if dataset is not None:
        df = df.filter(pl.col("source_dataset") == dataset)

    if len(df) == 0:
        typer.echo("No data matches the filters.", err=True)
        raise typer.Exit(1)

    # Organize data: dataset > sample index > transformations
    # Group by dataset and original text to find all transformations
    data_by_dataset = defaultdict(lambda: defaultdict(list))

    for row in df.iter_rows(named=True):
        ds = row["source_dataset"]
        original = row["original_text"]
        data_by_dataset[ds][original].append(row)

    # Build markdown output
    lines = []
    lines.append("# Semantic Duplicates Dataset Overview")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Source files**: {', '.join(f'`{f.name}`' for f in valid_files)}  ")
    lines.append(f"**Total records**: {len(df)}  ")
    lines.append(f"**Datasets**: {', '.join(sorted(data_by_dataset.keys()))}  ")
    lines.append("")

    # Process each dataset
    for ds_name in sorted(data_by_dataset.keys()):
        lines.append("---")
        lines.append("")
        lines.append(f"## {ds_name.upper()}")
        lines.append("")

        # Add dataset description from metadata
        if ds_name in dataset_metadata:
            meta = dataset_metadata[ds_name]
            lines.append(f"**{meta.get('full_name', meta['name'])}**")
            lines.append("")
            lines.append(meta.get('description', 'No description available.'))
            lines.append("")

            # Add source info
            if 'source' in meta:
                src = meta['source']
                authors = src.get('authors', 'Unknown')
                year = src.get('year', 'Unknown')
                lines.append(f"*Source*: {src.get('organization', 'Unknown')} ({year})")
                if 'paper_url' in src:
                    lines.append(f" | [Paper]({src['paper_url']})")
                if 'huggingface' in meta.get('urls', {}):
                    lines.append(f" | [HuggingFace]({meta['urls']['huggingface']})")
                lines.append("")

            # Add size info
            if 'size' in meta:
                size = meta['size']
                size_str = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in size.items() if k != 'avg_tests_per_problem')
                lines.append(f"*Size*: {size_str}")
                lines.append("")

        # Get sample original texts
        originals = list(data_by_dataset[ds_name].keys())
        num_samples = min(samples, len(originals))
        lines.append(f"**Showing {num_samples} of {len(originals)} samples**")
        lines.append("")

        # Show samples with their transformations
        for sample_idx, original_text in enumerate(originals[:num_samples], 1):
            transformations = data_by_dataset[ds_name][original_text]

            lines.append(f"### Sample {sample_idx}")
            lines.append("")

            # Original text (compact)
            lines.append("**Original:**")
            lines.append("```")
            # Truncate very long texts for readability
            display_text = original_text if len(original_text) <= 500 else original_text[:500] + "..."
            lines.append(display_text)
            lines.append("```")
            lines.append("")

            # Group transformations by level
            by_level = defaultdict(list)
            for t in transformations:
                by_level[t['sd_level']].append(t)

            # Show transformations organized by level
            for level in sorted(by_level.keys()):
                level_transforms = by_level[level]
                lines.append(f"**Level {level} Transformations ({len(level_transforms)})**:")
                lines.append("")

                for t in level_transforms:
                    variant = t['sd_variant']
                    sd_text = t['sd_text']
                    model = t['model_used']

                    # Parse metrics
                    metrics = json.loads(t['metrics'])
                    bigram = metrics['ngram_overlaps_pct'][1]
                    trigram = metrics['ngram_overlaps_pct'][2]
                    cosine = metrics['cosine_similarity']

                    lines.append(f"*{variant}* (model: `{model}`)")

                    # Show SD text (compact)
                    display_sd = sd_text if len(sd_text) <= 400 else sd_text[:400] + "..."
                    lines.append("```")
                    lines.append(display_sd)
                    lines.append("```")

                    # Compact metrics
                    lines.append(f"↳ Metrics: 2-gram={bigram:.1f}%, 3-gram={trigram:.1f}%, cosine={cosine:.3f}")
                    lines.append("")

            lines.append("")

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
    typer.echo("  - mbpp: Python programming problems (427 sanitized train items)")
    typer.echo("  - humaneval: Python code evaluation (164 test items)")
    typer.echo("  - popqa: Question answering (14,000 test items)")
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

    # Create plot(s)
    try:
        typer.echo(f"Creating corner plot(s) from {len(valid_files)} file(s)...")
        result = create_corner_plot(
            parquet_files=valid_files,
            transformations=transformation_filter,
            subsample_size=subsample,
            output_path=output,
            kde_levels=kde_levels,
            create_per_dataset=True,  # Always create both combined and per-dataset plots
        )
        
        if isinstance(result, dict):
            # Multiple plots created
            typer.echo(f"✓ Created {len(result)} plot(s):")
            for name, fig in result.items():
                if name == "all_datasets":
                    typer.echo(f"  - Combined plot (all datasets)")
                else:
                    typer.echo(f"  - Dataset: {name}")
            typer.echo(f"✓ Saved plots to {output}")
        else:
            # Single plot (shouldn't happen with create_per_dataset=True, but handle it)
            typer.echo(f"✓ Saved plot to {output}")
    except Exception as e:
        typer.echo(f"✗ Error creating plot: {e}", err=True)
        raise typer.Exit(1)


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
