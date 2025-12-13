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
        help="Dataset name: gsm8k, codeforces, allenai, mbpp, humaneval, popqa, bigbenchhard, zebralogic, agieval, or all",
    ),
    level: str = typer.Option(
        ...,
        "--level",
        "-l",
        help="Levels (e.g. '1', '2') OR specific variants (e.g. 'value_substitution'). Can be comma-separated list like '1,value_substitution'.",
    ),
    output_file: Path = typer.Option(
        "outputs/output.parquet",
        "--output",
        "-o",
        help="Output parquet file path",
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
    input_file: Path = typer.Option(
        None,
        "--input",
        "-i",
        help="Input parquet file (optional, overrides default dataset loader, mainly for zebralogic)",
    ),
    workers: int = typer.Option(
        4,
        "--workers",
        "-w",
        help="Number of concurrent workers (default: 4)",
    ),
) -> None:
    """Generate semantic duplicates for a dataset.

    By default, generation is resumable - if interrupted, you can re-run the same
    command and it will continue from where it left off. Errors are logged to
    generation_errors.log in the output directory.

    Selection (--level / -l):
    - "1": Run all variants in Level 1
    - "value_substitution": Run specific variant (searches all levels)
    - "1,condition_shuffle": Run all Level 1 variants AND condition_shuffle

    Examples:

        # Generate Level 1 SDs for GSM8K (test with 10 items)
        uv run python -m sdtd generate -d gsm8k -l 1 -n 10

        # Generate specific variants
        uv run python -m sdtd generate -d zebralogic -l "value_substitution,condition_shuffle"

        # Generate Level 1 AND specific Level 2 variant
        uv run python -m sdtd generate -d codeforces -l "1,fictional_setting"

        # Override model (uses this for ALL variants)
        uv run python -m sdtd generate -d gsm8k -l 1 -n 10 -m gpt-4-turbo

        # Generate for all datasets with checkpointing (can resume if interrupted)
        uv run python -m sdtd generate -d all -l 1,2 -o outputs/full_run/

        # Disable checkpointing (not recommended for large runs)
        uv run python -m sdtd generate -d gsm8k -l 1 -n 10 --no-checkpoint

        # Generate SDs for ZebraLogic from a local file
        uv run python -m sdtd generate -d zebralogic -l 2 -i datasets/my_zebralogic.parquet
        
        # Run with 8 concurrent workers
        uv run python -m sdtd generate -d gsm8k -l 1 -w 8
    """
    # Parse selection
    selection = [x.strip() for x in level.split(",")]

    # Run generation
    try:
        generate_sds(
            dataset,
            selection,
            output_file,
            limit,
            model_override=model,
            checkpoint_enabled=not no_checkpoint,
            input_file=input_file,
            workers=workers,
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
            # Use more backticks if content contains triple backticks
            backtick_count = 3
            display_text = original_text if len(original_text) <= 500 else original_text[:500] + "..."
            if "```" in display_text:
                backtick_count = 4
            backticks = "`" * backtick_count
            lines.append(backticks)
            lines.append(display_text)
            lines.append(backticks)
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
                    # Use more backticks if content contains triple backticks
                    sd_backtick_count = 3
                    if "```" in display_sd:
                        sd_backtick_count = 4
                    sd_backticks = "`" * sd_backtick_count
                    lines.append(sd_backticks)
                    lines.append(display_sd)
                    lines.append(sd_backticks)

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
    typer.echo("  - bigbenchhard: Challenging BIG-Bench tasks (6,511 items)")
    typer.echo("  - zebralogic: Logic grid puzzles (1,000 puzzles)")
    typer.echo("  - agieval: Human cognition & problem-solving exams (8,062 questions)")
    typer.echo("  - all: Process all datasets")

    typer.echo("\nPrompt files:")
    for level in [1, 2]:
        path = Path(f"prompts/level{level}.yaml")
        status = "✓" if path.exists() else "✗"
        typer.echo(f"  {status} prompts/level{level}.yaml")


@app.command()
def export_jsonl(
    input_type: str = typer.Option(
        ...,
        "--type",
        "-t",
        help="Input type: 'zebralogic' for original dataset, 'parquet' for generated parquet file",
    ),
    input_file: Path = typer.Option(
        None,
        "--input",
        "-i",
        help="Input parquet file (required if type is 'parquet')",
    ),
    output_file: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output JSONL file path",
    ),
    template_name: str = typer.Option(
        None,
        "--template",
        help="Template name from jsonl_templates.yaml (default: 'zebralogic_original' for zebralogic, 'zebralogic_generated' for parquet)",
    ),
    template_path: Path = typer.Option(
        "prompts/jsonl_templates.yaml",
        "--template-path",
        help="Path to templates YAML file",
    ),
    limit: int = typer.Option(
        None,
        "--limit",
        "-n",
        help="Limit number of items to export (only for zebralogic type)",
    ),
    dataset_filter: str = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Filter by dataset name (only for parquet type, e.g., 'zebralogic')",
    ),
    sd_variant: str = typer.Option(
        None,
        "--sd-variant",
        "-v",
        help="Filter by sd_variant (one or comma-separated list, e.g., 'value_substitution' or 'value_substitution,condition_shuffle')",
    ),
) -> None:
    """Export datasets to JSONL format for OpenAI fine-tuning.

    Converts either the original ZebraLogic dataset or generated parquet files
    to JSONL format using templates defined in jsonl_templates.yaml.

    Examples:

        # Export original ZebraLogic dataset
        uv run python -m sdtd export-jsonl -t zebralogic -o zebralogic_train.jsonl

        # Export with limit for testing
        uv run python -m sdtd export-jsonl -t zebralogic -o test.jsonl -n 10

        # Export generated parquet file
        uv run python -m sdtd export-jsonl -t parquet -i outputs/zebralogic_level12.parquet -o generated.jsonl

        # Export with dataset filter
        uv run python -m sdtd export-jsonl -t parquet -i outputs/all.parquet -o zebra.jsonl -d zebralogic

        # Export with sd_variant filter
        uv run python -m sdtd export-jsonl -t parquet -i outputs/zebralogic_level2.parquet -o output.jsonl -v value_substitution

        # Export multiple variants
        uv run python -m sdtd export-jsonl -t parquet -i outputs/zebralogic_level2.parquet -o output.jsonl -v value_substitution,condition_shuffle

        # Use custom template
        uv run python -m sdtd export-jsonl -t zebralogic -o output.jsonl --template custom_template
    """
    from sdtd.export import export_zebralogic_to_jsonl, export_parquet_to_jsonl

    if input_type == "zebralogic":
        # Export original ZebraLogic dataset
        if template_name is None:
            template_name = "zebralogic_original"
        
        try:
            export_zebralogic_to_jsonl(
                output_file=output_file,
                template_name=template_name,
                template_path=template_path,
                limit=limit,
            )
            typer.echo(f"✓ Exported ZebraLogic dataset to {output_file}")
        except Exception as e:
            typer.echo(f"✗ Error: {e}", err=True)
            raise typer.Exit(1)

    elif input_type == "parquet":
        # Export generated parquet file
        if input_file is None:
            typer.echo("Error: --input is required when type is 'parquet'", err=True)
            raise typer.Exit(1)
        
        if not input_file.exists():
            typer.echo(f"Error: Input file {input_file} not found", err=True)
            raise typer.Exit(1)

        if template_name is None:
            template_name = "zebralogic_generated"
        
        # Parse sd_variant filter (comma-separated list)
        variant_filter = None
        if sd_variant:
            variant_filter = [v.strip() for v in sd_variant.split(",")]
        
        try:
            export_parquet_to_jsonl(
                input_file=input_file,
                output_file=output_file,
                template_name=template_name,
                template_path=template_path,
                dataset_filter=dataset_filter,
                sd_variant_filter=variant_filter,
            )
            typer.echo(f"✓ Exported parquet file to {output_file}")
        except Exception as e:
            typer.echo(f"✗ Error: {e}", err=True)
            raise typer.Exit(1)

    else:
        typer.echo(f"Error: Invalid type '{input_type}'. Must be 'zebralogic' or 'parquet'", err=True)
        raise typer.Exit(1)


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


@app.command()
def generate_reasoning(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input parquet file with SDs",
    ),
    output_file: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output parquet file",
    ),
    model: str = typer.Option(
        "claude-3-5-sonnet-20241022",
        "--model",
        "-m",
        help="Model to use for reasoning",
    ),
    k: int = typer.Option(
        3,
        "--attempts",
        "-k",
        help="Max retry attempts",
    ),
    limit: int = typer.Option(
        None,
        "--limit",
        "-n",
        help="Limit number of items to process",
    ),
    no_checkpoint: bool = typer.Option(
        False,
        "--no-checkpoint",
        help="Disable checkpointing",
    ),
) -> None:
    """Enrich ZebraLogic SDs with correct reasoning traces.
    
    Iterates through the input parquet file containing ZebraLogic SDs,
    asks the model to solve each puzzle, checks the solution against the ground truth,
    and retries up to k times if incorrect. Adds the correct reasoning trace to the output.
    
    Examples:
    
        # Generate reasoning for ZebraLogic SDs
        uv run python -m sdtd generate-reasoning -i outputs/zebralogic_level2.parquet -o outputs/zebralogic_enriched.parquet
        
        # Test with limit and specific model
        uv run python -m sdtd generate-reasoning -i outputs/zebralogic_level2.parquet -o test.parquet -n 5 -m gpt-4o
    """
    from sdtd.reasoning import generate_reasoning_traces
    
    if not input_file.exists():
        typer.echo(f"Error: Input file {input_file} not found", err=True)
        raise typer.Exit(1)
        
    try:
        generate_reasoning_traces(
            input_file,
            output_file,
            model,
            k,
            limit,
            checkpoint_enabled=not no_checkpoint,
        )
        typer.echo("\n✓ Reasoning generation complete!")
    except Exception as e:
        typer.echo(f"\n✗ Error: {e}", err=True)
        raise typer.Exit(1)


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
