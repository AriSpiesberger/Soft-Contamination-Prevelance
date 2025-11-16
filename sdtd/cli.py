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
) -> None:
    """Generate semantic duplicates for a dataset.

    Examples:

        # Generate Level 1 SDs for GSM8K (test with 10 items)
        uv run python -m sdtd generate -d gsm8k -l 1 -n 10

        # Generate both levels for Codeforces
        uv run python -m sdtd generate -d codeforces -l 1,2 -n 5

        # Override model (uses this for ALL variants)
        uv run python -m sdtd generate -d gsm8k -l 1 -n 10 -m gpt-4-turbo

        # Generate for all datasets
        uv run python -m sdtd generate -d all -l 1,2 -o outputs/full_run/
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
        generate_sds(dataset, levels, output_dir, limit, model_override=model)
        typer.echo("\n✓ Generation complete!")
    except Exception as e:
        typer.echo(f"\n✗ Error: {e}", err=True)
        raise typer.Exit(1)


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


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
