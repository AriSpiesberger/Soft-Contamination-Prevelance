"""Core generation logic for semantic duplicates."""

import polars as pl
from pathlib import Path
from datetime import datetime
from typing import Any
import json
import re
import litellm
from litellm import completion
from litellm.caching import Cache

from sdtd.datasets import load_dataset
from sdtd.utils import load_prompts, format_prompt


# Set up disk cache for litellm
litellm.cache = Cache(disk_cache_dir=".cache/litellm")


# Map dataset names to their primary text field
DATASET_TEXT_FIELDS = {
    "gsm8k": "question",
    "codeforces": "description",
    "allenai": "text",
}


def calculate_ngram_overlap(text1: str, text2: str, n: int = 2) -> float:
    """Calculate n-gram overlap percentage between two texts.

    Args:
        text1: First text
        text2: Second text
        n: N-gram size (default: bigrams)

    Returns:
        Overlap percentage (0-100)
    """
    def get_ngrams(text: str, n: int) -> set:
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < n:
            return set()
        return set(zip(*[words[i:] for i in range(n)]))

    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)

    if not ngrams1 or not ngrams2:
        return 0.0

    overlap = len(ngrams1 & ngrams2)
    total = len(ngrams1 | ngrams2)

    return (overlap / total) * 100 if total > 0 else 0.0


def generate_sds(
    dataset_name: str,
    levels: list[int],
    output_dir: Path,
    limit: int | None = None,
    model_override: str | None = None,
) -> None:
    """Generate semantic duplicates for a dataset.

    Args:
        dataset_name: Name of dataset (gsm8k, codeforces, allenai, or all)
        levels: List of levels to generate (e.g., [1] or [1, 2])
        output_dir: Directory to save output files
        limit: Optional limit on number of items to process
        model_override: Optional model to use instead of YAML defaults
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_override:
        print(f"Model override: Using '{model_override}' for all variants\n")

    # Handle "all" datasets
    if dataset_name == "all":
        datasets = load_dataset("all", limit)
        for name, df in datasets.items():
            _generate_for_dataset(name, df, levels, output_dir, model_override)
    else:
        df = load_dataset(dataset_name, limit)
        _generate_for_dataset(dataset_name, df, levels, output_dir, model_override)


def _generate_for_dataset(
    dataset_name: str,
    df: pl.DataFrame,
    levels: list[int],
    output_dir: Path,
    model_override: str | None = None,
) -> None:
    """Generate SDs for a single dataset.

    Args:
        dataset_name: Name of dataset
        df: Dataset DataFrame
        levels: List of levels to generate
        output_dir: Output directory
        model_override: Optional model to use instead of YAML defaults
    """
    print(f"\n{'='*60}")
    print(f"Generating SDs for {dataset_name}")
    print(f"{'='*60}")

    # Load prompts for requested levels
    prompts = {}
    for level in levels:
        prompt_path = Path(f"prompts/level{level}.yaml")
        if not prompt_path.exists():
            print(f"Warning: {prompt_path} not found, skipping level {level}")
            continue
        prompts[level] = load_prompts(prompt_path)

    # Generate for each level
    all_results = []

    for level in levels:
        if level not in prompts:
            continue

        level_prompts = prompts[level].get(dataset_name)
        if not level_prompts:
            print(f"Warning: No prompts found for {dataset_name} level {level}")
            continue

        print(f"\nLevel {level}: {len(level_prompts)} variants")

        for variant_name, prompt_config in level_prompts.items():
            # Use model override if provided, otherwise use YAML default
            model = model_override if model_override else prompt_config["model"]
            print(f"  - {variant_name} ({model})...", end=" ", flush=True)

            variant_results = []
            for idx, row in enumerate(df.iter_rows(named=True)):
                try:
                    sd_text = generate_single(row, prompt_config, dataset_name, model_override)

                    # Get the primary text field for this dataset
                    text_field = DATASET_TEXT_FIELDS.get(dataset_name, "text")
                    original_text = row.get(text_field, "")

                    # Calculate n-gram overlap metrics
                    bigram_overlap = calculate_ngram_overlap(original_text, sd_text, n=2)
                    trigram_overlap = calculate_ngram_overlap(original_text, sd_text, n=3)
                    unigram_overlap = calculate_ngram_overlap(original_text, sd_text, n=1)

                    # Separate primary text from additional info
                    additional_info = {}
                    for k, v in row.items():
                        if k != text_field:
                            # Convert to JSON-serializable format
                            if isinstance(v, (list, dict)):
                                additional_info[k] = v
                            else:
                                additional_info[k] = str(v) if v is not None else None

                    result = {
                        "source_dataset": dataset_name,
                        "sd_level": level,
                        "sd_variant": variant_name,
                        "model_used": model,
                        "original_text": original_text,
                        "sd_text": sd_text,
                        "bigram_overlap_pct": round(bigram_overlap, 2),
                        "trigram_overlap_pct": round(trigram_overlap, 2),
                        "unigram_overlap_pct": round(unigram_overlap, 2),
                        "additional_info": json.dumps(additional_info),
                        "timestamp": datetime.now().isoformat(),
                    }

                    variant_results.append(result)

                except Exception as e:
                    print(f"\n    Error on item {idx}: {e}")
                    continue

            all_results.extend(variant_results)
            print(f"✓ {len(variant_results)} generated")

    # Save results
    if all_results:
        output_path = output_dir / f"{dataset_name}_level{''.join(map(str, levels))}.parquet"
        result_df = pl.DataFrame(all_results)
        result_df.write_parquet(output_path, compression="zstd", compression_level=3)
        print(f"\n✓ Saved {len(all_results)} SDs to {output_path} (zstd compressed)")
    else:
        print(f"\n⚠ No SDs generated for {dataset_name}")


def generate_single(
    row: dict,
    prompt_config: dict,
    dataset_name: str,
    model_override: str | None = None,
) -> str:
    """Generate a single semantic duplicate using litellm.

    Args:
        row: Data row dictionary
        prompt_config: Prompt configuration from YAML
        dataset_name: Name of dataset
        model_override: Optional model to use instead of YAML default

    Returns:
        Generated semantic duplicate text
    """
    # Format prompt with row data
    user_prompt = format_prompt(prompt_config["user"], row, dataset_name)

    # Build messages
    messages = []
    if "system" in prompt_config:
        messages.append({"role": "system", "content": prompt_config["system"]})
    messages.append({"role": "user", "content": user_prompt})

    # Use model override if provided, otherwise use YAML default
    model = model_override if model_override else prompt_config["model"]

    # Call litellm (with caching)
    response = completion(
        model=model,
        messages=messages,
        temperature=prompt_config.get("temperature", 0.7),
        max_tokens=prompt_config.get("max_tokens", 2048),
        caching=True,  # Enable caching
    )

    return response.choices[0].message.content.strip()
