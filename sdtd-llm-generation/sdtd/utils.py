"""Utility functions for SDTD."""

import yaml
from pathlib import Path
from typing import Any


def load_prompts(path: str | Path) -> dict[str, Any]:
    """Load prompts from YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary of prompts with dataset -> variant structure
    """
    with open(path) as f:
        return yaml.safe_load(f)


def format_prompt(template: str, row: dict, dataset_name: str) -> str:
    """Format prompt template with row data.

    Args:
        template: Prompt template string with {field} placeholders
        row: Data row dictionary
        dataset_name: Name of dataset (gsm8k, codeforces, allenai, mbpp, humaneval, popqa)

    Returns:
        Formatted prompt string
    """
    # Field mapping for each dataset
    field_map = {
        "gsm8k": "question",
        "codeforces": "description",
        "allenai": "text",
        "mbpp": "prompt",
        "humaneval": "prompt",
        "popqa": "question",
        "bigbenchhard": "input",
        "zebralogic": "puzzle",
        "agieval": "question",
    }

    field = field_map.get(dataset_name)
    if field and field in row:
        return template.format(**{field: row[field]})

    # Fallback: try to format with all row data
    try:
        return template.format(**row)
    except KeyError as e:
        raise ValueError(f"Missing field {e} in row for dataset {dataset_name}")
