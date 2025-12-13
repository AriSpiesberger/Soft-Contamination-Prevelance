"""Utility functions for SDTD."""

import logging
import threading
from openai import OpenAI
from pyarrow.util import os
import yaml
from pathlib import Path
from typing import Any
from functools import lru_cache


_openai_client = threading.local()
_openai_client_lock = threading.Lock()

logger = logging.getLogger(__name__)


# Initialize OpenAI client with Helicone configuration
def get_client():
    with _openai_client_lock:
        if hasattr(_openai_client, "_openai_client") and _openai_client._openai_client is not None:
            return _openai_client._openai_client
        key = os.getenv("HELICONE_API_KEY")
        base_url = os.getenv("HELICONE_BASE_URL", "https://ai-gateway.helicone.ai",)
        logger.info(f"Initializing OpenAI client with key: {key[:10]}...{key[-6:]} and base URL: {base_url}")
        if not key:
            raise ValueError("HELICONE_API_KEY environment variable not set")
        _openai_client._openai_client = OpenAI(api_key=key, base_url=base_url)
        return _openai_client._openai_client


def load_prompts(path: str | Path) -> dict[str, Any]:
    """Load prompts from YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary of prompts with dataset -> variant structure
    """
    with open(path) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=4)
def _load_prompts_cached(path: str) -> dict[str, Any]:
    """Cached version of load_prompts.
    
    Args:
        path: Path string (lru_cache needs hashable args)
    """
    return load_prompts(path)


def get_variant_config(dataset: str, level: int, variant: str) -> dict:
    """Get prompt configuration for a specific variant.
    
    Uses caching to avoid repeated file reads.
    
    Args:
        dataset: Dataset name
        level: Level number
        variant: Variant name
        
    Returns:
        Configuration dictionary for the variant
    """
    path = Path(f"prompts/level{level}.yaml")
    if not path.exists():
        return {}
        
    # Use cached loader (convert Path to str for lru_cache)
    data = _load_prompts_cached(str(path))
    
    return data.get(dataset, {}).get(variant, {})


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
