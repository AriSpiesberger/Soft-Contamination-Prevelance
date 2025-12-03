"""Dataset loading functions."""

import json
import polars as pl
from pathlib import Path
from datasets import load_dataset as hf_load_dataset


def load_dataset(name: str, limit: int | None = None) -> pl.DataFrame:
    """Load dataset by name.

    Args:
        name: Dataset name (gsm8k, codeforces, allenai, mbpp, humaneval, popqa, or all)
        limit: Optional limit on number of rows

    Returns:
        Polars DataFrame with dataset contents
    """
    if name == "gsm8k":
        return load_gsm8k(limit)
    elif name == "codeforces":
        return load_codeforces(limit)
    elif name == "allenai":
        return load_allenai(limit)
    elif name == "mbpp":
        return load_mbpp(limit)
    elif name == "humaneval":
        return load_humaneval(limit)
    elif name == "popqa":
        return load_popqa(limit)
    elif name == "all":
        # Return dict of all datasets
        return {
            "gsm8k": load_gsm8k(limit),
            "codeforces": load_codeforces(limit),
            "allenai": load_allenai(limit),
            "mbpp": load_mbpp(limit),
            "humaneval": load_humaneval(limit),
            "popqa": load_popqa(limit),
        }
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from: gsm8k, codeforces, allenai, mbpp, humaneval, popqa, all")


def load_gsm8k(limit: int | None = None) -> pl.DataFrame:
    """Load GSM8K math dataset.

    Args:
        limit: Optional limit on number of rows

    Returns:
        DataFrame with columns: question, answer
    """
    path = Path("datasets/openai_gsm8k_train-00000-of-00001.parquet")
    df = pl.read_parquet(path)
    if limit:
        df = df.head(limit)
    return df


def load_codeforces(limit: int | None = None) -> pl.DataFrame:
    """Load Codeforces programming problems dataset.

    Args:
        limit: Optional limit on number of rows

    Returns:
        DataFrame with columns: id, description, title, etc.
    """
    path = Path("datasets/open-r1_codeforces_train-00000-of-00011.parquet")
    df = pl.read_parquet(path)
    if limit:
        df = df.head(limit)
    return df


def load_allenai(limit: int | None = None) -> pl.DataFrame:
    """Load AllenAI educational text dataset.

    Args:
        limit: Optional limit on number of rows

    Returns:
        DataFrame with columns: text, id, audience, format, etc.
    """
    path = Path("datasets/allenai_dolmino-mix-1124_mathcoder2-synthmath_book_math.0000.0008.jsonl")
    records = []

    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            records.append(json.loads(line))

    return pl.DataFrame(records)


def load_mbpp(limit: int | None = None) -> pl.DataFrame:
    """Load MBPP (Mostly Basic Python Problems) dataset.

    Args:
        limit: Optional limit on number of rows

    Returns:
        DataFrame with columns: task_id, prompt, code, test_imports, test_list
    """
    # Load sanitized version which has improved task descriptions
    dataset = hf_load_dataset("google-research-datasets/mbpp", "sanitized", split="train")

    # Convert to polars DataFrame
    df = pl.DataFrame(dataset.to_dict())

    if limit:
        df = df.head(limit)

    return df


def load_humaneval(limit: int | None = None) -> pl.DataFrame:
    """Load HumanEval code evaluation dataset.

    Args:
        limit: Optional limit on number of rows

    Returns:
        DataFrame with columns: task_id, prompt, canonical_solution, test, entry_point
    """
    # Load the test split (HumanEval only has test split)
    dataset = hf_load_dataset("openai/openai_humaneval", split="test")

    # Convert to polars DataFrame
    df = pl.DataFrame(dataset.to_dict())

    if limit:
        df = df.head(limit)

    return df


def load_popqa(limit: int | None = None) -> pl.DataFrame:
    """Load PopQA question answering dataset.

    Args:
        limit: Optional limit on number of rows

    Returns:
        DataFrame with columns: question, possible_answers, subj, obj, prop, etc.
    """
    # Load the dataset (only has test split)
    dataset = hf_load_dataset("akariasai/PopQA", split="test")

    # Convert to polars DataFrame
    df = pl.DataFrame(dataset.to_dict())

    if limit:
        df = df.head(limit)

    return df
