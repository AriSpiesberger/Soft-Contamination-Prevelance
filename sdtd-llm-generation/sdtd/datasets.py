"""Dataset loading functions."""

import json
import polars as pl
from pathlib import Path
from datasets import load_dataset as hf_load_dataset


def load_dataset(name: str, limit: int | None = None, input_file: Path | str | None = None) -> pl.DataFrame:
    """Load dataset by name.

    Args:
        name: Dataset name (gsm8k, codeforces, allenai, mbpp, humaneval, popqa, bigbenchhard, zebralogic, agieval, or all)
        limit: Optional limit on number of rows
        input_file: Optional path to local input file (overrides default loading)

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
    elif name == "bigbenchhard":
        return load_bigbenchhard(limit)
    elif name == "zebralogic":
        return load_zebralogic(limit, input_file)
    elif name == "agieval":
        return load_agieval(limit)
    elif name == "all":
        # Return dict of all datasets
        return {
            "gsm8k": load_gsm8k(limit),
            "codeforces": load_codeforces(limit),
            "allenai": load_allenai(limit),
            "mbpp": load_mbpp(limit),
            "humaneval": load_humaneval(limit),
            "popqa": load_popqa(limit),
            "bigbenchhard": load_bigbenchhard(limit),
            "zebralogic": load_zebralogic(limit, input_file),
            # agieval: Not yet implemented due to HuggingFace compatibility issues
        }
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from: gsm8k, codeforces, allenai, mbpp, humaneval, popqa, bigbenchhard, zebralogic, agieval, all")


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


def load_bigbenchhard(limit: int | None = None) -> pl.DataFrame:
    """Load BIG-Bench Hard dataset.

    Args:
        limit: Optional limit on number of rows

    Returns:
        DataFrame with columns: input, target (and task name added)
    """
    # BBH has 27 separate tasks, we'll load them all and concatenate
    # Just load a few representative tasks to start
    tasks = ['boolean_expressions', 'causal_judgement', 'date_understanding',
             'logical_deduction_three_objects', 'word_sorting']

    dfs = []
    for task in tasks:
        dataset = hf_load_dataset("lukaemon/bbh", task, split="test")
        task_df = pl.DataFrame(dataset.to_dict())
        # Add task name as a column
        task_df = task_df.with_columns(pl.lit(task).alias("task"))
        dfs.append(task_df)

    # Concatenate all tasks
    df = pl.concat(dfs)

    if limit:
        df = df.head(limit)

    return df


def load_zebralogic(limit: int | None = None, input_file: Path | str | None = None) -> pl.DataFrame:
    """Load ZebraLogic dataset.

    Args:
        limit: Optional limit on number of rows
        input_file: Optional path to local parquet file

    Returns:
        DataFrame with columns: puzzle_id, puzzle, solution, clues, size_n, size_m, etc.
    """
    if input_file:
        df = pl.read_parquet(input_file)
    else:
        # Load the grid_mode configuration (full puzzle format with clues)
        dataset = hf_load_dataset("WildEval/ZebraLogic", "grid_mode", split="test")
        
        # Convert to polars DataFrame
        df = pl.DataFrame(dataset.to_dict())

    if limit:
        df = df.head(limit)

    return df


def load_agieval(limit: int | None = None) -> pl.DataFrame:
    """Load AGIEval English dataset.

    Args:
        limit: Optional limit on number of rows

    Returns:
        DataFrame with columns: question, answer, options, subject, etc.
    """
    # TODO: AGIEval dataset repositories have compatibility issues with current datasets library
    # This loader is a placeholder until a properly formatted repository is found
    raise NotImplementedError(
        "AGIEval dataset loader is not yet implemented. "
        "The available HuggingFace repositories have compatibility issues with the current datasets library. "
        "Please use other datasets for now."
    )
