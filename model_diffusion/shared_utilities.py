#!/usr/bin/env python3
"""
Shared utilities for semantic duplicate classifier training and inference.

Contains functions and constants used across training scripts (train_*_classifier.py)
and classification scripts (classify_mbpp.py, classify_codeforces.py).
"""

import numpy as np
import pandas as pd
import json
import re
import urllib.request
import tarfile
from pathlib import Path
from typing import List, Dict, Any


# ============================================================
# Chat format constants
# ============================================================

# Qwen-style chat format (Qwen3-30B, Qwen3-235B)
QWEN_CHAT_FORMAT = {
    "prompt_template": "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
    "completion_suffix": "<|im_end|>",
    "stop_token": "<|im_end|>",
}

# Llama/GPT-OSS-style chat format (GPT-OSS-20b)
LLAMA_CHAT_FORMAT = {
    "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "completion_suffix": "<|eot_id|>",
    "stop_token": "<|eot_id|>",
}


# ============================================================
# Prompt templates
# ============================================================

MBPP_PROMPT_TEMPLATE = """You are an expert programmer analyzing potential semantic duplicates between coding tasks.
## Task
Determine if the following two coding tasks are semantic duplicates - meaning they describe the same programming task, just potentially phrased differently.
## Test Task (from benchmark):
{test_text}
## Corpus Task (from training data):
{corpus_text}
## Guidelines:
1. **Focus on the TASK, not the solution** - ignore any code or solutions that may be present
2. **Mathematical equivalence counts as duplicate** - e.g., "sum 1 to n" and "sum n, n-1, ..., 1" are equivalent
3. **Corpus subsumes test = duplicate** - if the corpus task is strictly harder (asks for more), but solving it would trivially solve the test task, mark as duplicate
4. **Be calibrated** - use confidence primarily for ambiguous cases, tricky phrasing, or when you're uncertain
## Match Types:
- "exact": Nearly identical wording
- "equivalent": Different phrasing, same underlying task
- "subset": Corpus is a subset of test (test asks for more)
- "superset": Corpus is a superset of test (corpus asks for more, but solving it solves test)
- "unrelated": Different tasks entirely
Respond with valid JSON only."""
CODEFORCES_CHECKER_TEMPLATE = """You are an exceptionally talented programmer and mathematician with the goal to determine whether there are semantic duplicates between two chunks of texts. 

You will be provided with a benchmark problem and a corpus text problem, as well as a reason that they may or may not be semantic duplicates. The rational may be wrong. 

Please consider how both problems are solved. Your goal here is to determine whether or not it is the case that the corpus problem provides signifant algorithmic insight required to solve a major part of the test problem, thus making it a semantic duplcate. 

Respond with TRUE if it is a semantic duplicate
Provide your reason, in detail, as to why it is or isnt. 
Provide a score on the amount of duplication from 0 to 1.
"""
CODEFORCES_PROMPT_TEMPLATE = """You are an expert competitive programmer analyzing potential semantic duplicates between programming problems.

## Task
Determine if the following two competitive programming problems are semantically related - meaning exposure to the corpus problem during training could help solve the test problem.

## Test Problem (from benchmark):
{test_text}

## Corpus Problem (from training data):
{corpus_text}

## Analysis Steps:
1. **Check data quality first**: Is the corpus text a complete problem statement? If it's empty, fragmentary, or contains only code without a problem description, mark as "unrelated".
2. **Extract the core problem**: Strip away story/narrative framing. What is the actual computational task?
3. **Identify the key insight**: What algorithmic technique or observation is needed?
4. **Compare**: Is there meaningful overlap in what's being asked or how to solve it?

## Match Types:
- "exact": Nearly identical problem statements
- "equivalent": Different framing but identical algorithmic core
- "subset": Test is a special case of corpus
- "superset": Corpus is a special case of test
- "related": Corpus covers a component or shares key insight with test
- "unrelated": Different problems, or corpus data is unusable

## What counts as semantically related:
- Same computational task (any framing)
- One is a special case of the other
- Shared key insight or trick
- Corpus solves a significant component of test

## What is unrelated:
- Sharing only common techniques (DP, BFS) without structural similarity
- Unusable corpus data (empty, fragmentary, code-only)
- Genuinely different computational questions

Analyze the problems and provide your structured judgment."""

# Shorter codeforces prompt used by classify_codeforces.py
CODEFORCES_STRICT_JSON_PROMPT_TEMPLATE = """Analyze if these two competitive programming problems are semantic duplicates.

TEST PROBLEM:
{test_text}

CORPUS PROBLEM:
{corpus_text}

Categories: exact (identical/verbatim substring), equivalent (same algorithmic core), subset (test is special case of corpus), superset (corpus is simpler), related (shared key insight), unrelated (different problems or unusable corpus)

is_duplicate = true if category is NOT "unrelated"

RESPOND WITH ONLY THIS JSON, NO OTHER TEXT:
{{"reasoning": "<brief analysis>", "predicted_category": "<category>", "is_duplicate": <true/false>, "confidence": <0.0-1.0>}}"""


# ============================================================
# Shared training utilities
# ============================================================

def compute_batch_loss(fwdbwd_result, batch) -> float:
    """Compute weighted average loss per token from a forward-backward result.

    Used identically in all training scripts.

    Args:
        fwdbwd_result: Result from training_client.forward_backward()
        batch: List of types.Datum used in the forward-backward pass

    Returns:
        Weighted average negative log probability (loss)
    """
    logprobs = np.concatenate([
        output['logprobs'].tolist()
        for output in fwdbwd_result.loss_fn_outputs
    ])
    weights = np.concatenate([
        example.loss_fn_inputs['weights'].tolist()
        for example in batch
    ])
    total_weight = weights.sum()
    if total_weight == 0:
        return 0.0
    return -np.dot(logprobs, weights) / total_weight


def evaluate(training_client, eval_data, batch_size: int) -> float:
    """Evaluate loss on held-out data with pipelined forward passes.

    Used identically in all training scripts.

    Args:
        training_client: Tinker LoRA training client
        eval_data: List of types.Datum for evaluation
        batch_size: Batch size for forward passes

    Returns:
        Weighted average loss over all eval data
    """
    total_loss = 0.0
    total_weight = 0.0

    num_batches = (len(eval_data) + batch_size - 1) // batch_size

    # Fire off all forward passes at once (async)
    futures = []
    batches = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(eval_data))
        batch = eval_data[start_idx:end_idx]
        batches.append(batch)
        futures.append(training_client.forward(batch, "cross_entropy"))

    # Collect results
    for fwd_future, batch in zip(futures, batches):
        fwd_result = fwd_future.result()

        logprobs = np.concatenate([
            output['logprobs'].tolist()
            for output in fwd_result.loss_fn_outputs
        ])
        weights = np.concatenate([
            example.loss_fn_inputs['weights'].tolist()
            for example in batch
        ])

        total_loss += -np.dot(logprobs, weights)
        total_weight += weights.sum()

    return total_loss / total_weight if total_weight > 0 else 0.0


# ============================================================
# Shared classification utilities
# ============================================================

def download_checkpoint(tinker_path: str, local_dir: str = "./checkpoint_cache") -> str:
    """Download a checkpoint from Tinker to local storage.

    Args:
        tinker_path: Full tinker path to the checkpoint
        local_dir: Local directory to save the checkpoint

    Returns:
        Path to the downloaded checkpoint directory
    """
    import tinker  # lazy import - only needed if actually downloading

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    archive_path = local_dir / "checkpoint.tar"
    extract_dir = local_dir / "weights"

    # Check if already downloaded
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"  Checkpoint already downloaded at {extract_dir}")
        return str(extract_dir)

    print(f"  Downloading checkpoint from: {tinker_path}")

    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()

    # Get the signed download URL
    future = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path)
    checkpoint_archive_url_response = future.result()

    print(f"  URL expires: {checkpoint_archive_url_response.expires}")
    print(f"  Downloading archive...")

    # Download the archive
    urllib.request.urlretrieve(checkpoint_archive_url_response.url, str(archive_path))
    print(f"  Downloaded to: {archive_path}")

    # Extract the archive
    print(f"  Extracting...")
    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(path=extract_dir)

    print(f"  Extracted to: {extract_dir}")

    # Clean up archive
    archive_path.unlink()

    return str(extract_dir)


def parse_json_response(response: str) -> dict | None:
    """Parse a JSON response from the model, with regex fallback.

    Tries direct JSON parse first, then falls back to extracting JSON
    from surrounding text using regex.

    Args:
        response: Raw model output string

    Returns:
        Parsed dict, or None if all parsing fails
    """
    # Try direct JSON parse first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract JSON from response (text before/after)
    json_match = re.search(r'\{[^{}]*"is_duplicate"[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def get_prompt_template(source: str) -> str:
    """Get the appropriate prompt template for a data source.

    Args:
        source: 'mbpp' or 'codeforces'

    Returns:
        Prompt template string with {test_text} and {corpus_text} placeholders
    """
    if source == 'codeforces':
        return CODEFORCES_PROMPT_TEMPLATE
    return MBPP_PROMPT_TEMPLATE


def prepare_text_fields(row: Dict[str, Any], max_length: int = 2000) -> tuple:
    """Extract and truncate test_text and corpus_text from a row.

    Args:
        row: Dict with 'test_text' and 'corpus_text' keys
        max_length: Maximum character length for each text field

    Returns:
        Tuple of (test_text, corpus_text) strings
    """
    test_text = str(row.get('test_text', ''))[:max_length] if pd.notna(row.get('test_text')) else ''
    corpus_text = str(row.get('corpus_text', ''))[:max_length] if pd.notna(row.get('corpus_text')) else ''
    return test_text, corpus_text
