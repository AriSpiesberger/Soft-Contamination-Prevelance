"""Core generation logic for semantic duplicates."""

import polars as pl
from pathlib import Path
from datetime import datetime
from typing import Any, Callable
import json
import re
import math
import os
import requests
import time
import logging
import sys
from collections import Counter
import threading
from concurrent.futures import ThreadPoolExecutor

from sdtd.datasets import load_dataset
from sdtd.utils import get_client, load_prompts, format_prompt

# Global lock for file access
FILE_LOCK = threading.Lock()

# Map dataset names to their primary text field
DATASET_TEXT_FIELDS = {
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

# Embedding model
EMBEDDING_MODEL = "openrouter/qwen/qwen3-embedding-8b"


# Error handling configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0  # seconds
MAX_RETRY_DELAY = 60.0  # seconds
CHECKPOINT_INTERVAL = 10  # Save checkpoint every N items


def setup_error_logging(output_file: Path) -> None:
    """Set up detailed error logging to file.

    Args:
        output_file: Path to output file (log file will replace suffix with .log)
    """
    log_file = output_file.with_suffix(".log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )
    
    # Log header
    logging.info("=" * 60)
    logging.info(f"Session started at {datetime.now().isoformat()}")
    logging.info(f"Command: {' '.join(sys.argv)}")
    logging.info(f"Output file: {output_file}")
    logging.info("=" * 60)


def is_transient_error(error: Exception) -> bool:
    """Determine if an error is transient (should retry) or permanent.

    Args:
        error: Exception that occurred

    Returns:
        True if error is likely transient (network, timeout, rate limit)
    """
    error_str = str(error).lower()
    transient_indicators = [
        'timeout',
        'connection',
        'network',
        'rate limit',
        'too many requests',
        'service unavailable',
        '429',
        '503',
        '504',
        'temporarily unavailable',
    ]
    return any(indicator in error_str for indicator in transient_indicators)


def retry_with_backoff(func: Callable, *args, max_retries: int = MAX_RETRIES, **kwargs) -> Any:
    """Retry a function with exponential backoff on transient errors.

    Args:
        func: Function to call
        *args: Positional arguments for function
        max_retries: Maximum number of retry attempts
        **kwargs: Keyword arguments for function

    Returns:
        Result of function call

    Raises:
        Exception if all retries exhausted or permanent error encountered
    """
    delay = INITIAL_RETRY_DELAY

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Check if this is the last attempt or a permanent error
            if attempt == max_retries or not is_transient_error(e):
                raise

            # Log retry
            logging.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)

            # Exponential backoff with jitter
            delay = min(delay * 2 * (0.5 + 0.5 * time.time() % 1), MAX_RETRY_DELAY)


def get_existing_results(output_path: Path) -> set:
    """Get set of existing results from output file to skip duplicates.

    Returns:
        Set of (original_text_snippet, sd_level, sd_variant)
        where original_text_snippet is first 50 chars to avoid huge keys
    """
    if not output_path.exists():
        return set()
    
    try:
        with FILE_LOCK:
            df = pl.read_parquet(output_path)
        
        existing = set()
        for row in df.iter_rows(named=True):
            # Use original text snippet + variant info as unique key
            # ideally we would use a proper ID, but not all datasets have one
            # We can use the 'additional_info' ID if available, but fallback to text snippet
            
            # Try to get ID from additional_info first
            item_id = None
            if row.get("additional_info"):
                try:
                    info = json.loads(row["additional_info"])
                    item_id = info.get("id")
                except Exception:
                    pass
            
            if item_id:
                key = (str(item_id), row["sd_level"], row["sd_variant"])
            else:
                # Fallback to text snippet
                text = row.get("original_text", "")
                key = (text[:100], row["sd_level"], row["sd_variant"])
                
            existing.add(key)
            
        return existing
    except Exception as e:
        logging.warning(f"Failed to read existing results from {output_path}: {e}")
        return set()


def append_result_to_parquet(output_path: Path, result: dict) -> None:
    """Thread-safe append of a single result to parquet file.
    
    Args:
        output_path: Path to output file
        result: Result dictionary
    """
    with FILE_LOCK:
        try:
            df = pl.DataFrame([result])
            
            if output_path.exists():
                # Read existing schema/file to ensure compatibility
                # For simplicity in this append mode, we'll read, concat, write
                # This is not most efficient for huge files but safe and robust
                existing_df = pl.read_parquet(output_path)
                combined_df = pl.concat([existing_df, df])
                combined_df.write_parquet(output_path, compression="uncompressed")
            else:
                df.write_parquet(output_path, compression="uncompressed")
                
        except Exception as e:
            logging.error(f"Failed to append result to {output_path}: {e}")


def calculate_ngram_overlap(text1: str, text2: str, n: int = 2) -> float:
    """Calculate n-gram overlap percentage (Jaccard) between two texts.

    Args:
        text1: First text
        text2: Second text
        n: N-gram size

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


def calculate_char_ngram_overlap(text1: str, text2: str, n: int = 3) -> float:
    """Calculate character n-gram overlap percentage (Jaccard).

    Args:
        text1: First text
        text2: Second text
        n: Character n-gram size

    Returns:
        Overlap percentage (0-100)
    """
    def get_char_ngrams(text: str, n: int) -> set:
        text = text.lower()
        if len(text) < n:
            return set()
        return set(text[i:i+n] for i in range(len(text) - n + 1))

    ngrams1 = get_char_ngrams(text1, n)
    ngrams2 = get_char_ngrams(text2, n)

    if not ngrams1 or not ngrams2:
        return 0.0

    overlap = len(ngrams1 & ngrams2)
    total = len(ngrams1 | ngrams2)

    return (overlap / total) * 100 if total > 0 else 0.0


def calculate_rouge_l(text1: str, text2: str) -> float:
    """Calculate ROUGE-L F-measure (longest common subsequence).

    Args:
        text1: First text
        text2: Second text

    Returns:
        ROUGE-L F-measure (0-1)
    """
    def lcs_length(seq1: list, seq2: list) -> int:
        """Compute longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    words1 = re.findall(r'\b\w+\b', text1.lower())
    words2 = re.findall(r'\b\w+\b', text2.lower())

    if not words1 or not words2:
        return 0.0

    lcs_len = lcs_length(words1, words2)

    precision = lcs_len / len(words2) if words2 else 0.0
    recall = lcs_len / len(words1) if words1 else 0.0

    if precision + recall == 0:
        return 0.0

    f_measure = 2 * precision * recall / (precision + recall)
    return f_measure


def calculate_edit_distance_normalized(text1: str, text2: str) -> float:
    """Calculate normalized word-level Levenshtein distance.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Normalized edit distance (0-1)
    """
    def levenshtein(seq1: list, seq2: list) -> int:
        """Compute Levenshtein distance."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        return dp[m][n]

    words1 = re.findall(r'\b\w+\b', text1.lower())
    words2 = re.findall(r'\b\w+\b', text2.lower())

    if not words1 and not words2:
        return 0.0

    max_len = max(len(words1), len(words2))
    if max_len == 0:
        return 0.0

    distance = levenshtein(words1, words2)
    return distance / max_len


def calculate_tfidf_cosine(text1: str, text2: str) -> float:
    """Calculate TF-IDF weighted cosine similarity.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Cosine similarity (0-1)
    """
    def get_tokens(text: str) -> list:
        return re.findall(r'\b\w+\b', text.lower())

    tokens1 = get_tokens(text1)
    tokens2 = get_tokens(text2)

    if not tokens1 or not tokens2:
        return 0.0

    # Compute TF (term frequency)
    tf1 = Counter(tokens1)
    tf2 = Counter(tokens2)

    # Compute IDF (inverse document frequency) - treating two texts as corpus
    all_terms = set(tokens1) | set(tokens2)
    doc_count = {}
    for term in all_terms:
        count = (1 if term in tokens1 else 0) + (1 if term in tokens2 else 0)
        doc_count[term] = count

    idf = {term: math.log(2 / count) for term, count in doc_count.items()}

    # Compute TF-IDF vectors
    tfidf1 = {term: tf1[term] * idf[term] for term in tf1}
    tfidf2 = {term: tf2[term] * idf[term] for term in tf2}

    # Compute cosine similarity
    dot_product = sum(tfidf1.get(term, 0) * tfidf2.get(term, 0) for term in all_terms)

    norm1 = math.sqrt(sum(v ** 2 for v in tfidf1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in tfidf2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def check_number_preservation(text1: str, text2: str) -> dict:
    """Check if all numbers are preserved between texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Dictionary with preservation metrics
    """
    nums1 = set(re.findall(r'\b\d+\.?\d*\b', text1))
    nums2 = set(re.findall(r'\b\d+\.?\d*\b', text2))

    if not nums1 and not nums2:
        return {
            'all_preserved': True,
            'precision': 1.0,
            'recall': 1.0,
        }

    intersection = nums1 & nums2

    return {
        'all_preserved': nums1 == nums2,
        'precision': len(intersection) / len(nums2) if nums2 else 1.0,
        'recall': len(intersection) / len(nums1) if nums1 else 1.0,
    }


def calculate_length_ratio(text1: str, text2: str) -> float:
    """Calculate word count ratio.

    Args:
        text1: First text (original)
        text2: Second text (SD)

    Returns:
        Ratio of SD words to original words
    """
    words1 = re.findall(r'\b\w+\b', text1)
    words2 = re.findall(r'\b\w+\b', text2)

    if not words1:
        return 0.0

    return len(words2) / len(words1)


def calculate_jaccard_token(text1: str, text2: str) -> float:
    """Calculate token-level Jaccard similarity.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Jaccard similarity (0-1)
    """
    tokens1 = set(re.findall(r'\b\w+\b', text1.lower()))
    tokens2 = set(re.findall(r'\b\w+\b', text2.lower()))

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union) if union else 0.0


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """Get embedding for text using OpenRouter API directly (with retry logic).

    Args:
        text: Input text
        model: Embedding model name (with openrouter/ prefix)

    Returns:
        Embedding vector
    """
    def _get_embedding_inner():
        # litellm doesn't support embeddings via OpenRouter, so make direct API call
        if model.startswith("openrouter/"):
            actual_model = model.replace("openrouter/", "")

            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            data = {
                "model": actual_model,
                "input": text,
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers=headers,
                json=data,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            return result['data'][0]['embedding']
        else:
            # Fallback to OpenAI client for non-OpenRouter models
            response = get_client().embeddings.create(
                model=model,
                input=text,
            )
            return response.data[0].embedding

    # Use retry logic for embedding API calls
    return retry_with_backoff(_get_embedding_inner)


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (-1 to 1)
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same dimension")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a ** 2 for a in vec1))
    norm2 = math.sqrt(sum(b ** 2 for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def calculate_all_metrics(original_text: str, sd_text: str,
                         original_embedding: list[float], sd_embedding: list[float]) -> dict:
    """Calculate all similarity metrics.

    Args:
        original_text: Original text
        sd_text: Semantic duplicate text
        original_embedding: Original text embedding
        sd_embedding: SD text embedding

    Returns:
        Dictionary of all metrics
    """
    # N-gram overlaps (1-5)
    ngram_overlaps = [
        round(calculate_ngram_overlap(original_text, sd_text, n=i), 2)
        for i in range(1, 6)
    ]

    # Character n-gram overlaps (1-5)
    char_ngram_overlaps = [
        round(calculate_char_ngram_overlap(original_text, sd_text, n=i), 2)
        for i in range(1, 6)
    ]

    # Other metrics
    rouge_l = round(calculate_rouge_l(original_text, sd_text), 4)
    edit_dist = round(calculate_edit_distance_normalized(original_text, sd_text), 4)
    tfidf_cos = round(calculate_tfidf_cosine(original_text, sd_text), 4)
    length_ratio = round(calculate_length_ratio(original_text, sd_text), 4)
    jaccard = round(calculate_jaccard_token(original_text, sd_text), 4)

    num_preservation = check_number_preservation(original_text, sd_text)

    embedding_cosine = round(cosine_similarity(original_embedding, sd_embedding), 4)

    return {
        'ngram_overlaps_pct': ngram_overlaps,
        'char_ngram_overlaps_pct': char_ngram_overlaps,
        'rouge_l_f': rouge_l,
        'edit_distance_norm': edit_dist,
        'tfidf_cosine': tfidf_cos,
        'number_preservation': num_preservation['all_preserved'],
        'number_precision': round(num_preservation['precision'], 4),
        'number_recall': round(num_preservation['recall'], 4),
        'length_ratio': length_ratio,
        'jaccard_token': jaccard,
        'cosine_similarity': embedding_cosine,
        'cosine_similarity_model': EMBEDDING_MODEL,
    }


def process_item(
    item_idx: int,
    row: dict,
    dataset_name: str,
    variants: list[tuple[int, str, dict]], # (level, variant_name, prompt_config)
    output_path: Path,
    model_override: str | None = None,
) -> list[dict]:
    """Process a single item for multiple variants.
    
    Args:
        item_idx: Index of item in dataset
        row: Data row
        dataset_name: Dataset name
        variants: List of variants to process for this item
        output_path: Output file path to check/write
        model_override: Model override
        
    Returns:
        List of generated SD result dictionaries
    """
    results = []
    
    # Get existing results to skip
    existing_keys = get_existing_results(output_path)
    
    # Identify item ID for skipping check
    item_id = str(row.get("id", ""))
    text_field = DATASET_TEXT_FIELDS.get(dataset_name, "text")
    original_text = row.get(text_field, "")
    
    for level, variant_name, prompt_config in variants:
        # Check if already done
        # Try both ID and text snippet keys
        key_id = (item_id, level, variant_name)
        key_text = (original_text[:100], level, variant_name)
        
        if key_id in existing_keys or (not item_id and key_text in existing_keys):
            continue
            
        try:
            # Determine model
            yaml_model = prompt_config.get("model", "")
            if yaml_model == "none" and model_override is None:
                model = "none"
            else:
                model = model_override if model_override else yaml_model

            # Generate
            substitution_map = None
            transformed_solution = None
            sd_text = ""
            
            # Special handling for ZebraLogic transformations
            if dataset_name == "zebralogic":
                from sdtd.zebralogic_transforms import (
                    transform_category_substitution,
                    transform_condition_shuffle,
                    transform_shuffle_and_substitute,
                    transform_paraphrase,
                    transform_shuffle_and_paraphrase,
                    transform_shuffle_and_substitute_and_paraphrase,
                )
                
                puzzle = row.get("puzzle", "")
                solution = row.get("solution", {})
                original_reasoning = row.get("reasoning", "")

                if variant_name == "category_substitution":
                    sd_text, transformed_solution, substitution_map, sd_reasoning = transform_category_substitution(
                        puzzle, solution, original_reasoning, model, prompt_config.get("temperature"), level=level
                    )
                elif variant_name == "condition_shuffle":
                    sd_text, transformed_solution, substitution_map, sd_reasoning = transform_condition_shuffle(
                        puzzle, original_reasoning
                    )
                elif variant_name == "shuffle_and_substitute":
                    # Combined shuffle + category substitution
                    sd_text, transformed_solution, substitution_map, sd_reasoning = transform_shuffle_and_substitute(
                        puzzle, solution, original_reasoning, model, prompt_config.get("temperature"), level=level, prompt_template=prompt_config
                    )
                elif variant_name == "paraphrase":
                    sd_text, transformed_solution, substitution_map, sd_reasoning = transform_paraphrase(
                        puzzle, original_reasoning, model, prompt_config.get("temperature"), level=level, prompt_template=prompt_config
                    )
                elif variant_name == "shuffle_and_paraphrase":
                    sd_text, transformed_solution, substitution_map, sd_reasoning = transform_shuffle_and_paraphrase(
                        puzzle, solution, original_reasoning, model, prompt_config.get("temperature"), level=level, prompt_template=prompt_config
                    )
                elif variant_name == "shuffle_and_substitute_and_paraphrase":
                    sd_text, transformed_solution, substitution_map, sd_reasoning = transform_shuffle_and_substitute_and_paraphrase(
                        puzzle, solution, original_reasoning, model, prompt_config.get("temperature"), level=level, prompt_template=prompt_config
                    )
                else:
                    # Fallback to regular generation if not a specific logical transform
                    sd_text = generate_single(row, prompt_config, dataset_name, model_override)
                    transformed_solution = None
                    sd_reasoning = ""
            else:
                # Regular generation
                sd_text = generate_single(row, prompt_config, dataset_name, model_override)
                transformed_solution = None

            # Get embeddings
            original_embedding = get_embedding(original_text)
            sd_embedding = get_embedding(sd_text)

            # Calculate metrics
            metrics = calculate_all_metrics(original_text, sd_text,
                                           original_embedding, sd_embedding)

            # Prepare result
            additional_info = {}
            for k, v in row.items():
                if k != text_field:
                    if dataset_name == "zebralogic" and k == "solution":
                        continue
                    if isinstance(v, (list, dict)):
                        additional_info[k] = v
                    else:
                        additional_info[k] = str(v) if v is not None else None
            
            if dataset_name == "zebralogic":
                original_solution = row.get("solution", {})
                if original_solution:
                    additional_info["original_solution"] = original_solution
                if transformed_solution is not None:
                    additional_info["sd_solution"] = transformed_solution
                # Handle transformation metadata (value maps and clue permutations)
                if substitution_map and isinstance(substitution_map, dict):
                    # New format: dict with specific keys for clue_perm and/or value_map
                    if "clue_permutation" in substitution_map:
                        additional_info["clue_permutation"] = substitution_map["clue_permutation"]
                    if "value_category_map" in substitution_map:
                        additional_info["value_category_map"] = substitution_map["value_category_map"]
                    # Backward compat: if it's just a value map dict (old category_substitution format)
                    if not any(k in substitution_map for k in ["clue_permutation", "value_category_map"]):
                        additional_info["value_category_map"] = substitution_map
            elif transformed_solution is not None:
                additional_info["solution"] = transformed_solution

            result = {
                "source_dataset": dataset_name,
                "sd_level": level,
                "sd_variant": variant_name,
                "model_used": model,
                "original_text": original_text,
                "sd_text": sd_text,
                "original_reasoning": original_reasoning,  # NEW: top-level column
                "sd_reasoning": sd_reasoning,  # NEW: top-level column
                "original_embedding": original_embedding,
                "sd_embedding": sd_embedding,
                "embedding_model": EMBEDDING_MODEL,
                "metrics": json.dumps(metrics),
                "additional_info": json.dumps(additional_info),
                "timestamp": datetime.now().isoformat(),
            }
            
            # Add to results
            results.append(result)
            
        except Exception as e:
            error_msg = f"[{dataset_name}][L{level}][{variant_name}][Item {item_idx}] {type(e).__name__}: {e}"
            logging.error(error_msg)
            # We don't stop the whole process for one item error
            
    return results


def generate_sds(
    dataset_name: str,
    selection: list[str],
    output_file: Path,
    limit: int | None = None,
    model_override: str | None = None,
    checkpoint_enabled: bool = True, # Ignored, kept for compatibility
    input_file: Path | None = None,
    workers: int = 4,
) -> None:
    """Generate semantic duplicates for a dataset with parallel workers.

    Args:
        dataset_name: Name of dataset
        selection: List of levels (e.g. "1") or variant names
        output_file: Path to save output file
        limit: Optional limit on number of items
        model_override: Optional model override
        checkpoint_enabled: Deprecated, ignored (always resumes via file check)
        input_file: Optional input parquet file
        workers: Number of concurrent workers
    """
    output_file = Path(output_file)
    if output_file.parent:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    if model_override:
        print(f"Model override: Using '{model_override}' for all variants\n")

    # Handle "all" datasets
    if dataset_name == "all":
        datasets = load_dataset("all", limit, input_file=input_file)
        for name, df in datasets.items():
            # For "all", we need separate files, so we modify the filename
            # This is a bit tricky if user provided a specific file path
            # We'll append dataset name to stem
            ds_output_file = output_file.parent / f"{output_file.stem}_{name}{output_file.suffix}"
            _generate_for_dataset_parallel(name, df, selection, ds_output_file, model_override, workers)
    else:
        df = load_dataset(dataset_name, limit, input_file=input_file)
        _generate_for_dataset_parallel(dataset_name, df, selection, output_file, model_override, workers)


def _generate_for_dataset_parallel(
    dataset_name: str,
    df: pl.DataFrame,
    selection: list[str],
    output_path: Path,
    model_override: str | None = None,
    workers: int = 4,
) -> None:
    """Generate SDs for a single dataset using parallel workers.
    """
    print(f"\n{'='*60}")
    print(f"Generating SDs for {dataset_name} with {workers} workers")
    print(f"{'='*60}")

    setup_error_logging(output_path)

    # Parse selection
    levels_to_run_all = set()
    variants_to_run = set()
    
    for item in selection:
        item = str(item).strip()
        if item.isdigit():
            levels_to_run_all.add(int(item))
        else:
            variants_to_run.add(item)
    

    PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
    # Load prompts
    prompts = {}
    for level in [1, 2]:
        prompt_path = Path(f"{PROMPTS_DIR}/level{level}.yaml")
        prompts[level] = load_prompts(prompt_path)

    print(f"Output file: {output_path}")

    # Prepare list of variants to run
    variants_to_process = [] # List of (level, variant_name, prompt_config)
    
    for level, level_prompts in prompts.items():
        if dataset_name not in level_prompts:
            continue
            
        ds_prompts = level_prompts[dataset_name]
        
        # Determine active variants
        active_variants = []
        if level in levels_to_run_all:
            active_variants = list(ds_prompts.keys())
        else:
            for v in ds_prompts.keys():
                if v in variants_to_run:
                    active_variants.append(v)
                    
        for v_name in active_variants:
            variants_to_process.append((level, v_name, ds_prompts[v_name]))

    if not variants_to_process:
        print("No matching variants found to generate.")
        return

    print(f"Variants to process: {len(variants_to_process)}")
    for lvl, v, _ in variants_to_process:
        print(f"  - Level {lvl}: {v}")

    # Process items in parallel
    total_items = len(df)
    print(f"\nProcessing {total_items} items...")
    
    total_generated = 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for idx, row in enumerate(df.iter_rows(named=True)):
            futures.append(
                executor.submit(
                    process_item, 
                    idx, 
                    row, 
                    dataset_name, 
                    variants_to_process, 
                    output_path, 
                    model_override
                )
            )
            
        # Monitor progress
        completed = 0
        for future in futures:
            completed += 1
            try:
                results = future.result()
                
                # Write results
                for result in results:
                    append_result_to_parquet(output_path, result)
                    total_generated += 1
                    
                if completed % 10 == 0 or completed == total_items:
                    print(f"\rProgress: {completed}/{total_items} items processed (Total SDs generated: {total_generated})", end="")
            except Exception as e:
                logging.error(f"Worker exception: {e}")
                
    print(f"\n\nGeneration complete. Total SDs generated: {total_generated}")
    
    # Final consolidation and compression
    if output_path.exists():
        print("Consolidating and compressing output file...")
        try:
            # Re-read everything
            with FILE_LOCK:
                final_df = pl.read_parquet(output_path)
                # Write back compressed
                final_df.write_parquet(output_path, compression="zstd", compression_level=3)
            print(f"✓ Saved {len(final_df)} SDs to {output_path} (zstd compressed)")
        except Exception as e:
            logging.error(f"Failed to consolidate file: {e}")
            print(f"⚠ Failed to consolidate file: {e}")


def generate_single(
    row: dict,
    prompt_config: dict,
    dataset_name: str,
    model_override: str | None = None,
) -> str:
    """Generate a single semantic duplicate using OpenAI client via Helicone.

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
    if "system" in prompt_config and prompt_config["system"] and prompt_config["system"].strip():
        messages.append({"role": "system", "content": prompt_config["system"]})
    messages.append({"role": "user", "content": user_prompt})

    # Use model override if provided, otherwise use YAML default
    model = model_override if model_override else prompt_config["model"]

    def _generate_single_inner():
        # Call OpenAI client via Helicone
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": prompt_config.get("max_tokens", 2048),
        }
        
        # Only add temperature if provided (or default to None/provider default if explicitly removed from config)
        # Note: Previous default was 0.7. Now we rely on config or None.
        temp = prompt_config.get("temperature")
        if temp is not None:
            kwargs["temperature"] = temp
            
        response = get_client().chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    # Use retry logic for LLM calls
    return retry_with_backoff(_generate_single_inner)
