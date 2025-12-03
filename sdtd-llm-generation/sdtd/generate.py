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
from collections import Counter
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


def setup_error_logging(output_dir: Path) -> None:
    """Set up detailed error logging to file.

    Args:
        output_dir: Directory to store error log
    """
    log_file = output_dir / "generation_errors.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


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


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load checkpoint file if it exists.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary with checkpoint data or empty dict if no checkpoint
    """
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load checkpoint: {e}")
            return {}
    return {}


def save_checkpoint(checkpoint_path: Path, checkpoint_data: dict) -> None:
    """Save checkpoint data to file.

    Args:
        checkpoint_path: Path to checkpoint file
        checkpoint_data: Data to save
    """
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")


def save_partial_results(output_path: Path, results: list[dict]) -> None:
    """Save partial results to parquet file.

    Args:
        output_path: Path to output file
        results: List of result dictionaries
    """
    if not results:
        return

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df = pl.DataFrame(results)
        result_df.write_parquet(output_path, compression="zstd", compression_level=3)
        logging.info(f"Saved {len(results)} partial results to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save partial results: {e}")


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
            # Fallback to litellm for non-OpenRouter models
            from litellm import embedding
            response = embedding(
                model=model,
                input=[text],
                caching=True,
            )
            return response.data[0]['embedding']

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


def generate_sds(
    dataset_name: str,
    levels: list[int],
    output_dir: Path,
    limit: int | None = None,
    model_override: str | None = None,
    checkpoint_enabled: bool = True,
) -> None:
    """Generate semantic duplicates for a dataset with checkpointing.

    Args:
        dataset_name: Name of dataset (gsm8k, codeforces, allenai, or all)
        levels: List of levels to generate (e.g., [1] or [1, 2])
        output_dir: Directory to save output files
        limit: Optional limit on number of items to process
        model_override: Optional model to use instead of YAML defaults
        checkpoint_enabled: Enable checkpoint/resume functionality (default: True)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_override:
        print(f"Model override: Using '{model_override}' for all variants\n")

    if checkpoint_enabled:
        print(f"Checkpointing enabled (saves every {CHECKPOINT_INTERVAL} items)\n")

    # Handle "all" datasets
    if dataset_name == "all":
        datasets = load_dataset("all", limit)
        for name, df in datasets.items():
            _generate_for_dataset(name, df, levels, output_dir, model_override, checkpoint_enabled)
    else:
        df = load_dataset(dataset_name, limit)
        _generate_for_dataset(dataset_name, df, levels, output_dir, model_override, checkpoint_enabled)


def _generate_for_dataset(
    dataset_name: str,
    df: pl.DataFrame,
    levels: list[int],
    output_dir: Path,
    model_override: str | None = None,
    checkpoint_enabled: bool = True,
) -> None:
    """Generate SDs for a single dataset with checkpointing and error resilience.

    Args:
        dataset_name: Name of dataset
        df: Dataset DataFrame
        levels: List of levels to generate
        output_dir: Output directory
        model_override: Optional model to use instead of YAML defaults
        checkpoint_enabled: Enable checkpoint/resume functionality (default: True)
    """
    print(f"\n{'='*60}")
    print(f"Generating SDs for {dataset_name}")
    print(f"{'='*60}")

    # Set up error logging
    setup_error_logging(output_dir)

    # Load prompts for requested levels
    prompts = {}
    for level in levels:
        prompt_path = Path(f"prompts/level{level}.yaml")
        if not prompt_path.exists():
            logging.warning(f"{prompt_path} not found, skipping level {level}")
            continue
        prompts[level] = load_prompts(prompt_path)

    # Prepare checkpoint and output paths
    checkpoint_path = output_dir / f".checkpoint_{dataset_name}_level{''.join(map(str, levels))}.json"
    output_path = output_dir / f"{dataset_name}_level{''.join(map(str, levels))}.parquet"

    # Load checkpoint if enabled
    checkpoint = load_checkpoint(checkpoint_path) if checkpoint_enabled else {}
    completed_items = checkpoint.get("completed_items", {})

    # Generate for each level
    all_results = []
    total_errors = 0
    total_items = 0

    try:
        for level in levels:
            if level not in prompts:
                continue

            level_prompts = prompts[level].get(dataset_name)
            if not level_prompts:
                logging.warning(f"No prompts found for {dataset_name} level {level}")
                continue

            print(f"\nLevel {level}: {len(level_prompts)} variants")

            for variant_name, prompt_config in level_prompts.items():
                # Use model override if provided, otherwise use YAML default
                model = model_override if model_override else prompt_config["model"]
                variant_key = f"L{level}_{variant_name}"

                # Check if this variant was already completed
                if variant_key in completed_items:
                    num_completed = completed_items[variant_key]
                    print(f"  - {variant_name} ({model}): Resuming from item {num_completed}...")
                else:
                    print(f"  - {variant_name} ({model})...", end=" ", flush=True)
                    completed_items[variant_key] = 0

                variant_results = []
                variant_errors = 0
                start_idx = completed_items[variant_key]

                for idx, row in enumerate(df.iter_rows(named=True)):
                    # Skip already completed items
                    if idx < start_idx:
                        continue

                    total_items += 1

                    try:
                        sd_text = generate_single(row, prompt_config, dataset_name, model_override)

                        # Get the primary text field for this dataset
                        text_field = DATASET_TEXT_FIELDS.get(dataset_name, "text")
                        original_text = row.get(text_field, "")

                        # Get embeddings for both texts
                        original_embedding = get_embedding(original_text)
                        sd_embedding = get_embedding(sd_text)

                        # Calculate all metrics
                        metrics = calculate_all_metrics(original_text, sd_text,
                                                       original_embedding, sd_embedding)

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
                            "original_embedding": original_embedding,
                            "sd_embedding": sd_embedding,
                            "embedding_model": EMBEDDING_MODEL,
                            "metrics": json.dumps(metrics),
                            "additional_info": json.dumps(additional_info),
                            "timestamp": datetime.now().isoformat(),
                        }

                        variant_results.append(result)
                        completed_items[variant_key] = idx + 1

                        # Checkpoint periodically
                        if checkpoint_enabled and (idx + 1) % CHECKPOINT_INTERVAL == 0:
                            checkpoint["completed_items"] = completed_items
                            save_checkpoint(checkpoint_path, checkpoint)
                            # Also save partial results
                            all_temp = all_results + variant_results
                            save_partial_results(output_path, all_temp)

                    except Exception as e:
                        variant_errors += 1
                        total_errors += 1
                        error_msg = f"[{dataset_name}][L{level}][{variant_name}][Item {idx}] {type(e).__name__}: {e}"
                        logging.error(error_msg)

                        # For non-transient errors, print to console as well
                        if not is_transient_error(e):
                            print(f"\n    ! Permanent error on item {idx}: {type(e).__name__}")

                        continue

                all_results.extend(variant_results)

                # Print summary for this variant
                if variant_errors > 0:
                    print(f"✓ {len(variant_results)} generated ({variant_errors} errors)")
                else:
                    print(f"✓ {len(variant_results)} generated")

                # Save checkpoint after completing variant
                if checkpoint_enabled:
                    checkpoint["completed_items"] = completed_items
                    save_checkpoint(checkpoint_path, checkpoint)
                    save_partial_results(output_path, all_results)

    finally:
        # Always save results and final checkpoint, even if interrupted
        if all_results:
            result_df = pl.DataFrame(all_results)
            result_df.write_parquet(output_path, compression="zstd", compression_level=3)
            logging.info(f"Saved {len(all_results)} SDs to {output_path} (zstd compressed)")
            print(f"\n✓ Saved {len(all_results)} SDs to {output_path} (zstd compressed)")
        else:
            logging.warning(f"No SDs generated for {dataset_name}")
            print(f"\n⚠ No SDs generated for {dataset_name}")

        # Save final checkpoint
        if checkpoint_enabled:
            checkpoint["completed_items"] = completed_items
            checkpoint["finished"] = True
            save_checkpoint(checkpoint_path, checkpoint)

        # Print error summary
        if total_errors > 0:
            error_rate = (total_errors / total_items * 100) if total_items > 0 else 0
            logging.info(f"Generation completed with {total_errors} errors out of {total_items} items ({error_rate:.1f}%)")
            print(f"\n⚠ {total_errors} errors occurred (see generation_errors.log for details)")


def generate_single(
    row: dict,
    prompt_config: dict,
    dataset_name: str,
    model_override: str | None = None,
) -> str:
    """Generate a single semantic duplicate using litellm (with retry logic).

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

    def _generate_single_inner():
        # Call litellm (with caching)
        response = completion(
            model=model,
            messages=messages,
            temperature=prompt_config.get("temperature", 0.7),
            max_tokens=prompt_config.get("max_tokens", 2048),
            caching=True,  # Enable caching
        )
        return response.choices[0].message.content.strip()

    # Use retry logic for LLM calls
    return retry_with_backoff(_generate_single_inner)
