"""
Annotate semantic duplicates using Gemini API with thinking.

Features:
- Concurrent annotation with configurable workers (default: 30)
- Resume capability: checks existing annotations and skips completed ones
- Budget tracking: hard stop when budget is reached
- Individual JSON outputs per annotation
- Structured output via Pydantic schemas

Usage:
    python annotate_semantic_duplicates.py --benchmark mbpp --budget 100 --workers 30
    python annotate_semantic_duplicates.py --benchmark codeforces --budget 100 --workers 30
    
    # Resume with increased budget:
    python annotate_semantic_duplicates.py --benchmark mbpp --budget 200 --workers 30
"""

import argparse
import asyncio
import json
import os
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import csv
import threading

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Simple progress tracking (no tqdm dependency)

load_dotenv()

# =============================================================================
# API KEY MANAGEMENT
# =============================================================================

class APIKeyManager:
    """Manages multiple API keys with automatic failover."""
    def __init__(self):
        self.keys = []
        self.current_index = 0
        self.lock = threading.Lock()
        self.consecutive_failures = {}  # key -> failure count

        # Load keys from environment
        primary = os.getenv("GOOGLE_API_KEY")
        backup = os.getenv("GOOGLE_API_KEY_BACKUP")

        if primary:
            self.keys.append(primary)
        if backup:
            self.keys.append(backup)

        if not self.keys:
            raise ValueError("No API keys found. Set GOOGLE_API_KEY environment variable.")

        print(f"Loaded {len(self.keys)} API key(s)")

    def get_current_key(self) -> str:
        with self.lock:
            return self.keys[self.current_index]

    def record_failure(self, key: str) -> bool:
        """Record a failure. Returns True if switched to new key."""
        with self.lock:
            self.consecutive_failures[key] = self.consecutive_failures.get(key, 0) + 1

            # Switch after 3 consecutive failures on same key
            if self.consecutive_failures[key] >= 3 and len(self.keys) > 1:
                old_index = self.current_index
                self.current_index = (self.current_index + 1) % len(self.keys)
                if self.current_index != old_index:
                    print(f"\n[KEY SWITCH] Switching to backup API key after {self.consecutive_failures[key]} failures")
                    self.consecutive_failures[key] = 0
                    return True
            return False

    def record_success(self, key: str):
        with self.lock:
            self.consecutive_failures[key] = 0

    def create_client(self) -> genai.Client:
        return genai.Client(api_key=self.get_current_key())


# Global key manager
_key_manager: Optional[APIKeyManager] = None

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent / "data"
ANNOTATIONS_DIR = Path(__file__).parent / "annotations"

SAMPLED_FILES = {
    "mbpp": DATA_DIR / "sampled_mbpp_for_annotation.csv",
    "codeforces": DATA_DIR / "sampled_codeforces_for_annotation.csv",
}

MODEL_ID = "gemini-3-flash-preview"

# Pricing per 1M tokens (USD) - Gemini 3 Flash Preview paid tier
PRICING = {
    "input": 0.50,       # $0.50 per 1M input tokens
    "output": 3.00,      # $3.00 per 1M output tokens
    "thinking": 3.00,    # Same as output for thinking tokens
}

# Default model parameters
MODEL_PARAMS = {
    "thinking_level": "MEDIUM",
    "temperature": 1.0,
    "max_output_tokens": 8192,
    "response_format": "json",
    "schema": "SDAnnotation",
}

# Benchmark-specific model parameters (override defaults)
MODEL_PARAMS_BY_BENCHMARK = {
    "mbpp": {},  # Use defaults
    "codeforces": {
        "thinking_level": "HIGH",  # Codeforces needs deeper reasoning for key insight identification
    },
}


def get_model_params(benchmark: str) -> dict:
    """Get model parameters for a specific benchmark, merging defaults with overrides."""
    params = MODEL_PARAMS.copy()
    params.update(MODEL_PARAMS_BY_BENCHMARK.get(benchmark, {}))
    return params

# Retry configuration
RETRY_CONFIG = {
    "max_retries": 5,
    "base_delay": 1.0,        # Base delay in seconds
    "max_delay": 60.0,        # Maximum delay between retries
    "exponential_base": 2,    # Exponential backoff multiplier
    "jitter": 0.5,            # Random jitter factor (0-1)
}

# Retryable error patterns (rate limits, server errors)
RETRYABLE_ERRORS = [
    "429",           # Rate limit
    "500",           # Internal server error
    "502",           # Bad gateway
    "503",           # Service unavailable
    "504",           # Gateway timeout
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "DEADLINE_EXCEEDED",
    "rate limit",
    "quota",
    "overloaded",
]

# =============================================================================
# PYDANTIC SCHEMAS FOR STRUCTURED OUTPUT
# =============================================================================

class SDAnnotation(BaseModel):
    """Structured annotation for semantic duplicate detection."""
    is_sd: bool = Field(description="Is this a semantic duplicate? True if the corpus task is the same as or subsumes the test task.")
    confidence: float = Field(ge=0.0, le=1.0, description="Calibrated confidence in the verdict. 1.0 = 100% certain, 0.0 = 50-50 guess.")
    reasoning: str = Field(description="Explanation of the judgment, including key similarities/differences observed.")
    match_type: str = Field(description="Type of match: 'exact' (nearly identical), 'equivalent' (same task, different wording), 'subset' (test is subset of corpus), 'superset' (corpus is subset of test), 'unrelated' (different tasks)")


# =============================================================================
# PROMPTS
# =============================================================================

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
- "subset": Test task is a subset of corpus task (corpus is harder but solves test)
- "superset": Corpus task is a subset of test task (test is harder) - NOT a duplicate
- "unrelated": Different tasks entirely

Analyze the tasks and provide your structured judgment."""


CODEFORCES_PROMPT_TEMPLATE = """You are an expert competitive programmer analyzing potential semantic duplicates between programming problems.

## Task
Determine if the following two competitive programming problems are semantic duplicates - meaning exposure to the corpus problem during training would effectively leak how to solve the test problem.

## Test Problem (from benchmark):
{test_text}

## Corpus Problem (from training data):
{corpus_text}

## Analysis Steps:
1. **Check data quality first**: Is the corpus text a complete problem statement? If it's empty, fragmentary (just I/O format or constraints), or contains only solution code without a problem description, mark as "unrelated".
2. **Extract the core problem**: Strip away story/narrative framing. What is the actual computational task being asked?
3. **Identify the key insight**: What is the "trick" or non-obvious observation needed to solve each problem efficiently?
4. **Compare**: Do both problems require the same algorithmic approach AND the same key insight?

## Guidelines:

**What makes problems duplicates:**
- Same computational task after removing story framing
- Same key insight or "trick" needed to solve efficiently
- Corpus is a generalization that, once solved, trivially solves the test problem

**What does NOT make problems duplicates:**
- Sharing a common technique (BFS, DP, segment tree) - the structure of the problem must also match
- Similar input/output format
- Same problem category or tags
- Both involving graphs, math, or strings

**Constraints:**
- Vastly different constraints (n≤100 vs n≤10⁶) that require fundamentally different algorithmic complexity are usually different problems
- Minor constraint differences (n≤10⁵ vs n≤2×10⁵) with the same optimal approach are still duplicates

**Data quality issues → mark as "unrelated":**
- Corpus text is empty or nearly empty
- Corpus is just I/O format, examples, or constraints without the actual problem
- Corpus contains only solution code or editorial without problem statement
- Corpus text is corrupted or unintelligible

## Match Types:
- "exact": Nearly identical problem statements
- "equivalent": Different framing but identical algorithmic core and key insight
- "subset": Test is a special case of corpus; the corpus solution can be adapted to solve test with minimal changes
- "superset": Corpus is a special case of test; test requires more - NOT a duplicate
- "unrelated": Different problems, or corpus data is incomplete/empty/unusable

## Calibration:
- Use high confidence (0.8-1.0) when the algorithmic core clearly matches or clearly differs
- Use moderate confidence (0.5-0.8) when problems share techniques but the key insight may differ
- Use low confidence (0.3-0.5) for ambiguous cases or when corpus data quality is questionable

Analyze the problems and provide your structured judgment."""


PROMPTS = {
    "mbpp": MBPP_PROMPT_TEMPLATE,
    "codeforces": CODEFORCES_PROMPT_TEMPLATE,
}


# =============================================================================
# GEMINI API WITH STRUCTURED OUTPUT
# =============================================================================

# Thread-safe cost tracking
class CostTracker:
    def __init__(self, budget: float):
        self.budget = budget
        self.total_cost = 0.0
        self.lock = threading.Lock()
        self.budget_exceeded = False
    
    def add_cost(self, cost: float) -> bool:
        """Add cost and return True if still within budget."""
        with self.lock:
            self.total_cost += cost
            if self.total_cost >= self.budget:
                self.budget_exceeded = True
            return not self.budget_exceeded
    
    def get_total(self) -> float:
        with self.lock:
            return self.total_cost
    
    def is_exceeded(self) -> bool:
        with self.lock:
            return self.budget_exceeded


# Global rate limiter to coordinate backoff across all workers
class GlobalRateLimiter:
    """
    Coordinates rate limiting across all workers to prevent thundering herd.
    When 429s occur, ALL workers must wait before retrying.
    """
    def __init__(self, base_delay: float = 5.0, max_delay: float = 120.0):
        self.lock = threading.Lock()
        self.backoff_until = 0.0  # Unix timestamp when backoff ends
        self.consecutive_429s = 0
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def record_429(self) -> float:
        """
        Record a 429 error and compute global backoff.
        Returns the delay all workers should wait.
        """
        with self.lock:
            now = time.time()
            self.consecutive_429s += 1
            
            # Exponential backoff based on consecutive 429s
            delay = min(self.base_delay * (2 ** min(self.consecutive_429s - 1, 6)), self.max_delay)
            # Add jitter to prevent synchronized retries
            delay = delay * (1 + 0.3 * random.random())
            
            # Set global backoff time
            new_backoff_until = now + delay
            if new_backoff_until > self.backoff_until:
                self.backoff_until = new_backoff_until
            
            return delay
    
    def record_success(self):
        """Record a successful request, gradually reduce backoff."""
        with self.lock:
            # Decay consecutive 429s on success (but don't reset completely)
            self.consecutive_429s = max(0, self.consecutive_429s - 1)
    
    def wait_if_needed(self) -> float:
        """
        Wait if we're in a global backoff period.
        Returns the time waited (0 if no wait needed).
        """
        with self.lock:
            now = time.time()
            if now < self.backoff_until:
                wait_time = self.backoff_until - now
            else:
                wait_time = 0
        
        if wait_time > 0:
            time.sleep(wait_time)
        
        return wait_time
    
    def get_status(self) -> dict:
        """Get current rate limiter status."""
        with self.lock:
            now = time.time()
            return {
                "consecutive_429s": self.consecutive_429s,
                "backoff_remaining": max(0, self.backoff_until - now),
                "in_backoff": now < self.backoff_until,
            }


# Global instance (will be initialized per run)
_global_rate_limiter: Optional[GlobalRateLimiter] = None


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable (rate limit, server error, etc.)."""
    error_str = str(error).lower()
    return any(pattern.lower() in error_str for pattern in RETRYABLE_ERRORS)


def generate_annotation(
    client: genai.Client,
    prompt: str,
    model: str = MODEL_ID,
    model_params: dict = None,
    retry_config: dict = None,
    key_manager: Optional[APIKeyManager] = None,
) -> dict:
    """
    Generate structured annotation using Gemini with thinking.
    Includes retry logic with exponential backoff for transient errors.
    Uses global rate limiter to coordinate backoff across workers.
    Supports automatic API key switching on repeated failures.

    Returns dict with annotation, thoughts, usage, and cost.
    """
    global _global_rate_limiter

    if model_params is None:
        model_params = MODEL_PARAMS
    if retry_config is None:
        retry_config = RETRY_CONFIG

    max_retries = retry_config.get("max_retries", 5)
    base_delay = retry_config.get("base_delay", 1.0)
    max_delay = retry_config.get("max_delay", 60.0)
    exp_base = retry_config.get("exponential_base", 2)
    jitter = retry_config.get("jitter", 0.5)

    last_error = None
    current_client = client
    current_key = key_manager.get_current_key() if key_manager else None

    for attempt in range(max_retries + 1):
        # Wait for global rate limit before attempting request
        if _global_rate_limiter:
            waited = _global_rate_limiter.wait_if_needed()
            if waited > 0:
                print(f"  [Global backoff] Waited {waited:.1f}s before request")
        
        try:
            response_stream = current_client.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_level=model_params.get("thinking_level", "medium"),
                        include_thoughts=True
                    ),
                    temperature=model_params.get("temperature", 1.0),
                    max_output_tokens=model_params.get("max_output_tokens", 8192),
                    response_mime_type="application/json",
                    response_schema=SDAnnotation,
                )
            )
            
            # Collect all parts from stream
            thoughts = []
            answer_parts = []
            usage_metadata = None
            model_version = None
            
            for chunk in response_stream:
                if chunk.usage_metadata:
                    usage_metadata = chunk.usage_metadata
                if chunk.model_version:
                    model_version = chunk.model_version
                    
                if chunk.candidates and chunk.candidates[0].content:
                    for part in chunk.candidates[0].content.parts:
                        is_thought = getattr(part, 'thought', False)
                        text = getattr(part, 'text', None)
                        
                        if is_thought and text:
                            thoughts.append(text)
                        elif text:
                            answer_parts.append(text)
            
            raw_answer = "".join(answer_parts)
            
            # Handle thought delimiter
            THOUGHT_DELIMITER = "<|thought|>"
            if THOUGHT_DELIMITER in raw_answer:
                leaked_thought, clean_answer = raw_answer.split(THOUGHT_DELIMITER, 1)
                if leaked_thought.strip():
                    thoughts.append(f"\n[LEAKED THINKING]\n{leaked_thought}")
                raw_answer = clean_answer
            
            # Parse JSON response
            try:
                annotation_dict = json.loads(raw_answer)
                annotation = SDAnnotation(**annotation_dict)
            except (json.JSONDecodeError, ValueError) as e:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', raw_answer, re.DOTALL)
                if json_match:
                    annotation_dict = json.loads(json_match.group())
                    annotation = SDAnnotation(**annotation_dict)
                else:
                    raise ValueError(f"Could not parse annotation: {raw_answer[:500]}") from e
            
            # Calculate costs
            usage = {}
            cost = {}
            
            if usage_metadata:
                prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0) or 0
                answer_tokens = getattr(usage_metadata, 'candidates_token_count', 0) or 0
                thought_tokens = getattr(usage_metadata, 'thoughts_token_count', 0) or 0
                total_tokens = getattr(usage_metadata, 'total_token_count', 0) or 0
                
                prompt_cost = (prompt_tokens / 1_000_000) * PRICING["input"]
                thinking_cost = (thought_tokens / 1_000_000) * PRICING["thinking"]
                answer_cost = (answer_tokens / 1_000_000) * PRICING["output"]
                total_cost = prompt_cost + thinking_cost + answer_cost
                
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "answer_tokens": answer_tokens,
                    "thought_tokens": thought_tokens,
                    "total_tokens": total_tokens,
                }
                cost = {
                    "prompt": prompt_cost,
                    "thinking": thinking_cost,
                    "output": answer_cost,
                    "total": total_cost,
                }
            
            # Success - record it to decay global backoff
            if _global_rate_limiter:
                _global_rate_limiter.record_success()
            
            return {
                "success": True,
                "annotation": annotation.model_dump(),
                "thoughts": thoughts,
                "raw_answer": raw_answer,
                "usage": usage,
                "cost": cost,
                "model_version": model_version,
            }
            
        except Exception as e:
            last_error = e
            error_str = str(e)
            is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
            
            # Check if we should retry
            if attempt < max_retries and is_retryable_error(e):
                # Calculate per-request delay with exponential backoff + jitter
                delay = min(base_delay * (exp_base ** attempt), max_delay)
                delay = delay * (1 + jitter * (random.random() - 0.5))
                
                # For rate limits, also update global rate limiter
                if is_rate_limit and _global_rate_limiter:
                    global_delay = _global_rate_limiter.record_429()
                    # Use the larger of per-request or global delay
                    delay = max(delay, global_delay)
                    print(f"  [Retry {attempt + 1}/{max_retries}] RATE LIMITED: {error_str[:60]}...")
                    print(f"    -> Per-request backoff + global coordination: waiting {delay:.1f}s")
                else:
                    print(f"  [Retry {attempt + 1}/{max_retries}] {type(e).__name__}: {error_str[:50]}... waiting {delay:.1f}s")
                
                time.sleep(delay)
                continue
            
            # Non-retryable error or max retries exceeded
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "annotation": None,
                "thoughts": [],
                "raw_answer": None,
                "usage": {},
                "cost": {"total": 0},
                "model_version": None,
            }
    
    # Should not reach here, but just in case
    return {
        "success": False,
        "error": str(last_error) if last_error else "Unknown error after retries",
        "error_type": type(last_error).__name__ if last_error else "Unknown",
        "annotation": None,
        "thoughts": [],
        "raw_answer": None,
        "usage": {},
        "cost": {"total": 0},
        "model_version": None,
    }


# =============================================================================
# ANNOTATION PIPELINE
# =============================================================================

def get_annotation_path(benchmark: str, test_id: str, corpus_id: str, dataset: str) -> Path:
    """Get path for annotation JSON file."""
    out_dir = ANNOTATIONS_DIR / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize IDs for filename
    safe_test_id = re.sub(r'[^\w\-]', '_', str(test_id))
    safe_corpus_id = re.sub(r'[^\w\-]', '_', str(corpus_id))
    safe_dataset = re.sub(r'[^\w\-]', '_', str(dataset))
    return out_dir / f"{safe_dataset}__{safe_test_id}__{safe_corpus_id}.json"


def load_existing_annotations(benchmark: str) -> dict:
    """
    Load existing annotations and compute statistics.
    
    Returns dict with:
        - completed: set of completed keys
        - total_cost: total cost spent
        - num_sd: number of semantic duplicates found
        - num_not_sd: number of non-duplicates
        - sd_confidences: list of confidences for is_sd=True
        - avg_sd_confidence: average confidence for semantic duplicates
        - sd_percentage: percentage labeled as semantic duplicates
    """
    out_dir = ANNOTATIONS_DIR / benchmark
    result = {
        "completed": set(),
        "total_cost": 0.0,
        "num_sd": 0,
        "num_not_sd": 0,
        "sd_confidences": [],
        "errors": 0,
        "total_annotated": 0,
        "sd_percentage": 0.0,
        "avg_sd_confidence": 0.0,
    }
    
    if not out_dir.exists():
        return result
    
    for json_file in out_dir.glob("*.json"):
        # Skip summary file
        if json_file.stem.startswith("_"):
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract key from filename
            key = json_file.stem  # dataset__test_id__corpus_id
            result["completed"].add(key)
            
            # Add cost
            if "cost" in data and "total" in data["cost"]:
                result["total_cost"] += data["cost"]["total"]
            
            # Track SD stats
            if data.get("success") and data.get("annotation"):
                ann = data["annotation"]
                if ann.get("is_sd"):
                    result["num_sd"] += 1
                    if "confidence" in ann:
                        result["sd_confidences"].append(ann["confidence"])
                else:
                    result["num_not_sd"] += 1
            elif not data.get("success"):
                result["errors"] += 1
                
        except Exception:
            pass  # Skip corrupted files
    
    # Compute derived stats
    total_annotated = result["num_sd"] + result["num_not_sd"]
    result["total_annotated"] = total_annotated
    
    if total_annotated > 0:
        result["sd_percentage"] = (result["num_sd"] / total_annotated) * 100
    else:
        result["sd_percentage"] = 0.0
    
    if result["sd_confidences"]:
        result["avg_sd_confidence"] = sum(result["sd_confidences"]) / len(result["sd_confidences"])
    else:
        result["avg_sd_confidence"] = 0.0
    
    return result


def annotate_row(
    client: genai.Client,
    row: dict,
    benchmark: str,
    cost_tracker: CostTracker,
    sampling_params: dict,
    key_manager: Optional[APIKeyManager] = None,
) -> Optional[dict]:
    """Annotate a single row. Returns None if budget exceeded or already done."""

    # Check budget before starting
    if cost_tracker.is_exceeded():
        return None

    test_id = row["test_id"]
    corpus_id = row["corpus_id"]
    dataset = row["dataset"]

    # Check if already annotated
    out_path = get_annotation_path(benchmark, test_id, corpus_id, dataset)
    key = out_path.stem

    if out_path.exists():
        return {"skipped": True, "key": key}

    # Build prompt
    prompt_template = PROMPTS[benchmark]
    prompt = prompt_template.format(
        test_text=row["test_text"],
        corpus_text=row["corpus_text"],
    )

    # Get benchmark-specific model params
    model_params = get_model_params(benchmark)

    # Generate annotation with key manager for failover
    result = generate_annotation(client, prompt, model_params=model_params, key_manager=key_manager)
    
    # Update cost tracker
    cost = result["cost"].get("total", 0)
    within_budget = cost_tracker.add_cost(cost)
    
    # Build output
    output = {
        "test_id": test_id,
        "corpus_id": corpus_id,
        "dataset": dataset,
        "benchmark": row.get("benchmark", benchmark),
        "test_text": row["test_text"],
        "corpus_text": row["corpus_text"],
        "score": float(row["score"]),
        "weight": float(row["weight"]),
        "prompt": prompt,
        "thoughts": result["thoughts"],
        "annotation": result["annotation"],
        "raw_answer": result["raw_answer"],
        "success": result["success"],
        "error": result.get("error"),
        "error_type": result.get("error_type"),
        "usage": result["usage"],
        "cost": result["cost"],
        "metadata": {
            "model": MODEL_ID,
            "model_params": model_params,
            "sampling_params": sampling_params,
            "timestamp": datetime.now().isoformat(),
        },
    }
    
    # Only save successful annotations to the main directory
    # Failed annotations go to a separate _failed directory so they can be retried
    if result["success"]:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    else:
        # Save failures to a separate directory for debugging (but don't block retries)
        failed_dir = out_path.parent / "_failed"
        failed_dir.mkdir(parents=True, exist_ok=True)
        failed_path = failed_dir / f"{out_path.stem}__{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    return {
        "skipped": False,
        "key": key,
        "success": result["success"],
        "cost": cost,
        "is_sd": result["annotation"]["is_sd"] if result["annotation"] else None,
        "confidence": result["annotation"]["confidence"] if result["annotation"] else None,
        "within_budget": within_budget,
    }


def run_annotations(
    benchmark: str,
    budget: float,
    workers: int,
    dry_run: bool = False,
) -> dict:
    """Run annotation pipeline with concurrency."""
    
    # Load sampled CSV
    sampled_path = SAMPLED_FILES[benchmark]
    if not sampled_path.exists():
        raise FileNotFoundError(f"Sampled file not found: {sampled_path}. Run sample_for_annotation.py first.")
    
    print(f"Loading {sampled_path}...")
    rows = []
    with open(sampled_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Loaded {len(rows):,} rows to annotate")
    
    # Load existing annotations and stats
    existing = load_existing_annotations(benchmark)
    completed = existing["completed"]
    existing_cost = existing["total_cost"]
    
    print(f"\n{'='*60}")
    print("EXISTING ANNOTATIONS SUMMARY")
    print(f"{'='*60}")
    print(f"  Completed:        {len(completed):,} annotations")
    print(f"  Cost spent:       ${existing_cost:.4f}")
    if existing["total_annotated"] > 0:
        print(f"  Semantic dupes:   {existing['num_sd']:,} / {existing['total_annotated']:,} "
              f"({existing['sd_percentage']:.1f}%)")
        print(f"  Avg SD confidence: {existing['avg_sd_confidence']:.2f}")
    if existing["errors"] > 0:
        print(f"  Errors:           {existing['errors']:,}")
    print(f"{'='*60}\n")
    
    # Filter to pending rows
    pending_rows = []
    for row in rows:
        out_path = get_annotation_path(benchmark, row["test_id"], row["corpus_id"], row["dataset"])
        if out_path.stem not in completed:
            pending_rows.append(row)
    
    print(f"Pending: {len(pending_rows):,} rows")
    
    remaining_budget = budget - existing_cost
    if remaining_budget <= 0:
        print(f"Budget already exceeded! Spent ${existing_cost:.4f} >= ${budget:.2f}")
        return {"status": "budget_exceeded", "spent": existing_cost}
    
    print(f"Remaining budget: ${remaining_budget:.4f}")
    
    if dry_run:
        print("DRY RUN - not actually annotating")
        return {"status": "dry_run", "pending": len(pending_rows), "existing": existing}
    
    if len(pending_rows) == 0:
        print("No pending rows to annotate!")
        return {"status": "complete", "existing": existing}
    
    # Load sampling metadata (exclude file paths, keep only relevant params)
    meta_path = DATA_DIR / f"sampled_{benchmark}_metadata.json"
    sampling_params = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            full_meta = json.load(f)
        # Only keep relevant fields, exclude file paths
        sampling_params = {
            "benchmark": full_meta.get("benchmark"),
            "max_per_test": full_meta.get("max_per_test"),
            "seed": full_meta.get("seed"),
            "weight_formula": full_meta.get("weight_formula"),
            "filter": full_meta.get("filter"),
        }
    
    # Initialize
    global _global_rate_limiter, _key_manager
    _key_manager = APIKeyManager()
    cost_tracker = CostTracker(remaining_budget)
    _global_rate_limiter = GlobalRateLimiter(base_delay=5.0, max_delay=120.0)
    
    # Session stats (for this run only)
    session_stats = {
        "total": len(pending_rows),
        "completed": 0,
        "skipped": 0,
        "errors": 0,
        "duplicates_found": 0,
        "sd_confidences": [],
        "cost": 0.0,
    }
    
    print(f"\nStarting annotation with {workers} workers...")
    print("-" * 60)
    
    import sys
    start_time = time.time()
    
    def process_row(row):
        # Create client with current key (may switch on failures)
        client = _key_manager.create_client()
        return annotate_row(client, row, benchmark, cost_tracker, sampling_params, _key_manager)
    
    def print_progress(done, total, session_stats, existing, existing_cost, cost_tracker, checkpoint=False):
        """Print progress update."""
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        
        total_spent = existing_cost + cost_tracker.get_total()
        total_done = len(existing["completed"]) + done
        total_sd = existing["num_sd"] + session_stats["duplicates_found"]
        total_not_sd = existing["num_not_sd"] + (session_stats["completed"] - session_stats["duplicates_found"] - session_stats["errors"])
        total_annotated = total_sd + total_not_sd
        sd_pct = (total_sd / total_annotated * 100) if total_annotated > 0 else 0
        
        all_confidences = existing["sd_confidences"] + session_stats["sd_confidences"]
        avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        # Get rate limiter status
        rl_status = _global_rate_limiter.get_status() if _global_rate_limiter else {}
        
        if checkpoint:
            print(f"\n{'='*60}")
            print(f"CHECKPOINT @ {done} rows processed this session")
            print(f"{'='*60}")
            print(f"  Total completed:  {total_done} annotations")
            print(f"  Cost spent:       ${total_spent:.4f}")
            print(f"  Semantic dupes:   {total_sd} / {total_annotated} ({sd_pct:.1f}%)")
            print(f"  Avg SD confidence: {avg_conf:.2f}")
            print(f"  Rate:             {rate:.1f} rows/sec")
            print(f"  ETA:              {eta/60:.1f} min remaining")
            if session_stats["errors"] > 0:
                print(f"  Errors:           {session_stats['errors']}")
            if rl_status.get("consecutive_429s", 0) > 0:
                print(f"  Rate limit hits:  {rl_status['consecutive_429s']} consecutive 429s")
            print(f"{'='*60}")
            sys.stdout.flush()
        else:
            # Simple one-line progress
            pct = (done / total * 100) if total > 0 else 0
            rl_info = f" | 429s: {rl_status.get('consecutive_429s', 0)}" if rl_status.get("consecutive_429s", 0) > 0 else ""
            print(f"[{pct:5.1f}%] {done}/{total} | SDs: {session_stats['duplicates_found']} | Err: {session_stats['errors']} | ${total_spent:.4f} | {rate:.1f}/s{rl_info}")
            sys.stdout.flush()
    
    # Submit all tasks
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_row, row): row for row in pending_rows}
        
        for future in as_completed(futures):
            if cost_tracker.is_exceeded():
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break
            
            try:
                result = future.result(timeout=120)  # 2 min timeout per request
                
                if result is None:
                    continue
                
                if result.get("skipped"):
                    session_stats["skipped"] += 1
                else:
                    session_stats["completed"] += 1
                    session_stats["cost"] += result.get("cost", 0)
                    
                    if not result.get("success"):
                        session_stats["errors"] += 1
                    elif result.get("is_sd"):
                        session_stats["duplicates_found"] += 1
                        if result.get("confidence"):
                            session_stats["sd_confidences"].append(result["confidence"])
                
                # Progress updates
                done = session_stats["completed"] + session_stats["skipped"]
                
                # Detailed checkpoint every 500 rows
                if done > 0 and done % 500 == 0:
                    print_progress(done, session_stats["total"], session_stats, existing, existing_cost, cost_tracker, checkpoint=True)
                # Simple progress every 50 rows
                elif done > 0 and done % 50 == 0:
                    print_progress(done, session_stats["total"], session_stats, existing, existing_cost, cost_tracker, checkpoint=False)
                    
            except Exception as e:
                print(f"\n[ERROR] Worker error: {e}")
                sys.stdout.flush()
                session_stats["errors"] += 1
    
    # Compute final stats
    total_spent = existing_cost + cost_tracker.get_total()
    
    # Reload to get accurate final counts
    final = load_existing_annotations(benchmark)
    
    print("\n" + "=" * 60)
    print("ANNOTATION COMPLETE")
    print("=" * 60)
    print(f"\nSession stats:")
    print(f"  Completed this run:  {session_stats['completed']}")
    print(f"  Errors this run:     {session_stats['errors']}")
    print(f"  SDs found this run:  {session_stats['duplicates_found']}")
    print(f"  Session cost:        ${session_stats['cost']:.4f}")
    
    print(f"\nOverall stats:")
    print(f"  Total annotations:   {final['total_annotated']:,}")
    print(f"  Semantic duplicates: {final['num_sd']:,} ({final['sd_percentage']:.1f}%)")
    print(f"  Avg SD confidence:   {final['avg_sd_confidence']:.2f}")
    print(f"  Total cost:          ${final['total_cost']:.4f} / ${budget:.2f}")
    
    if cost_tracker.is_exceeded():
        print("\n[!] BUDGET EXCEEDED - annotation stopped early")
    
    # Save summary
    summary_path = ANNOTATIONS_DIR / benchmark / "_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark": benchmark,
            "timestamp": datetime.now().isoformat(),
            "budget": budget,
            "model": MODEL_ID,
            "model_params": get_model_params(benchmark),
            "session_stats": session_stats,
            "overall_stats": {
                "total_annotated": final["total_annotated"],
                "num_sd": final["num_sd"],
                "num_not_sd": final["num_not_sd"],
                "sd_percentage": final["sd_percentage"],
                "avg_sd_confidence": final["avg_sd_confidence"],
                "total_cost": final["total_cost"],
                "errors": final["errors"],
            },
        }, f, indent=2)
    
    return {
        "session": session_stats,
        "overall": final,
        "budget_exceeded": cost_tracker.is_exceeded(),
    }


def main():
    parser = argparse.ArgumentParser(description="Annotate semantic duplicates using Gemini")
    parser.add_argument(
        "--benchmark",
        choices=["mbpp", "codeforces"],
        required=True,
        help="Which benchmark to annotate",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=100.0,
        help="Budget cap in USD (default: 100)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=30,
        help="Number of concurrent workers (default: 30)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually annotate, just show what would be done",
    )
    
    args = parser.parse_args()
    
    run_annotations(
        benchmark=args.benchmark,
        budget=args.budget,
        workers=args.workers,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
