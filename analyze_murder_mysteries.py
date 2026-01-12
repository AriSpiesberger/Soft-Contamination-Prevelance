#!/usr/bin/env python3
"""
Murder Mystery Semantic Duplicate Analysis

This script analyzes potential semantic duplicates between MuSR murder mystery
samples and training data samples using a hybrid approach:

Workflow 1 (Direct): Ask Claude to directly compare test_text and corpus_text
Workflow 2 (Structural): Extract MMO structures, then compare structures

The hybrid approach:
1. Always run Workflow 1 first
2. If Workflow 1 returns is_sd=1 AND confidence < 0.7, run Workflow 2 for verification
3. Use Workflow 2 result as final verdict for uncertain cases

Output:
- Individual JSON files for each comparison
- Consolidated JSON with all results
- Consolidated CSV summary
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    # Try loading from multiple locations
    load_dotenv()
    load_dotenv(Path(__file__).parent.parent / ".env")
    load_dotenv(Path(__file__).parent.parent / ".env2")
except ImportError:
    pass

try:
    from anthropic import Anthropic
except ImportError:
    print("Please install anthropic: pip install anthropic")
    sys.exit(1)

from prompts import PROMPT_MM_DIRECT, PROMPT_MM_EXTRACT, PROMPT_MM_COMPARE

# =============================================================================
# Configuration
# =============================================================================

MODEL_OPUS = "claude-opus-4-5"
MODEL_SONNET = "claude-sonnet-4-5"

MAX_RETRIES = 3
RETRY_DELAY = 2.0
DEFAULT_CONCURRENCY = 5  # Conservative to avoid rate limits
WORKFLOW2_CONFIDENCE_THRESHOLD = 0.7  # Run Workflow 2 if confidence below this

# Output directory structure
OUTPUT_DIR = Path(__file__).parent / "analysis"

# Thread-safe locks
_file_lock = threading.Lock()
_print_lock = threading.Lock()
_stats_lock = threading.Lock()


# =============================================================================
# Utility Functions
# =============================================================================

def extract_dataset_name(csv_path: Path) -> str:
    """
    Extract clean dataset name from CSV filename.
    
    Examples:
    - "murder_mysteries-contamination_dolci_100pct-top_1000_contamination.csv.csv" -> "dolci_100pct"
    - "murder_mysteries-contamination_dolci_dpo_100pct-top_1000_contamination.csv.csv" -> "dolci_dpo_100pct"
    - "murder_mysteries-contamination_dolma3_dolmino_mix_1pct-top_1000_contamination.csv.csv" -> "dolma3_dolmino_mix_1pct"
    """
    filename = csv_path.stem.replace(".csv", "")  # Remove .csv extensions
    
    # Pattern: {benchmark}-contamination_{dataset}-top_1000_contamination
    parts = filename.split("-")
    if len(parts) >= 2:
        # Get the middle part and remove "contamination_" prefix
        middle = parts[1]
        if middle.startswith("contamination_"):
            return middle[len("contamination_"):]
        return middle
    return filename


def thread_print(*args, **kwargs):
    """Thread-safe print."""
    with _print_lock:
        print(*args, **kwargs)


def clean_json_response(text: str) -> str:
    """Clean markdown artifacts from JSON response."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def safe_json_parse(text: str) -> Optional[dict]:
    """Safely parse JSON, returning None on failure."""
    try:
        cleaned = clean_json_response(text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def truncate_text(text: str, max_chars: int = 50000) -> str:
    """Truncate text to avoid token limits."""
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[... TRUNCATED ...]"
    return text


# =============================================================================
# API Calls
# =============================================================================

def call_claude(
    client: Anthropic,
    prompt: str,
    model: str = MODEL_SONNET,
    max_tokens: int = 4096,
    timeout: float = 180.0
) -> Optional[str]:
    """
    Call Claude API with retry logic.
    Returns the response text or None on failure.
    """
    last_error = None
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                timeout=timeout,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # Rate limit - wait longer
            if "rate" in error_str or "429" in error_str:
                wait_time = RETRY_DELAY * (2 ** attempt)
                thread_print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            # Overloaded - wait and retry
            elif "overloaded" in error_str or "529" in error_str:
                wait_time = RETRY_DELAY * attempt
                thread_print(f"  API overloaded, waiting {wait_time}s...")
                time.sleep(wait_time)
            # Other error - shorter wait
            else:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
    
    thread_print(f"  API call failed after {MAX_RETRIES} attempts: {last_error}")
    return None


# =============================================================================
# Workflow 1: Direct Comparison
# =============================================================================

def workflow1_direct_comparison(
    client: Anthropic,
    test_text: str,
    corpus_text: str,
    verbose: bool = False
) -> dict:
    """
    Workflow 1: Direct comparison of test and corpus text.
    Returns dict with is_semantic_duplicate, confidence, reasoning, etc.
    """
    prompt = PROMPT_MM_DIRECT.format(
        test_text=truncate_text(test_text),
        corpus_text=truncate_text(corpus_text)
    )
    
    response = call_claude(client, prompt)
    
    if response is None:
        return {
            "workflow": 1,
            "success": False,
            "error": "API call failed",
            "is_semantic_duplicate": 0,
            "confidence": 0.0,
            "reasoning": "Failed to get API response"
        }
    
    parsed = safe_json_parse(response)
    
    if parsed is None:
        return {
            "workflow": 1,
            "success": False,
            "error": "JSON parse failed",
            "raw_response": response[:500],
            "is_semantic_duplicate": 0,
            "confidence": 0.0,
            "reasoning": "Failed to parse JSON response"
        }
    
    # Normalize the response
    result = {
        "workflow": 1,
        "success": True,
        "corpus_is_murder_mystery": parsed.get("corpus_is_murder_mystery", False),
        "corpus_type_if_not_mm": parsed.get("corpus_type_if_not_mm"),
        "is_semantic_duplicate": int(parsed.get("is_semantic_duplicate", 0)),
        "confidence": float(parsed.get("confidence", 0.0)),
        "reasoning": parsed.get("reasoning", "")
    }
    
    return result


# =============================================================================
# Workflow 2: Structural Extraction + Comparison
# =============================================================================

def workflow2_extract_structure(
    client: Anthropic,
    text: str,
    verbose: bool = False
) -> Optional[dict]:
    """
    Extract the logical structure from a murder mystery text.
    Returns the structure dict or None on failure.
    """
    prompt = PROMPT_MM_EXTRACT.format(text=truncate_text(text))
    
    response = call_claude(client, prompt)
    
    if response is None:
        return None
    
    parsed = safe_json_parse(response)
    return parsed


def workflow2_compare_structures(
    client: Anthropic,
    structure_a: dict,
    structure_b: dict,
    verbose: bool = False
) -> dict:
    """
    Compare two extracted structures for semantic duplication.
    """
    prompt = PROMPT_MM_COMPARE.format(
        structure_a=json.dumps(structure_a, indent=2),
        structure_b=json.dumps(structure_b, indent=2)
    )
    
    response = call_claude(client, prompt)
    
    if response is None:
        return {
            "workflow": 2,
            "success": False,
            "error": "API call failed",
            "is_semantic_duplicate": 0,
            "confidence": 0.0
        }
    
    parsed = safe_json_parse(response)
    
    if parsed is None:
        return {
            "workflow": 2,
            "success": False,
            "error": "JSON parse failed",
            "is_semantic_duplicate": 0,
            "confidence": 0.0
        }
    
    result = {
        "workflow": 2,
        "success": True,
        "both_valid_mysteries": parsed.get("both_valid_mysteries", False),
        "comparison_results": parsed.get("comparison_results", {}),
        "is_semantic_duplicate": int(parsed.get("is_semantic_duplicate", 0)),
        "confidence": float(parsed.get("confidence", 0.0)),
        "reasoning": parsed.get("reasoning", ""),
        "key_differences": parsed.get("key_differences", []),
        "key_similarities": parsed.get("key_similarities", [])
    }
    
    return result


def workflow2_full(
    client: Anthropic,
    test_text: str,
    corpus_text: str,
    verbose: bool = False
) -> dict:
    """
    Full Workflow 2: Extract structures from both texts, then compare.
    """
    # Extract structure from test text
    test_structure = workflow2_extract_structure(client, test_text, verbose)
    if test_structure is None:
        return {
            "workflow": 2,
            "success": False,
            "error": "Failed to extract test structure",
            "is_semantic_duplicate": 0,
            "confidence": 0.0
        }
    
    # Extract structure from corpus text
    corpus_structure = workflow2_extract_structure(client, corpus_text, verbose)
    if corpus_structure is None:
        return {
            "workflow": 2,
            "success": False,
            "error": "Failed to extract corpus structure",
            "test_structure": test_structure,
            "is_semantic_duplicate": 0,
            "confidence": 0.0
        }
    
    # Compare structures
    comparison = workflow2_compare_structures(client, test_structure, corpus_structure, verbose)
    
    # Add the structures to the result
    comparison["test_structure"] = test_structure
    comparison["corpus_structure"] = corpus_structure
    
    return comparison


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def analyze_single_row(
    client: Anthropic,
    row: dict,
    output_dir: Path,
    force: bool = False,
    verbose: bool = False
) -> dict:
    """
    Analyze a single row using the hybrid approach.
    
    1. Run Workflow 1 (direct comparison)
    2. If is_sd=1 AND confidence < threshold, run Workflow 2
    3. Return combined result
    """
    test_id = row["test_id"]
    corpus_id = row["corpus_id"]
    rank = row["rank"]
    score = row["score"]
    test_text = row["test_text"]
    corpus_text = row["corpus_text"]
    
    # Create unique filename
    safe_corpus_id = corpus_id[:50] if len(corpus_id) > 50 else corpus_id
    filename = f"{test_id}_rank{rank}_{safe_corpus_id}.json"
    output_path = output_dir / "individual" / filename
    
    # Check if already processed
    if not force and output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("final_verdict") is not None:
                return existing
        except (json.JSONDecodeError, KeyError):
            pass  # Re-process if file is corrupted
    
    # Initialize result
    result = {
        "test_id": test_id,
        "corpus_id": corpus_id,
        "rank": int(rank),
        "score": float(score),
        "test_text": test_text[:1000] + "..." if len(test_text) > 1000 else test_text,
        "corpus_text": corpus_text[:1000] + "..." if len(corpus_text) > 1000 else corpus_text,
        "timestamp": datetime.now().isoformat()
    }
    
    # Workflow 1: Direct comparison
    w1_result = workflow1_direct_comparison(client, test_text, corpus_text, verbose)
    result["workflow1"] = w1_result
    
    # Determine if we need Workflow 2
    needs_workflow2 = (
        w1_result["is_semantic_duplicate"] == 1 and
        w1_result["confidence"] < WORKFLOW2_CONFIDENCE_THRESHOLD
    )
    
    if needs_workflow2:
        if verbose:
            thread_print(f"  Running Workflow 2 (W1: is_sd=1, conf={w1_result['confidence']:.2f})")
        
        w2_result = workflow2_full(client, test_text, corpus_text, verbose)
        result["workflow2"] = w2_result
        result["workflow2_triggered"] = True
        
        # Use Workflow 2 as final verdict if successful
        if w2_result["success"]:
            result["final_verdict"] = {
                "is_semantic_duplicate": w2_result["is_semantic_duplicate"],
                "confidence": w2_result["confidence"],
                "source": "workflow2"
            }
        else:
            # Fall back to Workflow 1
            result["final_verdict"] = {
                "is_semantic_duplicate": w1_result["is_semantic_duplicate"],
                "confidence": w1_result["confidence"],
                "source": "workflow1_fallback"
            }
    else:
        result["workflow2_triggered"] = False
        result["final_verdict"] = {
            "is_semantic_duplicate": w1_result["is_semantic_duplicate"],
            "confidence": w1_result["confidence"],
            "source": "workflow1"
        }
    
    # Save individual result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    return result


def process_csv_file(
    client: Anthropic,
    csv_path: Path,
    output_dir: Path,
    concurrency: int = DEFAULT_CONCURRENCY,
    limit: Optional[int] = None,
    force: bool = False,
    verbose: bool = False
) -> list[dict]:
    """
    Process a CSV file containing murder mystery comparisons.
    """
    print(f"\nProcessing: {csv_path.name}")
    print("=" * 70)
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")
    
    if limit:
        df = df.head(limit)
        print(f"Limited to: {limit} rows")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "individual").mkdir(exist_ok=True)
    
    # Track statistics
    stats = {
        "total": len(df),
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "semantic_duplicates": 0,
        "workflow2_triggered": 0
    }
    
    results = []
    
    # Process rows with progress bar
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        
        for idx, row in df.iterrows():
            future = executor.submit(
                analyze_single_row,
                client,
                row.to_dict(),
                output_dir,
                force,
                verbose
            )
            futures[future] = idx
        
        with tqdm(total=len(futures), desc="Analyzing", unit="row") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    with _stats_lock:
                        stats["processed"] += 1
                        if result["final_verdict"]["is_semantic_duplicate"] == 1:
                            stats["semantic_duplicates"] += 1
                        if result.get("workflow2_triggered"):
                            stats["workflow2_triggered"] += 1
                    
                except Exception as e:
                    thread_print(f"\nError processing row {idx}: {e}")
                    with _stats_lock:
                        stats["errors"] += 1
                
                pbar.update(1)
    
    # Print statistics
    print(f"\nResults:")
    print(f"  Processed: {stats['processed']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Semantic duplicates found: {stats['semantic_duplicates']}")
    print(f"  Workflow 2 triggered: {stats['workflow2_triggered']}")
    
    return results, stats


def save_consolidated_results(results: list[dict], output_dir: Path, stats: dict):
    """
    Save consolidated JSON and CSV results.
    """
    # Consolidated JSON
    consolidated = {
        "generated_at": datetime.now().isoformat(),
        "stats": stats,
        "results": results
    }
    
    json_path = output_dir / "consolidated_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2, ensure_ascii=False)
    print(f"\nSaved consolidated JSON: {json_path}")
    
    # Consolidated CSV (summary)
    csv_path = output_dir / "consolidated_results.csv"
    csv_rows = []
    
    for r in results:
        csv_rows.append({
            "test_id": r["test_id"],
            "corpus_id": r["corpus_id"],
            "rank": r["rank"],
            "score": r["score"],
            "is_semantic_duplicate": r["final_verdict"]["is_semantic_duplicate"],
            "confidence": r["final_verdict"]["confidence"],
            "verdict_source": r["final_verdict"]["source"],
            "workflow2_triggered": r.get("workflow2_triggered", False),
            "corpus_is_mm": r.get("workflow1", {}).get("corpus_is_murder_mystery", None),
            "reasoning": r.get("workflow1", {}).get("reasoning", "")[:200]
        })
    
    df_out = pd.DataFrame(csv_rows)
    df_out.to_csv(csv_path, index=False)
    print(f"Saved consolidated CSV: {csv_path}")
    
    # Also save a summary of just the duplicates
    duplicates_df = df_out[df_out["is_semantic_duplicate"] == 1]
    if len(duplicates_df) > 0:
        duplicates_path = output_dir / "semantic_duplicates_only.csv"
        duplicates_df.to_csv(duplicates_path, index=False)
        print(f"Saved duplicates CSV: {duplicates_path}")


def generate_text_report(
    results_by_dataset: dict[str, list[dict]],
    stats_by_dataset: dict[str, dict],
    output_path: Path,
    limit_used: Optional[int] = None
) -> str:
    """
    Generate a human-readable text report of the analysis results.
    
    Args:
        results_by_dataset: Dict mapping dataset name to list of results
        stats_by_dataset: Dict mapping dataset name to stats dict
        output_path: Path to save the report
        limit_used: If a limit was applied, include it in the report
    
    Returns:
        The report text
    """
    lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Header
    lines.append("=" * 80)
    lines.append("MURDER MYSTERY SEMANTIC DUPLICATE ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {timestamp}")
    lines.append(f"Model: {MODEL_SONNET}")
    lines.append(f"Workflow 2 confidence threshold: {WORKFLOW2_CONFIDENCE_THRESHOLD}")
    if limit_used:
        lines.append(f"Row limit per dataset: {limit_used}")
    lines.append("")
    
    # Overall summary
    total_processed = sum(s.get("processed", 0) for s in stats_by_dataset.values())
    total_errors = sum(s.get("errors", 0) for s in stats_by_dataset.values())
    total_duplicates = sum(s.get("semantic_duplicates", 0) for s in stats_by_dataset.values())
    total_w2 = sum(s.get("workflow2_triggered", 0) for s in stats_by_dataset.values())
    
    lines.append("-" * 80)
    lines.append("OVERALL SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Total comparisons:        {total_processed}")
    lines.append(f"Total errors:             {total_errors}")
    lines.append(f"Semantic duplicates:      {total_duplicates}")
    lines.append(f"Duplicate rate:           {total_duplicates/max(total_processed,1)*100:.2f}%")
    lines.append(f"Workflow 2 triggered:     {total_w2} ({total_w2/max(total_processed,1)*100:.2f}%)")
    lines.append("")
    
    # Per-dataset breakdown
    lines.append("-" * 80)
    lines.append("BREAKDOWN BY DATASET")
    lines.append("-" * 80)
    lines.append("")
    
    for dataset_name in sorted(results_by_dataset.keys()):
        results = results_by_dataset[dataset_name]
        stats = stats_by_dataset.get(dataset_name, {})
        
        processed = stats.get("processed", len(results))
        errors = stats.get("errors", 0)
        duplicates = stats.get("semantic_duplicates", 0)
        w2_triggered = stats.get("workflow2_triggered", 0)
        
        # Count corpus types
        corpus_mm_count = sum(1 for r in results if r.get("workflow1", {}).get("corpus_is_murder_mystery", False))
        corpus_not_mm_count = processed - corpus_mm_count
        
        # Confidence distribution
        confidences = [r["final_verdict"]["confidence"] for r in results if "final_verdict" in r]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        
        lines.append(f"Dataset: {dataset_name}")
        lines.append(f"  Processed:              {processed}")
        lines.append(f"  Errors:                 {errors}")
        lines.append(f"  Semantic duplicates:    {duplicates} ({duplicates/max(processed,1)*100:.2f}%)")
        lines.append(f"  Corpus is murder mystery: {corpus_mm_count}")
        lines.append(f"  Corpus is NOT murder mystery: {corpus_not_mm_count}")
        lines.append(f"  Workflow 2 triggered:   {w2_triggered}")
        lines.append(f"  Average confidence:     {avg_conf:.3f}")
        lines.append("")
    
    # List all semantic duplicates found
    all_duplicates = []
    for dataset_name, results in results_by_dataset.items():
        for r in results:
            if r.get("final_verdict", {}).get("is_semantic_duplicate") == 1:
                all_duplicates.append({
                    "dataset": dataset_name,
                    "test_id": r["test_id"],
                    "corpus_id": r["corpus_id"][:30] + "..." if len(r.get("corpus_id", "")) > 30 else r.get("corpus_id", ""),
                    "rank": r["rank"],
                    "score": r["score"],
                    "confidence": r["final_verdict"]["confidence"],
                    "source": r["final_verdict"]["source"],
                    "reasoning": r.get("workflow1", {}).get("reasoning", "")[:100]
                })
    
    if all_duplicates:
        lines.append("-" * 80)
        lines.append(f"SEMANTIC DUPLICATES FOUND ({len(all_duplicates)} total)")
        lines.append("-" * 80)
        lines.append("")
        
        for i, dup in enumerate(all_duplicates, 1):
            lines.append(f"{i}. [{dup['dataset']}] {dup['test_id']}")
            lines.append(f"   Corpus: {dup['corpus_id']}")
            lines.append(f"   Rank: {dup['rank']}, Score: {dup['score']:.4f}, Confidence: {dup['confidence']:.2f}")
            lines.append(f"   Source: {dup['source']}")
            lines.append(f"   Reasoning: {dup['reasoning']}...")
            lines.append("")
    else:
        lines.append("-" * 80)
        lines.append("NO SEMANTIC DUPLICATES FOUND")
        lines.append("-" * 80)
        lines.append("")
    
    # Corpus type breakdown (what kinds of texts are in the training data)
    lines.append("-" * 80)
    lines.append("CORPUS TEXT TYPE BREAKDOWN (non-murder-mystery samples)")
    lines.append("-" * 80)
    lines.append("")
    
    corpus_types = {}
    for dataset_name, results in results_by_dataset.items():
        for r in results:
            w1 = r.get("workflow1", {})
            if not w1.get("corpus_is_murder_mystery", True):
                ctype = w1.get("corpus_type_if_not_mm", "unknown")
                if ctype:
                    # Truncate and normalize
                    ctype_short = ctype[:60] + "..." if len(ctype) > 60 else ctype
                    corpus_types[ctype_short] = corpus_types.get(ctype_short, 0) + 1
    
    if corpus_types:
        # Sort by count
        sorted_types = sorted(corpus_types.items(), key=lambda x: -x[1])[:20]
        for ctype, count in sorted_types:
            lines.append(f"  {count:4d}x  {ctype}")
    else:
        lines.append("  (All corpus texts were murder mysteries)")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    report_text = "\n".join(lines)
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"Saved text report: {output_path}")
    
    return report_text


def regenerate_report_from_existing(output_base_dir: Path, limit_used: Optional[int] = None) -> str:
    """
    Regenerate the text report from existing consolidated_results.json files.
    """
    results_by_dataset = {}
    stats_by_dataset = {}
    
    mm_dir = output_base_dir / "murder_mysteries"
    if not mm_dir.exists():
        print(f"No analysis directory found at {mm_dir}")
        return ""
    
    # Find all dataset directories (excluding _master)
    for dataset_dir in mm_dir.iterdir():
        if dataset_dir.is_dir() and dataset_dir.name != "_master":
            json_path = dataset_dir / "consolidated_results.json"
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results_by_dataset[dataset_dir.name] = data.get("results", [])
                stats_by_dataset[dataset_dir.name] = data.get("stats", {})
    
    if not results_by_dataset:
        print("No results found to generate report from")
        return ""
    
    report_path = mm_dir / "_master" / "analysis_report.txt"
    return generate_text_report(results_by_dataset, stats_by_dataset, report_path, limit_used)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze murder mystery samples for semantic duplication"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input CSV file or 'all' to process all murder mystery files"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: analysis/murder_mysteries/{dataset_name})"
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Number of concurrent workers (default: {DEFAULT_CONCURRENCY})"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of rows to process (for testing)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force reprocess all rows (ignore existing results)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run - don't make API calls, just show what would be processed"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only regenerate the text report from existing results (no API calls)"
    )
    
    args = parser.parse_args()
    
    # Handle --report-only mode
    if args.report_only:
        print("Regenerating report from existing results...")
        report = regenerate_report_from_existing(OUTPUT_DIR, args.limit)
        if report:
            print("\n" + report)
        sys.exit(0)
    
    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)
    
    # Find input files
    data_dir = Path(__file__).parent / "data"
    
    if args.input == "all" or args.input is None:
        # Find all murder mystery CSV files
        csv_files = list(data_dir.glob("murder_mysteries-*.csv*"))
        if not csv_files:
            print(f"No murder mystery CSV files found in {data_dir}")
            sys.exit(1)
        print(f"Found {len(csv_files)} murder mystery files")
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            input_path = data_dir / args.input
        if not input_path.exists():
            print(f"Input file not found: {args.input}")
            sys.exit(1)
        csv_files = [input_path]
    
    # Dry run mode
    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            print(f"\n{csv_file.name}:")
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            if args.limit:
                print(f"  Would process: {min(args.limit, len(df))} rows")
            else:
                print(f"  Would process: {len(df)} rows")
        print("\nDry run complete. Remove --dry-run to actually process.")
        sys.exit(0)
    
    # Initialize client
    client = Anthropic(api_key=api_key)
    
    # Process each file
    print("=" * 70)
    print("Murder Mystery Semantic Duplicate Analysis")
    print("=" * 70)
    print(f"Model: {MODEL_SONNET}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Workflow 2 threshold: confidence < {WORKFLOW2_CONFIDENCE_THRESHOLD}")
    
    all_results = []
    all_stats = {
        "total": 0,
        "processed": 0,
        "errors": 0,
        "semantic_duplicates": 0,
        "workflow2_triggered": 0
    }
    
    # Track results by dataset for the report
    results_by_dataset = {}
    stats_by_dataset = {}
    
    for csv_file in csv_files:
        # Extract dataset name from filename
        dataset_name = extract_dataset_name(csv_file)
        
        # Determine output directory
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = OUTPUT_DIR / "murder_mysteries" / dataset_name
        
        # Process the file
        results, stats = process_csv_file(
            client,
            csv_file,
            output_dir,
            concurrency=args.concurrency,
            limit=args.limit,
            force=args.force,
            verbose=args.verbose
        )
        
        # Save consolidated results for this file
        save_consolidated_results(results, output_dir, stats)
        
        # Track by dataset
        results_by_dataset[dataset_name] = results
        stats_by_dataset[dataset_name] = stats
        
        # Aggregate stats
        all_results.extend(results)
        for key in all_stats:
            all_stats[key] += stats.get(key, 0)
    
    # If processing multiple files, save a master consolidated result
    if len(csv_files) > 1:
        master_dir = OUTPUT_DIR / "murder_mysteries" / "_master"
        master_dir.mkdir(parents=True, exist_ok=True)
        save_consolidated_results(all_results, master_dir, all_stats)
    
    # Generate text report
    report_path = OUTPUT_DIR / "murder_mysteries" / "_master" / "analysis_report.txt"
    report_text = generate_text_report(results_by_dataset, stats_by_dataset, report_path, args.limit)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Total processed: {all_stats['processed']}")
    print(f"Total errors: {all_stats['errors']}")
    print(f"Total semantic duplicates: {all_stats['semantic_duplicates']}")
    print(f"Workflow 2 triggered: {all_stats['workflow2_triggered']} times")


if __name__ == "__main__":
    main()

