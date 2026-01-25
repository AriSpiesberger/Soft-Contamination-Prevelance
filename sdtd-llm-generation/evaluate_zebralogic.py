#!/usr/bin/env python3
"""Evaluate Claude Sonnet 4.5 on ZebraLogic puzzles from parquet files.

This script:
1. Reads puzzles from a parquet file (original or transformed)
2. Queries Claude Sonnet 4.5 for solutions
3. Compares solutions against ground truth
4. Stores results (reasoning + solution) in a parquet file
5. Supports parallel processing and re-running failed cases
"""

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from sdtd.utils import get_client

# Configuration
DEFAULT_MODEL = "claude-4.5-sonnet"
DEFAULT_WORKERS = 8
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.0

# Global lock for thread-safe file operations
FILE_LOCK = threading.Lock()

logger = logging.getLogger(__name__)


def setup_logging(output_file: Path) -> None:
    """Set up logging to both file and console."""
    log_file = output_file.with_suffix(".log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("=" * 60)
    logging.info(f"Evaluation started at {datetime.now().isoformat()}")
    logging.info(f"Output file: {output_file}")
    logging.info("=" * 60)


def normalize_solution(solution: dict) -> dict:
    """Normalize a solution dictionary for comparison.

    Converts the solution to a canonical form where keys are lowercase
    and values are sorted for consistent comparison.
    """
    if not solution or not isinstance(solution, dict):
        return {}

    # Handle both table format (header + rows) and House N format
    if "header" in solution and "rows" in solution:
        # Table format: normalize to House N format
        header = solution["header"]
        rows = solution["rows"]

        normalized = {}
        for row in rows:
            if len(row) != len(header):
                continue
            house_num = row[0]  # First column is house number
            house_key = f"House {house_num}"
            house_data = {}
            for i, col_name in enumerate(header[1:], 1):  # Skip first column (house number)
                house_data[col_name.lower()] = str(row[i]).lower().strip()
            normalized[house_key.lower()] = house_data
        return normalized
    else:
        # House N format: just normalize keys and values
        normalized = {}
        for house_key, house_data in solution.items():
            if isinstance(house_data, dict):
                normalized_house = {
                    k.lower(): str(v).lower().strip()
                    for k, v in house_data.items()
                }
                normalized[house_key.lower()] = normalized_house
        return normalized


def compare_solutions(pred_solution: dict, ground_truth: dict) -> dict:
    """Compare predicted solution with ground truth.

    Returns:
        dict with keys:
            - is_correct: bool, True if solutions match exactly
            - cell_accuracy: float, percentage of correct cells
            - total_cells: int, total number of cells
            - correct_cells: int, number of correct cells
            - errors: list of error descriptions
    """
    pred_norm = normalize_solution(pred_solution)
    truth_norm = normalize_solution(ground_truth)

    if not pred_norm:
        return {
            "is_correct": False,
            "cell_accuracy": 0.0,
            "total_cells": 0,
            "correct_cells": 0,
            "errors": ["Failed to parse solution"],
        }

    # Compare cell by cell
    total_cells = 0
    correct_cells = 0
    errors = []

    # Check all houses in ground truth
    for house_key, truth_house in truth_norm.items():
        if house_key not in pred_norm:
            errors.append(f"Missing {house_key}")
            total_cells += len(truth_house)
            continue

        pred_house = pred_norm[house_key]

        # Check all attributes in this house
        for attr_key, truth_value in truth_house.items():
            total_cells += 1
            pred_value = pred_house.get(attr_key)

            if pred_value == truth_value:
                correct_cells += 1
            else:
                errors.append(
                    f"{house_key}.{attr_key}: expected '{truth_value}', got '{pred_value}'"
                )

    cell_accuracy = (correct_cells / total_cells * 100) if total_cells > 0 else 0.0
    is_correct = correct_cells == total_cells and total_cells > 0

    return {
        "is_correct": is_correct,
        "cell_accuracy": cell_accuracy,
        "total_cells": total_cells,
        "correct_cells": correct_cells,
        "errors": errors[:10],  # Limit to first 10 errors
    }


def create_json_template(solution: dict) -> str:
    """Create a JSON template with placeholders based on the solution structure.

    Args:
        solution: Ground truth solution (either table or house format)

    Returns:
        JSON string with "___" placeholders
    """
    if not solution or not isinstance(solution, dict):
        raise ValueError(f"Invalid solution: expected dict, got {type(solution)}")

    # Parse solution format
    if "header" in solution and "rows" in solution:
        # Table format
        num_houses = len(solution["rows"])
        columns = solution["header"]
        if not columns or columns[0] != "House":
            raise ValueError(f"Invalid table format: first column should be 'House', got {columns[0] if columns else 'empty'}")
    else:
        # House format - convert to understand structure
        if not solution:
            raise ValueError("Solution dict is empty")
        num_houses = len(solution)
        # Get columns from first house
        first_house = list(solution.values())[0]
        if not isinstance(first_house, dict):
            raise ValueError(f"Invalid house format: expected dict for house data, got {type(first_house)}")
        columns = ["House"] + list(first_house.keys())

    # Create template
    json_template = {"reasoning": "___", "solution": {}}
    for i in range(num_houses):
        json_template["solution"][f"House {i+1}"] = {
            col: "___" for col in columns[1:]  # Skip "House" column
        }

    return json.dumps(json_template, indent=4)


def query_claude(
    puzzle: str,
    ground_truth: dict,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[str, dict | None, str]:
    """Query Claude with a ZebraLogic puzzle using the ZeroEval prompt template.

    Returns:
        tuple of (full_response, parsed_solution, reasoning)
    """
    client = get_client()

    # Create JSON template based on ground truth structure
    json_template = create_json_template(ground_truth)

    # Use the exact ZeroEval template structure
    user_prompt = f"""# Example Puzzle

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
    "solution": {{
        "House 1": {{
            "Name": "Arnold",
            "Drink": "tea"
        }},
        "House 2": {{
            "Name": "Peter",
            "Drink": "water"
        }},
        "House 3": {{
            "Name": "Eric",
            "Drink": "milk"
        }}
    }}
}}

# Puzzle to Solve

{puzzle}


# Instruction

Now please solve the above puzzle.

IMPORTANT: Output ONLY the JSON in the exact format below. Do not include any text before or after the JSON. All your reasoning must be inside the "reasoning" field of the JSON.

{json_template}
"""

    messages = [
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        full_response = response.choices[0].message.content.strip()

        # Try to parse JSON from response
        parsed = None

        # Strategy 1: Try parsing entire response as JSON (best case)
        try:
            parsed = json.loads(full_response)
            if "reasoning" in parsed and "solution" in parsed:
                solution = parsed.get("solution", {})
                reasoning = parsed.get("reasoning", "")
                return full_response, solution, reasoning
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: Try to extract from markdown code blocks
        if "```json" in full_response:
            start = full_response.find("```json") + 7
            end = full_response.find("```", start)
            json_str = full_response[start:end].strip()
            try:
                parsed = json.loads(json_str)
                solution = parsed.get("solution", {})
                reasoning = parsed.get("reasoning", "")
                return full_response, solution, reasoning
            except (json.JSONDecodeError, ValueError):
                pass
        elif "```" in full_response:
            start = full_response.find("```") + 3
            end = full_response.find("```", start)
            json_str = full_response[start:end].strip()
            try:
                parsed = json.loads(json_str)
                solution = parsed.get("solution", {})
                reasoning = parsed.get("reasoning", "")
                return full_response, solution, reasoning
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 3: Find JSON object in text
        brace_start = full_response.find('{')
        if brace_start != -1:
            potential_json = full_response[brace_start:].strip()
            try:
                parsed = json.loads(potential_json)
                if "reasoning" in parsed and "solution" in parsed:
                    solution = parsed.get("solution", {})
                    reasoning = parsed.get("reasoning", "")
                    return full_response, solution, reasoning
            except (json.JSONDecodeError, ValueError):
                pass

        # Failed to parse
        logger.warning(f"Failed to parse JSON from response")
        logger.debug(f"Response (first 500 chars): {full_response[:500]}...")
        return full_response, None, ""

    except Exception as e:
        logger.error(f"Error querying Claude: {e}")
        return f"ERROR: {e}", None, ""


def get_existing_results(output_path: Path) -> dict[str, dict]:
    """Load existing results from output file.

    Returns:
        dict mapping puzzle_id to result row
    """
    if not output_path.exists():
        return {}

    try:
        with FILE_LOCK:
            df = pl.read_parquet(output_path)

        existing = {}
        for row in df.iter_rows(named=True):
            puzzle_id = row.get("puzzle_id")
            if puzzle_id:
                existing[puzzle_id] = row

        return existing
    except Exception as e:
        logger.warning(f"Failed to read existing results from {output_path}: {e}")
        return {}


def append_result_to_parquet(output_path: Path, result: dict) -> None:
    """Thread-safe append of a single result to parquet file."""
    with FILE_LOCK:
        try:
            df = pl.DataFrame([result])

            if output_path.exists():
                existing_df = pl.read_parquet(output_path)
                combined_df = pl.concat([existing_df, df])
                combined_df.write_parquet(output_path, compression="uncompressed")
            else:
                df.write_parquet(output_path, compression="uncompressed")
        except Exception as e:
            logger.error(f"Failed to append result to {output_path}: {e}")


def process_puzzle(
    row: dict,
    model: str,
    max_tokens: int,
    temperature: float,
    output_path: Path,
    retry_failed: bool = False,
) -> dict | None:
    """Process a single puzzle.

    Returns:
        Result dictionary or None if skipped
    """
    # Extract data from row - handle both original and SD formats
    puzzle_id = None

    # Try to get ID from different locations
    if "id" in row:
        puzzle_id = row.get("id")
    else:
        # For SD parquet files, id is in additional_info
        additional_info = row.get("additional_info")
        if additional_info:
            try:
                info = json.loads(additional_info) if isinstance(additional_info, str) else additional_info
                puzzle_id = info.get("id")
            except Exception:
                pass

    if not puzzle_id:
        logger.warning(f"Skipping row: no puzzle ID found")
        return None

    # Get puzzle text - handle both original and SD formats
    if "sd_text" in row:
        puzzle = row.get("sd_text", "")
    else:
        puzzle = row.get("puzzle", "")

    # Get ground truth solution
    ground_truth = None
    additional_info = row.get("additional_info")
    if additional_info:
        try:
            info = json.loads(additional_info) if isinstance(additional_info, str) else additional_info
            # Try sd_solution first (for transformed puzzles), then original_solution
            ground_truth = info.get("sd_solution") or info.get("original_solution") or info.get("solution")
        except Exception as e:
            logger.warning(f"Error parsing additional_info for {puzzle_id}: {e}")

    if not ground_truth:
        ground_truth = row.get("solution")

    if not puzzle:
        logger.warning(f"Skipping puzzle {puzzle_id}: missing puzzle text")
        return None

    if not ground_truth or not isinstance(ground_truth, dict):
        logger.warning(f"Skipping puzzle {puzzle_id}: missing or invalid ground truth (type: {type(ground_truth)})")
        logger.debug(f"Row keys: {list(row.keys())}")
        if additional_info:
            try:
                info = json.loads(additional_info) if isinstance(additional_info, str) else additional_info
                logger.debug(f"additional_info keys: {list(info.keys())}")
            except Exception:
                pass
        return None

    # Query Claude
    try:
        full_response, parsed_solution, reasoning = query_claude(
            puzzle, ground_truth, model, max_tokens, temperature
        )
    except Exception as e:
        import traceback
        logger.error(f"Error processing puzzle {puzzle_id}: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return None

    # Compare solution
    comparison = compare_solutions(parsed_solution or {}, ground_truth)

    # Prepare result
    result = {
        "puzzle_id": puzzle_id,
        "puzzle": puzzle,
        "ground_truth_solution": json.dumps(ground_truth),
        "model_response": full_response,
        "model_reasoning": reasoning,
        "model_solution": json.dumps(parsed_solution) if parsed_solution else None,
        "is_correct": comparison["is_correct"],
        "cell_accuracy": comparison["cell_accuracy"],
        "total_cells": comparison["total_cells"],
        "correct_cells": comparison["correct_cells"],
        "errors": json.dumps(comparison["errors"]),
        "model": model,
        "timestamp": datetime.now().isoformat(),
    }

    # Append to output file
    append_result_to_parquet(output_path, result)

    return result


def evaluate_parquet(
    input_file: Path,
    output_file: Path,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    workers: int = DEFAULT_WORKERS,
    start_index: int | None = None,
    end_index: int | None = None,
    retry_failed: bool = False,
) -> None:
    """Evaluate Claude on puzzles from a parquet file.

    Args:
        input_file: Path to input parquet file (must have 'index' column)
        output_file: Path to output parquet file
        model: Model to use for evaluation
        max_tokens: Max tokens for model response
        temperature: Sampling temperature
        workers: Number of parallel workers
        start_index: Optional start index (inclusive)
        end_index: Optional end index (exclusive)
        retry_failed: If True, only retry puzzles that were incorrect
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    setup_logging(output_file)

    # Load input data
    logger.info(f"Loading puzzles from {input_file}")
    df = pl.read_parquet(input_file)

    # Validate index column exists
    if "index" not in df.columns:
        raise ValueError(
            f"Input file {input_file} must have an 'index' column. "
            "Use merge_zebralogic_shards.py to create indexed files."
        )

    # Filter by index range
    if start_index is not None or end_index is not None:
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = df["index"].max() + 1

        logger.info(f"Filtering to index range [{start_index}, {end_index})")
        df = df.filter((pl.col("index") >= start_index) & (pl.col("index") < end_index))

    logger.info(f"Loaded {len(df)} puzzles")

    # Get existing results
    existing_results = get_existing_results(output_file)
    logger.info(f"Found {len(existing_results)} existing results")

    # Filter puzzles to process
    puzzles_to_process = []
    for row_dict in df.iter_rows(named=True):
        # Extract puzzle ID - handle both original and SD formats
        puzzle_id = None
        if "id" in row_dict:
            puzzle_id = row_dict.get("id")
        else:
            # For SD parquet files, id is in additional_info
            additional_info = row_dict.get("additional_info")
            if additional_info:
                try:
                    info = json.loads(additional_info) if isinstance(additional_info, str) else additional_info
                    puzzle_id = info.get("id")
                except Exception:
                    pass

        if not puzzle_id:
            logger.warning("Skipping row without ID")
            continue

        # Skip if already processed and not retrying
        if puzzle_id in existing_results:
            if not retry_failed:
                continue
            # If retrying, only reprocess incorrect ones
            if existing_results[puzzle_id].get("is_correct"):
                continue

        puzzles_to_process.append(row_dict)

    logger.info(f"Processing {len(puzzles_to_process)} puzzles with {workers} workers")
    logger.info(f"Model: {model}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Max tokens: {max_tokens}")

    # Process puzzles in parallel
    completed = 0
    correct = 0
    total_cell_accuracy = 0.0

    def extract_puzzle_id(puzzle_dict):
        """Extract puzzle ID from either original or SD format."""
        if "id" in puzzle_dict:
            return puzzle_dict.get("id")
        additional_info = puzzle_dict.get("additional_info")
        if additional_info:
            try:
                info = json.loads(additional_info) if isinstance(additional_info, str) else additional_info
                return info.get("id", "unknown")
            except Exception:
                pass
        return "unknown"

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_puzzle = {
            executor.submit(
                process_puzzle,
                puzzle,
                model,
                max_tokens,
                temperature,
                output_file,
                retry_failed,
            ): extract_puzzle_id(puzzle)
            for puzzle in puzzles_to_process
        }

        # Process results as they complete
        for future in as_completed(future_to_puzzle):
            puzzle_id = future_to_puzzle[future]
            completed += 1

            try:
                result = future.result()
                if result:
                    if result["is_correct"]:
                        correct += 1
                    total_cell_accuracy += result["cell_accuracy"]

                    # Log progress
                    if completed % 10 == 0 or completed == len(puzzles_to_process):
                        avg_cell_acc = total_cell_accuracy / completed
                        puzzle_acc = correct / completed * 100
                        logger.info(
                            f"Progress: {completed}/{len(puzzles_to_process)} | "
                            f"Puzzle Acc: {puzzle_acc:.1f}% | "
                            f"Cell Acc: {avg_cell_acc:.1f}%"
                        )
            except Exception as e:
                logger.error(f"Error processing puzzle {puzzle_id}: {e}")

    # Final statistics
    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    logger.info(f"Total puzzles processed: {completed}")
    if completed > 0:
        logger.info(f"Correct puzzles: {correct} ({correct/completed*100:.1f}%)")
        logger.info(f"Average cell accuracy: {total_cell_accuracy/completed:.1f}%")
    else:
        logger.warning("No puzzles were successfully processed!")
    logger.info(f"Results saved to: {output_file}")
    logger.info("=" * 60)

    # Compress final output
    if output_file.exists():
        logger.info("Compressing output file...")
        with FILE_LOCK:
            try:
                final_df = pl.read_parquet(output_file)
                final_df.write_parquet(output_file, compression="zstd", compression_level=3)
                logger.info(f"Saved {len(final_df)} results to {output_file}")
            except Exception as e:
                logger.error(f"Failed to compress output: {e}")
    else:
        logger.warning(f"No output file created - all puzzles failed to process")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Claude Sonnet 4.5 on ZebraLogic puzzles"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input parquet file with puzzles",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output parquet file for results",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens for response (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--start",
        type=int,
        help="Start index (inclusive, default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        help="End index (exclusive, default: all)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only re-run puzzles that were incorrect",
    )

    args = parser.parse_args()

    evaluate_parquet(
        input_file=args.input,
        output_file=args.output,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        workers=args.workers,
        start_index=args.start,
        end_index=args.end,
        retry_failed=args.retry_failed,
    )


if __name__ == "__main__":
    main()
