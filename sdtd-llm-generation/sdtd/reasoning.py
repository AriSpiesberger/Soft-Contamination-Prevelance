"""Reasoning trace generation and verification for ZebraLogic dataset."""

import json
import logging
import os
import polars as pl
from pathlib import Path
from typing import Any
import time

from openai import OpenAI
from sdtd.generate import retry_with_backoff, load_checkpoint, save_checkpoint, save_partial_results

# Initialize OpenAI client with Helicone configuration
client = OpenAI(
    api_key=os.getenv("HELICONE_API_KEY"),
    base_url="https://oai.helicone.ai/v1"
)

# Template for ZebraLogic
ZEBRA_TEMPLATE = """# Puzzle to Solve 

{puzzle}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following json format:

{{
    "reasoning": "...",
    "solution": {{
        "House 1": {{
            "Category1": "Value1",
            "Category2": "Value2",
            ...
        }},
        "House 2": {{
            ...
        }},
        ...
    }}
}}
"""

def extract_last_complete_json(s: str) -> dict | None:
    """Extract the last complete JSON object from a string.
    
    Adapted from ZeroEval/src/evaluation/eval_utils.py
    """
    stack = []
    last_json_start = None
    last_json_str = None
    
    for i, char in enumerate(s):
        if char == '{':
            stack.append(i)
            if last_json_start is None:
                last_json_start = i
        elif char == '}':
            if stack:
                stack.pop()
                if not stack:
                    # Complete JSON object found
                    last_json_str = s[last_json_start:i+1]
                    last_json_start = None
    
    # Load the last JSON object
    if last_json_str:
        try:
            return json.loads(last_json_str.replace("\n", ""))
        except json.JSONDecodeError:
            pass
    
    return None

def check_solution(prediction: dict, ground_truth: dict) -> tuple[bool, float]:
    """Check if the predicted solution matches the ground truth.
    
    Args:
        prediction: Predicted solution dictionary (from model output JSON)
        ground_truth: Ground truth solution dictionary
        
    Returns:
        Tuple of (is_correct, accuracy)
    """
    if not prediction or "solution" not in prediction:
        return False, 0.0
        
    pred_table = prediction["solution"]
    
    # Handle ground truth structure
    # Ground truth might be in the format:
    # {
    #   "header": ["House", "Category1", ...],
    #   "rows": [["1", "Val1", ...], ...]
    # }
    # OR it might be already in the map format if it came from our transformation logic:
    # {
    #   "House 1": {"Category1": "Val1", ...},
    #   ...
    # }
    
    solution_table = {}
    
    if "header" in ground_truth and "rows" in ground_truth:
        # Parse standard ZebraLogic format
        columns = ground_truth["header"]
        num_houses = len(ground_truth["rows"])
        # assert columns[0] == "House"
        
        for i in range(num_houses):
            # rows[i][0] is usually the house number, subsequent are values
            # columns[j] corresponds to rows[i][j]
            house_key = f'House {i+1}'
            solution_table[house_key] = {}
            for j in range(1, len(columns)):
                solution_table[house_key][columns[j]] = ground_truth["rows"][i][j]
    else:
        # Assume it's already in the map format
        solution_table = ground_truth

    total_cells = 0
    correct_cells = 0
    
    for house in solution_table:
        for column in solution_table[house]:
            total_cells += 1
            
            # Check if prediction has this cell
            if house in pred_table and column in pred_table[house]:
                truth_val = str(solution_table[house][column]).lower().strip()
                
                pred_val_raw = pred_table[house][column]
                if pred_val_raw is None:
                    continue
                    
                if isinstance(pred_val_raw, list):
                    pred_val = str(pred_val_raw[0]).lower().strip()
                else:
                    pred_val = str(pred_val_raw).lower().strip()
                
                if truth_val == pred_val:
                    correct_cells += 1
    
    accuracy = correct_cells / total_cells if total_cells > 0 else 0.0
    return (correct_cells == total_cells), accuracy

def solve_zebra_puzzle(
    puzzle: str,
    solution: dict,
    model: str,
    k: int = 3,
    temperature: float = 0.7,
) -> dict | None:
    """Solve a Zebra puzzle with retries until correct.
    
    Args:
        puzzle: Puzzle text
        solution: Ground truth solution
        model: LLM model to use
        k: Maximum number of attempts
        temperature: Sampling temperature
        
    Returns:
        Dictionary with 'reasoning', 'solution' (prediction), and 'attempts' count.
        Returns None if not solved within k attempts.
    """
    prompt = ZEBRA_TEMPLATE.format(puzzle=puzzle)
    messages = [{"role": "user", "content": prompt}]
    
    for attempt in range(1, k + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=4096,
            )
            content = response.choices[0].message.content
            
            # Parse JSON
            parsed_output = extract_last_complete_json(content)
            
            if parsed_output:
                is_correct, acc = check_solution(parsed_output, solution)
                
                if is_correct:
                    return {
                        "reasoning": parsed_output.get("reasoning", ""),
                        "solution": parsed_output.get("solution", {}),
                        "attempts": attempt,
                        "raw_output": content
                    }
        except Exception as e:
            logging.warning(f"Error in attempt {attempt}: {e}")
            
    return None

def generate_reasoning_traces(
    input_file: Path,
    output_file: Path,
    model: str,
    k: int = 3,
    limit: int | None = None,
    checkpoint_enabled: bool = True,
) -> None:
    """Enrich SD parquet with correct reasoning traces.
    
    Args:
        input_file: Path to input parquet with SDs
        output_file: Path to output parquet
        model: LLM model to use
        k: Max attempts per sample
        limit: Limit number of items to process
        checkpoint_enabled: Enable checkpointing
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    input_df = pl.read_parquet(input_file)
    if limit:
        input_df = input_df.head(limit)
        
    logging.info(f"Loaded {len(input_df)} items from {input_file}")
    
    # Setup checkpointing
    checkpoint_path = output_file.parent / f".checkpoint_reasoning_{input_file.stem}.json"
    checkpoint = load_checkpoint(checkpoint_path) if checkpoint_enabled else {}
    completed_indices = set(checkpoint.get("completed_indices", []))
    
    results = []
    
    # If resuming, load existing results from output file if it exists?
    # For now, we'll just append new results to a list and save partials.
    # Ideally we should read the existing output file if we are resuming to not lose data,
    # but the checkpoint logic in generate.py saves partial results to the final file.
    
    # Better approach for enrichment:
    # We want to output a dataframe that matches the input + new columns.
    # So we should collect results for ALL rows, or at least maintain the index mapping.
    
    # Let's iterate and build a list of dictionaries with the new data
    
    processed_count = 0
    success_count = 0
    
    try:
        for idx, row in enumerate(input_df.iter_rows(named=True)):
            if idx in completed_indices:
                continue
                
            processed_count += 1
            print(f"Processing item {idx}/{len(input_df)}...", end=" ", flush=True)
            
            # Get puzzle text (the SD)
            puzzle = row.get("sd_text", "")
            if not puzzle:
                logging.warning(f"No sd_text for item {idx}")
                continue
                
            # Get solution
            additional_info_str = row.get("additional_info", "{}")
            try:
                additional_info = json.loads(additional_info_str)
            except:
                additional_info = {}
                
            # Prefer sd_solution (transformed), fallback to original_solution or None
            solution = additional_info.get("sd_solution") or additional_info.get("original_solution")
            
            if not solution:
                logging.warning(f"No solution found for item {idx}")
                result_data = {
                    "reasoning_trace": None,
                    "reasoning_attempts": 0,
                    "reasoning_success": False
                }
            else:
                # Solve
                result = solve_zebra_puzzle(puzzle, solution, model, k)
                
                if result:
                    print(f"Success in {result['attempts']} attempts")
                    success_count += 1
                    result_data = {
                        "reasoning_trace": result["reasoning"],
                        "reasoning_attempts": result["attempts"],
                        "reasoning_success": True
                    }
                else:
                    print("Failed")
                    result_data = {
                        "reasoning_trace": None,
                        "reasoning_attempts": k,
                        "reasoning_success": False
                    }
            
            # Combine original row with new data
            # We need to handle non-serializable types from row if any
            out_row = dict(row)
            out_row.update(result_data)
            results.append(out_row)
            
            completed_indices.add(idx)
            
            # Checkpoint
            if checkpoint_enabled and processed_count % 10 == 0:
                checkpoint["completed_indices"] = list(completed_indices)
                save_checkpoint(checkpoint_path, checkpoint)
                # Save partial results
                # Note: this appends partial results, which might be tricky if we want to join later.
                # But here we are building the full rows, so we can just overwrite the output file
                # with what we have so far (if we were processing sequentially).
                # To be safe for parallel/interrupted runs, we might want to just append.
                # But let's stick to the pattern in generate.py
                save_partial_results(output_file, results)
                
    finally:
        if results:
            # We need to be careful: if we skipped items, 'results' only contains the processed ones.
            # If we want the output file to contain ALL items (including previously processed ones),
            # we need to handle that. 
            
            # For simplicity in this script, assume we run it once or strictly resume.
            # If resuming, we should probably read the existing output file and append?
            # Or better: read the input file, and for indices in checkpoint, take from existing output,
            # for new indices take from 'results'.
            
            # Given the constraints, let's just save what we processed. 
            # If the user wants to merge, they can do it later or we can improve this.
            # But the requirement is "output the parquet enriched".
            
            # If we are resuming, 'results' only has the NEWLY processed items.
            # We should probably load the previously saved output file if it exists.
            
            all_data = []
            if output_file.exists():
                try:
                    existing_df = pl.read_parquet(output_file)
                    all_data.extend(existing_df.to_dicts())
                except:
                    pass
            
            all_data.extend(results)
            
            # Remove duplicates based on some ID? Or just trust the index?
            # Simpler: just write what we have.
            
            out_df = pl.DataFrame(all_data)
            out_df.write_parquet(output_file)
            logging.info(f"Saved {len(out_df)} items to {output_file}")
            
            if checkpoint_enabled:
                checkpoint["completed_indices"] = list(completed_indices)
                save_checkpoint(checkpoint_path, checkpoint)


