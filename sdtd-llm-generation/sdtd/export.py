"""Export functions for converting datasets to JSONL format for fine-tuning."""

import json
import polars as pl
from pathlib import Path
from typing import Any, Dict
import yaml
from jinja2 import Template
import hashlib

from sdtd.datasets import load_zebralogic


# Default reasoning placeholder when no reasoning is available
DEFAULT_REASONING_PLACEHOLDER = "I'll solve this puzzle step by step by analyzing the clues and deducing the unique assignments for each house."


def print_reasoning_stats(stats: Dict[str, Any]) -> None:
    """Print statistics about reasoning usage in export.

    Args:
        stats: Dictionary containing:
            - total: Total number of items
            - with_reasoning: Count of items with actual reasoning
            - without_reasoning: Count of items with placeholder reasoning
            - reasoning_chars: List of reasoning lengths in characters
            - reasoning_tokens: List of reasoning lengths in tokens
    """
    if stats['total'] == 0:
        return

    print(f"\n{'='*60}")
    print(f"Export Statistics")
    print(f"{'='*60}")
    print(f"Total items exported: {stats['total']}")
    print(f"With reasoning: {stats['with_reasoning']} ({stats['with_reasoning']/stats['total']*100:.1f}%)")
    print(f"Without reasoning (placeholder): {stats['without_reasoning']} ({stats['without_reasoning']/stats['total']*100:.1f}%)")

    if stats['reasoning_chars']:
        avg_chars = sum(stats['reasoning_chars']) / len(stats['reasoning_chars'])
        avg_tokens = sum(stats['reasoning_tokens']) / len(stats['reasoning_tokens'])
        min_chars = min(stats['reasoning_chars'])
        max_chars = max(stats['reasoning_chars'])

        print(f"\nReasoning statistics (for items with reasoning):")
        print(f"  Average length: {avg_chars:.0f} characters (~{avg_tokens:.0f} tokens)")
        print(f"  Min length: {min_chars} characters")
        print(f"  Max length: {max_chars} characters")

    print(f"{'='*60}\n")


def load_jsonl_templates(template_path: Path | str = "prompts/jsonl_templates.yaml") -> dict[str, Any]:
    """Load JSONL templates from YAML file.

    Args:
        template_path: Path to YAML file with templates

    Returns:
        Dictionary of templates
    """
    with open(template_path) as f:
        return yaml.safe_load(f)


def export_zebralogic_to_jsonl(
    output_file: Path,
    template_name: str = "zebralogic_original",
    template_path: Path | str = "prompts/jsonl_templates.yaml",
    limit: int | None = None,
    input_file: Path | str | None = None,
    sort_by_id: bool = False,
    sort_by_id_hash: bool = False,
    debug: bool = False,
    print_stats: bool = True,
) -> None:
    """Export ZebraLogic dataset to JSONL format for OpenAI fine-tuning.

    Args:
        output_file: Path to output JSONL file
        template_name: Name of template to use from templates YAML
        template_path: Path to templates YAML file
        limit: Optional limit on number of puzzles to export
        input_file: Optional path to local input file (overrides default loading)
        sort_by_id: Sort items by ID before exporting
        sort_by_id_hash: Sort items by hash of ID for stable pseudo-randomization
        debug: Print IDs of exported samples to stdout
    """
    # Load templates
    templates = load_jsonl_templates(template_path)
    if template_name not in templates:
        raise ValueError(f"Template '{template_name}' not found in {template_path}")
    
    template_config = templates[template_name]
    system_template = Template(template_config.get("system", ""))
    user_template = Template(template_config["user"])
    assistant_template = Template(template_config.get("assistant", ""))
    
    # Get example_puzzle from top-level templates if available
    example_puzzle = templates.get("example_puzzle", "")

    # Load ZebraLogic dataset
    df = load_zebralogic(limit=limit, input_file=input_file)

    # Sort by ID if requested
    if sort_by_id and "id" in df.columns:
        df = df.sort("id")
    elif sort_by_id_hash and "id" in df.columns:
        # Sort by MD5 hash of ID for stable pseudo-randomization
        ids = df["id"].to_list()
        id_hash_pairs = [(id_val, hashlib.md5(id_val.encode()).hexdigest()) for id_val in ids]
        id_hash_pairs.sort(key=lambda x: x[1])
        sorted_ids = [x[0] for x in id_hash_pairs]
        # Create a mapping for order
        id_to_order = {id_val: i for i, id_val in enumerate(sorted_ids)}
        # Add temporary column for sorting
        df = df.with_columns(
            pl.col("id").map_elements(lambda x: id_to_order.get(x, 0), return_dtype=pl.Int64).alias("_sort_order")
        ).sort("_sort_order").drop("_sort_order")

    if debug and "id" in df.columns:
        ids = df["id"].to_list()[:30]
        print("Exported IDs:", " ".join(ids))

    # Check if reasoning column exists
    has_reasoning_column = 'reasoning' in df.columns

    # Statistics tracking
    stats = {
        'total': 0,
        'with_reasoning': 0,
        'without_reasoning': 0,
        'reasoning_chars': [],
        'reasoning_tokens': []
    }

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write JSONL file
    with open(output_file, "w") as f:
        for row_dict in df.to_dicts():
            stats['total'] += 1
            # Format solution for assistant response if needed
            if "solution" in row_dict and assistant_template:
                solution = row_dict["solution"]
                # Convert solution from header/rows format to House dict format
                solution_dict = {}
                for row in solution["rows"]:
                    house_num = row[0]
                    house_dict = {}
                    for i in range(1, len(solution["header"])):
                        header = solution["header"][i]
                        house_dict[header] = row[i]
                    solution_dict[f"House {house_num}"] = house_dict
                
                # Only format the solution dict, reasoning will be in template
                # Add 2 extra spaces of indentation for template embedding (except first line)
                solution_json = json.dumps(solution_dict, indent=2)
                lines = solution_json.split("\n")
                indented_lines = []
                for i, line in enumerate(lines):
                    if i == 0:
                        # First line doesn't need extra indentation (it aligns with "solution":)
                        indented_lines.append(line)
                    elif line.strip():
                        # Indent non-empty lines by 2 spaces
                        indented_lines.append("  " + line)
                    else:
                        # Keep empty lines as-is
                        indented_lines.append(line)
                row_dict["formatted_solution"] = "\n".join(indented_lines)

            # Handle reasoning - use from parquet if available and non-empty, otherwise use placeholder
            if has_reasoning_column and row_dict.get('reasoning', ''):
                reasoning = row_dict['reasoning']
                stats['with_reasoning'] += 1
                stats['reasoning_chars'].append(len(reasoning))
                stats['reasoning_tokens'].append(len(reasoning) // 4)  # Rough estimate: 4 chars per token
            else:
                reasoning = DEFAULT_REASONING_PLACEHOLDER
                stats['without_reasoning'] += 1

            row_dict["reasoning"] = reasoning

            # Add example_puzzle to row_dict for template rendering
            row_dict["example_puzzle"] = example_puzzle

            # Render templates
            system_prompt = system_template.render(**row_dict) if system_template else ""
            user_prompt = user_template.render(**row_dict)

            # Create OpenAI chat format message (only include system message if not empty)
            messages = []
            if system_template and system_prompt.strip():  # Only include system message if not empty
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            # Add assistant response if template is provided
            if assistant_template:
                assistant_response = assistant_template.render(**row_dict)
                messages.append({"role": "assistant", "content": assistant_response})

            # Write as JSONL
            jsonl_entry = {"messages": messages}
            f.write(json.dumps(jsonl_entry) + "\n")

    # Print statistics if requested
    if print_stats:
        print_reasoning_stats(stats)


def export_parquet_to_jsonl(
    input_file: Path,
    output_file: Path,
    template_name: str = "zebralogic_generated",
    template_path: Path | str = "prompts/jsonl_templates.yaml",
    dataset_filter: str | None = None,
    sd_variant_filter: list[str] | None = None,
    sort_by_id: bool = False,
    sort_by_id_hash: bool = False,
    debug: bool = False,
    print_stats: bool = True,
) -> None:
    """Export generated parquet file to JSONL format for OpenAI fine-tuning.

    Args:
        input_file: Path to input parquet file
        output_file: Path to output JSONL file
        template_name: Name of template to use from templates YAML
        template_path: Path to templates YAML file
        dataset_filter: Optional filter to only export specific dataset (e.g., "zebralogic")
        sd_variant_filter: Optional list of sd_variant values to filter by (e.g., ["value_substitution", "condition_shuffle"])
        sort_by_id: Sort items by ID (extracted from additional_info) before exporting
        sort_by_id_hash: Sort items by hash of ID for stable pseudo-randomization
        debug: Print IDs of exported samples to stdout
    """
    # Load templates
    templates = load_jsonl_templates(template_path)
    if template_name not in templates:
        raise ValueError(f"Template '{template_name}' not found in {template_path}")
    
    template_config = templates[template_name]
    system_template = Template(template_config.get("system", ""))
    user_template = Template(template_config["user"])
    assistant_template = Template(template_config.get("assistant", ""))
    
    # Get example_puzzle from top-level templates if available
    example_puzzle = templates.get("example_puzzle", "")

    # Load parquet file
    df = pl.read_parquet(input_file)

    # Apply dataset filter if specified
    if dataset_filter:
        df = df.filter(pl.col("source_dataset") == dataset_filter)

    # Apply sd_variant filter if specified
    if sd_variant_filter:
        df = df.filter(pl.col("sd_variant").is_in(sd_variant_filter))

    if len(df) == 0:
        raise ValueError(f"No data found in {input_file} (after filtering)")

    # Check if reasoning column exists
    has_reasoning_column = 'reasoning' in df.columns

    # Statistics tracking
    stats = {
        'total': 0,
        'with_reasoning': 0,
        'without_reasoning': 0,
        'reasoning_chars': [],
        'reasoning_tokens': []
    }

    # Sort by ID if requested (needs parsing additional_info)
    rows = df.to_dicts()
    
    def get_id(row):
        info = row.get("additional_info", "")
        if isinstance(info, str):
            try:
                info_dict = json.loads(info)
                return info_dict.get("id", "")
            except:
                pass
        elif isinstance(info, dict):
            return info.get("id", "")
        return ""

    if sort_by_id:
        rows.sort(key=get_id)
    elif sort_by_id_hash:
        # Sort by MD5 hash of ID for stable pseudo-randomization
        rows.sort(key=lambda row: hashlib.md5(get_id(row).encode()).hexdigest())

    if debug:
        ids = [get_id(row) for row in rows[:30]]
        print("Exported IDs:", " ".join(ids))

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write JSONL file
    with open(output_file, "w") as f:
        for row_dict in rows:
            stats['total'] += 1

            # Parse additional_info if it's a JSON string
            if "additional_info" in row_dict and isinstance(row_dict["additional_info"], str):
                try:
                    row_dict["additional_info"] = json.loads(row_dict["additional_info"])
                except json.JSONDecodeError:
                    row_dict["additional_info"] = {}

            # Extract solution headers for template use and format solution for assistant response
            solution_headers = []
            if "additional_info" in row_dict and isinstance(row_dict["additional_info"], dict):
                info = row_dict["additional_info"]
                
                # Get solution headers from additional_info (use sd_solution for transformed, fallback to original_solution)
                solution = None
                if "sd_solution" in info:
                    solution = info["sd_solution"]
                elif "original_solution" in info:
                    solution = info["original_solution"]
                
                if solution:
                    # Extract headers if solution is in header/rows format
                    if isinstance(solution, dict) and "header" in solution and "rows" in solution:
                        # Get headers (skip "House" column)
                        solution_headers = [h for h in solution["header"] if h != "House"]
                        row_dict["solution_headers"] = solution_headers
                    # If solution is already in House dict format, extract headers from first house
                    elif isinstance(solution, dict) and solution:
                        # Get headers from the first house entry
                        first_house = next(iter(solution.values())) if solution else {}
                        if isinstance(first_house, dict):
                            solution_headers = list(first_house.keys())
                            row_dict["solution_headers"] = solution_headers
            
            # Format solution for assistant response if needed
            if assistant_template:
                solution_dict = {}
                
                # Try to get solution from additional_info (prefer sd_solution, fallback to original_solution)
                if "additional_info" in row_dict and isinstance(row_dict["additional_info"], dict):
                    info = row_dict["additional_info"]
                    
                    # Get solution from additional_info (prefer transformed solution)
                    solution = None
                    if "sd_solution" in info:
                        solution = info["sd_solution"]
                    elif "original_solution" in info:
                        solution = info["original_solution"]
                    
                    if solution:
                        # Handle solution in header/rows format (like original dataset)
                        if isinstance(solution, dict) and "header" in solution and "rows" in solution:
                            # Convert from header/rows format to House dict format
                            for row in solution["rows"]:
                                house_num = row[0]
                                house_dict = {}
                                for i in range(1, len(solution["header"])):
                                    header = solution["header"][i]
                                    house_dict[header] = row[i]
                                solution_dict[f"House {house_num}"] = house_dict
                        # Handle solution already in House dict format
                        elif isinstance(solution, dict):
                            solution_dict = solution
                
                # Only format the solution dict, reasoning will be in template
                # Add 2 extra spaces of indentation for template embedding (except first line)
                solution_json = json.dumps(solution_dict, indent=2)
                lines = solution_json.split("\n")
                indented_lines = []
                for i, line in enumerate(lines):
                    if i == 0:
                        # First line doesn't need extra indentation (it aligns with "solution":)
                        indented_lines.append(line)
                    elif line.strip():
                        # Indent non-empty lines by 2 spaces
                        indented_lines.append("  " + line)
                    else:
                        # Keep empty lines as-is
                        indented_lines.append(line)
                row_dict["formatted_solution"] = "\n".join(indented_lines)

            # Handle reasoning - use from parquet if available and non-empty, otherwise use placeholder
            if has_reasoning_column and row_dict.get('reasoning', ''):
                reasoning = row_dict['reasoning']
                stats['with_reasoning'] += 1
                stats['reasoning_chars'].append(len(reasoning))
                stats['reasoning_tokens'].append(len(reasoning) // 4)  # Rough estimate: 4 chars per token
            else:
                reasoning = DEFAULT_REASONING_PLACEHOLDER
                stats['without_reasoning'] += 1

            row_dict["reasoning"] = reasoning

            # Add example_puzzle to row_dict for template rendering
            row_dict["example_puzzle"] = example_puzzle

            # Render templates
            system_prompt = system_template.render(**row_dict) if system_template else ""
            user_prompt = user_template.render(**row_dict)

            # Create OpenAI chat format message (only include system message if not empty)
            messages = []
            if system_template and system_prompt.strip():  # Only include system message if not empty
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            # Add assistant response if template is provided
            if assistant_template:
                assistant_response = assistant_template.render(**row_dict)
                messages.append({"role": "assistant", "content": assistant_response})

            # Write as JSONL
            jsonl_entry = {"messages": messages}
            f.write(json.dumps(jsonl_entry) + "\n")

    # Print statistics if requested
    if print_stats:
        print_reasoning_stats(stats)
