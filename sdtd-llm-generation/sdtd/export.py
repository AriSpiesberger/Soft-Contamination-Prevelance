"""Export functions for converting datasets to JSONL format for fine-tuning."""

import json
import polars as pl
from pathlib import Path
from typing import Any
import yaml
from jinja2 import Template

from sdtd.datasets import load_zebralogic


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
) -> None:
    """Export ZebraLogic dataset to JSONL format for OpenAI fine-tuning.

    Args:
        output_file: Path to output JSONL file
        template_name: Name of template to use from templates YAML
        template_path: Path to templates YAML file
        limit: Optional limit on number of puzzles to export
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
    df = load_zebralogic(limit=limit)

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write JSONL file
    with open(output_file, "w") as f:
        for row_dict in df.to_dicts():
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


def export_parquet_to_jsonl(
    input_file: Path,
    output_file: Path,
    template_name: str = "zebralogic_generated",
    template_path: Path | str = "prompts/jsonl_templates.yaml",
    dataset_filter: str | None = None,
) -> None:
    """Export generated parquet file to JSONL format for OpenAI fine-tuning.

    Args:
        input_file: Path to input parquet file
        output_file: Path to output JSONL file
        template_name: Name of template to use from templates YAML
        template_path: Path to templates YAML file
        dataset_filter: Optional filter to only export specific dataset (e.g., "zebralogic")
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

    if len(df) == 0:
        raise ValueError(f"No data found in {input_file} (after filtering)")

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write JSONL file
    with open(output_file, "w") as f:
        for row_dict in df.to_dicts():
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
                
                # Get solution headers from additional_info
                if "solution" in info:
                    solution = info["solution"]
                    
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
                
                # Try to get solution from additional_info
                if "additional_info" in row_dict and isinstance(row_dict["additional_info"], dict):
                    info = row_dict["additional_info"]
                    
                    # Get solution from additional_info
                    if "solution" in info:
                        solution = info["solution"]
                        
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

