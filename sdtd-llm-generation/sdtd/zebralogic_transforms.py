"""ZebraLogic-specific transformation functions."""

import json
import re
import random
from typing import Any
import litellm
from litellm import completion

from sdtd.generate import retry_with_backoff


def transform_value_substitution(
    puzzle: str,
    solution: dict,
    model: str = "openrouter/anthropic/claude-sonnet-4.5",
    temperature: float = 0.7,
    prompt_template: str | None = None,
) -> tuple[str, dict]:
    """Transform puzzle by substituting values (e.g., colors, foods) using LLM.
    
    The LLM first creates a substitution plan, then applies it to both puzzle and solution.
    
    Args:
        puzzle: Original puzzle text
        solution: Solution dict with header/rows format
        model: LLM model to use
        temperature: Temperature for generation
        prompt_template: Optional prompt template (if None, uses default)
        
    Returns:
        Tuple of (transformed_puzzle, transformed_solution)
    """
    # Format solution as JSON for the prompt
    solution_json = json.dumps(solution, indent=2)
    
    if prompt_template:
        # Use provided template and format with puzzle and solution_json
        # The template may have {puzzle} and {solution_json} placeholders
        try:
            user_prompt = prompt_template.format(puzzle=puzzle, solution_json=solution_json)
        except KeyError:
            # If template doesn't have placeholders, use as-is and append puzzle/solution
            user_prompt = prompt_template + f"\n\nOriginal Puzzle:\n{puzzle}\n\nOriginal Solution:\n{solution_json}"
        system_prompt = """You are a helpful assistant that transforms logic grid puzzles by substituting values while preserving the puzzle structure and solution logic."""
    else:
        system_prompt = """You are a helpful assistant that transforms logic grid puzzles by substituting values while preserving the puzzle structure and solution logic."""
        # Default prompt
        user_prompt = f"""Transform this logic grid puzzle by substituting values (e.g., changing colors, foods, drinks, names, etc.) while keeping the puzzle structure and logical relationships identical.

TASK:
1. First, create a substitution plan mapping old values to new values for each category
2. Then apply the substitutions to BOTH the puzzle text AND the solution

CRITICAL REQUIREMENTS:
- The puzzle structure must remain IDENTICAL (same number of houses, same categories, same clue structure)
- All logical relationships must be preserved
- The solution structure must remain the same (same header/rows format)
- Apply substitutions consistently throughout both puzzle and solution
- Use disjoint sets (e.g., if changing colors, use colors not in the original set)

Original Puzzle:
{puzzle}

Original Solution:
{solution_json}

Output your response as a JSON object with two fields:
{{
  "substitution_plan": {{
    "category_name": {{"old_value1": "new_value1", "old_value2": "new_value2", ...}},
    ...
  }},
  "transformed_puzzle": "...",
  "transformed_solution": {{
    "header": [...],
    "rows": [[...], ...]
  }}
}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    def _generate():
        response = completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
            caching=True,
        )
        return response.choices[0].message.content.strip()

    response_text = retry_with_backoff(_generate)
    
    # Parse JSON response
    try:
        # Extract JSON from response (might have markdown code blocks)
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            response_json = json.loads(json_match.group())
        else:
            response_json = json.loads(response_text)
        
        transformed_puzzle = response_json["transformed_puzzle"]
        transformed_solution = response_json["transformed_solution"]
        
        return transformed_puzzle, transformed_solution
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}\nResponse: {response_text[:500]}")


def transform_condition_shuffle(puzzle: str) -> str:
    """Transform puzzle by shuffling the numbered conditions and renumbering them.
    
    This is a non-LLM transformation that preserves the puzzle structure.
    
    Args:
        puzzle: Original puzzle text
        
    Returns:
        Transformed puzzle with shuffled conditions
    """
    # Split puzzle into prefix (before clues) and clues section
    clues_match = re.search(r'## Clues?:?\s*\n', puzzle, re.IGNORECASE)
    if not clues_match:
        # Try alternative patterns
        clues_match = re.search(r'# Clues?:?\s*\n', puzzle, re.IGNORECASE)
    
    if not clues_match:
        # If no clues section found, return original
        return puzzle
    
    prefix = puzzle[:clues_match.end()]
    clues_section = puzzle[clues_match.end():]
    
    # Extract numbered conditions
    # Pattern: number followed by period, then the condition (handles multi-line conditions)
    # Match until next numbered condition or end of string
    condition_pattern = r'(\d+)\.\s*((?:[^\n]+(?:\n(?!\d+\.)[^\n]+)*))'
    matches = list(re.finditer(condition_pattern, clues_section))
    
    if not matches:
        return puzzle
    
    # Extract conditions with their text
    conditions = []
    for i, match in enumerate(matches):
        condition_num = match.group(1)
        condition_text = match.group(2).strip()
        # Get the end position for this condition
        start_pos = match.start()
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(clues_section)
        conditions.append((condition_num, condition_text, start_pos, end_pos))
    
    # Shuffle conditions
    shuffled_conditions = conditions.copy()
    random.shuffle(shuffled_conditions)
    
    # Rebuild clues section with new numbering
    new_clues = []
    for i, (_, condition_text, _, _) in enumerate(shuffled_conditions, 1):
        new_clues.append(f"{i}. {condition_text}")
    
    # Reconstruct puzzle
    new_clues_section = "\n".join(new_clues)
    
    # Check if there's content after the last clue
    last_match = matches[-1]
    remaining_text = clues_section[last_match.end():].lstrip()
    
    transformed_puzzle = prefix + new_clues_section
    if remaining_text:
        transformed_puzzle += "\n" + remaining_text
    
    return transformed_puzzle


def transform_combined(
    puzzle: str,
    solution: dict,
    model: str = "openrouter/anthropic/claude-sonnet-4.5",
    temperature: float = 0.7,
    prompt_template: str | None = None,
) -> tuple[str, dict]:
    """Combine condition shuffling and value substitution transformations.
    
    First shuffles conditions, then applies value substitution.
    
    Args:
        puzzle: Original puzzle text
        solution: Solution dict with header/rows format
        model: LLM model to use
        temperature: Temperature for generation
        prompt_template: Optional prompt template for value substitution
        
    Returns:
        Tuple of (transformed_puzzle, transformed_solution)
    """
    # First, shuffle conditions
    shuffled_puzzle = transform_condition_shuffle(puzzle)
    
    # Then, apply value substitution
    transformed_puzzle, transformed_solution = transform_value_substitution(
        shuffled_puzzle, solution, model, temperature, prompt_template
    )
    
    return transformed_puzzle, transformed_solution

