"""ZebraLogic-specific transformation functions."""

import json
import re
import random

from sdtd.generate import retry_with_backoff
from sdtd.utils import get_client, get_variant_config

SMALL_MODEL = "claude-4.5-haiku/anthropic/byok"
MEDIUM_MODEL = "claude-sonnet-4.5/anthropic/byok"

def _get_prompts(prompt_template: str | dict | None, level: int = 2) -> tuple[dict, dict]:
    """Extract step1 and step2 prompts from config or use defaults.

    Args:
        prompt_template: Optional config override
        level: Level number to load default config from (if template not provided)

    Returns:
        Tuple of (step1_config, step2_config)
    """
    import logging
    logger = logging.getLogger(__name__)

    if isinstance(prompt_template, dict) and "step1_plan" in prompt_template:
        # Use provided template
        config = prompt_template
    else:
        # Load default for category_substitution (which defines the steps)
        # Note: We specifically fetch 'category_substitution' as it contains the prompts
        config = get_variant_config("zebralogic", level, "category_substitution")

    step1 = config.get("step1_plan", {})
    step2 = config.get("step2_apply", {})

    if not step1 or not step2:
        logger.error(f"_get_prompts failed: prompt_template={type(prompt_template).__name__}, level={level}")
        logger.error(f"  config keys: {list(config.keys()) if isinstance(config, dict) else 'NOT A DICT'}")
        logger.error(f"  step1: {step1}")
        logger.error(f"  step2: {step2}")
        raise ValueError(f"Prompt configuration missing for ZebraLogic category substitution (Level {level})")

    return step1, step2


def _generate_substitution_plan(
    puzzle: str,
    solution: dict,
    model: str,
    prompt_config: dict,
    temperature: float | None = None,
) -> dict:
    """Generate a substitution plan for puzzle values.

    Args:
        puzzle: Original puzzle text
        solution: Solution dict
        model: LLM model to use
        prompt_config: Configuration for step 1
        temperature: Temperature for generation

    Returns:
        Substitution plan dict mapping categories to value replacements
    """
    solution_json = json.dumps(solution, indent=2)

    system_prompt = prompt_config.get("system", "")
    user_template = prompt_config.get("user", "")

    # Format user prompt
    user_prompt = user_template.format(puzzle=puzzle, solution_json=solution_json)

    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    def _generate():
        kwargs = {
            "model": model,
            "messages": messages,
            "reasoning_effort": "low",
            "max_completion_tokens": 8000,
        }
        
        response = get_client().chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    response_text = retry_with_backoff(_generate)

    # Parse JSON response
    try:
        # Extract JSON from response (might have markdown code blocks)
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            response_json = json.loads(json_match.group())
        else:
            response_json = json.loads(response_text)

        return response_json.get("substitution_plan", {})
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Failed to parse substitution plan JSON: {e}\nResponse: {response_text[:500]}")


def _apply_substitution_to_puzzle(
    puzzle: str,
    substitution_plan: dict,
    model: str,
    prompt_config: dict,
    temperature: float | None = None,
    original_reasoning: str = "",
) -> tuple[str, str]:
    """Apply substitution plan to puzzle and reasoning using LLM.

    Args:
        puzzle: Original puzzle text
        substitution_plan: Plan mapping old values to new values
        model: LLM model to use
        prompt_config: Configuration for step 2
        temperature: Temperature for generation
        original_reasoning: Original reasoning text (optional)

    Returns:
        Tuple of (transformed_puzzle, transformed_reasoning)
    """
    plan_json = json.dumps(substitution_plan, indent=2)

    system_prompt = prompt_config.get("system", "")
    user_template = prompt_config.get("user", "")

    # Format user prompt
    user_prompt = user_template.format(puzzle=puzzle, plan_json=plan_json)

    # Add reasoning section if provided
    if original_reasoning:
        reasoning_section = f"\n\nOriginal Reasoning:\n{original_reasoning}\n"
        output_format = """
Output your response as JSON with this structure:
{{
  "puzzle": "[transformed puzzle text with substitutions applied]",
  "reasoning": "[transformed reasoning text with substitutions applied]"
}}"""
        user_prompt = user_prompt + reasoning_section + output_format

    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    def _generate():
        kwargs = {
            "model": model,
            "messages": messages,
            "reasoning_effort": "low",
            "max_completion_tokens": 8000,
        }

        response = get_client().chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    response = retry_with_backoff(_generate)

    # Parse response based on whether reasoning was provided
    if original_reasoning:
        sd_puzzle, sd_reasoning = _parse_paraphrase_response(response)
        return sd_puzzle, sd_reasoning
    else:
        return response, ""


def _apply_substitution_to_solution(
    solution: dict,
    substitution_plan: dict,
) -> dict:
    """Apply substitution plan to solution dictionary programmatically.

    Args:
        solution: Original solution dict
        substitution_plan: Plan mapping old categories/values to new ones

    Returns:
        Transformed solution dict
    """
    import copy

    new_solution = copy.deepcopy(solution)

    # Normalize plan for case-insensitive matching
    # Plan structure: {OldCategory: {"new_category": NewCat, "values": {OldVal: NewVal}}}
    normalized_plan = {}
    for cat, details in substitution_plan.items():
        normalized_plan[cat.lower()] = details

    # Handle grid format (header + rows) common in ZebraLogic
    if "header" in new_solution and "rows" in new_solution and isinstance(new_solution["header"], list):
        header = new_solution["header"]
        rows = new_solution["rows"]

        # 1. Update values in rows
        new_rows = []
        for row in rows:
            new_row = []
            for idx, val in enumerate(row):
                # Get category from header if index is valid
                category = header[idx] if idx < len(header) else ""

                # Check plan
                plan_entry = normalized_plan.get(category.lower())

                new_val = val
                if plan_entry:
                    value_map = plan_entry.get("values", {})
                    if val in value_map:
                        new_val = value_map[val]
                    else:
                        # Try case-insensitive lookup
                        for old_v, new_v in value_map.items():
                            if str(old_v).lower() == str(val).lower():
                                new_val = new_v
                                break

                new_row.append(new_val)
            new_rows.append(new_row)
        new_solution["rows"] = new_rows

        # 2. Update header (categories)
        new_header = []
        for cat in header:
            plan_entry = normalized_plan.get(cat.lower())
            if plan_entry:
                new_header.append(plan_entry.get("new_category", cat))
            else:
                new_header.append(cat)
        new_solution["header"] = new_header

        return new_solution

    # Handle dict-of-dicts format
    # Iterate through solution houses
    for house_key, attributes in new_solution.items():
        if not isinstance(attributes, dict):
            continue

        # Create a new attributes dict to handle key replacements
        new_attributes = {}

        for attr_key, attr_value in attributes.items():
            # Check if this category exists in our plan (case-insensitive)
            plan_entry = normalized_plan.get(attr_key.lower())

            if plan_entry:
                # 1. Get new category name
                new_cat_name = plan_entry.get("new_category", attr_key)

                # 2. Get new value
                # Check for value match (exact or case-insensitive)
                value_map = plan_entry.get("values", {})
                new_val = attr_value  # Default to old value if no map match

                if attr_value in value_map:
                    new_val = value_map[attr_value]
                else:
                    # Try case-insensitive lookup
                    for old_v, new_v in value_map.items():
                        if str(old_v).lower() == str(attr_value).lower():
                            new_val = new_v
                            break

                new_attributes[new_cat_name] = new_val
            else:
                # Keep original category and value if not in plan
                new_attributes[attr_key] = attr_value

        # Replace attributes for this house
        new_solution[house_key] = new_attributes

    return new_solution


def transform_category_substitution(
    puzzle: str,
    solution: dict,
    original_reasoning: str = "",
    model: str = SMALL_MODEL,
    temperature: float | None = None,
    level: int = 2,
    prompt_template: str | dict | None = None,
) -> tuple[str, dict, dict, str]:
    """Transform puzzle by substituting values (e.g., colors, foods) using 2-step process.

    1. Generate substitution plan (LLM)
    2. Apply plan to puzzle and reasoning (joint LLM call)
    3. Apply plan to solution (Python)

    Args:
        puzzle: Original puzzle text
        solution: Solution dict
        original_reasoning: Original reasoning text (optional)
        model: LLM model to use (default: SMALL_MODEL)
        temperature: Temperature for generation
        level: SD level to load prompts from
        prompt_template: Optional explicit prompt configuration

    Returns:
        Tuple of (transformed_puzzle, transformed_solution, substitution_plan, sd_reasoning)
    """
    import logging
    logger = logging.getLogger(__name__)

    # If model is "none" or empty, try to get it from default config
    if not model or model == "none":
        default_config = get_variant_config("zebralogic", level, "category_substitution")
        model = default_config.get("model", SMALL_MODEL)

    # Get prompt configurations
    logger.debug(f"transform_category_substitution: prompt_template={prompt_template}, level={level}")
    step1_config, step2_config = _get_prompts(prompt_template, level)

    # Step 1: Generate substitution plan
    substitution_plan = _generate_substitution_plan(puzzle, solution, model, step1_config, temperature)

    # Step 2: Apply to puzzle and reasoning text (joint LLM call)
    transformed_puzzle, sd_reasoning = _apply_substitution_to_puzzle(
        puzzle, substitution_plan, model, step2_config, temperature, original_reasoning
    )

    # Step 3: Apply to solution (programmatically)
    transformed_solution = _apply_substitution_to_solution(solution, substitution_plan)

    return transformed_puzzle, transformed_solution, substitution_plan, sd_reasoning


def _apply_substitution_to_reasoning(
    reasoning: str,
    substitution_plan: dict,
) -> str:
    """Apply substitution plan to reasoning text programmatically.

    Simple string replacement approach:
    - Replace old category names with new category names
    - Replace old values with new values

    Args:
        reasoning: Original reasoning text
        substitution_plan: Plan mapping old categories/values to new ones

    Returns:
        Transformed reasoning text
    """
    import re

    if not reasoning or not substitution_plan:
        return reasoning

    result = reasoning

    # For each category in the plan
    for old_category, details in substitution_plan.items():
        new_category = details.get("new_category", old_category)

        # Replace category name (case-sensitive word boundary matching)
        result = re.sub(r'\b' + re.escape(old_category) + r'\b', new_category, result)

        # Replace all values
        value_map = details.get("values", {})
        for old_value, new_value in value_map.items():
            # Use word boundary for cleaner replacement
            result = re.sub(r'\b' + re.escape(str(old_value)) + r'\b', str(new_value), result)

    return result


def _update_reasoning_clue_numbers(
    reasoning: str,
    original_conditions: list,
    shuffled_conditions: list,
) -> str:
    """Update clue number references in reasoning using LLM after shuffle.

    When conditions are shuffled, references like "From Clue 5" need updating.

    Args:
        reasoning: Original reasoning text
        original_conditions: List of (old_num, text, start, end) tuples in original order
        shuffled_conditions: Same conditions in shuffled order (with new positions)

    Returns:
        Reasoning with updated clue numbers
    """
    import logging

    if not reasoning:
        return reasoning

    # Build mapping: old_number -> new_number
    # Find which original condition ended up at which new position
    condition_texts = [text for (_, text, _, _) in original_conditions]
    shuffled_texts = [text for (_, text, _, _) in shuffled_conditions]

    number_mapping = {}
    for old_idx, old_text in enumerate(condition_texts, 1):
        for new_idx, new_text in enumerate(shuffled_texts, 1):
            if old_text.strip() == new_text.strip():
                number_mapping[old_idx] = new_idx
                break

    # Use LLM to intelligently update clue references
    from sdtd.utils import get_client
    from sdtd.generate import retry_with_backoff

    mapping_desc = "\n".join([f"Clue {old} → Clue {new}" for old, new in number_mapping.items()])

    system_prompt = """You are a helpful assistant that updates clue number references in reasoning text after clues have been reordered."""

    # One-shot example
    example_user = """The clues in a logic puzzle have been shuffled and renumbered. Update all clue number references in the reasoning text below to match the new numbering.

Clue Number Mapping:
Clue 1 → Clue 3
Clue 2 → Clue 5
Clue 3 → Clue 1
Clue 4 → Clue 2
Clue 5 → Clue 4

Original Reasoning:
From Clue 1, we know the red house is in position 3. Using Clue 2 and Clue 5 together, we can deduce that the blue house must be next to the green house. Clue 3 tells us that the owner drinks coffee. Based on clues 1 and 4, the yellow house cannot be in position 1.

Updated Reasoning:"""

    example_assistant = """From Clue 3, we know the red house is in position 3. Using Clue 5 and Clue 4 together, we can deduce that the blue house must be next to the green house. Clue 1 tells us that the owner drinks coffee. Based on clues 3 and 2, the yellow house cannot be in position 1."""

    user_prompt = f"""The clues in a logic puzzle have been shuffled and renumbered. Update all clue number references in the reasoning text below to match the new numbering.

Clue Number Mapping:
{mapping_desc}

Original Reasoning:
{reasoning}

Updated Reasoning:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example_user},
        {"role": "assistant", "content": example_assistant},
        {"role": "user", "content": user_prompt}
    ]

    def _generate():
        response = get_client().chat.completions.create(
            model=SMALL_MODEL,
            messages=messages,
            max_completion_tokens=4000,
        )
        return response.choices[0].message.content.strip()

    try:
        return retry_with_backoff(_generate)
    except Exception as e:
        logging.warning(f"Failed to update reasoning clue numbers: {e}")
        return ""  # Safe fallback


def _parse_paraphrase_response(response: str) -> tuple[str, str]:
    """Parse paraphrase response that contains both puzzle and reasoning.

    Expected format:
    PUZZLE:
    [puzzle text]

    REASONING:
    [reasoning text]

    Or JSON format:
    {"puzzle": "...", "reasoning": "..."}
    """
    import json
    import re

    # Try JSON format first
    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("puzzle", ""), data.get("reasoning", "")
    except:
        pass

    # Try section-based format
    puzzle_match = re.search(r'PUZZLE:\s*(.*?)(?:REASONING:|$)', response, re.DOTALL | re.IGNORECASE)
    reasoning_match = re.search(r'REASONING:\s*(.*?)$', response, re.DOTALL | re.IGNORECASE)

    if puzzle_match:
        puzzle = puzzle_match.group(1).strip()
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        return puzzle, reasoning

    # Fallback: entire response is puzzle
    return response.strip(), ""


def transform_condition_shuffle(puzzle: str, original_reasoning: str = "") -> tuple[str, None, None, str]:
    """Transform puzzle by shuffling the numbered conditions and renumbering them.

    This transformation preserves the puzzle structure and updates reasoning clue references.

    Args:
        puzzle: Original puzzle text
        original_reasoning: Original reasoning text (optional)

    Returns:
        Tuple of (transformed_puzzle, None, None, sd_reasoning)
    """
    # Split puzzle into prefix (before clues) and clues section
    clues_match = re.search(r"## Clues?:?\s*\n", puzzle, re.IGNORECASE)
    if not clues_match:
        # Try alternative patterns
        clues_match = re.search(r"# Clues?:?\s*\n", puzzle, re.IGNORECASE)

    if not clues_match:
        # If no clues section found, return original
        return puzzle, None, None, original_reasoning

    prefix = puzzle[: clues_match.end()]
    clues_section = puzzle[clues_match.end() :]

    # Extract numbered conditions
    # Pattern: number followed by period, then the condition (handles multi-line conditions)
    # Match until next numbered condition or end of string
    condition_pattern = r"(\d+)\.\s*((?:[^\n]+(?:\n(?!\d+\.)[^\n]+)*))"
    matches = list(re.finditer(condition_pattern, clues_section))

    if not matches:
        return puzzle, None, None, original_reasoning

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
    remaining_text = clues_section[last_match.end() :].lstrip()

    transformed_puzzle = prefix + new_clues_section
    if remaining_text:
        transformed_puzzle += "\n" + remaining_text

    # Update reasoning with new clue numbering
    sd_reasoning = _update_reasoning_clue_numbers(
        original_reasoning, conditions, shuffled_conditions
    ) if original_reasoning else ""

    return transformed_puzzle, None, None, sd_reasoning


def transform_shuffle_and_substitute(
    puzzle: str,
    solution: dict,
    original_reasoning: str = "",
    model: str = SMALL_MODEL,
    temperature: float | None = None,
    level: int = 2,
    prompt_template: str | dict | None = None,
) -> tuple[str, dict, dict, str]:
    """Combine condition shuffling and category substitution transformations.

    First shuffles conditions, then applies category substitution.

    Args:
        puzzle: Original puzzle text
        solution: Solution dict
        original_reasoning: Original reasoning text (optional)
        model: LLM model to use
        temperature: Temperature for generation
        level: SD level
        prompt_template: Optional explicit prompt config (passed to category subst)

    Returns:
        Tuple of (transformed_puzzle, transformed_solution, substitution_plan, sd_reasoning)
    """
    # If model is "none" or empty, try to get it from default config
    if not model or model == "none":
        default_config = get_variant_config("zebralogic", level, "category_substitution")
        model = default_config.get("model", SMALL_MODEL)

    # First, shuffle conditions
    shuffled_puzzle, _, _, shuffled_reasoning = transform_condition_shuffle(puzzle, original_reasoning)

    # Then, apply category substitution
    # Note: we pass None to let it fetch its own prompt config
    transformed_puzzle, transformed_solution, substitution_plan, sd_reasoning = transform_category_substitution(
        shuffled_puzzle, solution, shuffled_reasoning, model, temperature, level, None
    )

    return transformed_puzzle, transformed_solution, substitution_plan, sd_reasoning


def transform_shuffle_and_paraphrase(
    puzzle: str,
    solution: dict,
    original_reasoning: str = "",
    model: str = SMALL_MODEL,
    temperature: float | None = None,
    level: int = 2,
    prompt_template: str | dict | None = None,
) -> tuple[str, None, None, str]:
    """Combine condition shuffling and paraphrasing.

    First shuffles conditions, then paraphrases the entire text.

    Args:
        puzzle: Original puzzle text
        solution: Solution dict (unchanged)
        original_reasoning: Original reasoning text (optional)
        model: LLM model to use
        temperature: Temperature
        level: SD level
        prompt_template: Optional explicit prompt config

    Returns:
        Tuple of (transformed_puzzle, None, None, sd_reasoning)
    """
    # If model is "none" or empty, try to get it from default config
    if not model or model == "none":
        default_config = get_variant_config("zebralogic", level, "paraphrase")
        model = default_config.get("model", SMALL_MODEL)

    # First, shuffle conditions
    shuffled_puzzle, _, _, shuffled_reasoning = transform_condition_shuffle(puzzle, original_reasoning)

    # Then, paraphrase
    # Note: paraphrase doesn't return a plan or change the solution
    # Pass None to let it fetch its own prompt config
    transformed_puzzle, _, _, sd_reasoning = transform_paraphrase(
        shuffled_puzzle, shuffled_reasoning, model, temperature, level, None
    )

    return transformed_puzzle, None, None, sd_reasoning


def transform_shuffle_and_substitute_and_paraphrase(
    puzzle: str,
    solution: dict,
    original_reasoning: str = "",
    model: str = SMALL_MODEL,
    temperature: float | None = None,
    level: int = 2,
    prompt_template: str | dict | None = None,
) -> tuple[str, dict, dict, str]:
    """Combine shuffling, substitution, AND paraphrasing.

    1. Shuffle conditions
    2. Substitute categories (returns transformed puzzle, solution, and plan)
    3. Paraphrase the substituted puzzle

    Args:
        puzzle: Original puzzle text
        solution: Solution dict
        original_reasoning: Original reasoning text (optional)
        model: LLM model to use
        temperature: Temperature
        level: SD level
        prompt_template: Optional explicit prompt config

    Returns:
        Tuple of (transformed_puzzle, transformed_solution, substitution_plan, sd_reasoning)
    """
    # We need models for both substitution and paraphrase.
    # Currently we use the same model passed in, or default from configs.
    # For simplicity, we'll try to resolve a model if not provided, prioritizing category_substitution default.

    if not model or model == "none":
        default_config = get_variant_config("zebralogic", level, "category_substitution")
        model = default_config.get("model", SMALL_MODEL)

    # 1. Shuffle
    shuffled_puzzle, _, _, shuffled_reasoning = transform_condition_shuffle(puzzle, original_reasoning)

    # 2. Substitute
    # This uses the 'category_substitution' prompts (handled inside transform_category_substitution)
    # Pass None to let it fetch its own prompt config
    subst_puzzle, subst_solution, subst_plan, subst_reasoning = transform_category_substitution(
        shuffled_puzzle, solution, shuffled_reasoning, model, temperature, level, None
    )

    # 3. Paraphrase
    # This uses the 'paraphrase' prompts. transform_paraphrase knows how to fetch them by level.
    # Note: We pass the same model. If paraphrase needs a different model, we might need more complex logic,
    # but using the same model (e.g. Sonnet 3.5) for both steps is usually fine and efficient.
    final_puzzle, _, _, final_reasoning = transform_paraphrase(
        subst_puzzle,
        subst_reasoning,
        model,
        temperature,
        level,
        None,  # Let it fetch its own prompt config
    )

    return final_puzzle, subst_solution, subst_plan, final_reasoning


def transform_paraphrase(
    puzzle: str,
    original_reasoning: str = "",
    model: str = SMALL_MODEL,
    temperature: float | None = None,
    level: int = 2,
    prompt_template: str | dict | None = None,
) -> tuple[str, None, None, str]:
    """Transform puzzle by paraphrasing the text while preserving logic.

    Args:
        puzzle: Original puzzle text
        original_reasoning: Original reasoning text (optional)
        model: LLM model to use
        temperature: Temperature for generation
        level: SD level
        prompt_template: Optional explicit prompt configuration

    Returns:
        Tuple of (transformed_puzzle, None, None, sd_reasoning)
    """
    if not model or model == "none":
        default_config = get_variant_config("zebralogic", level, "paraphrase")
        model = default_config.get("model", SMALL_MODEL)

    # Get prompts
    if isinstance(prompt_template, dict) and "user" in prompt_template and prompt_template.get("user"):
        config = prompt_template
    else:
        config = get_variant_config("zebralogic", level, "paraphrase")

    system_prompt = config.get("system", "")
    user_template = config.get("user", "")

    # Validate we got a valid user template
    if not user_template:
        raise ValueError(f"Paraphrase prompt configuration missing 'user' template for level {level}")

    # Format user prompt - add reasoning section if reasoning is provided
    user_prompt = user_template.format(puzzle=puzzle)

    if original_reasoning:
        # Add reasoning section and request JSON output
        reasoning_section = f"\n\nOriginal Reasoning:\n{original_reasoning}\n"
        output_format = """
Output your response as JSON with this structure:
{{
  "puzzle": "[rewritten puzzle text]",
  "reasoning": "[rewritten reasoning text]"
}}"""
        user_prompt = user_prompt + reasoning_section + output_format

    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    def _generate():
        kwargs = {
            "model": model,
            "messages": messages,
            "reasoning_effort": "low",
            "max_completion_tokens": 8000,
        }

        response = get_client().chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    response = retry_with_backoff(_generate)

    # Parse response based on whether reasoning was provided
    if original_reasoning:
        sd_text, sd_reasoning = _parse_paraphrase_response(response)
    else:
        sd_text, sd_reasoning = response, ""

    return sd_text, None, None, sd_reasoning
