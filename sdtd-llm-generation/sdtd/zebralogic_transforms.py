"""ZebraLogic-specific transformation functions."""

import json
import re
import random

from sdtd.generate import retry_with_backoff
from sdtd.utils import get_client, get_variant_config


def _get_prompts(prompt_template: str | dict | None, level: int = 2) -> tuple[dict, dict]:
    """Extract step1 and step2 prompts from config or use defaults.

    Args:
        prompt_template: Optional config override
        level: Level number to load default config from (if template not provided)

    Returns:
        Tuple of (step1_config, step2_config)
    """
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
) -> str:
    """Apply substitution plan to puzzle text using LLM.

    Args:
        puzzle: Original puzzle text
        substitution_plan: Plan mapping old values to new values
        model: LLM model to use
        prompt_config: Configuration for step 2
        temperature: Temperature for generation

    Returns:
        Transformed puzzle text
    """
    plan_json = json.dumps(substitution_plan, indent=2)

    system_prompt = prompt_config.get("system", "")
    user_template = prompt_config.get("user", "")

    # Format user prompt
    user_prompt = user_template.format(puzzle=puzzle, plan_json=plan_json)

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

    return retry_with_backoff(_generate)


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
    model: str = "helicone/claude-4.5-haiku",
    temperature: float | None = None,
    level: int = 2,
    prompt_template: str | dict | None = None,
) -> tuple[str, dict, dict]:
    """Transform puzzle by substituting values (e.g., colors, foods) using 2-step process.

    1. Generate substitution plan (LLM)
    2. Apply plan to puzzle (LLM)
    3. Apply plan to solution (Python)

    Args:
        puzzle: Original puzzle text
        solution: Solution dict
        model: LLM model to use (default: helicone/claude-4.5-haiku)
        temperature: Temperature for generation
        level: SD level to load prompts from
        prompt_template: Optional explicit prompt configuration

    Returns:
        Tuple of (transformed_puzzle, transformed_solution, substitution_plan)
    """
    # If model is "none" or empty, try to get it from default config
    if not model or model == "none":
        default_config = get_variant_config("zebralogic", level, "category_substitution")
        model = default_config.get("model", "helicone/claude-4.5-haiku")

    # Get prompt configurations
    step1_config, step2_config = _get_prompts(prompt_template, level)

    # Step 1: Generate substitution plan
    substitution_plan = _generate_substitution_plan(puzzle, solution, model, step1_config, temperature)

    # Step 2: Apply to puzzle text
    transformed_puzzle = _apply_substitution_to_puzzle(puzzle, substitution_plan, model, step2_config, temperature)

    # Step 3: Apply to solution (programmatically)
    transformed_solution = _apply_substitution_to_solution(solution, substitution_plan)

    return transformed_puzzle, transformed_solution, substitution_plan


def transform_condition_shuffle(puzzle: str) -> str:
    """Transform puzzle by shuffling the numbered conditions and renumbering them.

    This is a non-LLM transformation that preserves the puzzle structure.

    Args:
        puzzle: Original puzzle text

    Returns:
        Transformed puzzle with shuffled conditions
    """
    # Split puzzle into prefix (before clues) and clues section
    clues_match = re.search(r"## Clues?:?\s*\n", puzzle, re.IGNORECASE)
    if not clues_match:
        # Try alternative patterns
        clues_match = re.search(r"# Clues?:?\s*\n", puzzle, re.IGNORECASE)

    if not clues_match:
        # If no clues section found, return original
        return puzzle

    prefix = puzzle[: clues_match.end()]
    clues_section = puzzle[clues_match.end() :]

    # Extract numbered conditions
    # Pattern: number followed by period, then the condition (handles multi-line conditions)
    # Match until next numbered condition or end of string
    condition_pattern = r"(\d+)\.\s*((?:[^\n]+(?:\n(?!\d+\.)[^\n]+)*))"
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
    remaining_text = clues_section[last_match.end() :].lstrip()

    transformed_puzzle = prefix + new_clues_section
    if remaining_text:
        transformed_puzzle += "\n" + remaining_text

    return transformed_puzzle


def transform_shuffle_and_substitute(
    puzzle: str,
    solution: dict,
    model: str = "helicone/claude-4.5-haiku",
    temperature: float | None = None,
    level: int = 2,
    prompt_template: str | dict | None = None,
) -> tuple[str, dict, dict]:
    """Combine condition shuffling and category substitution transformations.

    First shuffles conditions, then applies category substitution.

    Args:
        puzzle: Original puzzle text
        solution: Solution dict
        model: LLM model to use
        temperature: Temperature for generation
        level: SD level
        prompt_template: Optional explicit prompt config (passed to category subst)

    Returns:
        Tuple of (transformed_puzzle, transformed_solution, substitution_plan)
    """
    # If model is "none" or empty, try to get it from default config
    if not model or model == "none":
        default_config = get_variant_config("zebralogic", level, "category_substitution")
        model = default_config.get("model", "helicone/claude-4.5-haiku")

    # First, shuffle conditions
    shuffled_puzzle = transform_condition_shuffle(puzzle)

    # Then, apply category substitution
    # Note: we pass the level so it can fetch the correct prompts if prompt_template is missing
    transformed_puzzle, transformed_solution, substitution_plan = transform_category_substitution(
        shuffled_puzzle, solution, model, temperature, level, prompt_template
    )

    return transformed_puzzle, transformed_solution, substitution_plan


def transform_shuffle_and_paraphrase(
    puzzle: str,
    solution: dict,
    model: str = "helicone/claude-4.5-haiku",
    temperature: float | None = None,
    level: int = 2,
    prompt_template: str | dict | None = None,
) -> tuple[str, dict, None]:
    """Combine condition shuffling and paraphrasing.

    First shuffles conditions, then paraphrases the entire text.

    Args:
        puzzle: Original puzzle text
        solution: Solution dict (unchanged)
        model: LLM model to use
        temperature: Temperature
        level: SD level
        prompt_template: Optional explicit prompt config

    Returns:
        Tuple of (transformed_puzzle, transformed_solution, None)
    """
    # If model is "none" or empty, try to get it from default config
    if not model or model == "none":
        default_config = get_variant_config("zebralogic", level, "paraphrase")
        model = default_config.get("model", "helicone/claude-4.5-haiku")

    # First, shuffle conditions
    shuffled_puzzle = transform_condition_shuffle(puzzle)

    # Then, paraphrase
    # Note: paraphrase doesn't return a plan or change the solution
    transformed_puzzle = transform_paraphrase(shuffled_puzzle, model, temperature, level, prompt_template)

    return transformed_puzzle, solution, None


def transform_shuffle_and_substitute_and_paraphrase(
    puzzle: str,
    solution: dict,
    model: str = "helicone/claude-4.5-haiku",
    temperature: float | None = None,
    level: int = 2,
    prompt_template: str | dict | None = None,
) -> tuple[str, dict, dict]:
    """Combine shuffling, substitution, AND paraphrasing.

    1. Shuffle conditions
    2. Substitute categories (returns transformed puzzle, solution, and plan)
    3. Paraphrase the substituted puzzle

    Args:
        puzzle: Original puzzle text
        solution: Solution dict
        model: LLM model to use
        temperature: Temperature
        level: SD level
        prompt_template: Optional explicit prompt config

    Returns:
        Tuple of (transformed_puzzle, transformed_solution, substitution_plan)
    """
    # We need models for both substitution and paraphrase.
    # Currently we use the same model passed in, or default from configs.
    # For simplicity, we'll try to resolve a model if not provided, prioritizing category_substitution default.

    if not model or model == "none":
        default_config = get_variant_config("zebralogic", level, "category_substitution")
        model = default_config.get("model", "helicone/claude-4.5-haiku")

    # 1. Shuffle
    shuffled_puzzle = transform_condition_shuffle(puzzle)

    # 2. Substitute
    # This uses the 'category_substitution' prompts (handled inside transform_category_substitution)
    subst_puzzle, subst_solution, subst_plan = transform_category_substitution(
        shuffled_puzzle, solution, model, temperature, level, prompt_template
    )

    # 3. Paraphrase
    # This uses the 'paraphrase' prompts. transform_paraphrase knows how to fetch them by level.
    # Note: We pass the same model. If paraphrase needs a different model, we might need more complex logic,
    # but using the same model (e.g. Sonnet 3.5) for both steps is usually fine and efficient.
    final_puzzle = transform_paraphrase(
        subst_puzzle,
        model,
        temperature,
        level,
        None,  # Let it fetch its own prompt config
    )

    return final_puzzle, subst_solution, subst_plan


def transform_paraphrase(
    puzzle: str,
    model: str = "helicone/claude-4.5-haiku",
    temperature: float | None = None,
    level: int = 2,
    prompt_template: str | dict | None = None,
) -> str:
    """Transform puzzle by paraphrasing the text while preserving logic.

    Args:
        puzzle: Original puzzle text
        model: LLM model to use
        temperature: Temperature for generation
        level: SD level
        prompt_template: Optional explicit prompt configuration

    Returns:
        Transformed puzzle text
    """
    if not model or model == "none":
        default_config = get_variant_config("zebralogic", level, "paraphrase")
        model = default_config.get("model", "helicone/claude-4.5-haiku")

    # Get prompts
    if isinstance(prompt_template, dict):
        config = prompt_template
    else:
        config = get_variant_config("zebralogic", level, "paraphrase")

    system_prompt = config.get("system", "")
    user_template = config.get("user", "")

    # Format user prompt
    user_prompt = user_template.format(puzzle=puzzle)

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

    return retry_with_backoff(_generate)
