"""
Level 0 semantic duplicates for Object Placements: story-only regeneration (no tree changes).

Goal:
- Regenerate the *context/story text* for existing Object Placements problems while keeping:
  - questions
  - answer indices
  - choices
  - intermediate_trees
  - intermediate_data (events/beliefs/actual_locs)
  unchanged.

Rationale:
- This mirrors `level0_generate_duplicates_with_no_tree_changes.py` for murder mystery:
  same reasoning structure -> different surface story text.

Reproducibility:
- Uses deterministic seeds per (sample_idx, variant_idx):
  seed = base_seed + sample_idx * 1000 + variant_idx
- Concurrent execution does not affect determinism (each variant has its own model instance).
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from src
SCRIPT_DIR = Path(__file__).parent.absolute()
MUSR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MUSR_DIR))

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(MUSR_DIR / ".env")

import argparse
import json
import random
import threading
import traceback
import re
from copy import deepcopy
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any, Set

from src import cache
from src.model import OpenAIModel
from src.logic_tree.tree import LogicTree, LogicNode
from src.utils.paths import OUTPUT_FOLDER
from src.utils.json_io import atomic_json_dump, load_json_array_tolerant
from src.dataset_types.object_placements_dataset import ObjectPlacementsDataset


# Thread-safe printing
_print_lock = threading.Lock()


def thread_safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def _respect_article(item: str, people: List[str]) -> str:
    if any([item.startswith(name) for name in people]):
        return item
    return f"the {item}"


def _respect_plural(item: str) -> str:
    if item.endswith("s"):
        return f"{item} are"
    return f"{item} is"


def _extract_first_question(sample: Dict) -> Dict:
    questions = sample.get("questions", [])
    if not questions:
        raise ValueError("Sample missing 'questions'.")
    return questions[0]


def _extract_tree_json(sample: Dict) -> Dict:
    q0 = _extract_first_question(sample)
    trees = q0.get("intermediate_trees", [])
    if not isinstance(trees, list) or not trees:
        raise ValueError("Sample missing 'intermediate_trees' on first question.")
    # For object placements, all questions share the same single tree.
    return trees[0]


def _extract_intermediate_state(sample: Dict) -> Dict[str, Any]:
    q0 = _extract_first_question(sample)
    inter = q0.get("intermediate_data", [])
    if not isinstance(inter, list) or not inter:
        raise ValueError("Sample missing 'intermediate_data' on first question.")
    state = inter[0]
    if not isinstance(state, dict):
        raise ValueError("Expected intermediate_data[0] to be a dict.")
    return state


def _extract_choices_locations(sample: Dict) -> List[str]:
    q0 = _extract_first_question(sample)
    choices = q0.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise ValueError("Sample missing 'choices' on first question.")
    return choices


def _extract_people_from_opening_scene(opening_node: LogicNode) -> List[str]:
    people: List[str] = []
    for c in opening_node.children:
        v = (c.value or "").strip()
        # Expected: "{Person} sees ..."
        if " sees " in v:
            person = v.split(" sees ", 1)[0].strip()
            if person:
                people.append(person)
    return sorted(list(set(people)))


def _flatten_obs_children_for_event(event_node: LogicNode) -> List[LogicNode]:
    """
    Matches the logic in `create_object_placements.py`:
    - For observation nodes with children, include their children (so we don't include "did not see ..." directly).
    - Otherwise include the observation node itself as a last resort.
    """
    flattened: List[LogicNode] = []
    for p in [x for x in event_node.children if "when moving" not in (x.value or "").lower()]:
        if len(p.children) > 0:
            flattened.extend(p.children)
        else:
            flattened.append(p)
    return flattened


def _facts_str__items_locations(actual_locs0: Dict[str, str], people: List[str]) -> str:
    return "\n".join(
        [
            f"- {_respect_article(_respect_plural(item), people)} at {_respect_article(loc, people)}"
            for item, loc in actual_locs0.items()
        ]
    )


def _facts_str__bullets(lines: List[str]) -> str:
    return "\n".join([f"- {x}" for x in lines if x and str(x).strip()])


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Best-effort extraction of a JSON object from a model response.

    We ask models to output JSON only, but in practice they may wrap it in prose.
    """
    if not text:
        raise ValueError("Empty response; expected JSON object.")
    s = text.strip()
    # Fast path: direct JSON.
    if s.startswith("{") and s.endswith("}"):
        return json.loads(s)
    # Fallback: find first {...} span.
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate JSON object in response.")
    return json.loads(s[start : end + 1])


def reconstruct_people_data_and_outline(
    original_context: str,
    people: List[str],
    extraction_model: OpenAIModel,
) -> Tuple[List[Dict[str, str]], str]:
    """
    Reconstruct the generation scaffolding that the original generator used:
    - people_data: [{name, role, motivation}, ...] length == 3
    - story_desc: 1 short paragraph outline

    This uses the existing story as source of truth to maximize in-distribution parity.
    """
    creator = ObjectPlacementsDataset()

    people_list = ", ".join(people)
    # We want the reconstructed outline to be usable as the "Description" input to the original
    # `create_object_placements.py` chapter prompts. In the original pipeline, this description
    # is produced *before* the move sequence is narrated; in our regen pipeline we must prevent
    # the outline from leaking future move details.
    # We only want to ban *move/epistemic leakage*, not common English prepositions.
    # Note: do NOT ban "to/from" here — that makes extraction nearly impossible and conflicts with our own examples.
    banned_outline_markers = [
        " moves ",
        " moved ",
        " relocate",
        " relocated",
        "unbeknownst",
        "did not see",
        "didn't see",
    ]
    # Few-shot extraction prompt (real examples) to improve format stability and avoid move leakage.
    # We accept the user's field names (characters/scenario_description/story_outline) and map them to our internal
    # (people_data/story_desc) representation for downstream parity prompts.
    prompt = f"""
You are analyzing stories to extract structured metadata. Given a completed story, extract character information, the original scenario description, and a story outline.

Return ONLY valid JSON (no markdown, no commentary).

Known people (names must match exactly, choose exactly these three and no others):
[{people_list}]

CRITICAL constraints for story_outline:
- Write a DETAILED, multi-sentence scenario description (4-6 sentences).
- The outline should mention ALL THREE characters and what they're doing together.
- Include ONE central activity or goal that connects the characters (e.g., "making coffee for a customer", "recording an album").
- Include character backgrounds, emotional states, and relationship dynamics.
- Do NOT mention any item movements/relocations (no "moved/relocated/from/to").
- Do NOT mention who moved what, when, or where items end up.
- Do NOT mention specific item locations (e.g., "on the table", "in the fridge") - those are provided separately.
- Avoid epistemic phrasing like "Unbeknownst" or "did not see".
- Match the rich, narrative style of the example outputs below.

=== EXAMPLE 1 ===
(Source: Opening scene example from create_object_placements.py prompt template, motivations enhanced)

STORY:

Alex and Carol were having a peaceful evening hanging out. Both getting a bit peckish, they decided to eat some ramen, which Alex had been practicing making for awhile now. Everyone knew Alex was the "Chef Friend", meaning that he was always cooking something delicious up. In fact, that's why a hungry friend, who showed up unannounced, Joshua decided to show up. All three of them noticed that the pans were already on the stove, perfect for ramen making! The kitchen knife was on the table, and the noodles were in the fridge.

OUTPUT:

{{
  "characters": [
    {{
      "name": "Alex",
      "role": "Aspiring home chef",
      "motivation": "Alex has been practicing his ramen-making skills and wants to impress his roommate and friends with a delicious homemade meal."
    }},
    {{
      "name": "Carol",
      "role": "Alex's roommate",
      "motivation": "Carol is spending a relaxing evening at home and is looking forward to enjoying Alex's cooking, as she knows he always makes something delicious."
    }},
    {{
      "name": "Joshua",
      "role": "Unexpected visitor",
      "motivation": "Joshua heard about Alex's cooking skills and decided to drop by unannounced, hoping to get a taste of whatever Alex was making."
    }}
  ],
  "scenario_description": "Alex is making Ramen and needs the noodles to cook.",
  "story_outline": "Alex and Carol were having a peaceful evening hanging out when they both got a bit peckish and decided to eat some ramen, which Alex had been practicing making for a while now. Everyone knew Alex was the 'Chef Friend', meaning that he was always cooking something delicious up. In fact, that's why a hungry friend named Joshua decided to show up unannounced. Carol was Alex's supportive roommate who was also hungry and looking forward to the meal. The kitchen was well-stocked and ready for their cooking session."
}}

=== EXAMPLE 2 ===
(Source: Scaffolding generation example from create_object_placements.py prompt template)

STORY:

Luis was having a super hard week with his paper deadline approaching and with all his experiments now failing he was in desperate need of his favorite cup of jo, with a dash of almond milk. He ordered it from Sarah who is about to start making it. Sarah is a skilled barista and has worked there for awhile, she loves making her customers feel welcomed. However, John is new and constantly fumbling around with things (which is expected since he's new). To keep him busy, Sarah put him on cleaning duty.

OUTPUT:

{{
  "characters": [
    {{
      "name": "Sarah",
      "role": "The Barista",
      "motivation": "Sarah wants to make Luis an almond milk coffee."
    }},
    {{
      "name": "Luis",
      "role": "A customer",
      "motivation": "Luis is having a rough week with deadlines looming over him, so he wanted his favorite coffee with almond milk."
    }},
    {{
      "name": "John",
      "role": "A cafe worker",
      "motivation": "John is having his first day working at the coffee cafe, he is working hard to make sure all the tables are clean for the customers."
    }}
  ],
  "scenario_description": "Sarah is making coffee at her work, she wants to use almond milk for her customer.",
  "story_outline": "Luis was having a super hard week with his paper deadline approaching and with all his experiments now failing, he was in desperate need of his favorite cup of jo, with a dash of almond milk. He ordered it from Sarah who is about to start making it. Sarah is a skilled barista and has worked there for a while, she loves making her customers feel welcomed. However, John is new and constantly fumbling around with things, which is expected since he's new. To keep him busy, Sarah put him on cleaning duty."
}}

=== EXAMPLE 3 ===
(Source: Real model output from create_object_placements.py run on 2025-12-17, custom_object_placements_scaffolding.json)

STORY:

Dave was diligently preparing for his critical meeting, involving the crucial software patches he stored in his encrypted USB drive. Normally, he kept it snuggly secured in his laptop bag, but today, for added safety, it resided in the safe. The laptop bag, normally his constant buddy, was sat in the conference room in readiness for the meeting. In the same corporate building, Dana, the IT professional, was waiting. After the meeting, Dana had a pivotal role to play, taking the encrypted USB patches to update specific systems, a task she was more than ready for. Carl, the ever-thorough janitor, was making his rounds. A personal point of pride, his dedication kept the office tidy and everything in its proper place. All of them were keenly aware of the placement of each item.

OUTPUT:

{{
  "characters": [
    {{
      "name": "Dave",
      "role": "Systems Analyst",
      "motivation": "Dave needs to prepare for an important meeting involving software patches and he wants to ensure everything goes smoothly."
    }},
    {{
      "name": "Dana",
      "role": "IT professional",
      "motivation": "Dana, who works in the same corporate building as Dave, needs to update specific systems with the software patches."
    }},
    {{
      "name": "Carl",
      "role": "Office Janitor",
      "motivation": "Carl is a responsible office janitor who takes pride in keeping the office tidy and organized."
    }}
  ],
  "scenario_description": "Dave, a systems analyst, always places his encrypted USB drive, which carries critical software patches, in his laptop bag. He needs it for his meeting the next day.",
  "story_outline": "Dave has a crucial meeting involving software patches which he carries in his encrypted USB. He usually stores his USB in his laptop bag for safe-keeping. Dana is the IT professional who, after Dave's meeting, will have to install the patches on the appropriate systems. Carl, the ever diligent janitor, has been entrusted with the task of keeping the office clean and tidy, and he ensures that all office equipment is neatly arranged."
}}

=== YOUR TURN ===

STORY:

{original_context}

OUTPUT:
    """.strip()

    retries = 0
    last_err = None
    while retries < 3:
        retries += 1
        raw, _ = creator.inference(prompt, extraction_model, temperature=0.2)
        try:
            obj = _extract_json_object(raw)
            # Accept either our internal field names or the few-shot field names.
            pd = obj.get("people_data", obj.get("characters"))
            sd = obj.get("story_desc", obj.get("story_outline"))
            if not isinstance(pd, list) or len(pd) != 3:
                raise ValueError("people_data/characters must be a list of length 3")
            if not isinstance(sd, str) or not sd.strip():
                raise ValueError("story_desc/story_outline must be a non-empty string")

            normalized: List[Dict[str, str]] = []
            seen = set()
            for entry in pd:
                if not isinstance(entry, dict):
                    raise ValueError("people_data entries must be objects")
                name = str(entry.get("name", "")).strip()
                role = str(entry.get("role", "")).strip()
                motivation = str(entry.get("motivation", "")).strip()
                if name not in people:
                    raise ValueError(f"Invalid name in people_data: '{name}'")
                if name in seen:
                    raise ValueError(f"Duplicate person in people_data: '{name}'")
                if not role or not motivation:
                    raise ValueError(f"Missing role/motivation for '{name}'")
                seen.add(name)
                normalized.append({"name": name, "role": role, "motivation": motivation})

            # Ensure same ordering as `people` list for downstream formatting stability.
            normalized.sort(key=lambda x: people.index(x["name"]))
            sd_clean = sd.strip()
            sd_l = sd_clean.lower()
            # Enforce that the outline doesn't leak move details; if it does, retry with stronger instruction.
            if any(m in sd_l for m in banned_outline_markers):
                raise ValueError("story_desc leaks move/epistemic details; must be high-level setup only")
            return normalized, sd_clean
        except Exception as e:
            last_err = e
            prompt = (
                prompt
                + f"\n\nYour last output was invalid JSON or did not match schema. Error: {e}\n"
                + "Return ONLY valid JSON matching the schema.\n"
                + "Reminder: story_desc must be a high-level setup only; do not mention moves/relocations, and avoid epistemic phrasing like 'Unbeknownst'/'did not see'.\n"
            )

    raise ValueError(f"Failed to reconstruct people_data/story_desc after retries: {last_err}")


def _parse_move_event_from_string(move_line: str, people: List[str], items: List[str], locations: List[str]) -> Tuple[str, str, str]:
    """
    Parse 'X moves the ITEM to the LOCATION.' style lines.

    Returns: (mover, item, to_location)
    """
    s = (move_line or "").strip().rstrip(".")
    # mover: first token before ' moves '
    if " moves " not in s:
        raise ValueError(f"Not a move line: {move_line}")
    mover = s.split(" moves ", 1)[0].strip()
    if mover not in people:
        # Some lines can start with whitespace; be defensive but strict.
        raise ValueError(f"Unrecognized mover '{mover}' in move line: {move_line}")

    # Destination location: try to match a known location at end (case-insensitive).
    lower = s.lower()
    to_loc = None
    for loc in sorted(locations, key=lambda x: len(x), reverse=True):
        candidate = loc.strip()
        if not candidate:
            continue
        if lower.endswith(candidate.lower()):
            to_loc = candidate
            break
    if not to_loc:
        # Fallback: split on ' to ' and take tail
        to_loc = s.split(" to ", 1)[-1].strip()

    # Item: choose which known item occurs in the string.
    item = None
    for it in sorted(items, key=lambda x: len(x), reverse=True):
        if it.lower() in lower:
            item = it
            break
    if not item:
        # Fallback: try regex between 'moves' and 'to'
        m = re.search(r"\bmoves\b\s+(?:the\s+)?(.+?)\s+\bto\b\s+", s, flags=re.IGNORECASE)
        if m:
            item = m.group(1).strip()

    if not item:
        raise ValueError(f"Could not parse item from move line: {move_line}")

    return mover, item, to_loc


def _ensure_because_clause(event_value: str, fallback_reason: str) -> str:
    """
    Ensure the move event string contains a 'Because,' clause (to match the original generator style).
    Does not mutate the tree; returns a new string.
    """
    v = (event_value or "").strip()
    if "because" in v.lower():
        return v
    reason = (fallback_reason or "").strip()
    if not reason:
        return v
    if not reason.lower().startswith("because"):
        reason = "Because, " + reason
    # Ensure punctuation between event and because.
    if not v.endswith("."):
        v += "."
    return f"{v} {reason}".strip()


def reconstruct_move_reason_from_context(
    original_context: str,
    mover: str,
    item: str,
    to_loc: str,
    other_item: Optional[str],
    locations: List[str],
    extraction_model: OpenAIModel,
) -> str:
    """
    Create a 1-sentence reason for a move, intended only as a fallback when the tree lacks a Because clause.

    Constraints:
    - Must not mention the other key item (if provided).
    - Must not introduce new locations outside the known list.
    - Must be a plausible personal motivation for the mover.
    """
    creator = ObjectPlacementsDataset()
    forbidden_items = [other_item] if other_item else []
    loc_list = ", ".join(locations)

    prompt = f"""
Return ONLY one sentence (no quotes, no bullet points).

Write a plausible reason for this move:
- Mover: {mover}
- Item: {item}
- Destination location: {to_loc}

Rules:
- Do NOT mention any other key items: {", ".join([x for x in forbidden_items if x])}
- Do NOT mention any locations outside this list: [{loc_list}]
- The reason should be consistent with the story below, but should not depend on another person's actions.

Story:
{original_context}
    """.strip()

    out, _ = creator.inference(prompt, extraction_model, temperature=0.3)
    sentence = out.strip().replace("\n", " ")
    # Minimal safety: strip leading/trailing quotes
    sentence = sentence.strip('"').strip("'").strip()
    return sentence


def _validate_move_coverage(story: str, moves: List[Tuple[str, str, str]]) -> List[str]:
    """
    Best-effort validation: each move's mover/item/location should appear in story text.
    Returns list of missing descriptors (empty means OK).
    """
    s = (story or "").lower()
    missing = []
    for mover, item, loc in moves:
        if mover.lower() not in s or item.lower() not in s or loc.lower() not in s:
            missing.append(f"{mover} / {item} / {loc}")
    return missing


def regenerate_object_placements_story_scaffolded_parity(
    tree: LogicTree,
    actual_locs: List[Dict[str, str]],
    events: List[List[str]],
    items: List[str],
    people: List[str],
    locations: List[str],
    original_context: str,
    generation_model: OpenAIModel,
    extraction_model: OpenAIModel,
    local_random: random.Random,
) -> Tuple[str, Dict[str, Any]]:
    """
    Scaffolded-parity Level 0 regeneration:
    - Reconstruct people_data + story_desc from original context
    - Keep instance (tree + intermediate_data) unchanged
    - Regenerate story using prompts that mirror the original chaptered workflow

    Returns: (story_text, debug_info)
    """
    creator = ObjectPlacementsDataset()

    people_data, story_desc = reconstruct_people_data_and_outline(original_context, people, extraction_model)

    # Parse the move list from intermediate_data events, which are canonical program strings.
    # events[0] is opening scene; events[1:] each begins with the move string.
    parsed_moves: List[Tuple[str, str, str]] = []
    for e in events[1:]:
        if not e:
            continue
        move_line = e[0]
        mover, item, to_loc = _parse_move_event_from_string(move_line, people, items, locations)
        parsed_moves.append((mover, item, to_loc))

    # Generate story chapters similar to create_object_placements.py
    story_so_far = ""
    debug = {
        "people_data": people_data,
        "story_desc": story_desc,
        "parsed_moves": parsed_moves,
    }

    # Build quick lookup for per-person info
    by_name = {x["name"]: x for x in people_data}

    # ---------------------------------------------------------------------
    # Pipeline-parity prompts copied (nearly verbatim) from:
    # `MuSR/musr_dataset_scripts/create_object_placements.py`
    # We keep the first-pass prompt identical; only retries append corrective feedback.
    # ---------------------------------------------------------------------

    def _has_forbidden_restart(text: str, story_prefix: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True
        # Guard against the model repeating large chunks or restarting the story.
        if "\n\n" in t:
            return True
        # If it includes the start of the story so far, it's almost certainly repeating.
        sp = (story_prefix or "").strip()
        if sp and len(sp) >= 80 and sp[:80] in t:
            return True
        return False

    def _contains_banned_move_language(text: str) -> bool:
        s = (text or "").lower()
        banned = [
            "unbeknownst",
            "didn't see",
            "did not see",
            "couldn't see",
            "could not see",
            "wasn't able to see",
            "was not able to see",
        ]
        return any(b in s for b in banned)

    # Opening scene (verbatim template + example)
    opening_retry = 0
    opening_output = ""
    facts_str = _facts_str__items_locations(actual_locs[0], people)
    while opening_retry < 3:
        opening_retry += 1
        opening_prompt = f"""
Create an opening scene description for a story.  It will be short.  Only write about the objects we list out and their location.  Your story MUST include each item and their location from the list.  Your story also MUST indicate that all the people we give you saw the location of all these items!

You may use the description to infer the correct scenery to describe, but are only allowed to talk about the facts presented in the list.

You must state that everyone knows where everything is, "They were all aware of each items location" or something like that is a safe way to ensure this condition is met.  Try to make this coherent with the story though.  For example, if someone doesn't know the exact location you could say "Everyone was aware that the item was somewhere in the location, and they definitely all knew that the other item was in the other location", or something like this.

Here is an example.

Description: Alex is making Ramen and needs the noodles to cook.

Items and Locations:
- The pans are at the stove.
- The noodles are at the fridge.
- The kitchen knife is at the table.

Character 1:
Name: Alex
Role in story: A want to be chef
Motivation in story: To make a bowl of ramen.

Character 2:
Name: Carol
Role in story: the roommate
Motivation in story: Hanging out with Alex, she is also hungry.

Character 3:
Name: Joshua
Role in story: A random visitor
Motivation in story: Joshua was a friend of Alex but showed up unannounced and hungry.

Output: Alex and Carol were having a peaceful evening hanging out.  Both getting a bit peckish, they decided to eat some ramen, which Alex had been practicing making for awhile now. Everyone knew Alex was the "Chef Friend", meaning that he was always cooking something delicious up. In fact, that's why a hungry friend, who showed up unannounced, Joshua decided to show up.  All three of them noticed that the pans were already on the stove, perfect for ramen making! The kitchen knife was on the table, and the noodles were in the fridge.

Your turn.

Description: {story_desc}

Items and Locations:
{facts_str}

Character 1:
Name: {people_data[0]['name']}
Role in story: {people_data[0]["role"]}
Motivation in story: {people_data[0]["motivation"]}

Character 2:
Name: {people_data[1]['name']}
Role in story: {people_data[1]["role"]}
Motivation in story: {people_data[1]["motivation"]}


Character 3:
Name: {people_data[2]['name']}
Role in story: {people_data[2]["role"]}
Motivation in story: {people_data[2]["motivation"]}

Output:
                                """.strip()

        opening_output, _ = creator.inference(opening_prompt, generation_model)
        opening_output = (opening_output or "").strip()

        # Minimal parity-safe validation: ensure every item/location is mentioned and global awareness is stated.
        lower_open = opening_output.lower()
        missing_pairs = []
        for it, loc in actual_locs[0].items():
            if it.lower() not in lower_open or loc.lower() not in lower_open:
                missing_pairs.append(f"{it}@{loc}")
        has_awareness = any(x in lower_open for x in ["everyone", "all three", "they were all aware", "all of them", "they all"])
        # Retry-only guard: opening should not mention move events (the original prompt intends this).
        # We make this strict to prevent "jump-ahead" openings that front-load later moves.
        opening_mentions_move = any(
            w in lower_open for w in [" moves ", " moved ", " relocate", " relocated"]
        )
        if missing_pairs or not has_awareness or opening_mentions_move:
            # Retry by appending feedback while keeping the base prompt intact on first pass.
            opening_prompt += (
                f"\n\nOne of your last outputs was this:\n\n{opening_output}\n\n"
                f"This is incorrect because it is missing one or more required item/location mentions ({missing_pairs}) "
                f"and/or it does not clearly state that all people know where everything is, "
                f"and/or it incorrectly mentions later move events (no 'moved/relocated').\n"
                "Try again. Return only the opening scene (no extra commentary).\n\nOutput:\n"
            )
            opening_output = ""
            continue
        break

    if not opening_output:
        # Fail safe: do not return empty story.
        opening_output = f"{story_desc}".strip()

    story_so_far += opening_output.strip()

    # Move + observation chapters, aligned with the tree order (opening scene node already handled).
    # We'll walk tree children in order; for each move event node, use its event string and facts.
    move_idx = -1
    for __n in tree.nodes[0].children:
        n = deepcopy(__n)
        if (n.value or "").strip().lower() == "opening scene":
            continue
        move_idx += 1

        # Move node value: ensure it contains a Because clause (fallback only if missing).
        # In practice, all valid samples have "Because" in the tree, so this fallback rarely triggers.
        event_value = (n.value or "").strip()
        if "because" not in event_value.lower():
            # Only call expensive extraction if we need a fallback reason
            mover, item, to_loc = parsed_moves[move_idx] if move_idx < len(parsed_moves) else (None, None, None)
            other_item = None
            if item and len(items) == 2:
                other_item = [x for x in items if x != item][0]
            fallback_reason = ""
            if mover and item and to_loc:
                fallback_reason = reconstruct_move_reason_from_context(
                    original_context, mover, item, to_loc, other_item, locations, extraction_model
                )
            event_value = _ensure_because_clause(event_value, fallback_reason)

        # Include "when moving" child facts
        when_moving_children = [x for x in n.children if "when moving" in (x.value or "").lower()]
        move_facts = [event_value, *[x.value for x in when_moving_children if x.value]]
        move_facts_str = _facts_str__bullets(move_facts)

        mover_name = (event_value or "").split(" ", 1)[0].strip()
        mover_info = by_name.get(mover_name, {"name": mover_name, "role": "", "motivation": ""})

        # Move prompt (verbatim template + examples)
        move_retry = 0
        move_output = ""
        while move_retry < 3:
            move_retry += 1
            move_prompt = f"""
You are going to continue our story that we have written by writing a short description of this event that will happen next.  Only write about the move, do not add any additional information.

Never say "someone didn't see something" or infer someones ability to infer where something is.  Never say "Unbeknownst" or anything like this!
Here is an example.

Only write one or two sentences.  It should be a very short continuation.

Description: Timmy was angry at Bob for cheating his way into the job Timmy deserved! So he started throwing away Bobs possessions.

Character:
Name: Timmy
Role in story: A recent graduate who is sharing an apartment.
Motivation in story: Timmy is angry because he interviewed for a job that his roommate got, but only because he cheated.

Event:
- Timmy moves the car keys to the trash bin. Because, Timmy was angry with Bob and wanted to throw away his keys.
- Timmy saw the iphone at the trash bin when moving the car keys.

Output: With an angry thrust, the keys clanked against the tin trash bin.  An unexpected *smack* followed though... curiosity overtaking his anger, Timmy looked in the trash and saw the iphone in there as well.

Here is another example.

Description: Carol had just moved into her new apartment, but, the previous tenant made a huge mess! The landlord wouldn't do anything, so it looks like she has to clean it all up herself.

Character:
Name: Carol
Role in story: Just moved into a new messy apartment.
Motivation in story: Carol wants to clean her new apartment that was left a mess by the previous tenant and has exactly no help from management.

Event:
- Carol moves the noodles to the pantry. Because, Carol was excited to have a clean apartment finally, and the noodles were the last step!

Output: Carol excitingly places the noodles back into the pantry.  What was once thought of as a never ending onslaught of trash and random items finally concluded and the apartment was finally clean again!

Your turn.

Description: {story_desc}

Character:
Name: {mover_info.get('name', mover_name)}
Role in story: {mover_info.get('role', '')}
Motivation in story: {mover_info.get('motivation', '')}

Event:
{move_facts_str}

Output:
{story_so_far}
                """.strip()

            move_output, _ = creator.inference(move_prompt, generation_model)
            move_output = (move_output or "").strip()

            # Parity-safe quality gates (retry-only):
            # - must be short (1-2 sentences), no multi-paragraph restarts
            # - must not contain forbidden phrasing explicitly banned by the original prompt
            if _has_forbidden_restart(move_output, story_so_far) or len(move_output) > 450 or _contains_banned_move_language(move_output):
                move_prompt += (
                    f"\n\nOne of your last outputs was this:\n\n{move_output}\n\n"
                    "This is incorrect because it either repeats earlier story text, is too long, contains multiple paragraphs, "
                    "or uses forbidden phrasing (e.g., 'Unbeknownst' / 'did not see').\n"
                    "Try again. Only write 1-2 short sentences continuing from the end of Output.\n\nOutput:\n"
                    f"{story_so_far}\n"
                )
                move_output = ""
                continue
            break

        if not move_output:
            # Fail-safe: at least state the canonical move line.
            move_output = (parsed_moves and move_idx < len(parsed_moves) and f"{parsed_moves[move_idx][0]} moves the {parsed_moves[move_idx][1]} to the {parsed_moves[move_idx][2]}.") or "The next move happens."

        story_so_far += f"\n\n{move_output}"

        # Observation chapter: use explicit leaf facts from flattened observation supports.
        flattened_obs_children = _flatten_obs_children_for_event(n)
        stree = deepcopy(tree)
        stree.nodes = [n]
        stree.nodes[0].children = flattened_obs_children
        obs_facts = sorted([x.value for x in stree.get_facts() if x.value], key=lambda s: str(s).lower())
        obs_facts_str = _facts_str__bullets(obs_facts)

        items_str = "\n".join([f"- {x}" for x in items])

        # Observation prompt (verbatim template + example + original retry logic)
        obs_retry = 0
        obs_output = ""
        obs_prompt_beginning = f"""
Continue the story we have written so far by writing about the observational facts below. Only write about the facts and do not add new information.  Never say "Someone saw" or "Did not notice" and never indicate if someone sees something, unless the only fact you have is "someone sees X".

Stick to the facts, there will be more information about the story that you can use to set the tone, but you should always use the facts as the main guide for the story.

Never mention the key items in the story:
{items_str}

{"There will be another event after this paragraph, so end this paragraph abruptly sticking only with the facts.  Make no general statements. The last sentence should be something about the facts we listed out only. It should be a complete sentence."
if move_idx < len(tree.nodes[0].children) - 2 else
"This is the end of the story, write a concluding sentence after your paragraph."}

Your story should take place during the most recent move.  So while this is happening:

"{move_output}"

the facts you will be writing about are happening at the same time.

IMPORTANT: Start your paragraph with a concurrent transition phrase like "Meanwhile,", "At the same time,", "Concurrently,", or "In the meantime," to indicate these events are happening simultaneously with the move. Write a full paragraph (3-5 sentences) about what the other characters are doing.

Here is an example.

Description: Jerry, Marry and Timmy are getting ready for the day.  Jerry has a huge meeting that he needs to prep for.  Marry is excited to help Jerry for his meeting and to hear about it later that day.  Timmy was getting ready for his test, but is being a bit inconsiderate to his dad, Jerry, with respect to his big meeting.

Character 1:
Name: Jerry
Role in story: the husband
Motivation in story: Jerry had a huge meeting coming up, one that could decide the fate of his career.

Character 2:
Name: Marry
Role in story: the wife
Motivation in story: Marry is always super supportive and wants the best for her family.

Character 3:
Name: Timmy
Role in story: the son
Motivation in story: Timmy has a huge test coming up in his school which he is nervous for and accidentally makes him a bit inconsiderate to everyone else.

Observational Facts:
- Jerry is cooking breakfast
- The trash bin is not in the kitchen.
- Marry is outside watering her garden.
- Marry has a window into the room with the trash bin.

Output:

Meanwhile, Jerry was hungry before he starts his day, so he was cooking his breakfast. The kitchen turned out to not have the trash bin though. Marry, always with her green thumb, was outside watering her garden. Through a nearby window, she had a clear view into the room with the trash bin. Timmy, nervous about his upcoming test, was pacing around the house, his mind elsewhere.

Your turn.

Description: {story_desc}

Character 1:
Name: {people_data[0]['name']}
Role in story: {people_data[0]["role"]}
Motivation in story: {people_data[0]["motivation"]}

Character 2:
Name: {people_data[1]['name']}
Role in story: {people_data[1]["role"]}
Motivation in story: {people_data[1]["motivation"]}

Character 3:
Name: {people_data[2]['name']}
Role in story: {people_data[2]["role"]}
Motivation in story: {people_data[2]["motivation"]}

Observational Facts:
{obs_facts_str}
                                """.strip()

        output_obs_prompt = f'''
Output:
{story_so_far}
'''

        good_obs = False
        while obs_retry < 3:
            obs_retry += 1
            obs_output, _ = creator.inference(f'{obs_prompt_beginning}\n\n{output_obs_prompt}', generation_model)
            # Original does NOT strip obs_output - preserve natural LLM paragraph structure
            # Only rstrip to remove trailing whitespace
            obs_output = (obs_output or "").rstrip()

            # Retry if key items appear (original behavior - the ONLY validation in create_object_placements.py)
            if any([x.lower() in obs_output.lower() for x in items]):
                obs_prompt_beginning += (
                    f"\n\nOne of your last outputs was this: \n\n{obs_output}\n\n"
                    f"This is incorrect because it mentions one of our key items: \n{items_str}\n\n"
                    "Make sure your next generation does not include mentioning our key items as that can confuse the reader."
                )
                continue

            # Only retry for story restarts/repetition (no length/paragraph constraints - original has none)
            if _has_forbidden_restart(obs_output, story_so_far):
                obs_prompt_beginning += (
                    f"\n\nOne of your last outputs was this: \n\n{obs_output}\n\n"
                    "This is incorrect because it repeats earlier story text. "
                    "Continue from where the story left off.\n"
                )
                continue

            good_obs = True
            break

        if good_obs and obs_output:
            # Add paragraph break before observation to match original story structure
            # (original stories have separate paragraphs for move and observation chapters)
            # The observation prompt now instructs LLM to start with "Meanwhile," etc.
            story_so_far += f'\n\n{obs_output}'

    # Keep event order fixed; local_random reserved for future options.
    _ = local_random
    return story_so_far.strip(), debug


def process_single_sample(
    sample: Dict,
    sample_idx: int,
    num_samples_actual: int,
    num_variants_per_sample: int,
    model_config: Dict,
    base_seed: int,
    variants_to_generate: Optional[List[int]] = None,
    overwrite_context: bool = False,
) -> Tuple[int, List[Dict], float]:
    thread_safe_print(f"\n{'='*80}")
    thread_safe_print(f"Sample {sample_idx + 1}/{num_samples_actual}")
    thread_safe_print(f"{'='*80}")

    tree_json = _extract_tree_json(sample)
    tree = LogicTree.from_json(tree_json)

    state = _extract_intermediate_state(sample)
    events = state.get("events", [])
    actual_locs = state.get("actual_locs", [])
    if not isinstance(actual_locs, list) or not actual_locs:
        raise ValueError("Expected intermediate_data[0]['actual_locs'] to be a non-empty list.")
    if not isinstance(events, list) or not events:
        raise ValueError("Expected intermediate_data[0]['events'] to be a non-empty list.")

    # Items are the keys of the initial world state.
    items = sorted(list(actual_locs[0].keys()))

    # Extract people from opening scene node in the tree
    opening_nodes = [x for x in tree.nodes[0].children if (x.value or "").strip().lower() == "opening scene"]
    if not opening_nodes:
        raise ValueError("Could not find 'opening scene' node in tree.")
    people = _extract_people_from_opening_scene(opening_nodes[0])

    original_context = sample.get("context", "")
    locations = _extract_choices_locations(sample)

    thread_safe_print(f"  [{sample_idx + 1}] People: {', '.join(people)}")
    thread_safe_print(f"  [{sample_idx + 1}] Items: {', '.join(items)}")
    thread_safe_print(f"  [{sample_idx + 1}] Variants: {num_variants_per_sample}")

    if variants_to_generate is None:
        variants_to_generate = list(range(num_variants_per_sample))

    sample_cost = 0.0
    variant_results: List[Dict] = []

    for variant_idx in variants_to_generate:
        local_seed = base_seed + sample_idx * 1000 + variant_idx
        local_random = random.Random(local_seed)

        # Create fresh model instances per variant (thread-safe).
        generation_model = OpenAIModel(
            engine=model_config["engine"],
            api_endpoint=model_config["api_endpoint"],
            api_max_attempts=model_config["api_max_attempts"],
            temperature=model_config["temperature"],
            top_p=model_config["top_p"],
            max_tokens=model_config["max_tokens"],
            num_samples=model_config["num_samples"],
            prompt_cost=model_config["prompt_cost"],
            completion_cost=model_config["completion_cost"],
        )

        extraction_cfg = model_config.get("extraction", {})
        extraction_model = OpenAIModel(
            engine=extraction_cfg.get("engine", model_config["engine"]),
            api_endpoint=extraction_cfg.get("api_endpoint", model_config["api_endpoint"]),
            api_max_attempts=extraction_cfg.get("api_max_attempts", model_config["api_max_attempts"]),
            temperature=extraction_cfg.get("temperature", 0.2),
            top_p=extraction_cfg.get("top_p", model_config["top_p"]),
            max_tokens=extraction_cfg.get("max_tokens", model_config["max_tokens"]),
            num_samples=extraction_cfg.get("num_samples", 1),
            # Costs are often unknown for custom engines; only track if provided.
            prompt_cost=extraction_cfg.get("prompt_cost"),
            completion_cost=extraction_cfg.get("completion_cost"),
        )

        thread_safe_print(f"    [{variant_idx + 1}/{num_variants_per_sample}] Regenerating story... (seed={local_seed})")
        new_story, debug_info = regenerate_object_placements_story_scaffolded_parity(
            tree=tree,
            actual_locs=actual_locs,
            events=events,
            items=items,
            people=people,
            locations=locations,
            original_context=original_context,
            generation_model=generation_model,
            extraction_model=extraction_model,
            local_random=local_random,
        )

        cost = float(generation_model.total_cost + extraction_model.total_cost)
        sample_cost += cost

        # Basic post-hoc validation (best-effort; does not change instance).
        validation_issues: List[str] = []
        if "parsed_moves" in debug_info:
            missing = _validate_move_coverage(new_story, debug_info["parsed_moves"])
            if missing:
                validation_issues.append(f"Move coverage missing for: {missing[:10]}")

        new_sample = deepcopy(sample)
        new_sample["original_sample_id"] = sample_idx
        new_sample["variant_index"] = variant_idx
        new_sample["random_seed"] = local_seed
        new_sample["modification_type"] = "object_placements_level0_scaffolded_parity"
        new_sample["original_story"] = sample.get("context", "")
        new_sample["new_story"] = new_story
        new_sample["generation_cost"] = cost
        new_sample["debug"] = debug_info  # Extracted scaffolding for debugging/analysis
        # Remove redundant 'context' field (already in 'original_story'), unless --overwrite-context
        if overwrite_context:
            new_sample["context"] = new_story
        else:
            new_sample.pop("context", None)
        if validation_issues:
            new_sample["validation_issues"] = validation_issues
        variant_results.append(new_sample)

        thread_safe_print(f"    [{variant_idx + 1}/{num_variants_per_sample}] Done (cost: ${cost:.4f})")

    thread_safe_print(f"  [{sample_idx + 1}] Complete | Cost: ${sample_cost:.4f}")
    return sample_idx, variant_results, sample_cost


def _load_existing_results(output_file: Path) -> Tuple[Dict[int, List[Dict]], Set[Tuple[int, int]]]:
    """
    Load existing results from a previous run and return:
    - results_by_sample: {original_sample_id: [variant_entries...]}
    - completed_pairs: set((original_sample_id, variant_index))

    Uses load_json_array_tolerant to handle truncated files gracefully.
    """
    if not output_file.exists():
        return {}, set()

    data, info = load_json_array_tolerant(output_file)
    if info.warning:
        thread_safe_print(f"WARNING (resume loader): {info.warning}")

    results_by_sample: Dict[int, List[Dict]] = {}
    completed_pairs: Set[Tuple[int, int]] = set()
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            sid = entry.get("original_sample_id")
            vidx = entry.get("variant_index")
            if sid is None or vidx is None:
                continue
            sid = int(sid)
            vidx = int(vidx)
            results_by_sample.setdefault(sid, []).append(entry)
            completed_pairs.add((sid, vidx))
    return results_by_sample, completed_pairs


def _merge_variants(existing: List[Dict], new_variants: List[Dict]) -> List[Dict]:
    """
    Merge variant entries for a sample by variant_index, preferring existing entries.

    Reason: in resume mode, existing entries represent completed work and should not be overwritten.
    """
    merged: Dict[int, Dict] = {}
    # Add new variants first...
    for v in new_variants:
        if isinstance(v, dict) and "variant_index" in v:
            merged[int(v["variant_index"])] = v
    # ...then existing overwrite (prefer existing).
    for v in existing:
        if isinstance(v, dict) and "variant_index" in v:
            merged[int(v["variant_index"])] = v
    return [merged[k] for k in sorted(merged.keys())]


def main():
    cache.disable()

    parser = argparse.ArgumentParser(description="Regenerate Object Placements stories from existing logic trees (no tree changes).")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to process (default: all).")
    parser.add_argument("--num-variants", type=int, default=2, help="Variants per sample (default: 2).")
    parser.add_argument("--max-workers", type=int, default=10, help="Concurrent samples (default: 10).")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed for reproducibility (default: 42).")
    parser.add_argument(
        "--extraction-engine",
        type=str,
        default=None,
        help="Model engine to use for extraction tasks (people/outline + fallback move reasons). Example: gpt-5.2. Default: same as generation engine.",
    )
    parser.add_argument(
        "--extraction-prompt-cost",
        type=float,
        default=None,
        help="Optional prompt cost per token for extraction engine (for cost tracking). If omitted, extraction costs are not tracked.",
    )
    parser.add_argument(
        "--extraction-completion-cost",
        type=float,
        default=None,
        help="Optional completion cost per token for extraction engine (for cost tracking). If omitted, extraction costs are not tracked.",
    )
    parser.add_argument(
        "--overwrite-context",
        action="store_true",
        help="If set, write regenerated story into 'context' (drop-in runnable). Otherwise store in 'new_story'.",
    )
    parser.add_argument(
        "--resume-output",
        type=str,
        default=None,
        help="Path to an existing partial output JSON to resume (will append missing variants).",
    )
    args = parser.parse_args()

    num_samples = args.num_samples
    num_variants_per_sample = args.num_variants
    max_workers = args.max_workers
    base_seed = args.base_seed
    overwrite_context = bool(args.overwrite_context)
    extraction_engine = args.extraction_engine
    extraction_prompt_cost = args.extraction_prompt_cost
    extraction_completion_cost = args.extraction_completion_cost

    input_file = OUTPUT_FOLDER / "object_placements.json"

    print("=" * 80)
    print("REGENERATING OBJECT PLACEMENTS STORIES FROM EXISTING TREES (LEVEL 0)")
    print("=" * 80)
    print(f"Input: {input_file}")
    print(f"Base seed: {base_seed}")

    print("\nLoading dataset...")
    with open(input_file, "r", encoding="utf-8") as f:
        original_dataset = json.load(f)

    samples_to_process = original_dataset if num_samples is None else original_dataset[:num_samples]
    num_samples_actual = len(samples_to_process)
    total_variants = num_samples_actual * num_variants_per_sample

    resuming = args.resume_output is not None
    resume_output_file = Path(args.resume_output).resolve() if resuming else None

    run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    if resuming:
        output_file = resume_output_file
        output_folder = output_file.parent
        metadata_file = output_folder / "run_metadata.txt"
        print(f"\nResuming from existing output: {output_file}")
    else:
        output_folder_name = f"object_placements_level0_samples-{num_samples_actual}_variants-{num_variants_per_sample}_{run_datetime}"
        output_folder = OUTPUT_FOLDER / output_folder_name
        output_folder.mkdir(parents=True, exist_ok=True)
        output_file = output_folder / f"object_placements_level0_samples-{num_samples_actual}_variants-{num_variants_per_sample}.json"
        metadata_file = output_folder / "run_metadata.txt"

    print(f"\nSamples to process: {num_samples_actual}")
    print(f"Variants per sample: {num_variants_per_sample}")
    print(f"Total variants to generate: {total_variants}")
    print(f"Concurrent workers: {max_workers}")
    print(f"Overwrite context: {overwrite_context}")
    print(f"Output folder: {output_folder}")
    print(f"Output file: {output_file}\n")

    model_config = {
        "engine": "gpt-4-0613",
        "api_endpoint": "chat",
        "api_max_attempts": 30,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 2400,
        "num_samples": 1,
        "prompt_cost": 0.03 / 1000,
        "completion_cost": 0.06 / 1000,
        "extraction": {
            "engine": extraction_engine or "gpt-4-0613",
            "api_endpoint": "chat",
            "api_max_attempts": 30,
            "temperature": 0.2,
            "top_p": 1.0,
            "max_tokens": 2400,
            "num_samples": 1,
            "prompt_cost": extraction_prompt_cost,
            "completion_cost": extraction_completion_cost,
        },
    }

    all_results_by_sample, completed_pairs = ({}, set())
    if resuming:
        all_results_by_sample, completed_pairs = _load_existing_results(output_file)
        already_done = len(completed_pairs)
        expected_total = num_samples_actual * num_variants_per_sample
        print(f"Found {already_done}/{expected_total} completed variants in existing output.\n")

    pending_work: List[Tuple[int, Dict, List[int]]] = []
    for idx, sample in enumerate(samples_to_process):
        missing = [v for v in range(num_variants_per_sample) if (idx, v) not in completed_pairs]
        if missing:
            pending_work.append((idx, sample, missing))

    if resuming and not pending_work:
        print("Nothing to do: all requested variants already exist in the output JSON.")
        return

    errors: List[Tuple[int, Exception, str]] = []
    total_cost = 0.0
    completed_samples = 0

    def _submit(executor, idx: int, sample: Dict, missing_variants: List[int]):
        return executor.submit(
            process_single_sample,
            sample,
            idx,
            num_samples_actual,
            num_variants_per_sample,
            model_config,
            base_seed,
            missing_variants,
            overwrite_context,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {_submit(executor, idx, sample, missing): idx for idx, sample, missing in pending_work}

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                sample_idx, variant_results, sample_cost = fut.result()
                total_cost += sample_cost
                existing = all_results_by_sample.get(sample_idx, [])
                all_results_by_sample[sample_idx] = _merge_variants(existing, variant_results)
                completed_samples += 1
                thread_safe_print(
                    f"\nSample {sample_idx + 1} complete ({completed_samples}/{len(pending_work)} samples)"
                )
                thread_safe_print(f"  Running total cost: ${total_cost:.4f}")
            except Exception as e:
                tb = traceback.format_exc()
                errors.append((idx, e, tb))
                thread_safe_print(f"\nERROR sample_idx={idx}: {e}")
                thread_safe_print(tb)

            # Persist frequently (helps with long runs + rate limit failures).
            flat_results: List[Dict] = []
            for sid in sorted(all_results_by_sample.keys()):
                flat_results.extend(all_results_by_sample[sid])
            # Use atomic write to prevent truncation on interrupt.
            with _print_lock:
                atomic_json_dump(flat_results, output_file, indent=2, ensure_ascii=False)

    # Calculate final stats
    end_datetime = datetime.now()
    flat_results_final: List[Dict] = []
    for sid in sorted(all_results_by_sample.keys()):
        flat_results_final.extend(all_results_by_sample[sid])
    successful_variants = len(flat_results_final)
    avg_cost_per_variant = total_cost / successful_variants if successful_variants > 0 else 0

    # Write metadata file (matching murder mystery level 0 format)
    metadata_content = f"""Run Metadata
============

Run Information:
  Start Time: {run_datetime}
  End Time: {end_datetime.strftime("%Y%m%d_%H%M%S")}
  Script: level0_object_placements_regenerate_stories_WIP.py
  Resumed: {resuming}
  Resume Output: {str(output_file) if resuming else "N/A"}

Configuration:
  Input File: {input_file}
  Output Folder: {output_folder}
  Num Samples: {num_samples_actual}
  Variants Per Sample: {num_variants_per_sample}
  Max Workers: {max_workers}
  Base Seed: {base_seed}
  Overwrite Context: {overwrite_context}

Model Configuration:
  Engine: {model_config['engine']}
  Temperature: {model_config['temperature']}
  Top P: {model_config['top_p']}
  Max Tokens: {model_config['max_tokens']}
  API Endpoint: {model_config['api_endpoint']}
  Extraction Engine: {model_config['extraction']['engine']}
  Extraction Temperature: {model_config['extraction']['temperature']}

Results:
  Total Variants Generated: {successful_variants}/{total_variants}
  Errors: {len(errors)}
  Total Cost (this run only): ${total_cost:.4f}
  Avg Cost Per Variant (this run only): ${avg_cost_per_variant:.4f}

Output Files:
  - {output_file.name}
  - {metadata_file.name}

Reproducibility:
  - Uses thread-local random.Random() with deterministic seeds
  - Seed formula: base_seed + sample_idx * 1000 + variant_idx
  - Same configuration always produces identical results
  - Concurrent execution does not affect reproducibility

Description:
  This run regenerated Object Placements stories from existing logic trees using temperature={model_config['temperature']}.
  The same trees produce different story text but with identical semantic content (facts).
  Uses scaffolded-parity approach: extracts character roles/motivations and story outline from
  original story, then regenerates using the same prompt templates as the original benchmark.

Methodology:
  1. Loads original samples with their logic trees and intermediate_data
  2. Extracts people_data (roles/motivations) and story_desc (outline) from original story
  3. Regenerates story text from UNCHANGED trees using original benchmark prompts
  4. Each variant has the same facts but different narrative text
  5. Questions/answers are unchanged (trees are unchanged)
"""

    if errors:
        metadata_content += "\nErrors:\n"
        for sample_idx, exc, _tb in errors[:200]:
            metadata_content += f"  - Sample {sample_idx + 1}: {exc}\n"

    with open(metadata_file, "w", encoding="utf-8") as f:
        f.write(metadata_content)

    # Final summary
    print(f"\n{'='*80}")
    print("REGENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Samples processed: {num_samples_actual}")
    print(f"Variants per sample: {num_variants_per_sample}")
    print(f"Total variants generated: {successful_variants}/{total_variants}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Avg cost per variant: ${avg_cost_per_variant:.4f}")
    print(f"Output folder: {output_folder}")
    print(f"Output file: {output_file.name}")
    print(f"Metadata file: {metadata_file.name}")
    print(f"\nEach variant has:")
    print(f"  - Same facts (trees unchanged)")
    print(f"  - New story text (regenerated with temperature={model_config['temperature']})")
    print(f"  - Same questions/answers (trees unchanged)")
    print(f"  - Reproducible results (deterministic seeding)")
    if errors:
        print(f"  - Errors: {len(errors)} (see metadata)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()


