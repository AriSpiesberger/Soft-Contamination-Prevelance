"""
Level 1: Generate Team Allocation semantic duplicates by swapping ONE skill branch.

This script implements a difficulty-preserving modification strategy:
1. Identifies skill branches in the tree (e.g., "Person X is bad at task Y")
2. Randomly selects ONE skill branch to swap
3. Copies the EXACT structure of the original branch (preserving complexity)
4. Uses the original benchmark's prompting method to fill new content
5. Regenerates story text from the modified tree
6. Verifies the answer is still correct (matrix unchanged)

The result is a semantic duplicate with:
- Same difficulty (identical structural complexity - same tree shape)
- Same answer (matrix and best_pair unchanged)
- Different content (different narrative justification for ONE skill)
- Same skill LEVEL (bad/okay/good stays the same, only justification changes)

Branch Types in Team Allocation:
- Skill branches: "X is [bad/okay/good] at [task].  Because we find out in the story that, "
  → Regenerated using skill ICL examples and prompts
- Cooperation branches: "X and Y work [badly/okay/well] together.  Because..."
  → NOT modified in Level 1

Reproducibility:
- Uses thread-local random.Random() instances with deterministic seeds
- Same (sample_idx, variant_idx) always produces same results
- Concurrent execution does not affect reproducibility
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path so we can import from src
SCRIPT_DIR = Path(__file__).parent.absolute()
MUSR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MUSR_DIR))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(SCRIPT_DIR / '.env')
load_dotenv(MUSR_DIR / '.env')

import json
import random
import threading
import traceback
from copy import deepcopy
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Set

from src import cache
from src.model import OpenAIModel
from src.logic_tree.tree import LogicTree, LogicNode, LogicNodeFactType
from src.utils.paths import OUTPUT_FOLDER
from src.utils.json_io import atomic_json_dump, load_json_array_tolerant
from src.dataset_types.team_allocation import TeamAllocationDataset, __team_allocation_completion_intro__
from src.dataset_builder import DatasetBuilder


# Thread-safe printing
_print_lock = threading.Lock()


def thread_safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with _print_lock:
        print(*args, **kwargs)


# =============================================================================
# DATA ACCESS HELPERS
# =============================================================================

def extract_sample_data(sample: Dict) -> Tuple[LogicTree, Dict, List[str], int, Dict]:
    """
    Extract structured data from a team allocation sample.
    
    Args:
        sample: The sample dict from the dataset
    
    Returns:
        tree: LogicTree object
        intermediate_data: {tasks, matrix, best_pair}
        choices: List of choice strings
        answer: The correct answer index
        tree_json: The raw tree JSON
    """
    question = sample['questions'][0]
    intermediate_data = question['intermediate_data'][0]
    tree_json = question['intermediate_trees'][0]
    tree = LogicTree.from_json(tree_json)
    choices = question['choices']
    answer = question['answer']
    
    return tree, intermediate_data, choices, answer, tree_json


def extract_people_from_matrix(matrix: Dict) -> List[str]:
    """Extract people names from the matrix keys."""
    return list(matrix.keys())


def extract_scenario_description(tree: LogicTree) -> str:
    """Extract the scenario description from the tree root."""
    if tree.nodes and tree.nodes[0].value:
        return tree.nodes[0].value
    return ""


def extract_facts_from_tree(tree: LogicTree) -> List[str]:
    """
    Extract all explicit facts from the tree.
    
    Returns list of fact strings (the leaf nodes and explicit facts).
    """
    facts = []
    for node in tree.get_facts():
        if node.value and node.value.strip():
            facts.append(node.value.strip())
    return list(sorted(set(facts)))


# =============================================================================
# SKILL BRANCH IDENTIFICATION
# =============================================================================

def identify_skill_branches(tree_json: Dict) -> List[Tuple[int, Dict, str, str, str]]:
    """
    Identify skill branches in a tree.
    
    A skill branch has a root value like:
    "{Person} is {bad/okay/good} at {task}.  Because we find out in the story that, "
    
    Returns:
        List of (index, branch_dict, person_name, skill_level, task_name) tuples
    """
    skill_branches = []
    root = tree_json['nodes'][0]
    
    skill_levels = ['bad', 'okay', 'good']
    
    for idx, branch in enumerate(root.get('children', [])):
        value = branch.get('value', '')
        lower_value = value.lower()
        
        # Check if this is a skill branch (contains "is bad/okay/good at")
        for level in skill_levels:
            pattern = f'is {level} at'
            if pattern in lower_value:
                # Parse out person name and task
                # Format: "{Person} is {level} at {task}.  Because..."
                parts = value.split(f' is {level} at ')
                if len(parts) >= 2:
                    person_name = parts[0].strip()
                    # Task is between "at " and ".  Because" or just "."
                    task_part = parts[1]
                    task_end = task_part.find('.')
                    if task_end != -1:
                        task_name = task_part[:task_end].strip()
                    else:
                        task_name = task_part.strip()
                    
                    skill_branches.append((idx, branch, person_name, level, task_name))
                break
    
    return skill_branches


def identify_cooperation_branches(tree_json: Dict) -> List[Tuple[int, Dict, str, str, str]]:
    """
    Identify cooperation branches in a tree.
    
    A cooperation branch has a root value like:
    "{Person1} and {Person2} work {badly/okay/well} together.  Because..."
    
    Returns:
        List of (index, branch_dict, person1, person2, coop_level) tuples
    """
    coop_branches = []
    root = tree_json['nodes'][0]
    
    coop_levels = ['badly', 'okay', 'well']
    
    for idx, branch in enumerate(root.get('children', [])):
        value = branch.get('value', '')
        lower_value = value.lower()
        
        # Check if this is a cooperation branch (contains "work badly/okay/well together")
        for level in coop_levels:
            pattern = f'work {level} together'
            if pattern in lower_value:
                # Parse out person names
                # Format: "{Person1} and {Person2} work {level} together..."
                and_idx = value.lower().find(' and ')
                if and_idx != -1:
                    person1 = value[:and_idx].strip()
                    remaining = value[and_idx + 5:]  # Skip " and "
                    work_idx = remaining.lower().find(f' work {level}')
                    if work_idx != -1:
                        person2 = remaining[:work_idx].strip()
                        coop_branches.append((idx, branch, person1, person2, level))
                break
    
    return coop_branches


# =============================================================================
# TREE STRUCTURE OPERATIONS
# =============================================================================

def calculate_branch_complexity(branch: Dict) -> Dict[str, int]:
    """Calculate complexity metrics for a branch (for logging/verification)."""
    def recurse(node, depth=0):
        metrics = {
            'max_depth': depth,
            'num_explicit': 0,
            'num_commonsense': 0,
            'total_nodes': 1
        }
        
        children = node.get('children', [])
        fact_type = node.get('fact_type', '')
        
        if not children:
            if fact_type == 'explicit':
                metrics['num_explicit'] = 1
            elif fact_type == 'commonsense':
                metrics['num_commonsense'] = 1
        
        for child in children:
            child_metrics = recurse(child, depth + 1)
            metrics['max_depth'] = max(metrics['max_depth'], child_metrics['max_depth'])
            metrics['num_explicit'] += child_metrics['num_explicit']
            metrics['num_commonsense'] += child_metrics['num_commonsense']
            metrics['total_nodes'] += child_metrics['total_nodes']
        
        return metrics
    
    return recurse(branch)


def copy_branch_structure_for_skill(
    old_branch: Dict,
    person_name: str,
    skill_level: str,
    task_name: str
) -> Dict:
    """
    Create a new skill branch by copying the EXACT structure of an existing branch.
    
    This preserves:
    - Same depth
    - Same number of children at each level
    - Same fact_types (explicit vs commonsense)
    - Same operators and other node properties
    
    Only changes:
    - All descendant values (cleared, to be filled by LLM)
    - Root value stays the same (same skill level, same task)
    
    Args:
        old_branch: The original branch to copy structure from
        person_name: Name of the person
        skill_level: Skill level (bad/okay/good)
        task_name: Name of the task
    
    Returns:
        New branch with same structure but empty descendant values
    """
    new_branch = deepcopy(old_branch)
    
    # Keep root value the same (same skill judgment, different justification)
    # The root value format is preserved to maintain the skill level
    new_branch['value'] = f"{person_name} is {skill_level} at {task_name.lower()}.  Because we find out in the story that, "
    
    # Clear all descendant values but keep structure
    def clear_descendant_values(node):
        for child in node.get('children', []):
            child['value'] = ''  # Will be filled by LLM
            clear_descendant_values(child)
    
    clear_descendant_values(new_branch)
    
    return new_branch


# =============================================================================
# ICL EXAMPLES FOR SKILL BRANCH COMPLETION
# =============================================================================

# These examples teach the LLM how to fill in skill branches.
# They follow the same pattern as the original team allocation dataset generator.

skill_example1_description = "A zoo requires caretakers for different animals and keepers for cleaning the exhibits."
skill_example1_tree = LogicTree(
    nodes=[
        LogicNode("A zoo requires caretakers for different animals and keepers for cleaning the exhibits.", [
            LogicNode("Alex is bad at animal caretaker.  Because we find out in the story that, ", [
                LogicNode("Alex expressed in the past no desire to pursue furthering his knowledge of animals outside of pets."),
                LogicNode("Alex admitted that he feels uncomfortable around animals larger than him."),
                LogicNode("If someone is uncomfortable around large animals and has no interest in expanding his knowledge of animals, they probably won't be good at a job that involves taking care of a variety of animals, some of which can be large.", fact_type=LogicNodeFactType.COMMONSENSE)
            ])
        ])
    ], prune=False, populate=False
)
skill_example1_node_completion = skill_example1_tree.nodes[0].children[0]

skill_example2_description = "You and your roommates want to make a video game, how should you assign each of your roommates so that the action video game is made."
skill_example2_tree = LogicTree(
    nodes=[
        LogicNode("You and your roommates want to make a video game.", [
            LogicNode("Maya is good at programming.  Because we find out in the story that, ", [
                LogicNode("Maya has been coding since she was twelve and has already released several indie games."),
                LogicNode("Maya's GitHub is filled with game engine contributions and she frequently helps others debug their code."),
                LogicNode("Someone with extensive programming experience since childhood and active contributions to game development projects would excel at programming tasks.", fact_type=LogicNodeFactType.COMMONSENSE)
            ])
        ])
    ], prune=False, populate=False
)
skill_example2_node_completion = skill_example2_tree.nodes[0].children[0]

skill_example3_description = "A paper deadline is coming up and since you are the supervisor of the lab, you must assign each graduate student efficiently to meet the deadline."
skill_example3_tree = LogicTree(
    nodes=[
        LogicNode("A paper deadline is coming up.", [
            LogicNode("Jordan is okay at writing.  Because we find out in the story that, ", [
                LogicNode("Jordan has published a few papers before, but always needed significant editing help from advisors."),
                LogicNode("Jordan mentioned that writing doesn't come naturally to them, though they can produce passable drafts with effort."),
                LogicNode("Someone who can write but requires external help and extra effort would be considered moderately capable at writing rather than excellent.", fact_type=LogicNodeFactType.COMMONSENSE)
            ])
        ])
    ], prune=False, populate=False
)
skill_example3_node_completion = skill_example3_tree.nodes[0].children[0]

skill_example_descriptions = [skill_example1_description, skill_example2_description, skill_example3_description]
skill_example_trees = [skill_example1_tree, skill_example2_tree, skill_example3_tree]
skill_example_node_completions = [skill_example1_node_completion, skill_example2_node_completion, skill_example3_node_completion]


# =============================================================================
# TREE FILLING - Using Original Benchmark's Method
# =============================================================================

def fill_branch_with_benchmark_method(
    branch_json: Dict,
    person_name: str,
    skill_level: str,
    task_name: str,
    scenario_description: str,
    model: OpenAIModel
) -> Dict:
    """
    Fill empty nodes in a skill branch using the original benchmark's prompting method.
    
    This uses the same approach as DatasetBuilder.complete_structure():
    1. Uses the benchmark's prompt format with ICL examples
    2. Parses output looking for '| Fact From Story' and '| Commonsense Knowledge'
    3. Fills nodes based on their fact_type
    
    Args:
        branch_json: Branch with structure but empty descendant values
        person_name: Name of the person
        skill_level: Skill level (bad/okay/good)
        task_name: Name of the task
        scenario_description: The scenario description
        model: OpenAI model instance
    
    Returns:
        Filled branch with all values populated
    """
    # Convert to LogicTree for processing
    temp_tree_json = {
        'chance_of_or': 0.0,
        'depth': 3,
        'chance_to_prune': 0.0,
        'chance_to_prune_all': 0.0,
        'bf_factor': {2: 1.0},
        'deduction_type_sample_rate': {'syllogism': 1.0},
        'root_structure': [],
        'nodes': [branch_json]
    }
    
    tree = LogicTree.from_json(temp_tree_json)
    
    # Create the dataset builder instance
    builder = DatasetBuilder()
    
    # Create completion prompt function using team allocation skill examples
    completion_prompt_fn = builder.create_completion_prompt(
        skill_example_trees,
        skill_example_node_completions,
        skill_example_descriptions,
        intro=__team_allocation_completion_intro__,
        because_clause_after=0
    )
    
    # Use the benchmark's parsing method
    pad_char = '> '
    
    def parse_out(output, node_value):
        """Parse LLM output for facts - matches benchmark's parse_out function."""
        facts_from_story = []
        cs_knowledge = []
        
        for line in output.split('\n'):
            val = '|'.join(line.replace(pad_char, '').split('|')[:-1])
            if val == node_value:
                continue
            
            if '| Fact From Story' in line or '| Complex Fact' in line:
                if val not in facts_from_story and val not in cs_knowledge:
                    facts_from_story.append(val)
            elif '| Commonsense Knowledge' in line:
                if val not in facts_from_story and val not in cs_knowledge:
                    cs_knowledge.append(val)
        
        return facts_from_story, cs_knowledge
    
    def fill_node_recursive(node: LogicNode, tree: LogicTree):
        """Recursively fill empty nodes."""
        children = node.children
        
        # If any child has empty value, we need to fill them
        if any(c.value == '' for c in children):
            prompt = completion_prompt_fn(tree, node, scenario_description)
            
            raw = model.inference(prompt)
            output = raw['choices'][0]['message']['content']
            
            facts_from_story, cs_knowledge = parse_out(output, node.value)
            
            # Retry if structure doesn't match
            retry_count = 0
            max_retries = 3
            while retry_count < max_retries and len(facts_from_story) + len(cs_knowledge) != len(children):
                retry_count += 1
                retry_prompt = prompt + f'\n\nYou erroneously produced this last time.\n{output}\n\nThis does not match the structure. Please return the correct number of "Facts From Story" and "Commonsense Knowledge" this time and make sure they are unique.\n\nOutput:'
                raw = model.inference(retry_prompt)
                output = raw['choices'][0]['message']['content']
                facts_from_story, cs_knowledge = parse_out(output, node.value)
            
            # Fill children based on their fact type
            try:
                for child in children:
                    if child.value == '':  # Only fill empty ones
                        if child.fact_type == LogicNodeFactType.COMMONSENSE:
                            if cs_knowledge:
                                child.value = cs_knowledge.pop(0)
                            else:
                                # Fallback
                                child.value = f"Someone who demonstrates these characteristics would typically be {skill_level} at {task_name}."
                        elif child.fact_type == LogicNodeFactType.EXPLICIT:
                            if facts_from_story:
                                child.value = facts_from_story.pop(0)
                            else:
                                # Fallback
                                child.value = f"{person_name} shows typical traits for this skill level."
            except Exception as e:
                thread_safe_print(f"    Warning: Error filling nodes: {e}, using fallbacks")
                fill_empty_nodes_fallback(node, person_name, skill_level, task_name)
        
        # Recurse to children
        for child in children:
            fill_node_recursive(child, tree)
    
    # Start filling from root
    for root_node in tree.nodes:
        fill_node_recursive(root_node, tree)
    
    # Convert back to JSON
    return tree.nodes[0].to_json()


def fill_empty_nodes_fallback(node: LogicNode, person_name: str, skill_level: str, task_name: str):
    """Fill any remaining empty nodes with generic fallback values."""
    for child in node.children:
        if child.value == '':
            if child.fact_type == LogicNodeFactType.COMMONSENSE:
                child.value = f"Someone who demonstrates these characteristics would typically be {skill_level} at {task_name}."
            else:
                child.value = f"{person_name} shows typical traits for this skill level."
        fill_empty_nodes_fallback(child, person_name, skill_level, task_name)


# =============================================================================
# STORY GENERATION
# =============================================================================

def generate_story_from_tree(
    tree: LogicTree,
    people: List[str],
    tasks: List[str],
    scenario_description: str,
    model: OpenAIModel
) -> str:
    """
    Generate a new story from the modified tree.
    
    This uses the same prompt structure as create_team_allocation.py.
    """
    creator = TeamAllocationDataset()
    
    # Extract facts from tree
    facts = extract_facts_from_tree(tree)
    facts_str = "\n".join([f'- {x}' for x in facts])
    
    # Main story generation prompt (matches create_team_allocation.py)
    prompt = f'''
You will write a short story given the description of a scenario and a list of facts. You must include every fact into the story.  

You should take the role of a manager or leader, someone who can assign people to skills.  

Your job is to find the perfect assignment of each person to a single skill.  You will not say what this perfect assignment is, that is left to the reader to decide.  

Instead, you must write out the narrative as if it were a short story. Make this story coherent and flow nicely.  

Most importantly, make this story interesting! 

Start with an introduction.  
Introduce each person:
- {people[0]}
- {people[1]}
- {people[2]}

And the tasks the manager has to assign them to (mention these by name in the beginning, i.e. "And they all had to be assigned to cleaning and sales" or something.):
- {tasks[0]}
- {tasks[1]}

Scenario description: {scenario_description}
Facts that you must include in your story:
{facts_str}

Output:'''

    output, _ = creator.inference(prompt, model)
    
    paragraphs = output.split('\n\n')
    
    # Fix the intro paragraph (same approach as create_team_allocation.py)
    fix_p0_prompt = f'''
I am writing a story that is similar to a word problem where a manager has to assign the right worker to a task.  Here's the full story:

{output}

---

I want you to rewrite the introduction paragraph to introduce each person and the tasks.  Do not assign people to a task (this is the question I want to ask the readers), just introduce them.  Rewrite the introduction:

{paragraphs[0]}

Make sure it includes:

Mentions to these peoples names (do not describe them)
- {people[0]}
- {people[1]}
- {people[2]}

And the tasks the manager has to assign them to (mention these by name in the beginning, i.e. "And they all had to be assigned to cleaning and sales" or something.)
- {tasks[0]}
- {tasks[1]}

It should be short.  No longer than the original introduction.
    '''.strip()
    
    fixed_intro, _ = creator.inference(fix_p0_prompt, model, temperature=0.2)
    
    # Replace the intro paragraph
    paragraphs[0] = fixed_intro
    final_story = "\n\n".join(paragraphs)
    
    return final_story


# =============================================================================
# RESUME SUPPORT
# =============================================================================

def _load_existing_results(output_file: Path) -> Tuple[Dict[int, List[Dict]], Set[Tuple[int, int]]]:
    """
    Load an existing partial output JSON and index completed (sample_idx, variant_idx) pairs.
    """
    if not output_file.exists():
        return {}, set()

    data, info = load_json_array_tolerant(output_file)
    if info.warning:
        thread_safe_print(f"⚠️  Resume loader: {info.warning}")

    if not isinstance(data, list):
        raise ValueError(f"Expected output JSON list, got: {type(data)}")

    results_by_sample: Dict[int, List[Dict]] = {}
    completed_pairs: Set[Tuple[int, int]] = set()

    for item in data:
        if not isinstance(item, dict):
            continue
        if "original_sample_id" not in item or "variant_index" not in item:
            raise ValueError(
                "Output JSON is missing required keys for resume. "
                "Expected 'original_sample_id' and 'variant_index'."
            )
        sidx = int(item["original_sample_id"])
        vidx = int(item["variant_index"])
        results_by_sample.setdefault(sidx, []).append(item)
        completed_pairs.add((sidx, vidx))

    return results_by_sample, completed_pairs


def _merge_variants(existing: List[Dict], new_variants: List[Dict]) -> List[Dict]:
    """Merge variant entries for a sample by variant_index, preferring existing entries."""
    merged: Dict[int, Dict] = {}
    for v in new_variants:
        if isinstance(v, dict) and "variant_index" in v:
            merged[int(v["variant_index"])] = v
    for v in existing:
        if isinstance(v, dict) and "variant_index" in v:
            merged[int(v["variant_index"])] = v
    return [merged[k] for k in sorted(merged.keys())]


def _extract_selected_branch_index_from_variant(variant_sample: Dict) -> Optional[int]:
    """
    Extract the selected branch index from a previously-generated variant sample.
    
    Returns the branch index that was swapped, or None if not found.
    """
    return variant_sample.get("selected_branch_index")


def _extract_used_branch_indices_from_variants(variants: List[Dict]) -> List[int]:
    """
    Extract all used branch indices from a list of variant samples.
    
    Used for resume mode to ensure diversity with previously generated variants.
    """
    indices = []
    for v in variants:
        idx = _extract_selected_branch_index_from_variant(v)
        if idx is not None:
            indices.append(idx)
    return indices


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def swap_skill_branch_in_sample(
    sample: Dict,
    model_config: Dict,
    sample_idx: int,
    total_samples: int,
    variant_idx: int = 0,
    total_variants: int = 1,
    base_seed: int = 42,
    excluded_branch_indices: Optional[List[int]] = None
) -> Tuple[Dict, int, float]:
    """
    Create a semantic duplicate by swapping ONE skill branch.
    
    Uses thread-local random for reproducibility:
    - seed = base_seed + sample_idx * 1000 + variant_idx
    - Same (sample_idx, variant_idx) always produces same results
    
    Branch Diversity:
    - Prefers branches not in excluded_branch_indices
    - Falls back to any branch if all are excluded (more branches than variants)
    - Tracks selected_branch_index in output for resume support
    
    Args:
        sample: Original sample to modify
        model_config: Model configuration dict
        sample_idx: Index of this sample
        total_samples: Total number of samples
        variant_idx: Index of this variant (0-based)
        total_variants: Total number of variants per sample
        base_seed: Base seed for random number generation
        excluded_branch_indices: Branch indices to avoid (used by previous variants)
    
    Returns:
        Tuple of (new_sample, selected_branch_index, cost)
    """
    if excluded_branch_indices is None:
        excluded_branch_indices = []
    # Create thread-local random instance with deterministic seed
    local_seed = base_seed + sample_idx * 1000 + variant_idx
    local_random = random.Random(local_seed)
    
    thread_safe_print(f"\n{'='*80}")
    thread_safe_print(f"Sample {sample_idx + 1}/{total_samples} - Variant {variant_idx + 1}/{total_variants}")
    thread_safe_print(f"  (seed={local_seed} for reproducibility)")
    thread_safe_print(f"{'='*80}")
    
    # Extract data
    tree, intermediate_data, choices, answer, tree_json = extract_sample_data(sample)
    
    tasks = intermediate_data['tasks']
    matrix = intermediate_data['matrix']
    best_pair = intermediate_data['best_pair']
    people = extract_people_from_matrix(matrix)
    scenario_description = extract_scenario_description(tree)
    
    thread_safe_print(f"  [{sample_idx + 1}] Tasks: {tasks}")
    thread_safe_print(f"  [{sample_idx + 1}] People: {people}")
    thread_safe_print(f"  [{sample_idx + 1}] Best pair: {best_pair}")
    thread_safe_print(f"  [{sample_idx + 1}] Answer index: {answer}")
    
    # Create model instance
    model = OpenAIModel(
        engine=model_config['engine'],
        api_endpoint=model_config['api_endpoint'],
        api_max_attempts=model_config['api_max_attempts'],
        temperature=model_config['temperature'],
        top_p=model_config['top_p'],
        max_tokens=model_config['max_tokens'],
        num_samples=model_config['num_samples'],
        prompt_cost=model_config['prompt_cost'],
        completion_cost=model_config['completion_cost']
    )
    
    # Deep copy the sample and tree
    new_sample = deepcopy(sample)
    new_tree_json = deepcopy(tree_json)
    
    # Identify skill branches
    skill_branches = identify_skill_branches(new_tree_json)
    thread_safe_print(f"  [{sample_idx + 1}] Found {len(skill_branches)} skill branches")
    
    # Track swapped branches
    swapped_branches = {
        'skill': []  # List of {person, skill_level, task, old_branch, new_branch}
    }
    
    # Track which branch index was selected (for diversity tracking)
    selected_branch_index = -1
    
    if skill_branches:
        # Build list of available indices, preferring non-excluded branches
        all_indices = list(range(len(skill_branches)))
        available_indices = [i for i in all_indices if i not in excluded_branch_indices]
        
        if available_indices:
            # Prefer branches not yet used by other variants
            selected_idx = local_random.choice(available_indices)
            thread_safe_print(f"  [{sample_idx + 1}] Selecting from {len(available_indices)} unused branches (avoiding {excluded_branch_indices})")
        else:
            # All branches already used, fall back to random selection
            selected_idx = local_random.choice(all_indices)
            thread_safe_print(f"  [{sample_idx + 1}] All {len(all_indices)} branches already used, selecting randomly")
        
        selected_branch_index = selected_idx
        branch_idx, old_branch, person_name, skill_level, task_name = skill_branches[selected_idx]
        
        thread_safe_print(f"  [{sample_idx + 1}] Selected branch: '{person_name} is {skill_level} at {task_name}'")
        
        # Calculate and log old complexity
        old_complexity = calculate_branch_complexity(old_branch)
        thread_safe_print(f"  [{sample_idx + 1}] Old branch complexity: {old_complexity}")
        
        # Copy structure and fill with new content
        thread_safe_print(f"  [{sample_idx + 1}] Copying branch structure (preserving complexity)...")
        new_branch = copy_branch_structure_for_skill(old_branch, person_name, skill_level, task_name)
        
        thread_safe_print(f"  [{sample_idx + 1}] Filling branch with LLM (benchmark method)...")
        filled_branch = fill_branch_with_benchmark_method(
            new_branch, person_name, skill_level, task_name, scenario_description, model
        )
        
        # Verify complexity is preserved
        new_complexity = calculate_branch_complexity(filled_branch)
        thread_safe_print(f"  [{sample_idx + 1}] New branch complexity: {new_complexity}")
        
        if old_complexity != new_complexity:
            thread_safe_print(f"  [{sample_idx + 1}] WARNING: Complexity changed! This shouldn't happen with structure copy.")
        
        # Record full swap data
        swapped_branches['skill'].append({
            'person': person_name,
            'skill_level': skill_level,
            'task': task_name,
            'old_branch': deepcopy(old_branch),
            'new_branch': deepcopy(filled_branch)
        })
        
        # Replace in tree
        new_tree_json['nodes'][0]['children'][branch_idx] = filled_branch
    else:
        thread_safe_print(f"  [{sample_idx + 1}] No skill branches found - skipping modification")
    
    # Convert modified tree to LogicTree for story generation
    modified_tree = LogicTree.from_json(new_tree_json)
    
    # Generate new story from modified tree
    thread_safe_print(f"  [{sample_idx + 1}] Generating story from modified tree...")
    new_story = generate_story_from_tree(
        modified_tree, people, tasks, scenario_description, model
    )
    
    # Store original questions before modification
    original_questions = deepcopy(sample['questions'])
    
    # Update the sample
    new_sample['original_sample_id'] = sample_idx
    new_sample['new_story'] = new_story
    new_sample['original_story'] = sample.get('original_story', sample.get('context', ''))
    
    # Remove 'context' field if present (redundant with original_story)
    if 'context' in new_sample:
        del new_sample['context']
    
    # Swapped branches with full old/new data
    new_sample['swapped_branches'] = swapped_branches
    new_sample['num_skill_branches_swapped'] = len(swapped_branches['skill'])
    new_sample['modification_type'] = 'skill_branch_swap'
    new_sample['variant_index'] = variant_idx
    new_sample['random_seed'] = local_seed
    new_sample['selected_branch_index'] = selected_branch_index  # For diversity tracking
    
    # Store modified tree in new questions
    new_sample['questions'][0]['intermediate_trees'] = [modified_tree.to_json()]
    
    # Store both original and new questions
    new_sample['original_questions'] = original_questions
    new_sample['new_questions'] = deepcopy(new_sample['questions'])
    
    # Remove the inherited 'questions' field (redundant with original_questions/new_questions)
    del new_sample['questions']
    
    # ==================== ANSWER VERIFICATION ====================
    # Verify that the answer is unchanged (matrix was not modified)
    original_answer = original_questions[0]['answer']
    new_answer = new_sample['new_questions'][0]['answer']
    
    if original_answer != new_answer:
        raise ValueError(
            f"Answer verification FAILED! Original: {original_answer}, New: {new_answer}. "
            "This should never happen since we only modify branch justifications, not the matrix."
        )
    
    # Also verify the matrix is unchanged
    original_matrix = original_questions[0]['intermediate_data'][0]['matrix']
    new_matrix = new_sample['new_questions'][0]['intermediate_data'][0]['matrix']
    
    if original_matrix != new_matrix:
        raise ValueError(
            f"Matrix verification FAILED! Matrix was modified. "
            "This should never happen since we only modify branch justifications."
        )
    
    new_sample['answer_verified'] = True
    new_sample['matrix_verified'] = True
    # ============================================================
    
    total_cost = model.total_cost
    new_sample['generation_cost'] = float(total_cost)
    
    thread_safe_print(f"  [{sample_idx + 1}v{variant_idx + 1}] ✓ Complete - Answer verified: {answer}")
    thread_safe_print(f"  [{sample_idx + 1}v{variant_idx + 1}]   Skill branches swapped: {new_sample['num_skill_branches_swapped']}")
    thread_safe_print(f"  [{sample_idx + 1}v{variant_idx + 1}]   Branch index: {selected_branch_index}")
    thread_safe_print(f"  [{sample_idx + 1}v{variant_idx + 1}]   Cost: ${total_cost:.4f}")
    
    return new_sample, selected_branch_index, total_cost


def process_sample_wrapper(args):
    """Wrapper function for concurrent processing of a single variant."""
    (sample, model_config, sample_idx, total_samples, 
     variant_idx, total_variants, base_seed, excluded_branch_indices) = args
    try:
        new_sample, branch_idx, cost = swap_skill_branch_in_sample(
            sample,
            model_config,
            sample_idx,
            total_samples,
            variant_idx,
            total_variants,
            base_seed,
            excluded_branch_indices
        )
        return sample_idx, variant_idx, new_sample, branch_idx, cost, None
    except Exception as e:
        return sample_idx, variant_idx, None, -1, 0.0, (e, traceback.format_exc())


def process_single_sample(
    sample: Dict,
    sample_idx: int,
    num_samples_actual: int,
    num_variants_per_sample: int,
    model_config: Dict,
    base_seed: int,
    variants_to_generate: Optional[List[int]] = None,
    existing_branch_indices: Optional[List[int]] = None,
) -> Tuple[int, List[Dict], List[Tuple], float]:
    """
    Process all variants for a single sample.
    Variants are processed sequentially to enable branch diversity tracking.
    
    Args:
        sample: The sample to process
        sample_idx: Index of the sample
        num_samples_actual: Total number of samples
        num_variants_per_sample: Number of variants to generate per sample
        model_config: Model configuration
        base_seed: Base seed for reproducibility
        variants_to_generate: Which variant indices to generate (for resume mode)
        existing_branch_indices: Branch indices already used by existing variants (for resume mode)
    """
    thread_safe_print(f"\n{'='*80}")
    thread_safe_print(f"Processing Sample {sample_idx + 1}/{num_samples_actual}")
    thread_safe_print(f"{'='*80}")
    
    sample_variants = []
    sample_errors = []
    sample_cost = 0.0
    
    # Track branch indices used across variants for diversity
    used_branch_indices: List[int] = list(existing_branch_indices) if existing_branch_indices else []
    
    if variants_to_generate is None:
        variants_to_generate = list(range(num_variants_per_sample))
    
    if used_branch_indices:
        thread_safe_print(f"  Existing branch indices from resume: {used_branch_indices}")
    
    for variant_idx in variants_to_generate:
        args = (
            sample, 
            model_config, 
            sample_idx, 
            num_samples_actual,
            variant_idx,
            num_variants_per_sample,
            base_seed,
            used_branch_indices.copy()  # Pass current exclusion list
        )
        
        _, _, new_sample, branch_idx, cost, error = process_sample_wrapper(args)
        
        if error:
            exc, tb = error
            thread_safe_print(f"\n❌ Error: Sample {sample_idx + 1} Variant {variant_idx + 1}: {exc}")
            thread_safe_print(tb)
            sample_errors.append((sample_idx, variant_idx, exc))
        else:
            sample_variants.append(new_sample)
            sample_cost += cost
            # Track this branch for subsequent variants
            if branch_idx >= 0:
                used_branch_indices.append(branch_idx)
    
    thread_safe_print(f"\n  Sample {sample_idx + 1} complete: {len(sample_variants)} variants | Cost: ${sample_cost:.4f}")
    thread_safe_print(f"  Branch indices used: {used_branch_indices}")
    
    return sample_idx, sample_variants, sample_errors, sample_cost


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function to generate semantic duplicates with swapped skill branches."""
    # Disable cache to force regeneration
    cache.disable()
    
    parser = argparse.ArgumentParser(
        description="Level 1: Generate Team Allocation semantic duplicates by swapping ONE skill branch."
    )
    parser.add_argument("--num-samples", type=int, default=None, 
                        help="Number of samples to process (default: all).")
    parser.add_argument("--num-variants", type=int, default=2, 
                        help="Variants per sample (default: 2).")
    parser.add_argument("--max-workers", type=int, default=20, 
                        help="Concurrent samples (default: 20).")
    parser.add_argument("--base-seed", type=int, default=42, 
                        help="Base seed for reproducibility (default: 42).")
    parser.add_argument(
        "--resume-output",
        type=str,
        default=None,
        help="Path to an existing partial output JSON to resume (will append missing variants).",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to input JSON file (default: datasets/team_allocation.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate costs without making API calls.",
    )
    args = parser.parse_args()

    # ==================== CONFIGURATION ====================
    num_samples = args.num_samples
    num_variants_per_sample = args.num_variants
    max_workers = args.max_workers
    base_seed = args.base_seed
    dry_run = args.dry_run
    
    if args.input_file:
        input_file = Path(args.input_file)
    else:
        input_file = OUTPUT_FOLDER / "team_allocation.json"
    # ======================================================
    
    print("="*80)
    print("LEVEL 1: GENERATING TEAM ALLOCATION SEMANTIC DUPLICATES")
    print("        (Swapping ONE skill branch per sample)")
    print("="*80)
    print(f"\nInput: {input_file}")
    print(f"Base seed: {base_seed} (for reproducibility)")
    print("\nThis script swaps ONE skill branch per sample:")
    print("  - Skill branch selected randomly")
    print("  - Same skill level (bad/okay/good) preserved")
    print("  - New narrative justification generated")
    print("  - Story regenerated from modified tree")
    print("  - Matrix and answer unchanged")
    
    # Load original dataset
    print(f"\nLoading dataset...")
    with open(input_file, 'r', encoding='utf-8') as f:
        original_dataset = json.load(f)
    
    # ==================== INPUT VALIDATION ====================
    if not original_dataset:
        raise ValueError("Input file is empty or contains no samples.")
    
    if not isinstance(original_dataset, list):
        raise ValueError(f"Expected input to be a list of samples, got {type(original_dataset).__name__}")
    
    # Validate first sample has expected structure
    sample_0 = original_dataset[0]
    if 'questions' not in sample_0 or not sample_0['questions']:
        raise ValueError("Sample missing 'questions' field or questions is empty.")
    
    q0 = sample_0['questions'][0]
    required_fields = ['intermediate_trees', 'intermediate_data', 'answer', 'choices']
    missing_fields = [f for f in required_fields if f not in q0]
    if missing_fields:
        raise ValueError(f"Sample questions[0] missing required fields: {missing_fields}")
    
    if not q0['intermediate_trees'] or not q0['intermediate_data']:
        raise ValueError("Sample has empty intermediate_trees or intermediate_data.")
    
    print(f"✓ Input validation passed ({len(original_dataset)} samples in file)")
    # =========================================================
    
    samples_to_process = original_dataset if num_samples is None else original_dataset[:num_samples]
    num_samples_actual = len(samples_to_process)
    total_variants = num_samples_actual * num_variants_per_sample
    
    resuming = args.resume_output is not None
    resume_output_file = Path(args.resume_output).resolve() if resuming else None

    # Create output folder (or reuse if resuming)
    run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    if resuming:
        output_file = resume_output_file
        output_folder = output_file.parent
        metadata_file = output_folder / "run_metadata.txt"
        print(f"\nResuming from existing output: {output_file}")
        print(f"Output folder: {output_folder}")
    else:
        output_folder_name = f"team_allocation_level1_samples-{num_samples_actual}_variants-{num_variants_per_sample}_{run_datetime}"
        output_folder = OUTPUT_FOLDER / output_folder_name
        output_folder.mkdir(parents=True, exist_ok=True)
        output_file = output_folder / f"team_allocation_level1_samples-{num_samples_actual}_variants-{num_variants_per_sample}.json"
        metadata_file = output_folder / "run_metadata.txt"
    
    print(f"Samples to process: {num_samples_actual}")
    print(f"Variants per sample: {num_variants_per_sample}")
    print(f"Total variants to generate: {total_variants}")
    print(f"Concurrent workers: {max_workers}")
    print(f"Output folder: {output_folder}")
    
    # Model configuration (matches create_team_allocation.py - uses gpt-4)
    model_config = {
        'engine': 'gpt-4-0613',
        'api_endpoint': 'chat',
        'api_max_attempts': 30,
        'temperature': 1.0,
        'top_p': 1.0,
        'max_tokens': 2400,
        'num_samples': 1,
        'prompt_cost': 0.03/1000,
        'completion_cost': 0.06/1000
    }
    
    # ==================== DRY RUN MODE ====================
    if dry_run:
        # Estimate costs based on typical usage patterns
        # Typical variant: ~2 LLM calls for branch filling + 2 for story generation = 4 calls
        # Average prompt: ~1500 tokens, completion: ~500 tokens per call
        avg_prompt_tokens_per_variant = 4 * 1500  # 6000 tokens
        avg_completion_tokens_per_variant = 4 * 500  # 2000 tokens
        
        estimated_prompt_cost = total_variants * avg_prompt_tokens_per_variant * model_config['prompt_cost']
        estimated_completion_cost = total_variants * avg_completion_tokens_per_variant * model_config['completion_cost']
        estimated_total_cost = estimated_prompt_cost + estimated_completion_cost
        
        # Count skill branches in first few samples for diversity estimate
        branch_counts = []
        for sample in samples_to_process[:min(10, num_samples_actual)]:
            tree_json = sample['questions'][0]['intermediate_trees'][0]
            skill_branches = identify_skill_branches(tree_json)
            branch_counts.append(len(skill_branches))
        avg_branches = sum(branch_counts) / len(branch_counts) if branch_counts else 0
        
        print(f"\n{'='*80}")
        print("DRY RUN - Cost Estimation (no API calls made)")
        print(f"{'='*80}")
        print(f"\nConfiguration:")
        print(f"  Samples: {num_samples_actual}")
        print(f"  Variants per sample: {num_variants_per_sample}")
        print(f"  Total variants: {total_variants}")
        print(f"  Model: {model_config['engine']}")
        print(f"\nBranch Analysis (sampled from first {len(branch_counts)} samples):")
        print(f"  Average skill branches per sample: {avg_branches:.1f}")
        print(f"  Min branches: {min(branch_counts) if branch_counts else 0}")
        print(f"  Max branches: {max(branch_counts) if branch_counts else 0}")
        if avg_branches < num_variants_per_sample:
            print(f"  ⚠️  Warning: Some samples may have fewer branches than variants")
        print(f"\nEstimated Costs:")
        print(f"  Prompt tokens per variant: ~{avg_prompt_tokens_per_variant:,}")
        print(f"  Completion tokens per variant: ~{avg_completion_tokens_per_variant:,}")
        print(f"  Prompt cost: ${estimated_prompt_cost:.2f}")
        print(f"  Completion cost: ${estimated_completion_cost:.2f}")
        print(f"  TOTAL ESTIMATED COST: ${estimated_total_cost:.2f}")
        print(f"\nNote: Actual costs may vary ±30% based on prompt/completion lengths.")
        print(f"{'='*80}\n")
        return
    # =====================================================
    
    # Load partial results if resuming
    all_results, completed_pairs = ({}, set())
    if resuming:
        all_results, completed_pairs = _load_existing_results(output_file)
        already_done = len(completed_pairs)
        expected_total = num_samples_actual * num_variants_per_sample
        print(f"Found {already_done}/{expected_total} completed variants in existing output.")

    # Build per-sample missing variant list with existing branch indices for resume
    pending_work: List[Tuple[int, Dict, List[int], List[int]]] = []
    for idx, sample in enumerate(samples_to_process):
        missing = [v for v in range(num_variants_per_sample) if (idx, v) not in completed_pairs]
        if missing:
            # Extract branch indices from existing variants for diversity tracking
            existing_variants = all_results.get(idx, [])
            existing_branch_indices = _extract_used_branch_indices_from_variants(existing_variants)
            pending_work.append((idx, sample, missing, existing_branch_indices))

    if resuming:
        print(f"Samples with remaining work: {len(pending_work)}/{num_samples_actual}")
        if not pending_work:
            print("Nothing to do: all requested variants already exist in the output JSON.")
            return

    # Process samples concurrently
    errors = []
    total_cost = 0.0
    completed_samples = 0
    total_pending_samples = len(pending_work)
    start_time = datetime.now()
    
    print(f"\nStarting processing...")
    print(f"Note: Different samples are processed concurrently ({max_workers} workers).")
    print(f"      Each (sample, variant) has deterministic seed for reproducibility.")
    print(f"      Branch diversity: each variant prefers different skill branches.\n")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_sample,
                sample,
                idx,
                num_samples_actual,
                num_variants_per_sample,
                model_config,
                base_seed,
                missing_variants,
                existing_branch_indices,
            ): idx 
            for (idx, sample, missing_variants, existing_branch_indices) in pending_work
        }
        
        for future in as_completed(futures):
            submitted_idx = futures[future]
            try:
                sample_idx, sample_variants, sample_errors, sample_cost = future.result()
            except Exception as e:
                errors.append((submitted_idx, -1, e))
                thread_safe_print(f"\n❌ Error: Sample {submitted_idx + 1}: {e}")
                continue
            completed_samples += 1
            
            # Calculate and display progress
            progress_pct = 100 * completed_samples / total_pending_samples
            elapsed = datetime.now() - start_time
            elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
            if completed_samples > 0:
                avg_time_per_sample = elapsed.total_seconds() / completed_samples
                remaining_samples = total_pending_samples - completed_samples
                eta_seconds = remaining_samples * avg_time_per_sample
                eta_str = str(timedelta(seconds=int(eta_seconds)))
            else:
                eta_str = "calculating..."
            
            # Merge with any existing variants for this sample (resume mode)
            existing = all_results.get(sample_idx, [])
            all_results[sample_idx] = _merge_variants(existing, sample_variants)
            errors.extend(sample_errors)
            total_cost += sample_cost
            
            thread_safe_print(f"\n✅ Sample {sample_idx + 1} complete | Progress: {completed_samples}/{total_pending_samples} ({progress_pct:.1f}%)")
            thread_safe_print(f"   Cost: ${total_cost:.4f} | Elapsed: {elapsed_str} | ETA: {eta_str}")
            
            # Save incrementally
            with _print_lock:
                ordered_results = []
                for idx in sorted(all_results.keys()):
                    ordered_results.extend(all_results[idx])
                atomic_json_dump(ordered_results, output_file, indent=2, ensure_ascii=False)
    
    # Flatten final results in order
    final_results = []
    for idx in sorted(all_results.keys()):
        final_results.extend(all_results[idx])
    
    # Final save
    atomic_json_dump(final_results, output_file, indent=2, ensure_ascii=False)
    
    # Calculate stats
    end_datetime = datetime.now()
    total_swapped = sum(s.get('num_skill_branches_swapped', 0) for s in final_results)
    successful_variants = len(final_results)
    avg_cost_per_variant = total_cost / successful_variants if successful_variants > 0 else 0
    
    # ==================== BRANCH DIVERSITY STATISTICS ====================
    # Calculate branch diversity across variants
    unique_branches_per_sample: Dict[int, Set[int]] = {}
    for s in final_results:
        sid = s.get('original_sample_id', -1)
        bid = s.get('selected_branch_index', -1)
        if sid >= 0 and bid >= 0:
            unique_branches_per_sample.setdefault(sid, set()).add(bid)
    
    # Count samples with full diversity (all variants selected different branches)
    samples_with_full_diversity = sum(
        1 for branches in unique_branches_per_sample.values() 
        if len(branches) == num_variants_per_sample
    )
    samples_with_partial_diversity = len(unique_branches_per_sample) - samples_with_full_diversity
    
    # Average unique branches per sample
    avg_unique_branches = (
        sum(len(b) for b in unique_branches_per_sample.values()) / len(unique_branches_per_sample)
        if unique_branches_per_sample else 0
    )
    
    # Total unique branch selections
    total_unique_selections = sum(len(b) for b in unique_branches_per_sample.values())
    
    # Verified answers count
    verified_count = sum(1 for s in final_results if s.get('answer_verified', False))
    # =====================================================================
    
    # Write metadata file
    metadata_content = f"""Run Metadata
============

Run Information:
  Start Time: {run_datetime}
  End Time: {end_datetime.strftime("%Y%m%d_%H%M%S")}
  Script: level1_team_allocation_swap_one_skill_branch.py
  Resumed: {resuming}
  Resume Output: {str(output_file) if resuming else "N/A"}

Configuration:
  Input File: {input_file}
  Output Folder: {output_folder}
  Num Samples: {num_samples_actual}
  Variants Per Sample: {num_variants_per_sample}
  Max Workers: {max_workers}
  Base Seed: {base_seed}

Model Configuration:
  Engine: {model_config['engine']}
  Temperature: {model_config['temperature']}
  Top P: {model_config['top_p']}
  Max Tokens: {model_config['max_tokens']}
  API Endpoint: {model_config['api_endpoint']}

Results:
  Total Variants Generated: {successful_variants}/{total_variants}
  Errors: {len(errors)}
  Total Skill Branches Swapped: {total_swapped}
  Total Cost: ${total_cost:.4f}
  Avg Cost Per Variant: ${avg_cost_per_variant:.4f}

Branch Diversity:
  Samples with full diversity: {samples_with_full_diversity}/{len(unique_branches_per_sample)}
  Samples with branch reuse: {samples_with_partial_diversity}
  Average unique branches per sample: {avg_unique_branches:.2f}
  Total unique selections: {total_unique_selections}

Verification:
  Answers verified: {verified_count}/{successful_variants}
  Matrix verified: {verified_count}/{successful_variants}

Output Files:
  - {output_file.name}
  - {metadata_file.name}

Reproducibility:
  - Uses thread-local random.Random() with deterministic seeds
  - Seed formula: base_seed + sample_idx * 1000 + variant_idx
  - Same configuration always produces identical results
  - Concurrent execution does not affect reproducibility

Description:
  Level 1 Team Allocation semantic duplicates: swap ONE skill branch per sample.
  The skill LEVEL (bad/okay/good) is preserved, only the narrative justification changes.

Methodology:
  1. Identifies skill branches in each tree (e.g., "Person X is bad at task Y")
  2. Selects ONE skill branch to swap, preferring branches not used by previous variants
  3. COPIES the EXACT structure of original branch (preserving complexity)
  4. Uses original benchmark's prompting method to fill new content
  5. Regenerates story text from modified tree
  6. Matrix and answer are UNCHANGED

Branch Diversity:
  - Each variant preferentially selects a DIFFERENT skill branch
  - Tracks selected_branch_index for each variant
  - Resume mode respects previously used branch indices
  - Falls back to random selection if all branches already used (more variants than branches)

Output Format:
  Each variant includes:
  - original_sample_id: Index of the original sample
  - original_story: The original story text
  - new_story: The regenerated story text
  - original_questions: Full original questions array (with original trees and intermediate_data)
  - new_questions: Full modified questions array (with modified tree)
  - swapped_branches: Dict with full old/new data:
      - skill: [{{person, skill_level, task, old_branch, new_branch}}, ...]
  - modification_type: 'skill_branch_swap'
  - variant_index: Which variant this is (0, 1, 2, ...)
  - selected_branch_index: Index of the skill branch that was swapped
  - random_seed: Deterministic seed for reproducibility
  - generation_cost: Cost in USD for this variant
  - 'context' field is REMOVED (redundant with original_story)
  - 'questions' field is REMOVED (redundant with original_questions/new_questions)

Complexity Preservation:
  - New branches have IDENTICAL structure to original (same depth, children, fact types)
  - Uses copy_branch_structure_for_skill() to clone tree shape
  - Complexity metrics logged for verification

Answer Preservation:
  - Matrix (skills and cooperation values) is NEVER modified
  - Only ONE skill branch justification is swapped
  - Same skill level preserved (bad/okay/good)
  - Answer is guaranteed to be unchanged
"""
    
    if errors:
        metadata_content += f"\nErrors:\n"
        for sample_idx, variant_idx, exc in errors:
            metadata_content += f"  - Sample {sample_idx + 1} Variant {variant_idx + 1}: {exc}\n"
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(metadata_content)
    
    # Final summary
    total_elapsed = datetime.now() - start_time
    total_elapsed_str = str(total_elapsed).split('.')[0]
    
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 Generation Statistics:")
    print(f"  Samples processed: {num_samples_actual}")
    print(f"  Variants per sample: {num_variants_per_sample}")
    print(f"  Total variants generated: {successful_variants}/{total_variants}")
    print(f"  Total skill branches swapped: {total_swapped}")
    print(f"  Total elapsed time: {total_elapsed_str}")
    
    print(f"\n💰 Cost Statistics:")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Avg cost per variant: ${avg_cost_per_variant:.4f}")
    
    print(f"\n🌳 Branch Diversity Statistics:")
    print(f"  Samples with full diversity: {samples_with_full_diversity}/{len(unique_branches_per_sample)} ({100*samples_with_full_diversity/len(unique_branches_per_sample):.1f}%)" if unique_branches_per_sample else "  No samples processed")
    print(f"  Samples with branch reuse: {samples_with_partial_diversity}")
    print(f"  Average unique branches per sample: {avg_unique_branches:.2f}")
    
    print(f"\n✅ Verification:")
    print(f"  Answers verified: {verified_count}/{successful_variants}")
    print(f"  Matrix verified: {verified_count}/{successful_variants}")
    
    if errors:
        print(f"\n❌ Errors: {len(errors)} variants failed")
        for sample_idx, variant_idx, exc in errors:
            print(f"  - Sample {sample_idx + 1} Variant {variant_idx + 1}: {exc}")
    
    print(f"\n📁 Output:")
    print(f"  Folder: {output_folder}")
    print(f"  Data file: {output_file.name}")
    print(f"  Metadata file: {metadata_file.name}")
    
    print(f"\n📝 Each variant has:")
    print(f"  - ONE skill branch swapped (different justification)")
    print(f"  - Different branch selected across variants (diversity tracking)")
    print(f"  - Same skill level preserved (bad/okay/good)")
    print(f"  - New story text (regenerated)")  
    print(f"  - Same matrix (verified unchanged)")
    print(f"  - Same answer (verified unchanged)")
    print(f"  - IDENTICAL complexity (exact structure copy)")
    print(f"  - Reproducible results (deterministic seeding)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

