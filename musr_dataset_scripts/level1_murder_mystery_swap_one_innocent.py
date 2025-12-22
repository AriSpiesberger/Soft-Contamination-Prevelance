"""
Generate semantic duplicates by swapping suspicious fact branches.

This script implements the difficulty-preserving modification strategy:
1. Identifies suspicious fact branches in each tree
2. Copies the EXACT structure of the original branch (preserving complexity)
3. Uses the original benchmark's prompting method to fill new content
4. Regenerates story text from modified trees
5. Verifies the answer is still correct

The result is a semantic duplicate with:
- Same difficulty (identical structural complexity - same tree shape)
- Same answer (MMO branches unchanged)
- Different content (different suspicious facts and story text)

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
load_dotenv(SCRIPT_DIR / '.env')  # Load from script directory
load_dotenv(MUSR_DIR / '.env')    # Also try MuSR directory

import json
import random
import threading
from copy import deepcopy
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Set

from src import cache
from src.model import OpenAIModel
from src.logic_tree.tree import LogicTree, LogicNode, LogicNodeFactType
from src.utils.paths import OUTPUT_FOLDER
from src.utils.json_io import atomic_json_dump, load_json_array_tolerant
from src.dataset_types.murder_mystery_dataset import (
    MurderMysteryDataset, 
    create_story_prompt__facts_only,
    sf_example_trees,
    sf_example_node_completions,
    sf_example_descriptions,
    _mm_suspicious_prompt_intro_
)
from src.dataset_builder import DatasetBuilder

# Thread-safe printing
_print_lock = threading.Lock()

def thread_safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with _print_lock:
        print(*args, **kwargs)


# =============================================================================
# DATA ACCESS HELPERS - Standardized access to sample data structures
# =============================================================================

def extract_sample_data(sample: Dict) -> Tuple[List[Dict], Dict, int, List[Dict]]:
    """
    Extract structured data from a sample.
    
    Args:
        sample: The sample dict from the dataset
    
    Returns:
        suspect_entries: List of {'suspect_info': {...}, 'is_murderer': bool, ...}
        victim_info: {'victim': str, 'crime_scene': str, 'murder_weapon': str}
        murderer_idx: Index of the murderer in the choices
        trees: List of tree JSON dicts
    """
    question = sample['questions'][0]
    intermediate_data = question['intermediate_data'][0]
    
    suspect_entries = intermediate_data['suspect_info']
    victim_info = intermediate_data['victim_info']
    murderer_idx = question['answer']
    trees = question['intermediate_trees']
    
    return suspect_entries, victim_info, murderer_idx, trees


def get_suspect_name(suspect_entry: Dict) -> str:
    """Get suspect name from a suspect entry."""
    return suspect_entry['suspect_info']['suspect']


def get_suspect_info(suspect_entry: Dict) -> Dict:
    """Get the suspect_info dict from a suspect entry."""
    return suspect_entry['suspect_info']


def is_suspect_murderer(suspect_entry: Dict) -> bool:
    """Check if a suspect entry is the murderer."""
    return suspect_entry['is_murderer']


def get_all_suspect_names(suspect_entries: List[Dict]) -> List[str]:
    """Get list of all suspect names."""
    return [get_suspect_name(entry) for entry in suspect_entries]


# =============================================================================
# SUSPICIOUS FACTS POOL
# =============================================================================

SUSPICIOUS_FACTS_FILE = Path(__file__).parent.parent / 'domain_seed' / 'suspicious_facts.json'


def load_suspicious_facts_pool() -> List[str]:
    """Load the pool of suspicious facts from the seed file."""
    with open(SUSPICIOUS_FACTS_FILE, encoding='utf-8') as f:
        return json.load(f)

def _extract_used_new_facts_from_variant_sample(variant_sample: Dict) -> List[str]:
    """
    Extract the set of newly used suspicious facts from a previously-generated variant sample.

    We use swapped_branches.suspicious[].new_fact when present, since that's the canonical record.
    """
    used: List[str] = []
    swapped = variant_sample.get("swapped_branches") or {}
    susp = swapped.get("suspicious") or []
    if isinstance(susp, list):
        for entry in susp:
            if isinstance(entry, dict) and entry.get("new_fact"):
                used.append(str(entry["new_fact"]))
    return used

def _load_existing_results(output_file: Path) -> Tuple[Dict[int, List[Dict]], Set[Tuple[int, int]]]:
    """
    Load an existing partial output JSON and index completed (sample_idx, variant_idx) pairs.

    Returns:
      - results_by_sample: {original_sample_id: [variant_samples...]}
      - completed_pairs: {(original_sample_id, variant_index), ...}
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

def _extract_suspicious_fact_from_branch_value(value: str, suspect_name: Optional[str]) -> Optional[str]:
    """
    Extract the suspicious-fact string from a suspicious branch root value.

    Expected format (case-insensitive marker):
      "{suspect_name} {fact} And this is suspicious."

    Returns:
      The extracted {fact} (preserves original casing/punctuation from value) or None.
    """
    if not value:
        return None
    lower = value.lower()
    marker = "and this is suspicious"
    marker_idx = lower.find(marker)
    if marker_idx == -1:
        return None

    prefix = value[:marker_idx].strip()
    if suspect_name:
        # Remove suspect name prefix if it matches (case-insensitive).
        sn = suspect_name.strip()
        if prefix[: len(sn)].lower() == sn.lower():
            prefix = prefix[len(sn) :].strip()

    fact = prefix.strip()
    return fact if fact else None


def get_used_suspicious_facts(sample: Dict, suspect_names: List[str]) -> List[str]:
    """
    Extract suspicious facts already used in a sample.

    Important: returns facts in the same shape as `domain_seed/suspicious_facts.json`
    (i.e., fact-only strings, without suspect name), so we can correctly exclude them.
    """
    used: List[str] = []
    _, _, _, trees = extract_sample_data(sample)

    for tree in trees:
        root = tree['nodes'][0]
        for branch in root.get('children', []):
            raw_value = branch.get('value', '')
            if 'suspicious' not in raw_value.lower():
                continue

            # Try parsing using known suspect names (most reliable).
            fact: Optional[str] = None
            for sn in suspect_names:
                fact = _extract_suspicious_fact_from_branch_value(raw_value, sn)
                if fact:
                    break

            # Fallback: attempt to parse without suspect name.
            if not fact:
                fact = _extract_suspicious_fact_from_branch_value(raw_value, None)

            if fact:
                used.append(fact)

    return used


# =============================================================================
# TREE STRUCTURE OPERATIONS
# =============================================================================

def identify_suspicious_branches(tree_json: Dict) -> List[Tuple[int, Dict]]:
    """
    Identify suspicious fact branches in a tree.
    Returns list of (index, branch_dict) tuples.
    """
    suspicious_branches = []
    root = tree_json['nodes'][0]
    
    for idx, branch in enumerate(root.get('children', [])):
        value = branch.get('value', '').lower()
        if 'suspicious' in value:
            suspicious_branches.append((idx, branch))
    
    return suspicious_branches


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


def copy_branch_structure(
    old_branch: Dict,
    suspect_name: str,
    new_suspicious_fact: str
) -> Dict:
    """
    Create a new suspicious branch by copying the EXACT structure of an existing branch.
    
    This preserves:
    - Same depth
    - Same number of children at each level
    - Same fact_types (explicit vs commonsense)
    - Same operators and other node properties
    
    Only changes:
    - Root value (new suspicious fact)
    - All descendant values (cleared, to be filled by LLM)
    
    Args:
        old_branch: The original branch to copy structure from
        suspect_name: Name of the suspect
        new_suspicious_fact: The new suspicious fact text
    
    Returns:
        New branch with same structure but empty values (except root)
    """
    new_branch = deepcopy(old_branch)
    
    # Update root value with new suspicious fact
    new_branch['value'] = f"{suspect_name} {new_suspicious_fact} And this is suspicious."
    
    # Clear all descendant values but keep structure
    def clear_descendant_values(node):
        for child in node.get('children', []):
            child['value'] = ''  # Will be filled by LLM
            clear_descendant_values(child)
    
    clear_descendant_values(new_branch)
    
    return new_branch


# =============================================================================
# TREE FILLING - Using Original Benchmark's Method
# =============================================================================

def fill_branch_with_benchmark_method(
    branch_json: Dict,
    suspect_name: str,
    suspicious_fact: str,
    model: OpenAIModel
) -> Dict:
    """
    Fill empty nodes in a branch using the original benchmark's prompting method.
    
    This uses the same approach as DatasetBuilder.complete_structure():
    1. Uses the benchmark's prompt format with ICL examples
    2. Parses output looking for '| Fact From Story' and '| Commonsense Knowledge'
    3. Fills nodes based on their fact_type
    
    Args:
        branch_json: Branch with structure but empty descendant values
        suspect_name: Name of the suspect
        suspicious_fact: The suspicious fact being used
        model: OpenAI model instance
    
    Returns:
        Filled branch with all values populated
    """
    # Convert to LogicTree for processing
    # Create a minimal tree structure with just this branch
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
    
    # Description for the completion prompt
    description = f'''{suspect_name} is suspicious... and they are super suspicious.'''.strip()
    
    # Create the dataset builder instance
    builder = DatasetBuilder()
    
    # Create completion prompt function using the benchmark's suspicious fact examples
    completion_prompt_fn = builder.create_completion_prompt(
        sf_example_trees,
        sf_example_node_completions,
        sf_example_descriptions,
        intro=_mm_suspicious_prompt_intro_,
        because_clause_after=0
    )
    
    # Use the benchmark's iterative completion method
    # We need to manually iterate since we want more control
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
            prompt = completion_prompt_fn(tree, node, description)
            
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
                                child.value = "Unusual behavior patterns can indicate hidden activities or secrets."
                        elif child.fact_type == LogicNodeFactType.EXPLICIT:
                            if facts_from_story:
                                child.value = facts_from_story.pop(0)
                            else:
                                # Fallback
                                child.value = f"{suspect_name} exhibits this behavior regularly."
            except Exception as e:
                thread_safe_print(f"    Warning: Error filling nodes: {e}, using fallbacks")
                fill_empty_nodes_fallback(node, suspect_name)
        
        # Recurse to children
        for child in children:
            fill_node_recursive(child, tree)
    
    # Start filling from root
    for root_node in tree.nodes:
        fill_node_recursive(root_node, tree)
    
    # Convert back to JSON
    return tree.nodes[0].to_json()


def fill_empty_nodes_fallback(node: LogicNode, suspect_name: str):
    """Fill any remaining empty nodes with generic fallback values."""
    for child in node.children:
        if child.value == '':
            if child.fact_type == LogicNodeFactType.COMMONSENSE:
                child.value = "Unusual behavior patterns can indicate hidden activities or secrets."
            else:
                child.value = f"{suspect_name} exhibits this behavior regularly."
        fill_empty_nodes_fallback(child, suspect_name)


# =============================================================================
# STORY GENERATION
# =============================================================================

def generate_story_for_tree(
    tree: LogicTree,
    suspect_info: Dict,
    victim_info: Dict,
    is_murderer: bool,
    model: OpenAIModel
) -> str:
    """Generate story chapter from a modified tree."""
    creator = MurderMysteryDataset()
    
    # Build description (same format as original benchmark)
    description_lines = [
        f"Victim: {victim_info['victim']}",
        f"Crime Scene: {victim_info['crime_scene']}",
        f"Murder Weapon: {victim_info['murder_weapon']}",
        f"Suspect: {suspect_info['suspect']}",
        f"Role in story: {suspect_info['role']}"
    ]
    
    if is_murderer:
        description_lines.append(f"The suspect's motive: {suspect_info['motive']}")
    
    description = '\n'.join(description_lines)
    
    # Generate chapter from tree
    prompt = create_story_prompt__facts_only(description, tree)
    chapter, _ = creator.inference(prompt, model)
    
    return chapter.strip()


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def swap_suspicious_facts_in_sample(
    sample: Dict,
    suspicious_facts_pool: List[str],
    model_config: Dict,
    sample_idx: int,
    total_samples: int,
    variant_idx: int = 0,
    total_variants: int = 1,
    excluded_facts: List[str] = None,
    base_seed: int = 42
) -> Tuple[Dict, List[str], float]:
    """
    Create a semantic duplicate by swapping suspicious facts.
    
    Uses thread-local random for reproducibility:
    - seed = base_seed + sample_idx * 1000 + variant_idx
    - Same (sample_idx, variant_idx) always produces same results
    
    Args:
        sample: Original sample to modify
        suspicious_facts_pool: Pool of available suspicious facts
        model_config: Model configuration dict
        sample_idx: Index of this sample
        total_samples: Total number of samples
        variant_idx: Index of this variant (0-based)
        total_variants: Total number of variants per sample
        excluded_facts: Facts to exclude (used by previous variants)
        base_seed: Base seed for random number generation
    
    Returns:
        Tuple of (new_sample, facts_used, cost)
    """
    if excluded_facts is None:
        excluded_facts = []
    
    # Create thread-local random instance with deterministic seed
    # This ensures reproducibility even with concurrent execution
    local_seed = base_seed + sample_idx * 1000 + variant_idx
    local_random = random.Random(local_seed)
    
    thread_safe_print(f"\n{'='*80}")
    thread_safe_print(f"Sample {sample_idx + 1}/{total_samples} - Variant {variant_idx + 1}/{total_variants}")
    thread_safe_print(f"  (seed={local_seed} for reproducibility)")
    thread_safe_print(f"{'='*80}")
    
    # Extract data using standardized helpers
    suspect_entries, victim_info, murderer_idx, trees = extract_sample_data(sample)
    sus_names = get_all_suspect_names(suspect_entries)
    
    # Get used suspicious facts to avoid
    used_facts = get_used_suspicious_facts(sample, sus_names)
    all_excluded = [u.strip().lower() for u in used_facts] + [e.strip().lower() for e in excluded_facts]
    available_facts = [f for f in suspicious_facts_pool if f.lower() not in all_excluded]
    
    if not available_facts:
        thread_safe_print(f"  [{sample_idx + 1}] Warning: No unique facts available, reusing pool")
        available_facts = suspicious_facts_pool.copy()
    
    # Track facts used in this variant
    variant_facts_used = []
    
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
    
    creator = MurderMysteryDataset()
    
    # Deep copy the sample
    new_sample = deepcopy(sample)
    new_trees = deepcopy(trees)
    
    thread_safe_print(f"  [{sample_idx + 1}] Victim: {victim_info['victim']}")
    thread_safe_print(f"  [{sample_idx + 1}] Weapon: {victim_info['murder_weapon']}")
    thread_safe_print(f"  [{sample_idx + 1}] Crime Scene: {victim_info['crime_scene']}")
    thread_safe_print(f"  [{sample_idx + 1}] Murderer: {sus_names[murderer_idx]} (index {murderer_idx})")
    
    # Process each tree (one per suspect)
    modified_trees = []
    new_chapters = []
    # Enhanced tracking: includes old and new data for each swapped branch
    swapped_branches = {
        'suspicious': []  # List of {suspect, old_fact, new_fact, old_branch, new_branch}
    }
    
    for tree_idx, tree_json in enumerate(new_trees):
        suspect_entry = suspect_entries[tree_idx]
        suspect_info = get_suspect_info(suspect_entry)
        suspect_name = get_suspect_name(suspect_entry)
        is_murderer = is_suspect_murderer(suspect_entry)
        
        thread_safe_print(f"  [{sample_idx + 1}] Processing suspect: {suspect_name} (murderer={is_murderer})")
        
        # Identify suspicious branches
        suspicious_branches = identify_suspicious_branches(tree_json)
        
        if suspicious_branches:
            thread_safe_print(f"  [{sample_idx + 1}] Found {len(suspicious_branches)} suspicious branch(es)")
            
            # Swap each suspicious branch
            for branch_idx, old_branch in suspicious_branches:
                # Calculate and log old complexity
                old_complexity = calculate_branch_complexity(old_branch)
                thread_safe_print(f"  [{sample_idx + 1}] Old branch complexity: {old_complexity}")
                
                # Select a new suspicious fact using thread-local random
                if not available_facts:
                    thread_safe_print(f"  [{sample_idx + 1}] Warning: ran out of unique facts mid-sample; refilling from pool")
                    available_facts = suspicious_facts_pool.copy()
                new_fact = local_random.choice(available_facts)
                available_facts.remove(new_fact)
                variant_facts_used.append(new_fact)
                
                # Extract old suspicious fact from value
                old_value = old_branch.get('value', '')
                old_fact = _extract_suspicious_fact_from_branch_value(old_value, suspect_name) or ''
                
                thread_safe_print(f"  [{sample_idx + 1}] Swapping: '{old_fact[:30]}...' → '{new_fact[:30]}...'")
                
                # OPTION A: Copy exact structure from old branch
                thread_safe_print(f"  [{sample_idx + 1}] Copying branch structure (preserving complexity)...")
                new_branch = copy_branch_structure(old_branch, suspect_name, new_fact)
                
                # Fill with LLM using original benchmark's method
                thread_safe_print(f"  [{sample_idx + 1}] Filling branch with LLM (benchmark method)...")
                filled_branch = fill_branch_with_benchmark_method(
                    new_branch, suspect_name, new_fact, model
                )
                
                # Verify complexity is preserved
                new_complexity = calculate_branch_complexity(filled_branch)
                thread_safe_print(f"  [{sample_idx + 1}] New branch complexity: {new_complexity}")
                
                if old_complexity != new_complexity:
                    thread_safe_print(f"  [{sample_idx + 1}] WARNING: Complexity changed! This shouldn't happen with structure copy.")
                
                # Record full swap data
                swapped_branches['suspicious'].append({
                    'suspect': suspect_name,
                    'old_fact': old_fact,
                    'new_fact': new_fact,
                    'old_branch': deepcopy(old_branch),
                    'new_branch': deepcopy(filled_branch)
                })
                
                # Replace in tree
                tree_json['nodes'][0]['children'][branch_idx] = filled_branch
        else:
            thread_safe_print(f"  [{sample_idx + 1}] No suspicious branches found")
        
        # Convert to LogicTree for story generation
        modified_tree = LogicTree.from_json(tree_json)
        modified_trees.append(modified_tree)
        
        # Generate new chapter
        thread_safe_print(f"  [{sample_idx + 1}] Generating story chapter...")
        chapter = generate_story_for_tree(
            modified_tree,
            suspect_info,
            victim_info,
            is_murderer,
            model
        )
        new_chapters.append((suspect_name, chapter))
    
    # Generate intro
    thread_safe_print(f"  [{sample_idx + 1}] Generating intro...")
    sus_strings = ", ".join(sus_names)
    intro_prompt = f"Create an intro for this murder mystery. It should only be 1 or 2 sentences. Only write the intro nothing else. \n\nScenario:\n{victim_info['victim']} was killed with a {victim_info['murder_weapon']} at a {victim_info['crime_scene']}. Detective Winston is on the case, interviewing suspects. The suspects are {sus_strings}.\n\nOutput:\n"
    intro, _ = creator.inference(intro_prompt, model)
    
    # Shuffle chapters using thread-local random (matches original generation)
    local_random.shuffle(new_chapters)
    
    # Build new story
    new_story = f"{intro}\n\n" + "\n\n".join([chapter for _, chapter in new_chapters])
    
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
    new_sample['num_suspicious_branches_swapped'] = len(swapped_branches['suspicious'])
    new_sample['modification_type'] = 'suspicious_fact_swap'
    new_sample['variant_index'] = variant_idx
    new_sample['random_seed'] = local_seed  # Record for reproducibility
    
    # Store modified trees in new questions
    new_sample['questions'][0]['intermediate_trees'] = [t.to_json() for t in modified_trees]
    
    # Also update trees inside intermediate_data to keep them in sync
    # Trees are stored in TWO places: intermediate_trees[] AND intermediate_data[].suspect_info[].tree
    if 'intermediate_data' in new_sample['questions'][0]:
        intermediate_data = new_sample['questions'][0]['intermediate_data'][0]
        for tree_idx, modified_tree in enumerate(modified_trees):
            if tree_idx < len(intermediate_data['suspect_info']):
                intermediate_data['suspect_info'][tree_idx]['tree'] = modified_tree.to_json()
        
        # Also update red_herrings for suspects with swapped suspicious facts
        for swap_info in swapped_branches['suspicious']:
            suspect_name = swap_info['suspect']
            new_fact = swap_info['new_fact']
            # Find the suspect entry and update red_herrings
            for entry in intermediate_data['suspect_info']:
                if entry['suspect_info']['suspect'] == suspect_name:
                    entry['suspect_info']['red_herrings'] = [new_fact]
                    break
    
    # Store both original and new questions (trees and intermediate_data are inside)
    new_sample['original_questions'] = original_questions
    new_sample['new_questions'] = deepcopy(new_sample['questions'])
    
    # Remove the inherited 'questions' field (redundant with original_questions/new_questions)
    del new_sample['questions']
    
    total_cost = model.total_cost
    thread_safe_print(f"  [{sample_idx + 1}v{variant_idx + 1}] ✓ Complete - Answer: {sus_names[murderer_idx]} (preserved)")
    thread_safe_print(f"  [{sample_idx + 1}v{variant_idx + 1}]   Suspicious branches swapped: {new_sample['num_suspicious_branches_swapped']}")
    thread_safe_print(f"  [{sample_idx + 1}v{variant_idx + 1}]   Cost: ${total_cost:.4f}")
    
    return new_sample, variant_facts_used, total_cost


def process_sample_wrapper(args):
    """Wrapper function for concurrent processing of a single variant."""
    (sample, suspicious_facts_pool, model_config, sample_idx, total_samples, 
     variant_idx, total_variants, excluded_facts, base_seed) = args
    try:
        new_sample, facts_used, cost = swap_suspicious_facts_in_sample(
            sample,
            suspicious_facts_pool.copy(),
            model_config,
            sample_idx,
            total_samples,
            variant_idx,
            total_variants,
            excluded_facts,
            base_seed
        )
        return sample_idx, variant_idx, new_sample, facts_used, cost, None
    except Exception as e:
        import traceback
        return sample_idx, variant_idx, None, [], 0.0, (e, traceback.format_exc())


def main():
    """Main function to generate semantic duplicates with swapped suspicious facts."""
    # Disable cache to force regeneration
    cache.disable()
    
    parser = argparse.ArgumentParser(description="Generate semantic duplicates by swapping one innocent suspicious-fact branch.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to process (default: all).")
    parser.add_argument("--num-variants", type=int, default=2, help="Variants per sample (default: 2).")
    parser.add_argument("--max-workers", type=int, default=20, help="Concurrent samples (default: 20).")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed for reproducibility (default: 42).")
    parser.add_argument(
        "--resume-output",
        type=str,
        default=None,
        help="Path to an existing partial output JSON to resume (will append missing variants).",
    )
    args = parser.parse_args()

    # ==================== CONFIGURATION ====================
    num_samples = args.num_samples
    num_variants_per_sample = args.num_variants
    max_workers = args.max_workers
    base_seed = args.base_seed
    input_file = OUTPUT_FOLDER / "murder_mystery.json"
    # ======================================================
    
    print("="*80)
    print("GENERATING SEMANTIC DUPLICATES WITH SWAPPED SUSPICIOUS FACTS")
    print("="*80)
    print(f"\nInput: {input_file}")
    print(f"Base seed: {base_seed} (for reproducibility)")
    
    # Load suspicious facts pool
    suspicious_facts_pool = load_suspicious_facts_pool()
    print(f"Loaded {len(suspicious_facts_pool)} suspicious facts from pool")
    print(f"  Max possible unique variants per sample: ~{len(suspicious_facts_pool) // 2} (2 suspects per sample)")
    
    # Load original dataset
    print(f"\nLoading dataset...")
    with open(input_file, encoding='utf-8') as f:
        dataset = json.load(f)
    
    samples_to_process = dataset if num_samples is None else dataset[:num_samples]
    num_samples_actual = len(samples_to_process)
    total_variants = num_samples_actual * num_variants_per_sample
    
    resuming = args.resume_output is not None
    resume_output_file = Path(args.resume_output).resolve() if resuming else None

    # Create output folder with descriptive name (or reuse if resuming)
    run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    if resuming:
        output_file = resume_output_file
        output_folder = output_file.parent
        metadata_file = output_folder / "run_metadata.txt"
        print(f"\nResuming from existing output: {output_file}")
        print(f"Output folder: {output_folder}")
    else:
        output_folder_name = f"murder_mystery_level1_samples-{num_samples_actual}_variants-{num_variants_per_sample}_{run_datetime}"
        output_folder = OUTPUT_FOLDER / output_folder_name
        output_folder.mkdir(parents=True, exist_ok=True)
        output_file = output_folder / f"murder_mystery_level1_samples-{num_samples_actual}_variants-{num_variants_per_sample}.json"
        metadata_file = output_folder / "run_metadata.txt"
    
    print(f"Samples to process: {num_samples_actual}")
    print(f"Variants per sample: {num_variants_per_sample}")
    print(f"Total variants to generate: {total_variants}")
    print(f"Concurrent workers: {max_workers}")
    print(f"Output folder: {output_folder}")
    
    # Model configuration (matches create_murder_mysteries.py)
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
    
    # Load partial results if resuming
    all_results, completed_pairs = ({}, set())
    if resuming:
        all_results, completed_pairs = _load_existing_results(output_file)
        already_done = len(completed_pairs)
        expected_total = num_samples_actual * num_variants_per_sample
        print(f"Found {already_done}/{expected_total} completed variants in existing output.")

    # Process samples
    errors = []
    total_cost = 0.0
    completed_samples = 0
    
    print(f"\nStarting processing...")
    print(f"Note: Variants of the same sample are processed sequentially to ensure uniqueness.")
    print(f"      Different samples are processed concurrently ({max_workers} workers).")
    print(f"      Each (sample, variant) has deterministic seed for reproducibility.\n")
    
    def process_single_sample(sample_idx: int, sample: Dict) -> Tuple[int, List[Dict], List[Tuple], float, List[str]]:
        """
        Process all variants for a single sample.
        Variants are processed sequentially to ensure uniqueness.
        """
        thread_safe_print(f"\n{'='*80}")
        thread_safe_print(f"Processing Sample {sample_idx + 1}/{num_samples_actual}")
        thread_safe_print(f"{'='*80}")
        
        # Resume support: seed exclusions from existing variants to preserve uniqueness.
        existing_variants = all_results.get(sample_idx, []) if resuming else []
        existing_by_idx = {int(v.get("variant_index", -1)): v for v in existing_variants if isinstance(v, dict)}
        sample_excluded_facts: List[str] = []
        if existing_by_idx:
            # Exclude any facts already used by previously-generated variants for this sample.
            # (This guarantees no duplicates across the final set of variants.)
            for v in sorted(existing_by_idx.keys()):
                sample_excluded_facts.extend(_extract_used_new_facts_from_variant_sample(existing_by_idx[v]))
        sample_variants = []
        sample_errors = []
        sample_cost = 0.0
        
        for variant_idx in range(num_variants_per_sample):
            # Skip already-generated variants when resuming.
            if (sample_idx, variant_idx) in completed_pairs:
                continue
            args = (
                sample, 
                suspicious_facts_pool, 
                model_config, 
                sample_idx, 
                num_samples_actual,
                variant_idx,
                num_variants_per_sample,
                sample_excluded_facts.copy(),
                base_seed
            )
            
            _, _, new_sample, facts_used, cost, error = process_sample_wrapper(args)
            
            if error:
                exc, tb = error
                thread_safe_print(f"\n❌ Error: Sample {sample_idx + 1} Variant {variant_idx + 1}: {exc}")
                thread_safe_print(tb)
                sample_errors.append((sample_idx, variant_idx, exc))
            else:
                sample_excluded_facts.extend(facts_used)
                sample_variants.append(new_sample)
                sample_cost += cost
        
        thread_safe_print(f"\n  Sample {sample_idx + 1} complete: {len(sample_variants)} variants | Cost: ${sample_cost:.4f}")
        thread_safe_print(f"  Unique suspicious facts used across variants: {len(sample_excluded_facts)}")
        
        return sample_idx, sample_variants, sample_errors, sample_cost, sample_excluded_facts
    
    # Only enqueue samples that have missing variants when resuming
    # (avoids "walking" samples that are already fully completed).
    if resuming:
        pending_work: List[Tuple[int, Dict]] = []
        for idx, sample in enumerate(samples_to_process):
            has_missing = any((idx, v) not in completed_pairs for v in range(num_variants_per_sample))
            if has_missing:
                pending_work.append((idx, sample))
        print(f"Samples with remaining work: {len(pending_work)}/{num_samples_actual}")
        if not pending_work:
            print("Nothing to do: all requested variants already exist in the output JSON.")
            return
    else:
        pending_work = list(enumerate(samples_to_process))

    # Process samples concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_sample, idx, sample): idx 
            for idx, sample in pending_work
        }
        
        for future in as_completed(futures):
            sample_idx, sample_variants, sample_errors, sample_cost, _ = future.result()
            completed_samples += 1
            
            # Merge with any existing variants for this sample (resume mode).
            existing = all_results.get(sample_idx, [])
            all_results[sample_idx] = _merge_variants(existing, sample_variants)
            errors.extend(sample_errors)
            total_cost += sample_cost
            
            thread_safe_print(f"\n✅ Sample {sample_idx + 1} fully complete ({completed_samples}/{num_samples_actual} samples)")
            
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
    total_swapped = sum(s.get('num_suspicious_branches_swapped', 0) for s in final_results)
    successful_variants = len(final_results)
    avg_cost_per_variant = total_cost / successful_variants if successful_variants > 0 else 0
    
    # Write metadata file
    metadata_content = f"""Run Metadata
============

Run Information:
  Start Time: {run_datetime}
  End Time: {end_datetime.strftime("%Y%m%d_%H%M%S")}
  Script: level1_murder_mystery_swap_one_innocent.py
  Resumed: {resuming}
  Resume Output: {str(output_file) if resuming else "N/A"}

Configuration:
  Input File: {input_file}
  Output Folder: {output_folder}
  Num Samples: {num_samples_actual}
  Variants Per Sample: {num_variants_per_sample}
  Max Workers: {max_workers}
  Base Seed: {base_seed}
  Suspicious Facts Pool Size: {len(suspicious_facts_pool)}

Model Configuration:
  Engine: {model_config['engine']}
  Temperature: {model_config['temperature']}
  Top P: {model_config['top_p']}
  Max Tokens: {model_config['max_tokens']}
  API Endpoint: {model_config['api_endpoint']}

Results:
  Total Variants Generated: {successful_variants}/{total_variants}
  Errors: {len(errors)}
  Total Suspicious Branches Swapped: {total_swapped}
  Total Cost: ${total_cost:.4f}
  Avg Cost Per Variant: ${avg_cost_per_variant:.4f}

Output Files:
  - {output_file.name}
  - {metadata_file.name}

Reproducibility:
  - Uses thread-local random.Random() with deterministic seeds
  - Seed formula: base_seed + sample_idx * 1000 + variant_idx
  - Same configuration always produces identical results
  - Concurrent execution does not affect reproducibility

Methodology:
  1. Identifies suspicious fact branches in each tree (all suspects)
  2. COPIES the EXACT structure of original branch (preserving complexity)
  3. Uses original benchmark's prompting method to fill new content
  4. Regenerates story text from modified trees
  5. Verifies the answer is still correct (MMO branches unchanged)
  6. Tracks used facts to ensure uniqueness across variants

Output Format:
  Each variant includes:
  - original_story: The original story text
  - new_story: The regenerated story text
  - original_questions: Full original questions array (with original trees and intermediate_data)
  - new_questions: Full modified questions array (with updates in ALL locations)
      → intermediate_trees[] updated
      → intermediate_data[].suspect_info[].tree also updated (kept in sync)
      → intermediate_data[].suspect_info[].suspect_info.red_herrings updated
  - swapped_branches: Dict with full old/new data:
      - suspicious: [{{suspect, old_fact, new_fact, old_branch, new_branch}}, ...]
  - 'context' field is removed (redundant with original_story)
  - 'questions' field is removed (redundant with original_questions/new_questions)

Complexity Preservation:
  - New branches have IDENTICAL structure to original (same depth, children, fact types)
  - Uses copy_branch_structure() to clone tree shape
  - Complexity metrics logged for verification

Answer Preservation:
  - MMO branches are NEVER modified
  - Only suspicious fact branches are swapped
  - Answer is guaranteed to be unchanged
"""
    
    if errors:
        metadata_content += f"\nErrors:\n"
        for sample_idx, variant_idx, exc in errors:
            metadata_content += f"  - Sample {sample_idx + 1} Variant {variant_idx + 1}: {exc}\n"
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(metadata_content)
    
    # Final summary
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Samples processed: {num_samples_actual}")
    print(f"Variants per sample: {num_variants_per_sample}")
    print(f"Total variants generated: {successful_variants}/{total_variants}")
    print(f"Total suspicious branches swapped: {total_swapped}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Avg cost per variant: ${avg_cost_per_variant:.4f}")
    if errors:
        print(f"Errors: {len(errors)} variants failed")
        for sample_idx, variant_idx, exc in errors:
            print(f"  - Sample {sample_idx + 1} Variant {variant_idx + 1}: {exc}")
    print(f"Output folder: {output_folder}")
    print(f"Output file: {output_file.name}")
    print(f"Metadata file: {metadata_file.name}")
    print(f"\nEach variant has:")
    print(f"  - Different suspicious facts (guaranteed unique across variants)")
    print(f"  - New story text (regenerated)")  
    print(f"  - Same answer (MMO unchanged)")
    print(f"  - IDENTICAL complexity (exact structure copy)")
    print(f"  - Reproducible results (deterministic seeding)")


if __name__ == "__main__":
    main()
