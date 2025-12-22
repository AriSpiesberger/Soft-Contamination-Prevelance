"""
Level 0: Regenerate Team Allocation stories from existing logic trees.

This script demonstrates that with temperature=1.0, the same trees produce
different story text but with identical semantic content (facts).

The approach mirrors level0_generate_duplicates_with_no_tree_changes.py for
murder mysteries:
- Trees are UNCHANGED
- Matrix (people_levels) is UNCHANGED  
- Answer (best_pair) is UNCHANGED
- Only the story narrative text is regenerated

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
from copy import deepcopy
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Set

from src import cache
from src.model import OpenAIModel
from src.logic_tree.tree import LogicTree
from src.utils.paths import OUTPUT_FOLDER
from src.utils.json_io import atomic_json_dump, load_json_array_tolerant
from src.dataset_types.team_allocation import TeamAllocationDataset


# Thread-safe printing
_print_lock = threading.Lock()


def thread_safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with _print_lock:
        print(*args, **kwargs)


# =============================================================================
# DATA ACCESS HELPERS
# =============================================================================

def extract_sample_data(sample: Dict) -> Tuple[LogicTree, Dict, List[str], List, Dict]:
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


def extract_people_from_matrix(matrix: Dict) -> List[str]:
    """Extract people names from the matrix keys."""
    return list(matrix.keys())


def extract_scenario_description(tree: LogicTree) -> str:
    """Extract the scenario description from the tree root."""
    if tree.nodes and tree.nodes[0].value:
        return tree.nodes[0].value
    return ""


# =============================================================================
# STORY GENERATION
# =============================================================================

def generate_story_from_facts(
    facts: List[str],
    people: List[str],
    tasks: List[str],
    scenario_description: str,
    model: OpenAIModel
) -> Tuple[str, str]:
    """
    Generate a new story from the facts.
    
    This uses the same prompt structure as create_team_allocation.py.
    
    Args:
        facts: List of fact strings from the tree
        people: List of people names
        tasks: List of task names
        scenario_description: The scenario description
        model: OpenAI model instance
    
    Returns:
        Tuple of (story_text, intro_paragraph)
    """
    creator = TeamAllocationDataset()
    
    # Format facts as bullet points
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
    
    return final_story, fixed_intro


def generate_single_variant(
    tree: LogicTree,
    intermediate_data: Dict,
    original_context: str,
    sample_idx: int,
    variant_idx: int,
    num_variants: int,
    model_config: Dict,
    base_seed: int = 42
) -> Tuple[str, float]:
    """
    Generate a single story variant.
    
    Uses thread-local random for reproducibility:
    - seed = base_seed + sample_idx * 1000 + variant_idx
    - Same (sample_idx, variant_idx) always produces same results
    
    Args:
        tree: LogicTree object
        intermediate_data: {tasks, matrix, best_pair}
        original_context: Original story text
        sample_idx: Index of this sample
        variant_idx: Index of this variant (0-based)
        num_variants: Total variants per sample
        model_config: Dict with model configuration
        base_seed: Base seed for random number generation
    
    Returns:
        Tuple of (story_text, cost)
    """
    # Create thread-local random instance with deterministic seed
    local_seed = base_seed + sample_idx * 1000 + variant_idx
    local_random = random.Random(local_seed)
    _ = local_random  # Mark as intentionally unused (for future use if needed)
    
    # Create a fresh model instance for this thread
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
    
    # Extract data
    tasks = intermediate_data['tasks']
    matrix = intermediate_data['matrix']
    people = extract_people_from_matrix(matrix)
    scenario_description = extract_scenario_description(tree)
    facts = extract_facts_from_tree(tree)
    
    thread_safe_print(f"    [{variant_idx + 1}/{num_variants}] Generating story... (seed={local_seed})")
    thread_safe_print(f"    [{variant_idx + 1}/{num_variants}] Facts count: {len(facts)}")
    
    # Generate new story
    story, intro = generate_story_from_facts(
        facts, people, tasks, scenario_description, model
    )
    
    cost = model.total_cost
    thread_safe_print(f"    [{variant_idx + 1}/{num_variants}] ✓ Complete (cost: ${cost:.4f})")
    
    return story, cost


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


# =============================================================================
# SAMPLE PROCESSING
# =============================================================================

def process_single_sample(
    sample: Dict,
    sample_idx: int,
    num_samples_actual: int,
    num_variants_per_sample: int,
    model_config: Dict,
    base_seed: int,
    variants_to_generate: Optional[List[int]] = None,
) -> Tuple[int, List[Dict], float]:
    """
    Process all variants for a single sample.
    
    Args:
        sample: The sample to process
        sample_idx: Index of this sample
        num_samples_actual: Total number of samples
        num_variants_per_sample: Number of variants to generate
        model_config: Model configuration dict
        base_seed: Base seed for random number generation
        variants_to_generate: Optional list of specific variant indices to generate
    
    Returns:
        Tuple of (sample_idx, list of variant samples, total cost)
    """
    thread_safe_print(f"\n{'='*80}")
    thread_safe_print(f"Sample {sample_idx + 1}/{num_samples_actual}")
    thread_safe_print(f"{'='*80}")
    
    # Extract data
    tree, intermediate_data, choices, answer, tree_json = extract_sample_data(sample)
    
    tasks = intermediate_data['tasks']
    matrix = intermediate_data['matrix']
    best_pair = intermediate_data['best_pair']
    people = extract_people_from_matrix(matrix)
    
    thread_safe_print(f"  [{sample_idx + 1}] Tasks: {tasks}")
    thread_safe_print(f"  [{sample_idx + 1}] People: {people}")
    thread_safe_print(f"  [{sample_idx + 1}] Best pair: {best_pair}")
    thread_safe_print(f"  [{sample_idx + 1}] Answer index: {answer}")
    thread_safe_print(f"  [{sample_idx + 1}] Generating {num_variants_per_sample} variants...")
    
    original_context = sample.get('context', '')
    
    sample_cost = 0.0
    variant_results = []
    
    if variants_to_generate is None:
        variants_to_generate = list(range(num_variants_per_sample))

    # Generate variants sequentially
    for variant_idx in variants_to_generate:
        story, cost = generate_single_variant(
            tree,
            intermediate_data,
            original_context,
            sample_idx,
            variant_idx,
            num_variants_per_sample,
            model_config,
            base_seed
        )
        
        # Create sample entry for this variant
        new_sample = deepcopy(sample)
        local_seed = base_seed + sample_idx * 1000 + variant_idx
        
        new_sample['original_sample_id'] = sample_idx
        new_sample['new_story'] = story
        new_sample['original_story'] = original_context
        new_sample['modification_type'] = 'regenerated'
        new_sample['variant_index'] = variant_idx
        new_sample['random_seed'] = local_seed
        new_sample['generation_cost'] = float(cost)
        
        # Remove 'context' field since it's redundant with 'original_story'
        if 'context' in new_sample:
            del new_sample['context']
        
        variant_results.append(new_sample)
        sample_cost += cost
    
    thread_safe_print(f"\n  [{sample_idx + 1}] ✓ Complete: {len(variant_results)} variants | Cost: ${sample_cost:.4f}")
    
    return sample_idx, variant_results, sample_cost


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function to regenerate team allocation stories from existing trees."""
    # Disable cache to force regeneration
    cache.disable()
    
    parser = argparse.ArgumentParser(
        description="Level 0: Regenerate Team Allocation stories from existing logic trees (no tree changes)."
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
    args = parser.parse_args()

    # ==================== CONFIGURATION ====================
    num_samples = args.num_samples
    num_variants_per_sample = args.num_variants
    max_workers = args.max_workers
    base_seed = args.base_seed
    
    if args.input_file:
        input_file = Path(args.input_file)
    else:
        input_file = OUTPUT_FOLDER / "team_allocation.json"
    # ======================================================
    
    print("="*80)
    print("LEVEL 0: REGENERATING TEAM ALLOCATION STORIES FROM EXISTING TREES")
    print("="*80)
    print(f"\nInput: {input_file}")
    print(f"Base seed: {base_seed} (for reproducibility)")
    
    # Load original dataset
    print(f"\nLoading dataset...")
    with open(input_file, 'r', encoding='utf-8') as f:
        original_dataset = json.load(f)
    
    # Take first N samples (or all if num_samples is None)
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
        output_folder_name = f"team_allocation_level0_samples-{num_samples_actual}_variants-{num_variants_per_sample}_{run_datetime}"
        output_folder = OUTPUT_FOLDER / output_folder_name
        output_folder.mkdir(parents=True, exist_ok=True)
        output_file = output_folder / f"team_allocation_level0_samples-{num_samples_actual}_variants-{num_variants_per_sample}.json"
        metadata_file = output_folder / "run_metadata.txt"
    
    print(f"Samples to process: {num_samples_actual}")
    print(f"Variants per sample: {num_variants_per_sample}")
    print(f"Total variants to generate: {total_variants}")
    print(f"Concurrent workers: {max_workers}")
    print(f"Output folder: {output_folder}")
    print("This will show that same trees -> different story text\n")
    
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
    
    # Load partial results if resuming
    all_results, completed_pairs = ({}, set())
    if resuming:
        all_results, completed_pairs = _load_existing_results(output_file)
        already_done = len(completed_pairs)
        expected_total = num_samples_actual * num_variants_per_sample
        print(f"Found {already_done}/{expected_total} completed variants in existing output.")

    # Process samples concurrently
    errors = []
    total_cost = 0.0
    completed_samples = 0
    
    print(f"\nStarting processing...")
    print(f"Note: Different samples are processed concurrently ({max_workers} workers).")
    print(f"      Each (sample, variant) has deterministic seed for reproducibility.\n")
    
    # Build per-sample missing variant list
    pending_work: List[Tuple[int, Dict, List[int]]] = []
    for idx, sample in enumerate(samples_to_process):
        missing = [v for v in range(num_variants_per_sample) if (idx, v) not in completed_pairs]
        if missing:
            pending_work.append((idx, sample, missing))

    if resuming:
        print(f"Samples with remaining work: {len(pending_work)}/{num_samples_actual}")
        if not pending_work:
            print("Nothing to do: all requested variants already exist in the output JSON.")
            return

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
            ): idx 
            for (idx, sample, missing_variants) in pending_work
        }
        
        for future in as_completed(futures):
            submitted_idx = futures[future]
            try:
                sample_idx, sample_variants, sample_cost = future.result()
            except Exception as e:
                import traceback
                errors.append((submitted_idx, e, traceback.format_exc()))
                thread_safe_print(f"\n❌ Error: Sample {submitted_idx + 1}: {e}")
                continue
            completed_samples += 1
            
            # Merge with any existing variants for this sample (resume mode)
            existing = all_results.get(sample_idx, [])
            all_results[sample_idx] = _merge_variants(existing, sample_variants)
            total_cost += sample_cost
            
            thread_safe_print(f"\n✅ Sample {sample_idx + 1} fully complete ({completed_samples}/{len(pending_work)} samples)")
            thread_safe_print(f"  Running total cost: ${total_cost:.4f}")
            
            # Save incrementally
            with _print_lock:
                ordered_results = []
                for idx in sorted(all_results.keys()):
                    ordered_results.extend(all_results[idx])
                atomic_json_dump(ordered_results, output_file, indent=2, ensure_ascii=False)
    
    # Flatten final results in order
    all_results_list = []
    for idx in sorted(all_results.keys()):
        all_results_list.extend(all_results[idx])
    
    # Final save
    atomic_json_dump(all_results_list, output_file, indent=2, ensure_ascii=False)
    
    # Calculate final stats
    end_datetime = datetime.now()
    successful_variants = len(all_results_list)
    avg_cost_per_variant = total_cost / successful_variants if successful_variants > 0 else 0
    
    # Write metadata file
    metadata_content = f"""Run Metadata
============

Run Information:
  Start Time: {run_datetime}
  End Time: {end_datetime.strftime("%Y%m%d_%H%M%S")}
  Script: level0_team_allocation_regenerate_stories.py
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
  This run regenerated Team Allocation stories from existing logic trees using temperature={model_config['temperature']}.
  The same trees produce different story text but with identical semantic content (facts).

Methodology:
  1. Loads original samples with their logic trees
  2. Extracts facts from the tree (skill facts + cooperation facts)
  3. Regenerates story text from UNCHANGED facts/trees
  4. Each variant has the same facts but different narrative text
  5. Answer is unchanged (matrix and best_pair are unchanged)

What's Preserved:
  - Logic tree (intermediate_trees)
  - Skill/cooperation matrix (intermediate_data.matrix)
  - Best pair assignment (intermediate_data.best_pair)
  - Tasks (intermediate_data.tasks)
  - Answer index
  - Choices

What's Changed:
  - Story text - regenerated with temperature=1.0
  - 'context' field removed (redundant with 'original_story')

Output Format:
  Each variant includes:
  - original_sample_id: Index of the original sample
  - original_story: The original story text (from 'context')
  - new_story: The regenerated story text
  - modification_type: 'regenerated'
  - variant_index: Which variant this is (0, 1, 2, ...)
  - random_seed: Deterministic seed for reproducibility
  - generation_cost: Cost in USD for this variant
  - 'context' field is REMOVED (redundant with original_story)
"""

    if errors:
        metadata_content += "\nErrors:\n"
        for sample_idx, exc, _tb in errors:
            metadata_content += f"  - Sample {sample_idx + 1}: {exc}\n"
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
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
    print(f"  - Same matrix (skills/cooperation unchanged)")
    print(f"  - Same answer (best_pair unchanged)")
    print(f"  - New story text (regenerated with temperature=1.0)")
    print(f"  - Reproducible results (deterministic seeding)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

