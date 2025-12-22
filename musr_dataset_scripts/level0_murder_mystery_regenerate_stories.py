"""
Regenerate stories from existing logic trees to demonstrate that with temperature=1.0,
the same trees produce different story text but with identical semantic content (facts).

This script takes samples from murder_mystery.json and regenerates the story text 
using the same logic trees, but with temperature=1.0 for variety.

Follows the same workflow as create_murder_mysteries.py including random.shuffle()
of chapters.

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
from src.logic_tree.tree import LogicTree
from src.utils.paths import OUTPUT_FOLDER
from src.utils.json_io import atomic_json_dump, load_json_array_tolerant
from src.dataset_types.murder_mystery_dataset import MurderMysteryDataset, create_story_prompt__facts_only


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
        trees_json: List of tree JSON dicts
    """
    question = sample['questions'][0]
    intermediate_data = question['intermediate_data'][0]
    
    suspect_entries = intermediate_data['suspect_info']
    victim_info = intermediate_data['victim_info']
    murderer_idx = question['answer']
    trees_json = question['intermediate_trees']
    
    return suspect_entries, victim_info, murderer_idx, trees_json


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
# STORY GENERATION
# =============================================================================

def generate_single_story(
    trees: List[LogicTree],
    suspect_entries: List[Dict],
    victim_info: Dict,
    sus_names: List[str],
    sus_strings: str,
    sample_idx: int,
    variant_idx: int,
    num_variants: int,
    model_config: Dict,
    base_seed: int = 42
) -> Tuple[str, float]:
    """
    Generate a single regenerated story.
    
    Uses thread-local random for reproducibility:
    - seed = base_seed + sample_idx * 1000 + variant_idx
    - Same (sample_idx, variant_idx) always produces same results
    
    Args:
        trees: List of LogicTree objects
        suspect_entries: List of suspect entry dicts
        victim_info: Victim info dict
        sus_names: List of suspect names
        sus_strings: Comma-separated suspect names
        sample_idx: Index of this sample
        variant_idx: Index of this variant (0-based)
        num_variants: Total variants per sample (for display)
        model_config: Dict with model configuration
        base_seed: Base seed for random number generation
    
    Returns:
        Tuple of (story_text, cost)
    """
    # Create thread-local random instance with deterministic seed
    local_seed = base_seed + sample_idx * 1000 + variant_idx
    local_random = random.Random(local_seed)
    
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
    
    creator = MurderMysteryDataset()
    
    # Generate intro
    thread_safe_print(f"    [{variant_idx + 1}/{num_variants}] Generating intro... (seed={local_seed})")
    intro_prompt = f"Create an intro for this murder mystery. It should only be 1 or 2 sentences. Only write the intro nothing else. \n\nScenario:\n{victim_info['victim']} was killed with a {victim_info['murder_weapon']} at a {victim_info['crime_scene']}. Detective Winston is on the case, interviewing suspects. The suspects are {sus_strings}.\n\nOutput:\n"
    intro, _ = creator.inference(intro_prompt, model)
    
    # Generate chapters
    new_chapters = []
    for tree_idx, (tree, suspect_entry) in enumerate(zip(trees, suspect_entries)):
        suspect_name = get_suspect_name(suspect_entry)
        suspect_info = get_suspect_info(suspect_entry)
        is_murderer = is_suspect_murderer(suspect_entry)
        
        thread_safe_print(f"    [{variant_idx + 1}/{num_variants}] Generating chapter for {suspect_name}...")
        
        # Build description (exclude motive unless they're the murderer)
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
        
        new_chapters.append((suspect_name, chapter.strip()))
    
    # Shuffle chapter order using thread-local random (matches original generation)
    local_random.shuffle(new_chapters)
    
    # Build story
    story = f"{intro}\n\n" + "\n\n".join([chapter for _, chapter in new_chapters])
    
    cost = model.total_cost
    thread_safe_print(f"    [{variant_idx + 1}/{num_variants}] ✓ Complete (cost: ${cost:.4f})")
    
    return story, cost


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
            # Old/unexpected format; can't safely resume.
            raise ValueError(
                f"Output JSON is missing required keys for resume. "
                f"Expected 'original_sample_id' and 'variant_index'."
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
    
    Returns:
        Tuple of (sample_idx, list of variant samples, total cost)
    """
    thread_safe_print(f"\n{'='*80}")
    thread_safe_print(f"Sample {sample_idx + 1}/{num_samples_actual}")
    thread_safe_print(f"{'='*80}")
    
    # Extract data using standardized helpers
    suspect_entries, victim_info, murderer_idx, trees_json = extract_sample_data(sample)
    
    # Convert trees from JSON
    trees = [LogicTree.from_json(tree_json) for tree_json in trees_json]
    
    sus_names = get_all_suspect_names(suspect_entries)
    sus_strings = ", ".join(sus_names)
    
    thread_safe_print(f"  [{sample_idx + 1}] Victim: {victim_info['victim']}")
    thread_safe_print(f"  [{sample_idx + 1}] Weapon: {victim_info['murder_weapon']}")
    thread_safe_print(f"  [{sample_idx + 1}] Crime Scene: {victim_info['crime_scene']}")
    thread_safe_print(f"  [{sample_idx + 1}] Murderer: {sus_names[murderer_idx]} (index {murderer_idx})")
    thread_safe_print(f"  [{sample_idx + 1}] Generating {num_variants_per_sample} versions...")
    
    sample_cost = 0.0
    variant_results = []
    
    if variants_to_generate is None:
        variants_to_generate = list(range(num_variants_per_sample))

    # Generate variants sequentially (keeps API pressure lower and output cleaner)
    for variant_idx in variants_to_generate:
        story, cost = generate_single_story(
            trees,
            suspect_entries,
            victim_info,
            sus_names,
            sus_strings,
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
        new_sample['original_story'] = sample.get('context', '')
        new_sample['modification_type'] = 'regenerated'
        new_sample['variant_index'] = variant_idx
        new_sample['random_seed'] = local_seed
        # Store per-variant cost so resumed runs can compute totals more accurately.
        new_sample['generation_cost'] = float(cost)
        
        variant_results.append(new_sample)
        sample_cost += cost
    
    thread_safe_print(f"\n  [{sample_idx + 1}] ✓ Complete: {num_variants_per_sample} variants | Cost: ${sample_cost:.4f}")
    
    return sample_idx, variant_results, sample_cost


def main():
    """Main function to regenerate stories from existing trees."""
    # Disable cache to force regeneration
    cache.disable()
    
    parser = argparse.ArgumentParser(description="Regenerate stories from existing logic trees (no tree changes).")
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
    print("REGENERATING STORIES FROM EXISTING TREES")
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
        output_folder_name = f"murder_mystery_level0_samples-{num_samples_actual}_variants-{num_variants_per_sample}_{run_datetime}"
        output_folder = OUTPUT_FOLDER / output_folder_name
        output_folder.mkdir(parents=True, exist_ok=True)
        output_file = output_folder / f"murder_mystery_level0_samples-{num_samples_actual}_variants-{num_variants_per_sample}.json"
        metadata_file = output_folder / "run_metadata.txt"
    
    print(f"Samples to process: {num_samples_actual}")
    print(f"Variants per sample: {num_variants_per_sample}")
    print(f"Total variants to generate: {total_variants}")
    print(f"Concurrent workers: {max_workers}")
    print(f"Output folder: {output_folder}")
    print("This will show that same trees -> different story text\n")
    
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
            
            # Merge with any existing variants for this sample (resume mode).
            existing = all_results.get(sample_idx, [])
            all_results[sample_idx] = _merge_variants(existing, sample_variants)
            total_cost += sample_cost
            
            thread_safe_print(f"\n✅ Sample {sample_idx + 1} fully complete ({completed_samples}/{num_samples_actual} samples)")
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
  Script: level0_murder_mystery_regenerate_stories.py
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
  This run regenerated stories from existing logic trees using temperature={model_config['temperature']}.
  The same trees produce different story text but with identical semantic content (facts).

Methodology:
  1. Loads original samples with their logic trees
  2. Regenerates story text from UNCHANGED trees
  3. Each variant has the same facts but different narrative text
  4. Answer is unchanged (trees are unchanged)
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
    print(f"  - New story text (regenerated with temperature=1.0)")
    print(f"  - Same answer (trees unchanged)")
    print(f"  - Reproducible results (deterministic seeding)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
