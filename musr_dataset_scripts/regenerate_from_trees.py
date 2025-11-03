"""
Regenerate stories from existing logic trees to demonstrate that with temperature=1.0,
the same trees produce different story text but with identical semantic content (facts).

This script takes the first 10 samples from murder_mystery.json and regenerates
the story text using the same logic trees, but with temperature=1.0 for variety.

Follows the same workflow as create_murder_mysteries.py including random.shuffle()
of chapters.

Uses concurrent generation with ThreadPoolExecutor for faster processing.
"""

import json
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

random.seed(0)

from src import cache
from src.model import OpenAIModel
from src.logic_tree.tree import LogicTree
from src.utils.paths import OUTPUT_FOLDER
from src.dataset_types.murder_mystery_dataset import MurderMysteryDataset, create_story_prompt__facts_only


def generate_single_story(trees, suspect_infos, victim_info, sus_names, sus_strings, regen_idx, num_regenerations, model_config, reused_intro=None):
    """
    Generate a single regenerated story. This function is thread-safe and can be run concurrently.
    
    Args:
        trees: List of LogicTree objects
        suspect_infos: List of suspect info dicts
        victim_info: Victim info dict
        sus_names: List of suspect names
        sus_strings: Comma-separated suspect names
        regen_idx: Index of this regeneration (for display)
        num_regenerations: Total regenerations (for display)
        model_config: Dict with model configuration
        reused_intro: Optional pre-generated intro to reuse (for consecutive samples with same setup)
    
    Returns:
        Tuple of (story_text, intro_text, cost)
    """
    # Create a fresh model instance for this thread
    model = OpenAIModel(
        engine=model_config['engine'],
        api_endpoint=model_config['api_endpoint'],
        api_max_attempts=model_config['api_max_attempts'],
        temperature=model_config['temperature'],
        max_tokens=model_config['max_tokens'],
        num_samples=model_config['num_samples'],
        prompt_cost=model_config['prompt_cost'],
        completion_cost=model_config['completion_cost']
    )
    
    creator = MurderMysteryDataset()
    
    # Use reused intro if provided, otherwise generate new one
    if reused_intro is not None:
        print(f"    [{regen_idx + 1}/{num_regenerations}] Reusing intro from previous sample...")
        intro = reused_intro
    else:
        print(f"    [{regen_idx + 1}/{num_regenerations}] Generating intro...")
        intro_prompt = f"Create an intro for this murder mystery.  It should only be 1 or 2 sentences.  Only write the intro nothing else. \n\nScenario:\n{victim_info['victim']} was killed with a {victim_info['murder_weapon']} at a {victim_info['crime_scene']}. Detective Winston is on the case, interviewing suspects. The suspects are {sus_strings}.\n\nOutput:\n"
        intro, _ = creator.inference(intro_prompt, model)
    
    # Generate chapters
    new_chapters = []
    for tree_idx, (tree, suspect_info_dict) in enumerate(zip(trees, suspect_infos)):
        suspect_name = suspect_info_dict['suspect_info']['suspect']
        print(f"    [{regen_idx + 1}/{num_regenerations}] Generating chapter for {suspect_name}...")
        
        # Build description (exclude motive unless they're the murderer)
        description_lines = [
            f"Victim: {victim_info['victim']}",
            f"Crime Scene: {victim_info['crime_scene']}",
            f"Murder Weapon: {victim_info['murder_weapon']}",
            f"Suspect: {suspect_info_dict['suspect_info']['suspect']}",
            f"Role in story: {suspect_info_dict['suspect_info']['role']}"
        ]
        
        if suspect_info_dict['is_murderer']:
            description_lines.append(f"The suspect's motive: {suspect_info_dict['suspect_info']['motive']}")
        
        description = '\n'.join(description_lines)
        
        # Generate chapter from tree
        prompt = create_story_prompt__facts_only(description, tree)
        chapter, _ = creator.inference(prompt, model)
        
        new_chapters.append((suspect_name, chapter.strip()))
    
    # Shuffle chapter order (matches line 259 of create_murder_mysteries.py)
    random.shuffle(new_chapters)
    
    # Build story
    story = f"{intro}\n\n" + "\n\n".join([chapter for _, chapter in new_chapters])
    
    cost = model.total_cost
    print(f"    [{regen_idx + 1}/{num_regenerations}] ✓ Complete (cost: ${cost:.4f})")
    
    return story, intro, cost


def main():
    # Disable cache to force regeneration
    cache.disable()
    
    # ==================== CONFIGURATION ====================
    # Number of dataset samples to process
    num_samples = 1000
    
    # Number of regenerated stories per sample (to show variety from same trees)
    num_regenerations_per_sample = 3
    
    # Number of concurrent workers (API calls in parallel)
    # Adjust based on API rate limits. 5-10 is usually safe for OpenAI
    max_workers = 10
    # ======================================================
    
    # Load the original dataset
    input_file = OUTPUT_FOLDER / 'murder_mystery.json'
    output_file = OUTPUT_FOLDER / f'murder_mystery_regenerated_first{num_samples}.json'
    
    print(f"Loading original dataset from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        original_dataset = json.load(f)
    
    # Take first N samples
    samples_to_regenerate = original_dataset[:num_samples]
    
    print(f"\nRegenerating {num_samples} samples with {num_regenerations_per_sample} versions each")
    print(f"Total stories to generate: {num_samples * num_regenerations_per_sample}")
    print(f"Using {max_workers} concurrent workers for parallel generation")
    print("This will show that same trees -> different story text\n")
    
    # Model configuration (will be used to create instances in each thread)
    model_config = {
        'engine': 'gpt-4',
        'api_endpoint': 'chat',
        'api_max_attempts': 30,
        'temperature': 1.0,
        'max_tokens': 2400,
        'num_samples': 1,
        'prompt_cost': 0.03/1000,
        'completion_cost': 0.06/1000
    }
    
    regenerated_dataset = []
    total_cost = 0.0
    
    # Track previous sample's info for intro reuse
    previous_story_prefix = None
    previous_intros = None
    
    for idx, sample in enumerate(samples_to_regenerate):
        print(f"\n{'='*80}")
        print(f"Sample {idx + 1}/{num_samples}")
        print(f"{'='*80}")
        
        # Extract data from the original sample (same for all regenerations)
        question = sample['questions'][0]
        trees_json = question['intermediate_trees']
        intermediate_data = question['intermediate_data'][0]
        suspect_infos = intermediate_data['suspect_info']
        victim_info = intermediate_data['victim_info']
        
        # Convert trees from JSON
        trees = [LogicTree.from_json(tree_json) for tree_json in trees_json]
        
        sus_names = [s['suspect_info']['suspect'] for s in suspect_infos]
        sus_strings = ", ".join(sus_names)
        
        # Check if this sample has the same setup as previous (first 50 chars match)
        current_story_prefix = sample['context'][:50]
        can_reuse_intros = (previous_story_prefix is not None and 
                           current_story_prefix == previous_story_prefix)
        
        if can_reuse_intros:
            print(f"\n  ♻️  Detected same setup as previous sample - reusing {num_regenerations_per_sample} intros!")
            intros_to_use = previous_intros
        else:
            print(f"\n  Generating {num_regenerations_per_sample} versions concurrently...")
            intros_to_use = [None] * num_regenerations_per_sample  # Generate new intros
        
        regenerated_stories = []
        generated_intros = []
        sample_cost = 0.0
        
        # Submit all regeneration tasks to the executor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create futures for all regenerations
            futures = []
            for regen_idx in range(num_regenerations_per_sample):
                future = executor.submit(
                    generate_single_story,
                    trees,
                    suspect_infos,
                    victim_info,
                    sus_names,
                    sus_strings,
                    regen_idx,
                    num_regenerations_per_sample,
                    model_config,
                    intros_to_use[regen_idx]  # Pass reused intro or None
                )
                futures.append((regen_idx, future))
            
            # Collect results as they complete
            results = [None] * num_regenerations_per_sample
            intros = [None] * num_regenerations_per_sample
            for regen_idx, future in futures:
                story, intro, cost = future.result()
                results[regen_idx] = story
                intros[regen_idx] = intro
                sample_cost += cost
            
            regenerated_stories = results
            generated_intros = intros
        
        # Store current sample info for next iteration
        previous_story_prefix = current_story_prefix
        previous_intros = generated_intros
        
        # Create comparison entry with all regenerated versions
        regenerated_entry = {
            'sample_number': idx + 1,
            'original_story': sample['context'],
            'regenerated_stories': regenerated_stories,  # List of all versions
            'num_regenerations': num_regenerations_per_sample,
            'suspects': [s['suspect_info']['suspect'] for s in suspect_infos],
            'victim': victim_info['victim'],
            'weapon': victim_info['murder_weapon'],
            'crime_scene': victim_info['crime_scene'],
            'murderer': sus_names[question['answer']],
            'questions': sample['questions']  # Keep original for reference
        }
        
        regenerated_dataset.append(regenerated_entry)
        
        total_cost += sample_cost
        print(f"\n  Sample total cost: ${sample_cost:.4f} | Running total: ${total_cost:.4f}")
        
        # Save incrementally
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(regenerated_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"✓ Regeneration complete!")
    print(f"  Total samples: {num_samples}")
    print(f"  Regenerations per sample: {num_regenerations_per_sample}")
    print(f"  Total stories generated: {num_samples * num_regenerations_per_sample}")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Avg per sample: ${total_cost/num_samples:.4f}")
    print(f"  Avg per story: ${total_cost/(num_samples * num_regenerations_per_sample):.4f}")
    print(f"  Output: {output_file}")
    print(f"\nEach sample has {num_regenerations_per_sample} regenerated versions showing")
    print(f"how the same trees produce different narrative text with temperature=1.0!")
    print(f"\n♻️  Intro reuse optimization: Consecutive samples with matching setups")
    print(f"   automatically reuse intros, saving API calls and cost!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
