"""
CONCURRENT VERSION: Murder Mystery Dataset Generator with 6 Workers

This is a parallelized version of create_murder_mysteries.py that uses
ThreadPoolExecutor to generate multiple stories concurrently.

Usage: python create_murder_mysteries_concurrent.py

NOTE: Expects your openai api key to be in the environment.
NOTE: By default, datasets go into "{ROOT_FOLDER}/datasets/{dataset_name}.json"
"""

import copy
import json
import sys
import threading
import traceback
from pathlib import Path
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path so we can import from src
SCRIPT_DIR = Path(__file__).parent.absolute()
MUSR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MUSR_DIR))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(SCRIPT_DIR / '.env')
load_dotenv(MUSR_DIR / '.env')

from src import cache
from src.model import OpenAIModel
from src.logic_tree.tree import LogicTree, LogicNode, LogicNodeFactType
from src.madlib.madlib import Madlib
from src.utils.paths import OUTPUT_FOLDER, ROOT_FOLDER

from src.dataset_types.murder_mystery_dataset import MurderMysteryDataset

# ============================================================================
# ICL EXAMPLES for creating deductions (same as original)
# ============================================================================

example1_description = """

Victim: Victoria
Crime scene: Home
Murder weapon: Gun
Suspect: James
Suspect's role in story: Brother
Suspect's motive: Financial gain
"""

example1_tree = LogicTree(
nodes=[
    LogicNode('James is a murderer.', [
        LogicNode('James has a means', [
            LogicNode('James has practiced shooting guns.'),
            LogicNode('James owns guns'),
            LogicNode('If you both own and practice using guns then you have the ability to murder someone.', fact_type=LogicNodeFactType.COMMONSENSE)
        ]),
        LogicNode('James has a motive.', [
            LogicNode('James was violently desperate for cash.'),
            LogicNode('James was violently desperate for Victoria\'s cash'),
            LogicNode(
                'When someone is violently desperate they may go to extreme measures to accomplish a task, including murderer.',
                fact_type=LogicNodeFactType.COMMONSENSE)
        ]),
        LogicNode('James has a opportunity.')
    ])
], prune=False, populate=False
)

example1_node_completion = LogicNode('James has a opportunity.', [
    LogicNode('James has access to Victoria\'s house.'),
    LogicNode('Having access to someones house gives you the opportunity to murder them.', fact_type=LogicNodeFactType.COMMONSENSE)
])

example2_description = """
Story Information:
Victim: Harry
Crime scene: Racetrack
Murder weapon: Shovel
Suspect: Claire
Suspect's role in story: Running buddy
Suspects motive: To prevent someone else harm
"""
example2_tree = LogicTree(
nodes=[
    LogicNode('Claire is a murderer.', [
        LogicNode('Claire has a means.', [
            LogicNode('Claire is a farmer'),
            LogicNode(
                'Farmers typically use gardening tools like shovels in their work.',
                fact_type=LogicNodeFactType.COMMONSENSE)
        ]),
        LogicNode('Claire has a motive.'),
        LogicNode('Claire has an opportunity')
    ])
], prune=False, populate=False
)

example2_node_completion = LogicNode('Claire has a motive.', [
    LogicNode('Claire loves Brian deeply.'),
    LogicNode('Harry threatened Brian.'),
    LogicNode('Deep and passionate love can push people to do extreme things like murder when that loved one is threatened.', fact_type=LogicNodeFactType.COMMONSENSE)
])

example3_description = """
Victim: Jared
Crime scene: Public park bench
Murder weapon: Heroin overdose
Suspect: Jose
Suspect's role in story: Drug user
Suspects motive: Public humiliation
"""
example3_tree = LogicTree(
nodes=[
    LogicNode('Jose is a murderer.', [
        LogicNode('Jose has a means.'),
        LogicNode('Jose has a motive'),
        LogicNode('Jose has an opportunity.')
    ])
], prune=False, populate=False
)

example3_node_completion = LogicNode('Jose has a means.', [
    LogicNode('Jose has access to heroin.'),
    LogicNode('Jose knows how much heroin is needed for an overdose.'),
    LogicNode('Having access to heroin and knowing how much heroin is required to overdose implies you could have intentionally given the victim a dose of lethal heroin providing a means for murder.', fact_type=LogicNodeFactType.COMMONSENSE)
])

example_trees = [example1_tree, example2_tree, example3_tree]
example_node_completions = [example1_node_completion, example2_node_completion, example3_node_completion]
example_descriptions = [example1_description, example2_description, example3_description]


# ============================================================================
# THREAD-SAFE PROGRESS TRACKING
# ============================================================================

class ProgressTracker:
    """Thread-safe progress tracker for concurrent story generation."""
    
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.total_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.lock = threading.Lock()
        self.start_time = datetime.now()
    
    def update(self, cost: float, prompt_tokens: int, completion_tokens: int, success: bool = True):
        with self.lock:
            if success:
                self.completed += 1
            else:
                self.failed += 1
            self.total_cost += cost
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            
            elapsed = datetime.now() - self.start_time
            rate = self.completed / elapsed.total_seconds() * 3600 if elapsed.total_seconds() > 0 else 0
            eta_hours = (self.total - self.completed) / rate if rate > 0 else 0
            
            print(f"[{self.completed}/{self.total}] "
                  f"Cost: ${cost:.2f} (Total: ${self.total_cost:.2f}) | "
                  f"Rate: {rate:.1f}/hr | ETA: {eta_hours:.1f}h | "
                  f"Failed: {self.failed}")


# ============================================================================
# WORKER FUNCTION: Generate a single story
# ============================================================================

def generate_single_story(
    story_config: dict,
    story_idx: int,
    config: dict,
    progress: ProgressTracker
) -> dict:
    """
    Generate a single murder mystery story.
    Each worker has its own model instances to avoid shared state issues.
    
    Args:
        story_config: Pre-generated madlib configuration for this story
        story_idx: Index of this story (for logging)
        config: Shared configuration (tree_depth, etc.)
        progress: Thread-safe progress tracker
        
    Returns:
        dict with 'samples', 'cost', 'prompt_tokens', 'completion_tokens'
    """
    
    # Create thread-local model instances
    local_gpt4 = OpenAIModel(
        engine='gpt-4-0613',
        api_max_attempts=30,
        api_endpoint='chat',
        temperature=1.0,
        top_p=1.0,
        max_tokens=2400,
        num_samples=1,
        prompt_cost=0.03/1000,
        completion_cost=0.06/1000
    )
    
    local_gpt16k35 = OpenAIModel(
        engine='gpt-3.5-turbo-16k',
        api_endpoint='chat',
        api_max_attempts=30,
        temperature=1.0,
        max_tokens=2400,
        num_samples=1,
        prompt_cost=0.003/1000,
        completion_cost=0.004/1000
    )
    
    model_to_use = local_gpt4
    
    # Create thread-local creator instance
    creator = MurderMysteryDataset()
    
    # Use thread-local random for shuffling (seeded by story_idx for reproducibility)
    local_random = random.Random(story_idx)
    
    try:
        # Extract pre-generated config
        victim_dict = story_config['victim_dict']
        suspect_dicts = story_config['suspect_dicts']
        
        # Generate suspect trees
        suspect_trees = creator.create_suspect_trees(
            model_to_use,
            victim_dict,
            suspect_dicts,
            example_trees,
            example_node_completions,
            example_descriptions,
            depth=config['tree_depth'],
            bf_factor={2: 1.0},
            chance_to_prune=0.0,
            chance_to_prune_all=0.0,
            max_num_of_suspicious_facts=config['max_num_suspicious_facts'],
            max_retries_on_error=config['max_structure_completion_retries'],
            retry_model=model_to_use,
            progress_bar=False,  # Disable per-story progress bars in concurrent mode
            use_validators=config['use_validators'],
            model_validator_model=local_gpt4,
            model_validator_early_escape_model=local_gpt16k35,
            test_completion_prompt=False
        )
        
        suspect_trees = creator.create_chapter_trees(
            suspect_trees,
            max_num_of_suspicious_facts=config['max_num_suspicious_facts']
        )
        
        suspect_trees = creator.create_chapter(
            model_to_use,
            suspect_trees,
            validate_model=model_to_use
        )
        
        # Create intro
        sus_strings = ", ".join([x['suspect_info']['suspect'] for x in suspect_trees])
        intro_prompt = (
            f"Create an intro for this murder mystery. It should only be 1 or 2 sentences. "
            f"Only write the intro nothing else.\n\n"
            f"Scenario:\n{victim_dict['victim']} was killed with a {victim_dict['murder_weapon']} "
            f"at a {victim_dict['crime_scene']}. Detective Winston is on the case, interviewing suspects. "
            f"The suspects are {sus_strings}.\n\nOutput:\n"
        )
        intro, _ = creator.inference(intro_prompt, model_to_use)
        
        # Generate samples (one per suspect as murderer)
        samples = []
        for murderer_idx in range(len(suspect_trees)):
            _suspect_trees = copy.deepcopy(suspect_trees)
            
            for sidx, s in enumerate(_suspect_trees):
                _suspect_trees[sidx]['used_chapter'] = _suspect_trees[sidx][
                    'innocent_chapter' if sidx != murderer_idx else 'murderer_chapter']
                _suspect_trees[sidx]['used_tree'] = _suspect_trees[sidx][
                    'innocent_tree' if sidx != murderer_idx else 'murderer_tree']
                _suspect_trees[sidx]['is_murderer'] = sidx == murderer_idx
            
            chapters = [(x['suspect_info']['suspect'], x['used_chapter'].strip()) for x in _suspect_trees]
            local_random.shuffle(chapters)
            
            story = f"{intro}\n\n" + "\n\n".join([x[1] for x in chapters])
            choices = [x['suspect_info']["suspect"] for x in _suspect_trees]
            
            # Use hash matching original implementation
            story_hash = hash(intro)
            
            safe_suspects_dict = [
                {k: v.to_json() if isinstance(v, LogicTree) else v for k, v in x.items()}
                for x in _suspect_trees
            ]
            
            sample = creator.create_dataset_question_object(
                context=story,
                questions=['Who is the most likely murderer?'],
                answers=[murderer_idx],
                choices=[choices],
                intermediate_trees=[[x['used_tree'] for x in _suspect_trees]],
                intermediate_data=[[{
                    'suspect_info': safe_suspects_dict,
                    'victim_info': victim_dict,
                    'story_hash_id': story_hash
                }]]
            )
            samples.append(sample)
        
        # Calculate costs
        cost = model_to_use.total_cost + local_gpt16k35.total_cost
        prompt_tokens = model_to_use.total_prompt_tokens + local_gpt16k35.total_prompt_tokens
        completion_tokens = model_to_use.total_completion_tokens + local_gpt16k35.total_completion_tokens
        
        # Update progress
        progress.update(cost, prompt_tokens, completion_tokens, success=True)
        
        return {
            'story_idx': story_idx,
            'samples': samples,
            'cost': cost,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'success': True
        }
        
    except Exception as e:
        print(f"[ERROR] Story {story_idx}: {str(e)}")
        traceback.print_exc()
        progress.update(0, 0, 0, success=False)
        return {
            'story_idx': story_idx,
            'samples': [],
            'cost': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'success': False,
            'error': str(e)
        }


# ============================================================================
# PRE-GENERATE ALL MADLIB CONFIGURATIONS
# ============================================================================

def pre_generate_story_configs(
    max_examples: int,
    max_number_of_suspects: int,
    max_num_suspicious_facts: int
) -> list:
    """
    Pre-generate all madlib configurations upfront.
    This avoids race conditions on previously_sampled_items.
    """
    
    print("Pre-generating story configurations...")
    
    # Set seed for reproducibility
    random.seed(0)
    
    creator = MurderMysteryDataset()
    
    madlib = Madlib({
        "male_names": ROOT_FOLDER / 'domain_seed/male_names.json',
        "female_names": ROOT_FOLDER / 'domain_seed/female_names.json',
        "male_relationships": ROOT_FOLDER / 'domain_seed/male_relationships.json',
        "female_relationships": ROOT_FOLDER / 'domain_seed/female_relationships.json',
        "motives": ROOT_FOLDER / 'domain_seed/strong_motives.json',
        "murder_weapons": ROOT_FOLDER / 'domain_seed/murder_weapons.json',
        "relationships": ROOT_FOLDER / 'domain_seed/relationships.json',
        "crime_scenes": ROOT_FOLDER / 'domain_seed/crime_scenes.json',
        'red_herrings': ROOT_FOLDER / 'domain_seed/suspicious_facts.json',
    })
    
    previously_sampled_items = []
    story_configs = []
    
    constant_sampled_items = [['male_names', 'female_names'], 'crime_scenes', 'murder_weapons']
    constant_sampled_names = ['victim', 'crime_scene', 'murder_weapon']
    variable_sampled_items = [['male_names,male_relationships', 'female_names,female_relationships'], 'motives', 'crime_scenes']
    variable_sampled_names = ['suspect', 'role', 'motive', 'alibi']
    
    description_string = "Victim: {victim}\nCrime Scene: {crime_scene}\nMurder Weapon: {murder_weapon}"
    variable_string = 'Suspect: {suspect}\nRole in story: {role}\nThe suspect\'s motive: {motive}'
    
    for example_idx in range(max_examples):
        # Sample victim info
        victim_string, victim_dict, sampled = creator.sample_madlib(
            madlib,
            constant_sampled_items,
            previously_sampled=previously_sampled_items,
            description_string_format=description_string,
            sampled_item_names=constant_sampled_names
        )
        victim_dict = victim_dict[0]
        previously_sampled_items = sampled
        
        # Sample suspect info
        suspect_strings, suspect_dicts, _ = creator.sample_madlib(
            madlib,
            variable_sampled_items,
            previously_sampled=[[None, None, None, victim_dict['crime_scene']]],
            n_samples=max_number_of_suspects,
            description_string_format=variable_string,
            sampled_item_names=variable_sampled_names
        )
        
        # Sample red herrings
        _, suspicious_fact_dicts, _ = creator.sample_madlib(
            madlib,
            ['red_herrings'],
            n_samples=max_num_suspicious_facts * len(suspect_dicts),
            description_string_format='{red_herrings}'
        )
        random.shuffle(suspicious_fact_dicts)
        
        for s in suspect_dicts:
            s['red_herrings'] = []
            for n in range(max_num_suspicious_facts):
                s['red_herrings'].append(suspicious_fact_dicts.pop()['red_herrings'])
        
        scenario = f'{victim_string[0]}\n'
        for idx, s in enumerate(suspect_strings):
            suspect_dicts[idx]['description'] = f"{scenario}{s}".strip()
        
        story_configs.append({
            'victim_dict': victim_dict,
            'suspect_dicts': suspect_dicts,
            'scenario': scenario
        })
    
    print(f"Generated {len(story_configs)} story configurations.")
    return story_configs


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # Concurrency settings
    MAX_WORKERS = 6  # Number of concurrent story generators
    
    # Dataset parameters
    max_examples = 125  # Number of stories (will generate 2x questions)
    tree_depth = 3
    max_number_of_suspects = 2
    max_structure_completion_retries = 3
    max_num_suspicious_facts = 1
    use_validators = True
    
    # Disable cache for fresh generation
    cache.disable()
    
    # ========================================================================
    # SETUP OUTPUT
    # ========================================================================
    
    run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder_name = f"murder_mystery_samples-{max_examples}_concurrent_{run_datetime}"
    output_folder = OUTPUT_FOLDER / output_folder_name
    output_folder.mkdir(parents=True, exist_ok=True)
    
    out_file = output_folder / f'murder_mystery_samples-{max_examples}.json'
    metadata_file = output_folder / 'run_metadata.txt'
    
    print("=" * 80)
    print("CONCURRENT MURDER MYSTERY DATASET GENERATOR")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Max workers: {MAX_WORKERS}")
    print(f"  Max examples (stories): {max_examples}")
    print(f"  Suspects per story: {max_number_of_suspects}")
    print(f"  Questions per story: {max_number_of_suspects}")
    print(f"  Total expected questions: {max_examples * max_number_of_suspects}")
    print(f"  Tree depth: {tree_depth}")
    print(f"  Model: gpt-4-0613")
    print(f"  Output folder: {output_folder}")
    print()
    
    # ========================================================================
    # PRE-GENERATE STORY CONFIGURATIONS
    # ========================================================================
    
    story_configs = pre_generate_story_configs(
        max_examples=max_examples,
        max_number_of_suspects=max_number_of_suspects,
        max_num_suspicious_facts=max_num_suspicious_facts
    )
    
    # ========================================================================
    # SHARED CONFIGURATION FOR WORKERS
    # ========================================================================
    
    config = {
        'tree_depth': tree_depth,
        'max_structure_completion_retries': max_structure_completion_retries,
        'max_num_suspicious_facts': max_num_suspicious_facts,
        'use_validators': use_validators,
    }
    
    # ========================================================================
    # CONCURRENT GENERATION
    # ========================================================================
    
    print(f"\nStarting concurrent generation with {MAX_WORKERS} workers...")
    print("-" * 80)
    
    start_time = datetime.now()
    progress = ProgressTracker(total=max_examples)
    
    all_results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all jobs
        futures = {
            executor.submit(
                generate_single_story,
                story_config,
                story_idx,
                config,
                progress
            ): story_idx
            for story_idx, story_config in enumerate(story_configs)
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            story_idx = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                
                # Periodic save (every 10 stories)
                if len(all_results) % 10 == 0:
                    # Sort and flatten samples
                    sorted_results = sorted(all_results, key=lambda x: x['story_idx'])
                    dataset = []
                    for r in sorted_results:
                        if r['success']:
                            dataset.extend(r['samples'])
                    
                    with open(out_file, 'w') as f:
                        json.dump(dataset, f, indent=2)
                    print(f"  [Checkpoint] Saved {len(dataset)} samples to {out_file.name}")
                    
            except Exception as e:
                print(f"[ERROR] Story {story_idx} failed: {e}")
    
    # ========================================================================
    # FINAL PROCESSING
    # ========================================================================
    
    print("-" * 80)
    print("Processing results...")
    
    # Sort results by story_idx to maintain order
    sorted_results = sorted(all_results, key=lambda x: x['story_idx'])
    
    # Flatten samples into dataset
    dataset = []
    for result in sorted_results:
        if result['success']:
            dataset.extend(result['samples'])
    
    # Calculate statistics
    end_time = datetime.now()
    total_elapsed = end_time - start_time
    total_elapsed_str = str(total_elapsed).split('.')[0]
    
    total_cost = progress.total_cost
    total_prompt_tokens = progress.total_prompt_tokens
    total_completion_tokens = progress.total_completion_tokens
    total_tokens = total_prompt_tokens + total_completion_tokens
    
    successful_stories = progress.completed
    failed_stories = progress.failed
    total_questions = len(dataset)
    
    avg_cost_per_story = total_cost / successful_stories if successful_stories > 0 else 0
    avg_cost_per_question = total_cost / total_questions if total_questions > 0 else 0
    avg_tokens_per_story = total_tokens / successful_stories if successful_stories > 0 else 0
    
    # ========================================================================
    # SAVE FINAL OUTPUT
    # ========================================================================
    
    with open(out_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Write metadata
    metadata_content = f"""Run Metadata
============

Run Information:
  Start Time: {start_time.strftime("%Y%m%d_%H%M%S")}
  End Time: {end_time.strftime("%Y%m%d_%H%M%S")}
  Total Elapsed: {total_elapsed_str}
  Script: create_murder_mysteries_concurrent.py
  Mode: CONCURRENT ({MAX_WORKERS} workers)

Configuration:
  Max Examples (Stories): {max_examples}
  Suspects Per Story: {max_number_of_suspects}
  Tree Depth: {tree_depth}
  Max Structure Completion Retries: {max_structure_completion_retries}
  Max Suspicious Facts: {max_num_suspicious_facts}
  Use Validators: {use_validators}

Model Configuration:
  Primary Model: gpt-4-0613
  Validator Model: gpt-4-0613
  Early Escape Model: gpt-3.5-turbo-16k
  Temperature: 1.0

Results:
  Stories Attempted: {max_examples}
  Stories Successful: {successful_stories}
  Stories Failed: {failed_stories}
  Total Questions: {total_questions}
  Questions Per Story: {max_number_of_suspects}

Token Statistics:
  Total Prompt Tokens: {total_prompt_tokens:,}
  Total Completion Tokens: {total_completion_tokens:,}
  Total Tokens: {total_tokens:,}
  Avg Tokens Per Story: {avg_tokens_per_story:,.0f}

Cost Statistics:
  Total Cost: ${total_cost:.4f}
  Avg Cost Per Story: ${avg_cost_per_story:.4f}
  Avg Cost Per Question: ${avg_cost_per_question:.4f}

Performance:
  Total Time: {total_elapsed_str}
  Avg Time Per Story: {(total_elapsed.total_seconds() / successful_stories):.1f}s (with {MAX_WORKERS}x concurrency)
  Effective Rate: {(successful_stories / total_elapsed.total_seconds() * 3600):.1f} stories/hour

Output Files:
  - {out_file.name}
  - {metadata_file.name}

Description:
  Concurrent murder mystery dataset generation with {MAX_WORKERS} parallel workers.
  Each story has {max_number_of_suspects} suspects and generates {max_number_of_suspects} questions.
"""
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(metadata_content)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print(f"\n{'=' * 80}")
    print("GENERATION COMPLETE")
    print(f"{'=' * 80}")
    
    print(f"\n📊 Generation Statistics:")
    print(f"  Stories successful: {successful_stories}/{max_examples}")
    print(f"  Stories failed: {failed_stories}")
    print(f"  Total questions: {total_questions}")
    print(f"  Total elapsed time: {total_elapsed_str}")
    print(f"  Effective rate: {(successful_stories / total_elapsed.total_seconds() * 3600):.1f} stories/hour")
    
    print(f"\n🔢 Token Statistics:")
    print(f"  Prompt tokens: {total_prompt_tokens:,}")
    print(f"  Completion tokens: {total_completion_tokens:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg tokens per story: {avg_tokens_per_story:,.0f}")
    
    print(f"\n💰 Cost Statistics:")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Avg cost per story: ${avg_cost_per_story:.4f}")
    print(f"  Avg cost per question: ${avg_cost_per_question:.4f}")
    
    print(f"\n📁 Output:")
    print(f"  Folder: {output_folder}")
    print(f"  Data file: {out_file.name}")
    print(f"  Metadata file: {metadata_file.name}")
    print(f"{'=' * 80}\n")
    
    if failed_stories > 0:
        print(f"⚠️  WARNING: {failed_stories} stories failed. Check logs for errors.")
        print("   You may want to run a supplementary generation to fill in the gaps.\n")


if __name__ == "__main__":
    main()

