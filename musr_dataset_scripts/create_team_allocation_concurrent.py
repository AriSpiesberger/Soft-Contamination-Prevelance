"""
CONCURRENT VERSION: Team Allocation Dataset Generator with 6 Workers

This is a parallelized version of create_team_allocation.py that uses
ThreadPoolExecutor to generate multiple samples concurrently.

Usage: python create_team_allocation_concurrent.py

NOTE: Expects your openai api key to be in the environment.
NOTE: By default, datasets go into "{ROOT_FOLDER}/datasets/{dataset_name}.json"
"""

import json
import sys
import threading
import traceback
from pathlib import Path
import random
from functools import partial
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
from src.utils.paths import OUTPUT_FOLDER

from src.dataset_types.team_allocation import TeamAllocationDataset

cLogicNode = partial(LogicNode, fact_type=LogicNodeFactType.COMMONSENSE)

# ============================================================================
# ICL EXAMPLES for creating deductions (same as original)
# ============================================================================

example1_description = 'Paul and Alice are at a karaoke bar.'
example1_tree = LogicTree(
    nodes=[
        LogicNode('Paul and Alice are at a karaoke bar.', [
            LogicNode('Opening Scene', [
                LogicNode('Paul sees the microphone at the stage.'),
                LogicNode('Alice sees the microphone at the stage.'),
                LogicNode('Paul sees the beer at the bar.'),
                LogicNode('Alice sees the beer at the bar.')
            ]),
            LogicNode('Paul moves the beer to the table.', [
                LogicNode('Alice did not see the beer move to the table.', [
                    LogicNode('Alice was facing away from the table.', [
                        LogicNode('Alice was talking to another patron.'),
                        LogicNode('The other patron was facing the table.'),
                        cLogicNode('Usually people talk to each other while facing each other, so if one person is looking in one direction the other person is looking in the opposite direction.')
                    ]),
                    cLogicNode('If someone is facing away from something else, they cannot see things transpire near that something else.')
                ])
            ]),
            LogicNode('Alice moves the microphone to the table', [
                LogicNode('Alice saw the beer at the table when moving the microphone.'),
                LogicNode('Paul saw the microphone move to the table.', [
                    cLogicNode("Paul was drinking the beer at the table."),
                    LogicNode("When something happens where a person is at, they usually see things that are happening there.")
                ])
            ]),
            LogicNode('Alice moves the beer to the trash can.', [
                LogicNode('Paul did not see the beer move to the trash can.')
            ])
        ])

    ], prune=False, populate=False
)
example1_node_completion_tree = LogicNode('Paul and Alice are at a karaoke bar.', [LogicNode('Alice moves the beer to the trash can.', [
            LogicNode('Paul did not see the beer move to the trash can.', [
                LogicNode("Alice tricked Paul into looking \"over there\"."),
                LogicNode("Alice pointed in the opposite direction of the trash can to Paul."),
                cLogicNode("If you trick someone into looking else where, they cannot see what happens in the other direction.")
            ])
        ])])


example2_description = 'Your dog has just pooped on the neighbours yard.  The neighbour glares in your direction and comes forward... he says "Hey you! What do you think you are letting your dog do on my nice yard right here!"'
example2_tree = LogicTree(
    nodes=[
        LogicNode('Punch the neighbour square in the nose.', [
                LogicNode('It\'ll look cool and this is a pro.', [
                    LogicNode('People think fighting is cool where you live.'),
                    LogicNode('You would be fighting.'),
                    LogicNode('Doing something people think is cool will make you cool too.',
                              fact_type=LogicNodeFactType.COMMONSENSE)
                ]),
                LogicNode('It\'ll look cool unless...', [])
            ]),
        LogicNode('Say, "I am so sorry mr. I am trying to train him."'),
        LogicNode('You feel threatened by the neighbour and think he might hurt you.'),
        LogicNode('The neighbour would leave you alone.')
        ], prune=False, populate=False
)
example2_node_completion_tree = LogicNode('Punch the neighbour square in the nose.',
                                     [LogicNode('It\'ll look cool unless...', [
                                        LogicNode('You could harm your neighbour.'),
                                        LogicNode('You are unprovoked'),
                                        cLogicNode('It\'s not cool if you hurt someone unprovoked.')
                                    ])])


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


example_descriptions = [example1_description, example2_description, example3_description]
example_trees = [example1_tree, example2_tree, example3_tree]
example_node_completions = [example1_node_completion_tree.children[0].children[0], example3_node_completion, example1_node_completion_tree.children[0]]


# ============================================================================
# THREAD-SAFE PROGRESS TRACKING
# ============================================================================

class ProgressTracker:
    """Thread-safe progress tracker for concurrent sample generation."""
    
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
# WORKER FUNCTION: Generate a single sample
# ============================================================================

def generate_single_sample(
    sample_config: dict,
    sample_idx: int,
    config: dict,
    progress: ProgressTracker
) -> dict:
    """
    Generate a single team allocation sample.
    Each worker has its own model instances to avoid shared state issues.
    
    Args:
        sample_config: Pre-generated configuration for this sample (description)
        sample_idx: Index of this sample (for logging)
        config: Shared configuration (tree_depth, etc.)
        progress: Thread-safe progress tracker
        
    Returns:
        dict with 'sample', 'cost', 'prompt_tokens', 'completion_tokens', 'success'
    """
    
    # Create thread-local model instance
    local_model = OpenAIModel(
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
    
    # Create thread-local creator instance
    creator = TeamAllocationDataset()
    
    # Use thread-local random for shuffling (seeded by sample_idx for reproducibility)
    local_random = random.Random(sample_idx)
    
    try:
        description = sample_config['description']
        
        # Flesh out the description to include peoples names and skill/task names.
        prompt = f'''
Use the given scenario description to create a list of three people, two tasks, and two skills.  Each skill should be associated with one of the tasks, and it should be that each skill is unique and orthogonal to the other and it's assigned task.

Rules
 1) Never indicate in someones name their title or that they are good at any particular skill or task.  For example, never say "Dr. Bob" as this implies they should be in charge of medical tasks.

Here's an example

Description: A heavy flux of customers walk into the coffee bar, you have to assign workers to the register and others to be baristas to handle the flow.

Output:

People: Sarah; Luis; John;
Tasks: Barista; Cashier
Skills: Can make coffee; Can handle customers

Your turn!

Description: {description}

Output:

        '''.strip()
        
        output, _ = creator.inference(prompt, local_model)
        
        people = []
        skills = []
        tasks = []
        
        # Parse the output into lists of people/tasks/skills.
        for line in output.split('\n'):
            if line.startswith('People:'):
                people.extend([x.strip() for x in line.replace('People:', '').strip().split(';') if x != ''])
            elif line.startswith('Tasks:'):
                tasks.extend([x.strip() for x in line.replace('Tasks:', '').strip().split(';') if x != ''])
            elif line.startswith('Skills:'):
                skills.extend([x.strip() for x in line.replace('Skills:', '').strip().split(';') if x != ''])
        
        # Validate parsing
        if len(people) < 3 or len(tasks) < 2 or len(skills) < 2:
            raise ValueError(f"Parsing failed: people={len(people)}, tasks={len(tasks)}, skills={len(skills)}")
        
        # Create the skill/relationships scores and then generate the fact set F
        people_levels, best_pair, all_pairs = creator.build_assignment(people)
        facts = creator.create_facts(people_levels, people, tasks)
        
        local_random.shuffle(facts)
        local_random.shuffle(all_pairs)
        
        tree = creator.create_fact_trees(
            local_model, facts, tasks, description, example_trees, example_node_completions,
            example_descriptions, depth=config['tree_depth'], bf_factor={2: 1.0}, 
            chance_to_prune=0.0, chance_to_prune_all=0.0,
            max_retries_on_error=config['structure_retries'], progress_bar=False, 
            test_complete_structure_prompt=False, retry_model=local_model
        )
        
        def fact_str(t):
            facts = list(sorted(list(set([x.value for x in t.get_facts()]))))
            facts_str = "\n".join([f'- {x}' for x in facts])
            return facts_str
        
        facts_text = fact_str(tree)
        
        # Create the story
        story_prompt = f'''
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

Scenario description: {description}
Facts that you must include in your story:
{facts_text}

Output:'''
        
        story_output, _ = creator.inference(story_prompt, local_model)
        
        paragraphs = story_output.split('\n\n')
        
        # Fix the first paragraph to properly introduce the scenario
        fix_p0_prompt = f'''
I am writing a story that is similar to a word problem where a manager has to assign the right worker to a task.  Here's the full story:

{story_output}

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
        
        paragraphs[0], _ = creator.inference(fix_p0_prompt, local_model, temperature=0.2)
        
        final_story = "\n\n".join(paragraphs)
        
        question = "Given the story, how would you uniquely allocate each person to make sure both tasks are accomplished efficiently?"
        
        choices = [
            f"{tasks[0]}: {all_pairs[0][0][0]}, {tasks[1]}: {all_pairs[0][1][0]} and {all_pairs[0][1][1]}",
            f"{tasks[0]}: {all_pairs[1][0][0]}, {tasks[1]}: {all_pairs[1][1][0]} and {all_pairs[1][1][1]}",
            f"{tasks[0]}: {all_pairs[2][0][0]}, {tasks[1]}: {all_pairs[2][1][0]} and {all_pairs[2][1][1]}",
        ]
        
        gold_idx = all_pairs.index(best_pair)
        
        sample = creator.create_dataset_question_object(
            final_story,
            questions=[question],
            answers=[gold_idx],
            choices=[choices],
            intermediate_trees=[[tree]],
            intermediate_data=[[{'tasks': tasks, 'matrix': people_levels, 'best_pair': best_pair}]]
        )
        
        # Calculate costs
        cost = local_model.total_cost
        prompt_tokens = local_model.total_prompt_tokens
        completion_tokens = local_model.total_completion_tokens
        
        # Update progress
        progress.update(cost, prompt_tokens, completion_tokens, success=True)
        
        return {
            'sample_idx': sample_idx,
            'sample': sample,
            'cost': cost,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'success': True
        }
        
    except Exception as e:
        print(f"[ERROR] Sample {sample_idx}: {str(e)}")
        traceback.print_exc()
        progress.update(0, 0, 0, success=False)
        return {
            'sample_idx': sample_idx,
            'sample': None,
            'cost': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'success': False,
            'error': str(e)
        }


# ============================================================================
# PRE-GENERATE SAMPLE CONFIGURATIONS
# ============================================================================

def pre_generate_sample_configs(max_examples: int) -> list:
    """
    Pre-generate all sample configurations (descriptions) upfront.
    This requires API calls to build the madlib, so it's done sequentially.
    """
    
    print("Pre-generating sample configurations...")
    
    # Set seed for reproducibility
    random.seed(0)
    
    # Create model for madlib building
    model = OpenAIModel(
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
    
    creator = TeamAllocationDataset()
    
    # Build the madlib (requires API calls)
    print("  Building madlib (this requires API calls)...")
    madlib = creator.build_madlib(
        model,
        things_to_create_names=['scenario_descriptions'],
        things_to_create_description=[
            "Scenarios where you have to assign people to groups that perform different tasks and the skills required to solve those tasks are fairly different, only give the scenario and DO NOT number them"
        ],
        examples_of_things_to_create=[
            [
                "You and your roommates want to make a video game, how should you assign each of your roommates so that the action video game is made.",
                "A paper deadline is coming up and since you are the supervisor of the lab, you must assign each graduate student efficiently to meet the deadline.",
                "They never said campaigning was easy, you want to be elected for President of the United States, so you have to assign your team effectively to help your chances.",
                "You are planning the perfect heist, how do you build your team such that the heist goes off without a hitch."
            ]
        ]
    )
    
    madlib_cost = model.total_cost
    madlib_prompt_tokens = model.total_prompt_tokens
    madlib_completion_tokens = model.total_completion_tokens
    print(f"  Madlib built. Cost: ${madlib_cost:.4f}")
    
    # Pre-sample all descriptions
    print("  Sampling descriptions...")
    previous_samples = [['']]
    sample_configs = []
    
    # Sample more than needed to account for potential failures
    num_to_sample = int(max_examples * 1.5)
    
    for i in range(num_to_sample):
        descriptions, _, previous_samples = creator.sample_madlib(
            madlib, 
            ['scenario_descriptions'],
            '{scenario_descriptions}',
            previously_sampled=previous_samples
        )
        description = descriptions[0]
        sample_configs.append({
            'description': description
        })
    
    print(f"  Generated {len(sample_configs)} sample configurations.")
    
    return sample_configs, madlib_cost, madlib_prompt_tokens, madlib_completion_tokens


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # Concurrency settings
    MAX_WORKERS = 6  # Number of concurrent sample generators
    
    # Dataset parameters
    max_examples = 125  # Number of samples to generate
    tree_depth = 2
    structure_retries = 1
    
    # Disable cache for fresh generation
    cache.disable()
    
    # ========================================================================
    # SETUP OUTPUT
    # ========================================================================
    
    run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder_name = f"team_allocation_samples-{max_examples}_concurrent_{run_datetime}"
    output_folder = OUTPUT_FOLDER / output_folder_name
    output_folder.mkdir(parents=True, exist_ok=True)
    
    out_file = output_folder / f'team_allocation_samples-{max_examples}.json'
    metadata_file = output_folder / 'run_metadata.txt'
    
    print("=" * 80)
    print("CONCURRENT TEAM ALLOCATION DATASET GENERATOR")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Max workers: {MAX_WORKERS}")
    print(f"  Max examples: {max_examples}")
    print(f"  Tree depth: {tree_depth}")
    print(f"  Structure retries: {structure_retries}")
    print(f"  Model: gpt-4-0613")
    print(f"  Output folder: {output_folder}")
    print()
    
    # ========================================================================
    # PRE-GENERATE SAMPLE CONFIGURATIONS
    # ========================================================================
    
    sample_configs, madlib_cost, madlib_prompt_tokens, madlib_completion_tokens = pre_generate_sample_configs(
        max_examples=max_examples
    )
    
    # ========================================================================
    # SHARED CONFIGURATION FOR WORKERS
    # ========================================================================
    
    config = {
        'tree_depth': tree_depth,
        'structure_retries': structure_retries,
    }
    
    # ========================================================================
    # CONCURRENT GENERATION
    # ========================================================================
    
    print(f"\nStarting concurrent generation with {MAX_WORKERS} workers...")
    print("-" * 80)
    
    start_time = datetime.now()
    progress = ProgressTracker(total=max_examples)
    
    all_results = []
    samples_needed = max_examples
    config_idx = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit initial batch of jobs
        futures = {}
        for i in range(min(len(sample_configs), samples_needed + MAX_WORKERS * 2)):
            future = executor.submit(
                generate_single_sample,
                sample_configs[i],
                i,
                config,
                progress
            )
            futures[future] = i
            config_idx = i + 1
        
        # Collect results as they complete
        for future in as_completed(futures):
            sample_idx = futures[future]
            try:
                result = future.result()
                if result['success']:
                    all_results.append(result)
                
                # If we don't have enough successful samples and have more configs, submit more
                if len(all_results) < samples_needed and config_idx < len(sample_configs):
                    new_future = executor.submit(
                        generate_single_sample,
                        sample_configs[config_idx],
                        config_idx,
                        config,
                        progress
                    )
                    futures[new_future] = config_idx
                    config_idx += 1
                
                # Periodic save (every 10 successful samples)
                if len(all_results) > 0 and len(all_results) % 10 == 0:
                    sorted_results = sorted(all_results, key=lambda x: x['sample_idx'])
                    dataset = [r['sample'] for r in sorted_results if r['sample'] is not None]
                    
                    with open(out_file, 'w') as f:
                        json.dump(dataset, f, indent=2)
                    print(f"  [Checkpoint] Saved {len(dataset)} samples to {out_file.name}")
                
                # Stop if we have enough
                if len(all_results) >= samples_needed:
                    break
                    
            except Exception as e:
                print(f"[ERROR] Sample {sample_idx} failed: {e}")
    
    # ========================================================================
    # FINAL PROCESSING
    # ========================================================================
    
    print("-" * 80)
    print("Processing results...")
    
    # Sort results by sample_idx to maintain order, take only what we need
    sorted_results = sorted(all_results, key=lambda x: x['sample_idx'])[:max_examples]
    
    # Extract samples
    dataset = [r['sample'] for r in sorted_results if r['sample'] is not None]
    
    # Calculate statistics
    end_time = datetime.now()
    total_elapsed = end_time - start_time
    total_elapsed_str = str(total_elapsed).split('.')[0]
    
    # Include madlib cost in totals
    total_cost = progress.total_cost + madlib_cost
    total_prompt_tokens = progress.total_prompt_tokens + madlib_prompt_tokens
    total_completion_tokens = progress.total_completion_tokens + madlib_completion_tokens
    total_tokens = total_prompt_tokens + total_completion_tokens
    
    successful_samples = len(dataset)
    failed_samples = progress.failed
    
    avg_cost_per_sample = total_cost / successful_samples if successful_samples > 0 else 0
    avg_tokens_per_sample = total_tokens / successful_samples if successful_samples > 0 else 0
    
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
  Script: create_team_allocation_concurrent.py
  Mode: CONCURRENT ({MAX_WORKERS} workers)

Configuration:
  Max Examples: {max_examples}
  Tree Depth: {tree_depth}
  Structure Retries: {structure_retries}

Model Configuration:
  Primary Model: gpt-4-0613
  Temperature: 1.0

Results:
  Samples Attempted: {progress.completed + progress.failed}
  Samples Successful: {successful_samples}
  Samples Failed: {failed_samples}

Token Statistics:
  Madlib Tokens: {madlib_prompt_tokens + madlib_completion_tokens:,}
  Generation Prompt Tokens: {progress.total_prompt_tokens:,}
  Generation Completion Tokens: {progress.total_completion_tokens:,}
  Total Prompt Tokens: {total_prompt_tokens:,}
  Total Completion Tokens: {total_completion_tokens:,}
  Total Tokens: {total_tokens:,}
  Avg Tokens Per Sample: {avg_tokens_per_sample:,.0f}

Cost Statistics:
  Madlib Cost: ${madlib_cost:.4f}
  Generation Cost: ${progress.total_cost:.4f}
  Total Cost: ${total_cost:.4f}
  Avg Cost Per Sample: ${avg_cost_per_sample:.4f}

Performance:
  Total Time: {total_elapsed_str}
  Avg Time Per Sample: {(total_elapsed.total_seconds() / successful_samples):.1f}s (with {MAX_WORKERS}x concurrency)
  Effective Rate: {(successful_samples / total_elapsed.total_seconds() * 3600):.1f} samples/hour

Output Files:
  - {out_file.name}
  - {metadata_file.name}

Description:
  Concurrent team allocation dataset generation with {MAX_WORKERS} parallel workers.
  Each sample has 3 people and 2 tasks. The model generates scenarios,
  creates skill matrices, builds reasoning trees, and generates story narratives.
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
    print(f"  Samples successful: {successful_samples}/{max_examples}")
    print(f"  Samples failed: {failed_samples}")
    print(f"  Total elapsed time: {total_elapsed_str}")
    print(f"  Effective rate: {(successful_samples / total_elapsed.total_seconds() * 3600):.1f} samples/hour")
    
    print(f"\n🔢 Token Statistics:")
    print(f"  Prompt tokens: {total_prompt_tokens:,}")
    print(f"  Completion tokens: {total_completion_tokens:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg tokens per sample: {avg_tokens_per_sample:,.0f}")
    
    print(f"\n💰 Cost Statistics:")
    print(f"  Madlib cost: ${madlib_cost:.4f}")
    print(f"  Generation cost: ${progress.total_cost:.4f}")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Avg cost per sample: ${avg_cost_per_sample:.4f}")
    
    print(f"\n📁 Output:")
    print(f"  Folder: {output_folder}")
    print(f"  Data file: {out_file.name}")
    print(f"  Metadata file: {metadata_file.name}")
    print(f"{'=' * 80}\n")
    
    if failed_samples > 0:
        print(f"⚠️  WARNING: {failed_samples} samples failed. Check logs for errors.")
        print("   The script automatically retried with additional configurations.\n")


if __name__ == "__main__":
    main()

