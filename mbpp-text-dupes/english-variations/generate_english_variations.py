#!/usr/bin/env python3
"""
MBPP English Variations Generator

Generates 10 English variations of MBPP task descriptions:

SUBJECT-BASED VARIATIONS (sub1-sub5):
Contextualizes the question in different domains while preserving:
- The same algorithmic requirement
- The same function name (if mentioned)
- The same difficulty/complexity/clarity

- sub1: Sports/Games context
- sub2: Shopping/Inventory context  
- sub3: School/Grades context
- sub4: Food/Nutrition context
- sub5: Accounting/Finance context

PARAPHRASE VARIATIONS (para1-para5):
Simple paraphrases that reword the question differently while keeping
the exact same meaning and context.

Uses Claude Opus 4.5 for high-quality semantic transformations.
"""

import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

try:
    from anthropic import Anthropic
except ImportError:
    print("Please install anthropic: pip install anthropic")
    sys.exit(1)

try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")
    sys.exit(1)


# Configuration
MODEL = "claude-opus-4-5"
MAX_RETRIES = 5
RETRY_DELAY = 2.0
DEFAULT_CONCURRENCY = 10
OUTPUT_DIR = Path(__file__).parent / "output"
INDIVIDUAL_DIR = OUTPUT_DIR / "individual"

# Subject contexts for subject-based variations
SUBJECTS = {
    'sub1': {
        'name': 'Sports/Games',
        'context': 'sports, games, athletics, competitions, players, teams, scores, matches, tournaments',
        'examples': 'players, teams, scores, game rounds, match statistics, tournament brackets'
    },
    'sub2': {
        'name': 'Shopping/Inventory',
        'context': 'shopping, retail, inventory, products, prices, orders, customers, stock',
        'examples': 'products, prices, shopping carts, inventory items, order quantities, customer purchases'
    },
    'sub3': {
        'name': 'School/Grades',
        'context': 'education, school, grades, students, classes, assignments, exams, courses',
        'examples': 'students, grades, test scores, assignments, class enrollments, course credits'
    },
    'sub4': {
        'name': 'Food/Nutrition',
        'context': 'food, cooking, nutrition, recipes, ingredients, meals, calories, diets',
        'examples': 'ingredients, recipes, calories, meal portions, nutritional values, cooking times'
    },
    'sub5': {
        'name': 'Accounting/Finance',
        'context': 'accounting, finance, budgets, transactions, expenses, revenue, investments',
        'examples': 'transactions, expenses, revenue, account balances, budget items, financial records'
    },
}

# Subject-based variation prompts
SUBJECT_PROMPT_TEMPLATE = """You are an expert at rewriting programming task descriptions to use different contextual domains while preserving the exact same algorithmic requirement.

ORIGINAL TASK:
{text}

YOUR TASK:
Rewrite this programming task description using a {subject_name} context ({subject_context}).

CRITICAL REQUIREMENTS:
1. The rewritten task must require EXACTLY the same algorithm/logic to solve
2. The difficulty, complexity, and clarity must remain IDENTICAL
3. If the original mentions a specific function name, KEEP THAT EXACT FUNCTION NAME
4. Use terminology from {subject_name}: {subject_examples}
5. The input/output types and structure must remain the same
6. Do NOT add extra requirements or simplify the task

EXAMPLES OF GOOD TRANSFORMATIONS:
- "Find the sum of elements in a list" → "Calculate the total score of all players in a team roster"
- "Count occurrences of an element" → "Count how many times a product appears in inventory"
- "Find maximum value" → "Find the highest grade in a class"

Output ONLY the rewritten task description. No explanations, no markdown, no extra text.

REWRITTEN TASK ({subject_name} context):"""

# Single prompt to generate ALL 5 paraphrases at once
# This ensures the model knows what variations it has already produced and diversifies
PARAPHRASE_BATCH_PROMPT = """You are an expert at paraphrasing programming task descriptions.

ORIGINAL TASK:
{text}

YOUR TASK:
Generate exactly 5 DISTINCT paraphrases of this programming task. Each paraphrase must:
1. Have COMPLETELY DIFFERENT wording from the others
2. Preserve the EXACT same meaning and requirements
3. Maintain the same level of technical detail and clarity
4. Keep any mentioned function names UNCHANGED

CRITICAL: Each paraphrase must be noticeably different from the others. Vary:
- Sentence structure (active vs passive, questions vs statements)
- Vocabulary choices (synonyms, different technical terms)
- Order of information presented
- Level of formality

AVOID:
- Starting multiple paraphrases the same way (e.g., don't start 3 with "Write a...")
- Simply swapping one or two words while keeping structure identical
- Adding or removing requirements not in the original
- Changing the programming language if one is specified
- Making the task ambiguous or less precise

Output EXACTLY in this JSON format (no extra text, no markdown):
{{
  "para1": "first paraphrase here",
  "para2": "second paraphrase here",
  "para3": "third paraphrase here",
  "para4": "fourth paraphrase here",
  "para5": "fifth paraphrase here"
}}

Generate the 5 diverse paraphrases now:"""

# Combined variation definitions
ENGLISH_VARIATIONS = {}

# Add subject-based variations
for key, subject in SUBJECTS.items():
    ENGLISH_VARIATIONS[key] = {
        'name': f"Subject: {subject['name']}",
        'type': 'subject',
        'subject_name': subject['name'],
        'subject_context': subject['context'],
        'subject_examples': subject['examples'],
    }

# Add paraphrase variations (these are generated together in one batch call)
PARAPHRASE_KEYS = ['para1', 'para2', 'para3', 'para4', 'para5']
for key in PARAPHRASE_KEYS:
    ENGLISH_VARIATIONS[key] = {
        'name': f"Paraphrase {key[-1]}",
        'type': 'paraphrase',
    }

# Thread-safe locks
_file_lock = threading.Lock()
_print_lock = threading.Lock()
_stats_lock = threading.Lock()


def setup_output_dirs():
    """Create output directories if they don't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    INDIVIDUAL_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "archive").mkdir(exist_ok=True)


def thread_print(*args, **kwargs):
    """Thread-safe print function."""
    with _print_lock:
        print(*args, **kwargs)


def load_existing_results() -> dict:
    """Load existing results for resume capability."""
    existing = {}
    for filepath in INDIVIDUAL_DIR.glob("task_*.json"):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                task_id = data['task_id']
                existing[task_id] = data
        except Exception:
            pass
    return existing


def load_mbpp_samples(num_samples: int = 10, all_splits: bool = False) -> list[dict]:
    """Load samples from the MBPP sanitized dataset."""
    if all_splits:
        splits_to_load = ["prompt", "test", "validation", "train"]
        print(f"Loading up to {num_samples} samples from ALL splits (sanitized)...")
    else:
        splits_to_load = ["test"]
        print(f"Loading up to {num_samples} samples from test split (sanitized)...")
    
    # Use sanitized config for cleaner, higher-quality samples
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized")
    
    samples = []
    for split_name in splits_to_load:
        if split_name not in dataset:
            continue
        split_data = dataset[split_name]
        print(f"  {split_name}: {len(split_data)} samples available")
        for sample in split_data:
            if len(samples) >= num_samples:
                break
            sample_dict = dict(sample)
            sample_dict['source_split'] = split_name
            sample_dict['source_config'] = 'sanitized'
            samples.append(sample_dict)
        if len(samples) >= num_samples:
            break
    
    print(f"Loaded {len(samples)} samples total")
    return samples


def generate_subject_variation(client: Anthropic, text: str, variation_code: str) -> tuple[Optional[str], int]:
    """Generate subject-based variation. Returns (varied_text, attempts)."""
    variation = ENGLISH_VARIATIONS[variation_code]
    
    prompt = SUBJECT_PROMPT_TEMPLATE.format(
        text=text,
        subject_name=variation['subject_name'],
        subject_context=variation['subject_context'],
        subject_examples=variation['subject_examples'],
    )
    
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                timeout=120.0,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.content[0].text.strip()
            # Clean up any markdown artifacts
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
            return result, attempt
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)  # Exponential backoff
    
    return None, MAX_RETRIES


def generate_all_paraphrases(client: Anthropic, text: str) -> tuple[dict[str, Optional[str]], int]:
    """
    Generate all 5 paraphrases in a single API call.
    This ensures the model diversifies across all variations.
    Returns (dict of para1-para5 -> text, attempts).
    """
    prompt = PARAPHRASE_BATCH_PROMPT.format(text=text)
    
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=2048,  # Larger for 5 paraphrases
                timeout=180.0,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.content[0].text.strip()
            
            # Clean up any markdown artifacts
            if result.startswith("```json"):
                result = result[7:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            
            # Parse JSON response
            paraphrases = json.loads(result)
            
            # Validate we got all 5
            output = {}
            for key in PARAPHRASE_KEYS:
                if key in paraphrases and paraphrases[key]:
                    output[key] = paraphrases[key].strip()
                else:
                    output[key] = None
            
            return output, attempt
            
        except json.JSONDecodeError as e:
            last_error = e
            # If JSON parsing fails, we retry
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    
    # Return empty dict on total failure
    return {key: None for key in PARAPHRASE_KEYS}, MAX_RETRIES


def generate_variation(client: Anthropic, text: str, variation_code: str) -> tuple[Optional[str], int]:
    """Generate a single subject-based variation. For paraphrases, use generate_all_paraphrases instead."""
    variation = ENGLISH_VARIATIONS[variation_code]
    
    if variation['type'] == 'subject':
        return generate_subject_variation(client, text, variation_code)
    else:
        # Paraphrases should use generate_all_paraphrases() for batch generation
        raise ValueError(f"Paraphrases should use generate_all_paraphrases(), not generate_variation()")


def process_sample(
    client: Anthropic,
    sample: dict,
    existing_results: dict,
    variations: list[str],
    verbose: bool = True
) -> dict:
    """Process a single sample to generate variations."""
    task_id = sample['task_id']
    text = sample['prompt']
    
    # Check existing results
    existing = existing_results.get(task_id, {})
    
    result = {
        'task_id': task_id,
        'source_split': sample.get('source_split', 'unknown'),
        'source_config': sample.get('source_config', 'sanitized'),
        'text': text,
        'code': sample.get('code', ''),
        'test_list': sample.get('test_list', []),
    }
    
    # Separate subject-based and paraphrase variations
    subject_vars = [v for v in variations if ENGLISH_VARIATIONS[v]['type'] == 'subject']
    paraphrase_vars = [v for v in variations if ENGLISH_VARIATIONS[v]['type'] == 'paraphrase']
    
    # Check which paraphrases need generation (not already successful)
    paraphrases_needed = []
    for var_code in paraphrase_vars:
        text_col = f'text_{var_code}'
        status_col = f'text_{var_code}_status'
        if text_col in existing and existing.get(status_col) == 'success':
            # Already done - copy to result
            result[text_col] = existing[text_col]
            result[status_col] = existing[status_col]
            result[f'text_{var_code}_attempts'] = existing.get(f'text_{var_code}_attempts', 1)
            result[f'text_{var_code}_type'] = 'paraphrase'
            if verbose:
                thread_print(f"  [Task {task_id}] [SKIP] {var_code} (already exists)")
        else:
            paraphrases_needed.append(var_code)
    
    # Generate ALL needed paraphrases in a single batch call (for diversity)
    if paraphrases_needed:
        if verbose:
            thread_print(f"  [Task {task_id}] Generating {len(paraphrases_needed)} paraphrases in batch...")
        
        paraphrases_dict, attempts = generate_all_paraphrases(client, text)
        
        for var_code in paraphrases_needed:
            text_col = f'text_{var_code}'
            status_col = f'text_{var_code}_status'
            attempts_col = f'text_{var_code}_attempts'
            type_col = f'text_{var_code}_type'
            
            varied_text = paraphrases_dict.get(var_code)
            
            if varied_text:
                result[text_col] = varied_text
                result[status_col] = 'success'
                result[attempts_col] = attempts
                result[type_col] = 'paraphrase'
                if verbose:
                    thread_print(f"  [Task {task_id}] [OK] {var_code}")
            else:
                result[text_col] = None
                result[status_col] = 'failed'
                result[attempts_col] = attempts
                result[type_col] = 'paraphrase'
                if verbose:
                    thread_print(f"  [Task {task_id}] [FAIL] {var_code}")
    
    # Generate subject-based variations one at a time (they are domain-specific, not inter-dependent)
    for var_code in subject_vars:
        text_col = f'text_{var_code}'
        status_col = f'text_{var_code}_status'
        attempts_col = f'text_{var_code}_attempts'
        type_col = f'text_{var_code}_type'
        
        # Check if already successfully processed
        if text_col in existing and existing.get(status_col) == 'success':
            result[text_col] = existing[text_col]
            result[status_col] = existing[status_col]
            result[attempts_col] = existing.get(attempts_col, 1)
            result[type_col] = 'subject'
            if verbose:
                thread_print(f"  [Task {task_id}] [SKIP] {var_code} (already exists)")
            continue
        
        var_info = ENGLISH_VARIATIONS[var_code]
        if verbose:
            thread_print(f"  [Task {task_id}] Generating {var_code} ({var_info['name']})...")
        
        varied_text, attempts = generate_subject_variation(client, text, var_code)
        
        if varied_text:
            result[text_col] = varied_text
            result[status_col] = 'success'
            result[attempts_col] = attempts
            result[type_col] = 'subject'
            if verbose:
                thread_print(f"  [Task {task_id}] [OK] {var_code}")
        else:
            result[text_col] = None
            result[status_col] = 'failed'
            result[attempts_col] = attempts
            result[type_col] = 'subject'
            if verbose:
                thread_print(f"  [Task {task_id}] [FAIL] {var_code}")
    
    # Save individual result
    save_individual_result(task_id, result)
    
    return result


def save_individual_result(task_id: int, result: dict):
    """Save individual sample result to JSON (thread-safe)."""
    filepath = INDIVIDUAL_DIR / f"task_{task_id}.json"
    with _file_lock:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


def save_master_json(results: list, metadata: dict):
    """Save master JSON with all results and metadata."""
    master_data = {
        "metadata": metadata,
        "results": results
    }
    
    # Save latest
    filepath = OUTPUT_DIR / "master_results.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(master_data, f, indent=2, ensure_ascii=False)
    
    # Save timestamped archive
    archive_dir = OUTPUT_DIR / "archive"
    archive_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"master_results_{timestamp_str}.json"
    with open(archive_path, 'w', encoding='utf-8') as f:
        json.dump(master_data, f, indent=2, ensure_ascii=False)
    
    return filepath, archive_path


def save_csv(results: list, variations: list[str]):
    """Save CSV with all variations."""
    filepath = OUTPUT_DIR / "english_variations.csv"
    
    # Prepare archive path
    archive_dir = OUTPUT_DIR / "archive"
    archive_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"english_variations_{timestamp_str}.csv"
    
    # Build field names
    fieldnames = ['task_id', 'source_split', 'source_config', 'text', 'code']
    for var in variations:
        fieldnames.extend([f'text_{var}', f'text_{var}_status', f'text_{var}_type'])
    
    # Write to both files
    for out_path in [filepath, archive_path]:
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            for r in results:
                row = {
                    'task_id': r['task_id'],
                    'source_split': r.get('source_split', ''),
                    'source_config': r.get('source_config', ''),
                    'text': r.get('text', ''),
                    'code': r.get('code', ''),
                }
                for var in variations:
                    row[f'text_{var}'] = r.get(f'text_{var}', '')
                    row[f'text_{var}_status'] = r.get(f'text_{var}_status', '')
                    row[f'text_{var}_type'] = r.get(f'text_{var}_type', '')
                writer.writerow(row)
    
    return filepath, archive_path


def generate_integrity_report(results: list, variations: list[str], stats: dict, metadata: dict):
    """Generate integrity report."""
    report_lines = []
    timestamp = datetime.now()
    
    report_lines.append("=" * 80)
    report_lines.append("MBPP ENGLISH VARIATIONS - INTEGRITY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {timestamp.isoformat()}")
    report_lines.append(f"Run timestamp: {metadata.get('run_timestamp', 'unknown')}")
    report_lines.append(f"Duration: {metadata.get('duration_seconds', 0):.1f}s")
    report_lines.append(f"Model: {MODEL}")
    report_lines.append(f"Dataset: MBPP Sanitized")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("OVERALL STATISTICS")
    report_lines.append("=" * 80)
    report_lines.append(f"Total tasks: {stats['total_samples']}")
    report_lines.append(f"Total variations requested: {len(variations)}")
    report_lines.append("")
    
    # Group by type
    subject_vars = [v for v in variations if ENGLISH_VARIATIONS[v]['type'] == 'subject']
    paraphrase_vars = [v for v in variations if ENGLISH_VARIATIONS[v]['type'] == 'paraphrase']
    
    report_lines.append("=" * 80)
    report_lines.append("SUBJECT-BASED VARIATIONS")
    report_lines.append("=" * 80)
    
    for var in subject_vars:
        var_stats = stats['variations'].get(var, {})
        success = var_stats.get('success', 0)
        failed = var_stats.get('failed', 0)
        total = success + failed
        rate = (success / total * 100) if total > 0 else 0
        info = ENGLISH_VARIATIONS[var]
        report_lines.append(f"{var} ({info['subject_name']}): {success}/{total} ({rate:.1f}%)")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("PARAPHRASE VARIATIONS")
    report_lines.append("=" * 80)
    
    for var in paraphrase_vars:
        var_stats = stats['variations'].get(var, {})
        success = var_stats.get('success', 0)
        failed = var_stats.get('failed', 0)
        total = success + failed
        rate = (success / total * 100) if total > 0 else 0
        report_lines.append(f"{var} (Paraphrase {var[-1]}): {success}/{total} ({rate:.1f}%)")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("VARIATION METHODS")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Subject-Based Variations:")
    report_lines.append("  Contextualizes the task in different domains while preserving")
    report_lines.append("  the exact same algorithmic requirement and difficulty.")
    report_lines.append("")
    for var in subject_vars:
        info = ENGLISH_VARIATIONS[var]
        report_lines.append(f"  {var}: {info['subject_name']}")
        report_lines.append(f"       Context: {info['subject_context']}")
    
    report_lines.append("")
    report_lines.append("Paraphrase Variations:")
    report_lines.append("  Simple rewording of the task while keeping the exact same meaning.")
    report_lines.append("")
    for var in paraphrase_vars:
        report_lines.append(f"  {var}: Different wording, same semantic meaning")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Save reports
    archive_dir = OUTPUT_DIR / "archive"
    archive_dir.mkdir(exist_ok=True)
    
    main_report_path = OUTPUT_DIR / "integrity_report.txt"
    with open(main_report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"integrity_report_{timestamp_str}.txt"
    with open(archive_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    return main_report_path, archive_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='MBPP English Variations Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Variation Types:

  SUBJECT-BASED (--subjects or --all):
    sub1: Sports/Games context
    sub2: Shopping/Inventory context
    sub3: School/Grades context
    sub4: Food/Nutrition context
    sub5: Accounting/Finance context

  PARAPHRASES (--paraphrases or --all):
    para1-para5: Different phrasings of the same task

Examples:
  # Generate all variations for 10 samples
  python generate_english_variations.py --num-samples 10 --all
  
  # Generate only subject-based variations
  python generate_english_variations.py --num-samples 100 --subjects
  
  # Generate only paraphrases
  python generate_english_variations.py --num-samples 100 --paraphrases
  
  # Generate specific variations
  python generate_english_variations.py --variation sub1,sub2,para1
  
  # Resume a previous run
  python generate_english_variations.py --num-samples 427 --all --resume
        """
    )
    
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples (max 427 for sanitized)')
    parser.add_argument('--all-splits', action='store_true', help='Load from all splits (427 total)')
    parser.add_argument('--variation', type=str, default=None,
                       help='Comma-separated variations (e.g., sub1,sub2,para1)')
    parser.add_argument('--all', action='store_true', help='Generate all 10 variations')
    parser.add_argument('--subjects', action='store_true', help='Generate all 5 subject-based variations')
    parser.add_argument('--paraphrases', action='store_true', help='Generate all 5 paraphrase variations')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY, help='Concurrent workers')
    parser.add_argument('--resume', action='store_true', help='Resume from previous run')
    parser.add_argument('--force', action='store_true', help='Force reprocess all')
    
    args = parser.parse_args()
    
    # Determine variations
    if args.all:
        variations = list(ENGLISH_VARIATIONS.keys())
    elif args.subjects:
        variations = [k for k in ENGLISH_VARIATIONS.keys() if k.startswith('sub')]
    elif args.paraphrases:
        variations = [k for k in ENGLISH_VARIATIONS.keys() if k.startswith('para')]
    elif args.variation:
        variations = [v.strip() for v in args.variation.split(',')]
        for var in variations:
            if var not in ENGLISH_VARIATIONS:
                print(f"ERROR: Unknown variation '{var}'.")
                print(f"Available subject variations: {', '.join(k for k in ENGLISH_VARIATIONS if k.startswith('sub'))}")
                print(f"Available paraphrase variations: {', '.join(k for k in ENGLISH_VARIATIONS if k.startswith('para'))}")
                sys.exit(1)
    else:
        # Default: all variations
        variations = list(ENGLISH_VARIATIONS.keys())
    
    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)
    
    client = Anthropic(api_key=api_key)
    
    print("=" * 70)
    print("MBPP English Variations Generator")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Dataset: MBPP Sanitized")
    print(f"Samples: {args.num_samples}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Resume: {args.resume}")
    print("")
    print(f"Variations to generate ({len(variations)} total):")
    
    subject_vars = [v for v in variations if ENGLISH_VARIATIONS[v]['type'] == 'subject']
    paraphrase_vars = [v for v in variations if ENGLISH_VARIATIONS[v]['type'] == 'paraphrase']
    
    if subject_vars:
        print(f"  Subject-based ({len(subject_vars)}):")
        for var in subject_vars:
            print(f"    - {var}: {ENGLISH_VARIATIONS[var]['subject_name']}")
    
    if paraphrase_vars:
        print(f"  Paraphrases ({len(paraphrase_vars)}):")
        for var in paraphrase_vars:
            print(f"    - {var}: Paraphrase {var[-1]}")
    
    print("=" * 70)
    
    setup_output_dirs()
    
    existing_results = load_existing_results() if args.resume and not args.force else {}
    if existing_results:
        print(f"Found {len(existing_results)} existing results")
    
    samples = load_mbpp_samples(args.num_samples, all_splits=args.all_splits)
    
    stats = {
        'total_samples': len(samples),
        'variations': {var: {'success': 0, 'failed': 0, 'skipped': 0} for var in variations}
    }
    
    start_time = datetime.now()
    completed = 0
    
    print(f"\nProcessing {len(samples)} samples with {args.concurrency} workers...")
    print("-" * 70)
    
    def process_with_stats(sample):
        return sample['task_id'], process_sample(client, sample, existing_results, variations)
    
    results_dict = {}
    
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_sample = {executor.submit(process_with_stats, s): s for s in samples}
        
        for future in as_completed(future_to_sample):
            sample = future_to_sample[future]
            task_id = sample['task_id']
            
            try:
                _, result = future.result()
                results_dict[task_id] = result
                
                with _stats_lock:
                    completed += 1
                    for var in variations:
                        status = result.get(f'text_{var}_status', '')
                        if status == 'success':
                            stats['variations'][var]['success'] += 1
                        elif status == 'failed':
                            stats['variations'][var]['failed'] += 1
                    
                    if completed % 10 == 0 or completed == len(samples):
                        elapsed = (datetime.now() - start_time).total_seconds()
                        rate = completed / elapsed if elapsed > 0 else 0
                        print(f"  Progress: {completed}/{len(samples)} ({rate:.2f}/s)")
                        
            except Exception as e:
                thread_print(f"  [Task {task_id}] [FATAL] {e}")
    
    results = [results_dict[s['task_id']] for s in samples if s['task_id'] in results_dict]
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    metadata = {
        'run_timestamp': start_time.isoformat(),
        'duration_seconds': duration,
        'num_samples': args.num_samples,
        'variations': variations,
        'model': MODEL,
        'dataset': 'mbpp_sanitized',
        'concurrency': args.concurrency,
        'stats': stats
    }
    
    master_path, master_archive = save_master_json(results, metadata)
    print(f"\n[OK] Master JSON: {master_path}")
    
    csv_path, csv_archive = save_csv(results, variations)
    print(f"[OK] CSV: {csv_path}")
    
    report_path, report_archive = generate_integrity_report(results, variations, stats, metadata)
    print(f"[OK] Report: {report_path}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Duration: {duration:.1f}s")
    print("")
    
    if subject_vars:
        print("Subject-based variations:")
        for var in subject_vars:
            s = stats['variations'][var]
            total = s['success'] + s['failed']
            rate = (s['success'] / total * 100) if total > 0 else 0
            print(f"  {var} ({ENGLISH_VARIATIONS[var]['subject_name']}): {s['success']}/{total} ({rate:.1f}%)")
    
    if paraphrase_vars:
        print("\nParaphrase variations:")
        for var in paraphrase_vars:
            s = stats['variations'][var]
            total = s['success'] + s['failed']
            rate = (s['success'] / total * 100) if total > 0 else 0
            print(f"  {var}: {s['success']}/{total} ({rate:.1f}%)")


if __name__ == "__main__":
    main()
