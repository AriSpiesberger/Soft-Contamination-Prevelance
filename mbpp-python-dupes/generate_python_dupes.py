#!/usr/bin/env python3
"""
MBPP Python Semantic Duplicates Generator

This script generates up to 5 semantically different Python solutions for each
MBPP problem, with line-by-line comments to increase semantic variation.

Features:
- Recursive generation: each new solution is prompted with all previous solutions
- Line-by-line comments (on line above) for semantic diversity
- Validation via test execution
- Retry with structural difference requests if solutions are too similar
- Parallel processing at sample level
- Output: Individual JSONs + Master JSON + CSV
"""

import csv
import json
import os
import re
import sys
import threading
import time
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(Path(__file__).parent.parent / ".env")
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

from validators import PythonValidator

# Configuration
MAX_DUPLICATES = 5  # Generate up to 5 semantic duplicates (python_1 to python_5)
MAX_GENERATION_ATTEMPTS = 3  # Attempts per duplicate with error feedback
MAX_SIMILARITY_RETRIES = 2  # Retries if solution is too similar to previous ones
SIMILARITY_THRESHOLD = 0.85  # Code similarity threshold (without comments)
API_RETRY_DELAY = 2  # seconds between API retries
DEFAULT_CONCURRENCY = 10
MODEL = "claude-opus-4-5"
OUTPUT_DIR = Path(__file__).parent / "output"
INDIVIDUAL_DIR = OUTPUT_DIR / "individual"

# Thread-safe locks
_file_lock = threading.Lock()
_print_lock = threading.Lock()
_stats_lock = threading.Lock()


def setup_output_dirs():
    """Create output directories if they don't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    INDIVIDUAL_DIR.mkdir(exist_ok=True)


def load_existing_results() -> dict:
    """Load existing results for resume capability."""
    master_json = OUTPUT_DIR / "master_results.json"
    if master_json.exists():
        with open(master_json, 'r') as f:
            data = json.load(f)
            return {r['task_id']: r for r in data.get('results', [])}
    return {}


def load_mbpp_samples(num_samples: int = 10, use_full: bool = True, all_splits: bool = False) -> list[dict]:
    """Load samples from the MBPP dataset.
    
    Args:
        num_samples: Number of samples to load
        use_full: If True, use 'full' config. If False, use 'sanitized' config.
        all_splits: If True, load from all splits (prompt, test, validation, train).
                    If False, load only from test split.
    
    Full config totals:
        - prompt: 10 samples (task_ids: 1-10)
        - test: 500 samples (task_ids: 11-510)  
        - validation: 90 samples (task_ids: 511-600)
        - train: 374 samples (task_ids: 601-974)
        - TOTAL: 974 samples
    """
    config = "full" if use_full else "sanitized"
    
    if all_splits:
        splits_to_load = ["prompt", "test", "validation", "train"]
        print(f"Loading up to {num_samples} samples from ALL splits (config={config})...")
    else:
        splits_to_load = ["test"]
        print(f"Loading up to {num_samples} samples from test split (config={config})...")
    
    dataset = load_dataset("google-research-datasets/mbpp", config)
    
    # Collect samples from specified splits
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
            sample_dict['source_split'] = split_name  # Track which split it came from
            sample_dict['source_config'] = config     # Track which config
            samples.append(sample_dict)
        if len(samples) >= num_samples:
            break
    
    print(f"Loaded {len(samples)} samples total")
    return samples


def thread_print(*args, **kwargs):
    """Thread-safe print function."""
    with _print_lock:
        print(*args, **kwargs)


def extract_function_name(python_code: str) -> str:
    """Extract the main function name from Python code."""
    match = re.search(r'def\s+(\w+)\s*\(', python_code)
    return match.group(1) if match else "unknown"


def strip_comments(code: str) -> str:
    """Remove comments from Python code for similarity comparison."""
    lines = []
    for line in code.split('\n'):
        # Remove inline comments
        line = re.sub(r'#.*$', '', line)
        # Skip empty lines and comment-only lines
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    return '\n'.join(lines)


def code_similarity(code1: str, code2: str) -> float:
    """Calculate similarity ratio between two code snippets (ignoring comments)."""
    clean1 = strip_comments(code1)
    clean2 = strip_comments(code2)
    return difflib.SequenceMatcher(None, clean1, clean2).ratio()


def clean_code_response(code: str) -> str:
    """Clean up markdown code blocks from LLM response."""
    code = code.strip()
    
    # Remove markdown code blocks
    patterns = ["```python", "```py", "```"]
    for pattern in patterns:
        if code.startswith(pattern):
            code = code[len(pattern):]
            break
    
    if code.endswith("```"):
        code = code[:-3]
    
    return code.strip()


def get_generation_prompt(
    task_description: str,
    original_code: str,
    test_list: list[str],
    previous_solutions: list[str],
    previous_error: Optional[str] = None,
    require_more_different: bool = False
) -> str:
    """Generate the prompt for creating a semantic duplicate."""
    
    test_context = "\n".join(test_list[:3])
    func_name = extract_function_name(original_code)
    
    base_prompt = f"""You are an expert Python programmer. Your task is to write a DIFFERENT Python solution for the following problem.

TASK DESCRIPTION:
{task_description}

ORIGINAL PYTHON SOLUTION (for reference - DO NOT COPY):
```python
{original_code}
```

PYTHON TEST EXAMPLES (your solution must pass these):
```python
{test_context}
```

"""
    
    # Add previous solutions if any
    if previous_solutions:
        base_prompt += "PREVIOUS SOLUTIONS YOU'VE ALREADY WRITTEN (your new solution must be STRUCTURALLY DIFFERENT from ALL of these):\n"
        for i, sol in enumerate(previous_solutions, 1):
            base_prompt += f"\n--- Solution {i} ---\n```python\n{sol}\n```\n"
        base_prompt += "\n"
    
    # Add error feedback if retry
    if previous_error:
        base_prompt += f"""YOUR PREVIOUS ATTEMPT HAD ERRORS:
{previous_error}

Please fix these errors in your new solution.

"""
    
    # Add stronger differentiation request if needed
    if require_more_different:
        base_prompt += """⚠️ IMPORTANT: Your previous solution was TOO SIMILAR to existing ones!
You MUST use a significantly DIFFERENT algorithmic approach. Consider:
- Using different data structures (list vs set vs dict vs deque)
- Using different iteration patterns (for vs while vs recursion vs comprehensions)
- Using different built-in functions or libraries
- Restructuring the logic flow completely

"""
    
    base_prompt += f"""REQUIREMENTS:
1. Write a COMPLETE Python solution that passes all tests
2. The function MUST be named EXACTLY: {func_name}
3. Use a DIFFERENT algorithmic approach or implementation style than the solutions shown above
4. ADD A COMMENT ON THE LINE ABOVE EACH LINE OF CODE explaining what it does
5. Comments should be ORIGINAL, CONCISE, and INSIGHTFUL - not generic
6. Make sure every substantive line has a comment above it
7. The comments should help distinguish this solution semantically from others

COMMENT STYLE EXAMPLE:
```python
# Initialize counter for tracking element frequency
count = 0
# Iterate through each item in the input sequence
for item in items:
    # Increment counter when condition is met
    if condition:
        count += 1
# Return the final accumulated count
return count
```

OUTPUT ONLY THE PYTHON CODE with comments. No markdown code blocks, no explanations outside the code."""

    return base_prompt


def generate_single_duplicate(
    client: Anthropic,
    task_description: str,
    original_code: str,
    test_list: list[str],
    previous_solutions: list[str],
    validator: PythonValidator,
    verbose: bool = True
) -> tuple[Optional[str], str, int]:
    """
    Generate a single semantic duplicate.
    
    Returns: (code or None, status, attempts)
    """
    previous_error = None
    require_more_different = False
    total_attempts = 0
    last_code = None  # Track the last generated code
    
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        for similarity_retry in range(MAX_SIMILARITY_RETRIES + 1):
            total_attempts += 1
            
            try:
                # Generate prompt
                prompt = get_generation_prompt(
                    task_description,
                    original_code,
                    test_list,
                    previous_solutions,
                    previous_error,
                    require_more_different
                )
                
                # Call API
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    timeout=240.0,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                code = clean_code_response(response.content[0].text)
                last_code = code  # Save the code even if validation fails
                
                # Check similarity to previous solutions
                if previous_solutions:
                    max_sim = max(code_similarity(code, prev) for prev in previous_solutions)
                    if max_sim >= SIMILARITY_THRESHOLD:
                        if similarity_retry < MAX_SIMILARITY_RETRIES:
                            require_more_different = True
                            if verbose:
                                thread_print(f"    [SIMILARITY] {max_sim:.2f} >= {SIMILARITY_THRESHOLD}, retrying for more different solution...")
                            continue
                        else:
                            if verbose:
                                thread_print(f"    [WARN] Could not generate sufficiently different solution after {MAX_SIMILARITY_RETRIES} retries")
                
                # Also check similarity to original
                orig_sim = code_similarity(code, original_code)
                if orig_sim >= SIMILARITY_THRESHOLD:
                    if similarity_retry < MAX_SIMILARITY_RETRIES:
                        require_more_different = True
                        if verbose:
                            thread_print(f"    [SIMILARITY] Too similar to original ({orig_sim:.2f}), retrying...")
                        continue
                
                # Validate the code
                result = validator.validate(code, test_list)
                
                if result.all_passed:
                    return code, 'success', total_attempts
                
                # Validation failed - prepare for retry
                previous_error = result.error_message
                if verbose:
                    thread_print(f"    [VALIDATION] Failed: {result.error_message[:100]}...")
                require_more_different = False  # Focus on fixing errors first
                break  # Exit similarity retry loop, continue with error retry
                
            except Exception as e:
                if verbose:
                    thread_print(f"    [ERROR] Exception: {str(e)[:100]}")
                previous_error = str(e)
                time.sleep(API_RETRY_DELAY)
                break  # Exit similarity retry loop
    
    # All attempts failed - return the last generated code as failed
    return last_code, 'failed', total_attempts


def process_sample(
    client: Anthropic,
    sample: dict,
    existing_results: dict,
    verbose: bool = True
) -> dict:
    """Process a single sample to generate semantic duplicates."""
    task_id = sample['task_id']
    task_description = sample['prompt']
    original_code = sample['code']
    test_list = sample['test_list']
    
    # Check existing results
    existing = existing_results.get(task_id, {})
    
    result = {
        'task_id': task_id,
        'source_split': sample.get('source_split', 'unknown'),
        'source_config': sample.get('source_config', 'unknown'),
        'prompt': task_description,
        'code_python': original_code,  # Original solution
        'test_list': test_list,
    }
    
    validator = PythonValidator(verbose=verbose)
    previous_solutions = [original_code]  # Start with original
    
    # Generate up to MAX_DUPLICATES semantic duplicates
    for dup_num in range(1, MAX_DUPLICATES + 1):
        code_col = f'python_{dup_num}'
        status_col = f'python_{dup_num}_status'
        attempts_col = f'python_{dup_num}_attempts'
        
        # Check if already successfully processed
        if code_col in existing:
            existing_status = existing.get(status_col, '')
            if existing_status == 'success':
                result[code_col] = existing[code_col]
                result[status_col] = existing[status_col]
                result[attempts_col] = existing.get(attempts_col, 1)
                previous_solutions.append(existing[code_col])
                if verbose:
                    thread_print(f"  [Task {task_id}] [SKIP] python_{dup_num} (already exists)")
                continue
        
        if verbose:
            thread_print(f"  [Task {task_id}] Generating python_{dup_num}...")
        
        code, status, attempts = generate_single_duplicate(
            client,
            task_description,
            original_code,
            test_list,
            previous_solutions,
            validator,
            verbose
        )
        
        result[code_col] = code
        result[status_col] = status
        result[attempts_col] = attempts
        
        if code and status == 'success':
            previous_solutions.append(code)
            if verbose:
                thread_print(f"  [Task {task_id}] [OK] python_{dup_num} generated (attempts: {attempts})")
        else:
            if verbose:
                thread_print(f"  [Task {task_id}] [FAIL] python_{dup_num} failed after {attempts} attempts")
    
    # Save individual result
    save_individual_result(task_id, result)
    
    return result


def save_individual_result(task_id: int, result: dict):
    """Save individual sample result to JSON (thread-safe)."""
    filepath = INDIVIDUAL_DIR / f"task_{task_id}.json"
    with _file_lock:
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)


def save_master_json(results: list, metadata: dict):
    """Save master JSON with all results and metadata."""
    master_data = {
        "metadata": metadata,
        "results": results
    }
    
    # Save latest (always overwritten)
    filepath = OUTPUT_DIR / "master_results.json"
    with open(filepath, 'w') as f:
        json.dump(master_data, f, indent=2)
    
    # Save timestamped archive
    archive_dir = OUTPUT_DIR / "archive"
    archive_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"master_results_{timestamp_str}.json"
    with open(archive_path, 'w') as f:
        json.dump(master_data, f, indent=2)
    
    return filepath, archive_path


def save_csv(results: list):
    """Save CSV with all duplicates (latest + timestamped archive)."""
    filepath = OUTPUT_DIR / "mbpp_python_dupes.csv"
    
    # Prepare archive path
    archive_dir = OUTPUT_DIR / "archive"
    archive_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"mbpp_python_dupes_{timestamp_str}.csv"
    
    # Build field names
    fieldnames = [
        'task_id',
        'source_split',
        'source_config',
        'prompt',
        'code_python',  # Original
        'test_list',
    ]
    
    # Add duplicate columns
    for i in range(1, MAX_DUPLICATES + 1):
        fieldnames.extend([
            f'python_{i}',
            f'python_{i}_status',
            f'python_{i}_attempts',
        ])
    
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
                    'prompt': r.get('prompt', ''),
                    'code_python': r.get('code_python', ''),
                    'test_list': json.dumps(r.get('test_list', [])),
                }
                
                # Add duplicates
                for i in range(1, MAX_DUPLICATES + 1):
                    row[f'python_{i}'] = r.get(f'python_{i}', '')
                    row[f'python_{i}_status'] = r.get(f'python_{i}_status', '')
                    row[f'python_{i}_attempts'] = r.get(f'python_{i}_attempts', 0)
                
                writer.writerow(row)
    
    return filepath, archive_path


def generate_integrity_report(results: list, stats: dict, metadata: dict):
    """Generate integrity report with statistics and characteristics."""
    report_lines = []
    timestamp = datetime.now()
    
    report_lines.append("=" * 80)
    report_lines.append("MBPP PYTHON SEMANTIC DUPLICATES - INTEGRITY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {timestamp.isoformat()}")
    report_lines.append(f"Run timestamp: {metadata.get('run_timestamp', 'unknown')}")
    report_lines.append(f"Duration: {metadata.get('duration_seconds', 0):.1f}s")
    report_lines.append("")
    
    # Overall statistics
    report_lines.append("=" * 80)
    report_lines.append("OVERALL STATISTICS")
    report_lines.append("=" * 80)
    report_lines.append(f"Total tasks: {stats['total_samples']}")
    report_lines.append(f"Total duplicate slots: {stats['total_samples'] * MAX_DUPLICATES}")
    report_lines.append(f"Successful duplicates: {stats['duplicates_generated']}")
    report_lines.append(f"Failed duplicates: {stats['duplicates_failed']}")
    report_lines.append(f"Skipped (resume): {stats.get('duplicates_skipped', 0)}")
    success_rate = stats['duplicates_generated'] / (stats['total_samples'] * MAX_DUPLICATES) * 100 if stats['total_samples'] > 0 else 0
    report_lines.append(f"Success rate: {success_rate:.2f}%")
    report_lines.append("")
    
    # Per-slot statistics
    report_lines.append("=" * 80)
    report_lines.append("PER-SLOT STATISTICS")
    report_lines.append("=" * 80)
    
    slot_stats = {}
    for i in range(1, MAX_DUPLICATES + 1):
        slot_stats[f'python_{i}'] = {'success': 0, 'failed': 0}
    
    for r in results:
        for i in range(1, MAX_DUPLICATES + 1):
            status = r.get(f'python_{i}_status', '')
            if status == 'success':
                slot_stats[f'python_{i}']['success'] += 1
            elif status == 'failed':
                slot_stats[f'python_{i}']['failed'] += 1
    
    for slot, s in slot_stats.items():
        total = s['success'] + s['failed']
        rate = (s['success'] / total * 100) if total > 0 else 0
        report_lines.append(f"{slot}: {s['success']} success, {s['failed']} failed ({rate:.1f}%)")
    report_lines.append("")
    
    # By split statistics
    report_lines.append("=" * 80)
    report_lines.append("BY SPLIT STATISTICS")
    report_lines.append("=" * 80)
    
    split_counts = {}
    for r in results:
        split = r.get('source_split', 'unknown')
        if split not in split_counts:
            split_counts[split] = {'total': 0, 'success': 0, 'failed': 0}
        split_counts[split]['total'] += 1
        for i in range(1, MAX_DUPLICATES + 1):
            status = r.get(f'python_{i}_status', '')
            if status == 'success':
                split_counts[split]['success'] += 1
            elif status == 'failed':
                split_counts[split]['failed'] += 1
    
    for split in ['prompt', 'test', 'validation', 'train', 'unknown']:
        if split in split_counts:
            s = split_counts[split]
            total_slots = s['total'] * MAX_DUPLICATES
            rate = (s['success'] / total_slots * 100) if total_slots > 0 else 0
            report_lines.append(f"{split}: {s['total']} tasks, {s['success']}/{total_slots} duplicates ({rate:.1f}%)")
    report_lines.append("")
    
    # Incomplete tasks
    report_lines.append("=" * 80)
    report_lines.append("INCOMPLETE TASKS (missing duplicates)")
    report_lines.append("=" * 80)
    
    incomplete = []
    for r in results:
        successes = sum(1 for i in range(1, MAX_DUPLICATES + 1) if r.get(f'python_{i}_status') == 'success')
        failures = sum(1 for i in range(1, MAX_DUPLICATES + 1) if r.get(f'python_{i}_status') == 'failed')
        if failures > 0:
            incomplete.append({
                'task_id': r['task_id'],
                'split': r.get('source_split', 'unknown'),
                'successes': successes,
                'failures': failures,
                'prompt': r.get('prompt', '')[:60]
            })
    
    report_lines.append(f"Total incomplete tasks: {len(incomplete)}")
    report_lines.append("")
    
    if incomplete:
        report_lines.append(f"{'Task ID':<10} {'Split':<12} {'Success':<10} {'Failed':<10} Prompt")
        report_lines.append("-" * 80)
        for t in sorted(incomplete, key=lambda x: x['successes'])[:30]:
            report_lines.append(f"{t['task_id']:<10} {t['split']:<12} {t['successes']}/5       {t['failures']}/5       {t['prompt']}...")
        if len(incomplete) > 30:
            report_lines.append(f"... and {len(incomplete) - 30} more incomplete tasks")
    report_lines.append("")
    
    # Data characteristics
    report_lines.append("=" * 80)
    report_lines.append("DATA CHARACTERISTICS")
    report_lines.append("=" * 80)
    
    # Code lengths
    orig_lengths = [len(r.get('code_python', '')) for r in results if r.get('code_python')]
    dup_lengths = []
    for r in results:
        for i in range(1, MAX_DUPLICATES + 1):
            code = r.get(f'python_{i}', '')
            if code and r.get(f'python_{i}_status') == 'success':
                dup_lengths.append(len(code))
    
    if orig_lengths:
        report_lines.append(f"Original code length: min={min(orig_lengths)}, max={max(orig_lengths)}, avg={sum(orig_lengths)/len(orig_lengths):.0f}")
    if dup_lengths:
        report_lines.append(f"Duplicate code length: min={min(dup_lengths)}, max={max(dup_lengths)}, avg={sum(dup_lengths)/len(dup_lengths):.0f}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Ensure archive directory exists
    archive_dir = OUTPUT_DIR / "archive"
    archive_dir.mkdir(exist_ok=True)
    
    # Write main report (overwritten each run)
    main_report_path = OUTPUT_DIR / "integrity_report.txt"
    with open(main_report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Write timestamped archive report
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"integrity_report_{timestamp_str}.txt"
    with open(archive_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    return main_report_path, archive_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='MBPP Python Semantic Duplicates Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate duplicates for 10 samples
  python generate_python_dupes.py --num-samples 10
  
  # Process all 427 samples with high concurrency
  python generate_python_dupes.py --num-samples 427 --concurrency 20
  
  # Resume a previous run
  python generate_python_dupes.py --num-samples 100 --resume
        """
    )
    
    parser.add_argument(
        '--num-samples', type=int, default=10,
        help='Number of samples to process (default: 10, max: 974 with --all-splits, 500 test-only)'
    )
    parser.add_argument(
        '--sanitized', action='store_true',
        help='Use sanitized dataset instead of full'
    )
    parser.add_argument(
        '--all-splits', action='store_true',
        help='Load from all splits (prompt+test+validation+train = 974 samples) instead of just test (500)'
    )
    parser.add_argument(
        '--concurrency', type=int, default=DEFAULT_CONCURRENCY,
        help=f'Number of concurrent workers (default: {DEFAULT_CONCURRENCY})'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from previous run (skip already-successful samples)'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force reprocess all samples (ignore resume)'
    )
    
    args = parser.parse_args()
    
    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)
    
    client = Anthropic(api_key=api_key)
    
    print("=" * 70)
    print("MBPP Python Semantic Duplicates Generator")
    print("=" * 70)
    print(f"Model: {MODEL}")
    if args.all_splits:
        max_samples = "427 sanitized" if args.sanitized else "974 full"
        print(f"Dataset: {'sanitized' if args.sanitized else 'full'}, ALL splits ({max_samples} max)")
    else:
        max_samples = "257 sanitized" if args.sanitized else "500 full"
        print(f"Dataset: {'sanitized' if args.sanitized else 'full'}, test split only ({max_samples} max)")
    print(f"Max duplicates per sample: {MAX_DUPLICATES}")
    print(f"Max generation attempts: {MAX_GENERATION_ATTEMPTS}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"Samples to process: {args.num_samples}")
    print(f"Concurrency: {args.concurrency} workers")
    print(f"Resume mode: {args.resume}")
    print("=" * 70)
    
    # Setup
    setup_output_dirs()
    
    # Load existing results for resume
    existing_results = load_existing_results() if args.resume and not args.force else {}
    if existing_results:
        print(f"Found {len(existing_results)} existing results")
    
    # Load samples
    use_full = not args.sanitized
    samples = load_mbpp_samples(args.num_samples, use_full=use_full, all_splits=args.all_splits)
    
    # Initialize stats
    stats = {
        'total_samples': len(samples),
        'duplicates_generated': 0,
        'duplicates_failed': 0,
        'duplicates_skipped': 0,
        'total_attempts': 0,
    }
    
    start_time = datetime.now()
    results = []
    completed = 0
    
    print(f"\nProcessing {len(samples)} samples with {args.concurrency} workers...")
    print("-" * 70)
    
    def process_with_stats(sample):
        """Wrapper to process sample and return with sample info."""
        return sample['task_id'], process_sample(
            client,
            sample,
            existing_results,
            verbose=True
        )
    
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_sample = {
            executor.submit(process_with_stats, sample): sample
            for sample in samples
        }
        
        results_dict = {}
        for future in as_completed(future_to_sample):
            sample = future_to_sample[future]
            task_id = sample['task_id']
            
            try:
                _, result = future.result()
                results_dict[task_id] = result
                
                # Update stats
                with _stats_lock:
                    completed += 1
                    
                    for i in range(1, MAX_DUPLICATES + 1):
                        status = result.get(f'python_{i}_status', '')
                        attempts = result.get(f'python_{i}_attempts', 0)
                        
                        if status == 'success':
                            stats['duplicates_generated'] += 1
                        elif status == 'failed':
                            stats['duplicates_failed'] += 1
                        elif status == 'skipped':
                            stats['duplicates_skipped'] += 1
                        
                        stats['total_attempts'] += attempts
                    
                    if completed % 5 == 0 or completed == len(samples):
                        elapsed = (datetime.now() - start_time).total_seconds()
                        rate = completed / elapsed if elapsed > 0 else 0
                        print(f"  Progress: {completed}/{len(samples)} samples ({rate:.2f}/s)")
                        
            except Exception as e:
                thread_print(f"  [Task {task_id}] [FATAL] Fatal exception: {e}")
                results_dict[task_id] = {
                    'task_id': task_id,
                    'prompt': sample['prompt'],
                    'code_python': sample['code'],
                    'test_list': sample['test_list'],
                    'error': str(e)
                }
    
    # Sort results by task_id
    results = [results_dict[s['task_id']] for s in samples if s['task_id'] in results_dict]
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Save master JSON
    metadata = {
        'run_timestamp': start_time.isoformat(),
        'duration_seconds': duration,
        'num_samples': args.num_samples,
        'max_duplicates': MAX_DUPLICATES,
        'max_generation_attempts': MAX_GENERATION_ATTEMPTS,
        'similarity_threshold': SIMILARITY_THRESHOLD,
        'model': MODEL,
        'concurrency': args.concurrency,
        'stats': stats
    }
    
    master_path, master_archive = save_master_json(results, metadata)
    print(f"\n[OK] Master JSON saved to: {master_path}")
    print(f"[OK] Archived to: {master_archive}")
    
    # Save CSV
    csv_path, csv_archive = save_csv(results)
    print(f"[OK] CSV saved to: {csv_path}")
    print(f"[OK] Archived to: {csv_archive}")
    
    # Generate integrity report
    report_path, report_archive = generate_integrity_report(results, stats, metadata)
    print(f"[OK] Integrity report saved to: {report_path}")
    print(f"[OK] Archived to: {report_archive}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Duration: {duration:.1f}s ({len(samples)/duration:.2f} samples/s)")
    print(f"Concurrency: {args.concurrency} workers")
    print(f"\nDuplicates:")
    print(f"  Generated (success): {stats['duplicates_generated']}")
    print(f"  Failed: {stats['duplicates_failed']}")
    print(f"  Skipped (resume): {stats['duplicates_skipped']}")
    print(f"  Total attempts: {stats['total_attempts']}")
    
    if stats['duplicates_generated'] + stats['duplicates_failed'] > 0:
        success_rate = stats['duplicates_generated'] / (stats['duplicates_generated'] + stats['duplicates_failed']) * 100
        print(f"  Success rate: {success_rate:.1f}%")
    
    print("\n" + "=" * 70)
    print("Output files:")
    print(f"  - {master_path}")
    print(f"  - {csv_path}")
    print(f"  - {report_path}")
    print(f"  - {INDIVIDUAL_DIR}/ (individual task JSONs)")
    print(f"  - output/archive/ (timestamped backups)")


if __name__ == "__main__":
    main()

