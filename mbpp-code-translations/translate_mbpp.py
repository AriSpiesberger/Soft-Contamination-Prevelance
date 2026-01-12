#!/usr/bin/env python3
"""
MBPP Multi-Language Translation Pipeline

This script:
1. Fetches samples from the MBPP dataset
2. Uses Claude Opus 4.5 to translate Python solutions to multiple languages
3. Validates translations by running tests with language-specific runtimes
4. Uses Claude Haiku for LLM validation of failed tests
5. Saves results as individual JSONs, master JSON, and CSV
6. Supports resume from previous runs
7. Supports concurrent processing for faster execution
"""

import csv
import json
import os
import re
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

# Import modular components
from translators import TRANSLATORS, ALL_LANGUAGES
from translators.base import TranslationResult
from validators import VALIDATORS
from validators.base import ValidationResult


# Configuration
DEFAULT_MAX_TRANSLATION_ATTEMPTS = 5  # Default attempts with error feedback
MAX_TRANSLATION_ATTEMPTS = DEFAULT_MAX_TRANSLATION_ATTEMPTS  # Will be overridden by CLI
API_RETRY_DELAY = 2  # seconds between API retries
DEFAULT_CONCURRENCY = 10
OUTPUT_DIR = Path(__file__).parent / "output"

# Output structure:
# output/
# ├── master/                    # Combined outputs
# │   ├── master_results.json
# │   └── master_translations.csv
# ├── {language}/                # Per-language outputs
# │   ├── individual/            # Per-task JSONs
# │   │   └── task_{id}.json
# │   ├── {language}_results.json
# │   └── {language}_translations.csv

# Thread-safe locks
_file_lock = threading.Lock()
_print_lock = threading.Lock()
_stats_lock = threading.Lock()


def setup_output_dirs(languages: list[str] = None):
    """Create output directories with per-language structure."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Create master output directory
    (OUTPUT_DIR / "master").mkdir(exist_ok=True)
    
    # Create per-language directories
    if languages:
        for lang in languages:
            lang_dir = OUTPUT_DIR / lang
            lang_dir.mkdir(exist_ok=True)
            (lang_dir / "individual").mkdir(exist_ok=True)


def get_failed_task_ids(languages: list[str]) -> set[int]:
    """Get task IDs that failed for any of the specified languages."""
    failed_ids = set()
    
    for lang in languages:
        lang_dir = OUTPUT_DIR / lang / "individual"
        if not lang_dir.exists():
            continue
            
        for json_file in lang_dir.glob("task_*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    task_id = data.get('task_id')
                    status = data.get(f'code_{lang}_status', '')
                    if task_id and status == 'failed':
                        failed_ids.add(task_id)
            except (json.JSONDecodeError, IOError):
                pass
    
    return failed_ids


def load_existing_results(languages: list[str] = None) -> dict:
    """
    Load existing results for resume capability.
    Merges results from per-language directories.
    """
    results = {}
    
    # Load from master JSON first
    master_json = OUTPUT_DIR / "master" / "master_results.json"
    if master_json.exists():
        with open(master_json, 'r') as f:
            data = json.load(f)
            for r in data.get('results', []):
                results[r['task_id']] = r
    
    # Also check legacy location
    legacy_master = OUTPUT_DIR / "master_results.json"
    if legacy_master.exists() and not master_json.exists():
        with open(legacy_master, 'r') as f:
            data = json.load(f)
            for r in data.get('results', []):
                results[r['task_id']] = r
    
    # Load from per-language directories (takes precedence)
    if languages:
        for lang in languages:
            lang_dir = OUTPUT_DIR / lang / "individual"
            if lang_dir.exists():
                for json_file in lang_dir.glob("task_*.json"):
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                            task_id = data.get('task_id')
                            if task_id and data.get(f'code_{lang}_status') == 'success':
                                # Merge language-specific results
                                if task_id not in results:
                                    results[task_id] = data
                                else:
                                    # Update only the language-specific fields
                                    for key in data:
                                        if key.startswith(f'code_{lang}') or key.startswith(f'test_{lang}'):
                                            results[task_id][key] = data[key]
                    except (json.JSONDecodeError, IOError):
                        pass
    
    return results


def load_mbpp_samples(num_samples: int = 10, use_sanitized: bool = True, all_splits: bool = False) -> list[dict]:
    """
    Load samples from the MBPP dataset with proper source tracking.
    
    Args:
        num_samples: Number of samples to load
        use_sanitized: If True, use 'sanitized' config. If False, use 'full' config.
        all_splits: If True, load from all splits. If False, only test split.
    
    Sanitized config totals:
        - prompt: 10 samples
        - test: 257 samples
        - validation: 90 samples  
        - train: 70 samples
        - TOTAL: 427 samples
    """
    config = "sanitized" if use_sanitized else "full"
    
    if all_splits:
        splits_to_load = ["prompt", "test", "validation", "train"]
        print(f"Loading up to {num_samples} samples from ALL splits (config={config})...")
    else:
        splits_to_load = ["test"]
        print(f"Loading up to {num_samples} samples from test split (config={config})...")
    
    try:
        dataset = load_dataset("google-research-datasets/mbpp", config)
    except Exception:
        try:
            dataset = load_dataset("Muennighoff/mbpp", config, trust_remote_code=True)
        except Exception:
            dataset = load_dataset("Muennighoff/mbpp", config)
    
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
            sample_dict['source_split'] = split_name  # Track which split
            sample_dict['source_config'] = config     # Track which config
            samples.append(sample_dict)
        if len(samples) >= num_samples:
            break
    
    print(f"Loaded {len(samples)} samples total")
    return samples


def extract_function_name(python_code: str) -> str:
    """Extract the main function name from Python code."""
    match = re.search(r'def\s+(\w+)\s*\(', python_code)
    return match.group(1) if match else "unknown"


def thread_print(*args, **kwargs):
    """Thread-safe print function."""
    with _print_lock:
        print(*args, **kwargs)


def analyze_error_with_sonnet(
    client: Anthropic,
    python_code: str,
    translated_code: str,
    language: str,
    error_message: str,
    test_list: list[str]
) -> str:
    """
    Use Sonnet to analyze why the translation failed and provide detailed feedback.
    """
    prompt = f"""You are a code debugging expert. A Python function was translated to {language}, but the translation is producing wrong results.

ORIGINAL PYTHON CODE:
```python
{python_code}
```

TRANSLATED {language.upper()} CODE:
```
{translated_code}
```

TESTS THAT FAILED:
{chr(10).join(test_list[:3])}

ERROR/FAILURE MESSAGE:
{error_message}

Analyze why the {language} translation is wrong and provide specific, actionable feedback for fixing it. Focus on:
1. Logic errors in the translation
2. Language-specific issues (type handling, syntax, built-in functions)
3. Edge cases that may be handled differently
4. The exact fix needed

Be concise but specific. Your feedback will be used to improve the translation."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            timeout=120.0,  # 120 second timeout for error analysis
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        # Fallback to raw error if Sonnet fails
        return f"Error analysis failed: {e}\n\nRaw error: {error_message}"


def translate_with_retry(
    translator,
    validator,
    sample: dict,
    func_name: str,
    task_description: str,
    client: Anthropic = None,
    max_attempts: int = None  # Don't use global as default - it's evaluated at definition time!
) -> TranslationResult:
    """
    Translate code with intelligent retry mechanism.
    Uses Sonnet to analyze errors and provide feedback for Opus retries.
    
    KEY IMPROVEMENT: Now passes ALL previous attempts with their errors to the model,
    not just the last one. This gives the model full context of what has been tried.
    """
    # Use global if not specified (evaluated at runtime, not definition time)
    if max_attempts is None:
        max_attempts = MAX_TRANSLATION_ATTEMPTS
    
    # Track ALL previous attempts as list of {code, error} dicts
    attempt_history = []
    error_history = []  # Keep for result
    last_code = None
    last_result = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            # Translate (with FULL error history if retry)
            code = translator.translate_with_api_retries(
                task_description,
                sample['code'],
                sample['test_list'],
                attempt_history=attempt_history if attempt_history else None
            )
            
            last_code = code
            
            # Validate
            result = validator.validate(
                code,
                sample['test_list'],
                func_name,
                task_description
            )
            last_result = result
            
            if result.all_passed:
                return TranslationResult(
                    code=code,
                    status='success',
                    attempts=attempt,
                    error_history=error_history,
                    test_results=result.to_dict()
                )
            
            # Get intelligent error analysis from Sonnet (if client available)
            if client:
                sonnet_analysis = analyze_error_with_sonnet(
                    client,
                    sample['code'],
                    code,
                    translator.LANGUAGE_NAME,
                    result.error_message,
                    sample['test_list']
                )
                error_text = f"ANALYSIS FROM CODE REVIEWER:\n{sonnet_analysis}\n\nRAW ERROR:\n{result.error_message}"
            else:
                error_text = result.error_message
            
            # Add this attempt to history (for next attempt to see ALL previous tries)
            attempt_history.append({
                'code': code,
                'error': error_text
            })
            error_history.append(error_text)
            
        except Exception as e:
            error_text = str(e)
            error_history.append(error_text)
            # Add even failed API calls to history
            if last_code:
                attempt_history.append({
                    'code': last_code,
                    'error': error_text
                })
    
    # All attempts failed - return last result
    return TranslationResult(
        code=last_code or "",
        status='failed',
        attempts=max_attempts,
        error_history=error_history,
        test_results=last_result.to_dict() if last_result else {}
    )


def process_sample(
    client: Anthropic,
    sample: dict,
    languages: list[str],
    existing_results: dict,
    skip_validation: bool = False,
    verbose: bool = True,
    capture_errors: bool = False
) -> dict:
    """Process a single sample for all specified languages."""
    task_id = sample['task_id']
    func_name = extract_function_name(sample['code'])
    task_description = sample['prompt']
    
    # Check existing results
    existing = existing_results.get(task_id, {})
    
    result = {
        'task_id': task_id,
        'source_split': sample.get('source_split', 'unknown'),
        'source_config': sample.get('source_config', 'sanitized'),
        'prompt': task_description,
        'python_code': sample['code'],
        'test_list': sample['test_list'],
    }
    
    # Process code translations
    for lang in languages:
        code_col = f'code_{lang}'
        status_col = f'code_{lang}_status'
        attempts_col = f'code_{lang}_attempts'
        passed_col = f'code_{lang}_tests_passed'
        failed_col = f'code_{lang}_tests_failed'
        
        # Check if already successfully processed
        if code_col in existing:
            existing_status = existing.get(status_col, '')
            if existing_status == 'success':
                result[code_col] = existing[code_col]
                result[status_col] = existing[status_col]
                result[attempts_col] = existing.get(attempts_col, 1)
                result[passed_col] = existing.get(passed_col, 0)
                result[failed_col] = existing.get(failed_col, 0)
                if verbose:
                    thread_print(f"  [Task {task_id}] [SKIP] {lang} skipped (already passed)")
                continue
        
        # Get translator and validator
        translator_cls = TRANSLATORS.get(lang)
        validator_cls = VALIDATORS.get(lang)
        
        if not translator_cls or not validator_cls:
            result[code_col] = None
            result[status_col] = 'unsupported'
            result[attempts_col] = 0
            result[passed_col] = 0
            result[failed_col] = 0
            continue
        
        translator = translator_cls(client, verbose=verbose)
        validator = validator_cls(client, verbose=verbose)
        
        # Check runtime availability
        if not skip_validation and not validator.check_runtime_available():
            result[code_col] = None
            result[status_col] = 'runtime_unavailable'
            result[attempts_col] = 0
            result[passed_col] = 0
            result[failed_col] = 0
            if verbose:
                thread_print(f"  [Task {task_id}] [WARN] {lang} runtime not available")
            continue
        
        try:
            if skip_validation:
                # Just translate without validation
                code = translator.translate_with_api_retries(
                    task_description,
                    sample['code'],
                    sample['test_list']
                )
                result[code_col] = code
                result[status_col] = 'unvalidated'
                result[attempts_col] = 1
                result[passed_col] = 0
                result[failed_col] = 0
            else:
                # Translate with retry and validation
                trans_result = translate_with_retry(
                    translator,
                    validator,
                    sample,
                    func_name,
                    task_description,
                    client=client  # Pass client for Sonnet error analysis
                )
                
                result[code_col] = trans_result.code
                result[status_col] = trans_result.status
                result[attempts_col] = trans_result.attempts
                
                test_results = trans_result.test_results
                passed = test_results.get('passed', 0) + test_results.get('passed_llm_validated', 0)
                failed = test_results.get('failed', 0)
                
                result[passed_col] = passed
                result[failed_col] = failed
                
                # Save generated test cases (LLM-converted tests)
                test_outputs = test_results.get('outputs', [])
                if test_outputs:
                    # Store each generated test case
                    generated_tests = []
                    for i, test_output in enumerate(test_outputs):
                        if 'generated_test_code' in test_output:
                            test_entry = {
                                'python_test': test_output.get('python_test', ''),
                                'generated_test': test_output.get('generated_test_code', ''),
                                'passed': test_output.get('passed', False),
                                'llm_validated': test_output.get('llm_validated', False)
                            }
                            # Include error traces when --capture-errors is enabled
                            if capture_errors and not test_output.get('passed', False):
                                test_entry['expected_value'] = test_output.get('expected_value', '')
                                test_entry['actual_value'] = test_output.get('actual_value', '')
                                if test_output.get('error'):
                                    test_entry['error'] = test_output.get('error', '')
                            generated_tests.append(test_entry)
                    if generated_tests:
                        result[f'tests_{lang}'] = generated_tests
                
                # Status indicator
                if trans_result.status == 'success':
                    status = "[OK]"
                elif passed > 0:
                    status = "[PARTIAL]"
                else:
                    status = "[FAIL]"
                
                if verbose:
                    thread_print(f"  [Task {task_id}] {status} {lang}: {passed}/{passed+failed} tests (attempt {trans_result.attempts})")
                    
        except Exception as e:
            result[code_col] = None
            result[status_col] = 'error'
            result[attempts_col] = MAX_TRANSLATION_ATTEMPTS
            result[passed_col] = 0
            result[failed_col] = 0
            if verbose:
                thread_print(f"  [Task {task_id}] [FAIL] {lang} failed: {e}")
    
    # Save individual result (pass languages for per-language success saving)
    save_individual_result(task_id, result, languages)
    
    return result


def save_individual_result(task_id: int, result: dict, languages: list[str] = None):
    """Save individual sample result to per-language folders (thread-safe)."""
    
    # Save to each language's individual folder
    if languages:
        for lang in languages:
            code = result.get(f'code_{lang}')
            if code is None:
                continue  # Skip if no code generated
                
            lang_file = OUTPUT_DIR / lang / "individual" / f"task_{task_id}.json"
            
            # Create a language-specific result with all relevant fields
            lang_result = {
                'task_id': task_id,
                'source_split': result.get('source_split', 'unknown'),
                'source_config': result.get('source_config', 'sanitized'),
                'prompt': result.get('prompt', ''),
                'python_code': result.get('python_code', ''),
                'test_list': result.get('test_list', []),
                # Code translation
                f'code_{lang}': code,
                f'code_{lang}_status': result.get(f'code_{lang}_status', ''),
                f'code_{lang}_attempts': result.get(f'code_{lang}_attempts', 1),
                f'code_{lang}_tests_passed': result.get(f'code_{lang}_tests_passed', 0),
                f'code_{lang}_tests_failed': result.get(f'code_{lang}_tests_failed', 0),
            }
            
            # Include generated test cases if present
            tests_key = f'tests_{lang}'
            if tests_key in result:
                lang_result[tests_key] = result[tests_key]
            
            # Include text translations if present
            for key in result:
                if key.startswith('text_'):
                    lang_result[key] = result[key]
            
            with _file_lock:
                with open(lang_file, 'w') as f:
                    json.dump(lang_result, f, indent=2)


def save_master_json(results: list, metadata: dict, languages: list[str] = None):
    """Save master JSON with all results and per-language JSONs."""
    master_data = {
        "metadata": metadata,
        "results": results
    }
    
    # Save combined master JSON
    master_dir = OUTPUT_DIR / "master"
    master_path = master_dir / "master_results.json"
    with open(master_path, 'w') as f:
        json.dump(master_data, f, indent=2)
    
    # Save timestamped archive
    archive_dir = master_dir / "archive"
    archive_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"master_results_{timestamp_str}.json"
    with open(archive_path, 'w') as f:
        json.dump(master_data, f, indent=2)
    
    # Save per-language master JSONs
    if languages:
        for lang in languages:
            lang_results = []
            for r in results:
                if r.get(f'code_{lang}'):
                    # Filter to only include this language's fields
                    lang_result = {
                        'task_id': r['task_id'],
                        'source_split': r.get('source_split', 'unknown'),
                        'source_config': r.get('source_config', 'sanitized'),
                        'prompt': r.get('prompt', ''),
                        'python_code': r.get('python_code', ''),
                        'test_list': r.get('test_list', []),
                        f'code_{lang}': r.get(f'code_{lang}'),
                        f'code_{lang}_status': r.get(f'code_{lang}_status', ''),
                        f'code_{lang}_attempts': r.get(f'code_{lang}_attempts', 0),
                        f'code_{lang}_tests_passed': r.get(f'code_{lang}_tests_passed', 0),
                        f'code_{lang}_tests_failed': r.get(f'code_{lang}_tests_failed', 0),
                    }
                    # Include generated tests
                    if f'tests_{lang}' in r:
                        lang_result[f'tests_{lang}'] = r[f'tests_{lang}']
                    # Include text translations
                    for key in r:
                        if key.startswith('text_'):
                            lang_result[key] = r[key]
                    lang_results.append(lang_result)
            
            lang_master_path = OUTPUT_DIR / lang / f"{lang}_results.json"
            lang_metadata = {**metadata, 'language': lang, 'total_results': len(lang_results)}
            with open(lang_master_path, 'w') as f:
                json.dump({"metadata": lang_metadata, "results": lang_results}, f, indent=2)
    
    return master_path, archive_path


def save_csv(results: list, samples_by_id: dict, languages: list[str]):
    """Save master CSV and per-language CSVs (latest + timestamped archive)."""
    
    # ===== MASTER CSV (all languages) =====
    master_dir = OUTPUT_DIR / "master"
    master_csv_path = master_dir / "master_translations.csv"
    
    # Prepare archive path
    archive_dir = master_dir / "archive"
    archive_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_csv_path = archive_dir / f"master_translations_{timestamp_str}.csv"
    
    # Build field names for master CSV
    master_fieldnames = [
        'task_id',
        'source_split',
        'source_config',
        'text',
        'code_python',
        'test_list',
    ]
    
    # Add code language columns
    for lang in languages:
        master_fieldnames.extend([
            f'code_{lang}',
            f'code_{lang}_status',
            f'code_{lang}_attempts',
            f'code_{lang}_tests_passed',
            f'code_{lang}_tests_failed',
        ])
    
    # Write master CSV to both latest and archive
    for out_path in [master_csv_path, archive_csv_path]:
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=master_fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            for r in results:
                task_id = r['task_id']
                original = samples_by_id.get(task_id, {})
                
                row = {
                    'task_id': task_id,
                    'source_split': r.get('source_split', original.get('source_split', '')),
                    'source_config': r.get('source_config', original.get('source_config', '')),
                    'text': r.get('prompt', original.get('prompt', '')),
                    'code_python': r.get('python_code', original.get('code', '')),
                    'test_list': json.dumps(r.get('test_list', original.get('test_list', []))),
                }
                
                # Add code translations
                for lang in languages:
                    row[f'code_{lang}'] = r.get(f'code_{lang}', '')
                    row[f'code_{lang}_status'] = r.get(f'code_{lang}_status', '')
                    row[f'code_{lang}_attempts'] = r.get(f'code_{lang}_attempts', 0)
                    row[f'code_{lang}_tests_passed'] = r.get(f'code_{lang}_tests_passed', 0)
                    row[f'code_{lang}_tests_failed'] = r.get(f'code_{lang}_tests_failed', 0)
                
                writer.writerow(row)
    
    # ===== PER-LANGUAGE CSVs =====
    for lang in languages:
        lang_csv_path = OUTPUT_DIR / lang / f"{lang}_translations.csv"
        
        lang_fieldnames = [
            'task_id',
            'source_split',
            'source_config',
            'text',
            'code_python',
            'test_list',
            f'code_{lang}',
            f'code_{lang}_status',
            f'code_{lang}_attempts',
            f'code_{lang}_tests_passed',
            f'code_{lang}_tests_failed',
            f'tests_{lang}',  # JSON-encoded generated tests
        ]
        
        with open(lang_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=lang_fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            for r in results:
                if not r.get(f'code_{lang}'):
                    continue  # Skip if no code for this language
                    
                task_id = r['task_id']
                original = samples_by_id.get(task_id, {})
                
                row = {
                    'task_id': task_id,
                    'source_split': r.get('source_split', original.get('source_split', '')),
                    'source_config': r.get('source_config', original.get('source_config', '')),
                    'text': r.get('prompt', original.get('prompt', '')),
                    'code_python': r.get('python_code', original.get('code', '')),
                    'test_list': json.dumps(r.get('test_list', original.get('test_list', []))),
                    f'code_{lang}': r.get(f'code_{lang}', ''),
                    f'code_{lang}_status': r.get(f'code_{lang}_status', ''),
                    f'code_{lang}_attempts': r.get(f'code_{lang}_attempts', 0),
                    f'code_{lang}_tests_passed': r.get(f'code_{lang}_tests_passed', 0),
                    f'code_{lang}_tests_failed': r.get(f'code_{lang}_tests_failed', 0),
                    f'tests_{lang}': json.dumps(r.get(f'tests_{lang}', [])),
                }
                
                writer.writerow(row)
    
    return master_csv_path, archive_csv_path


def generate_integrity_report(results: list, languages: list[str], stats: dict, metadata: dict):
    """Generate integrity report with statistics and characteristics."""
    report_lines = []
    timestamp = datetime.now()
    
    report_lines.append("=" * 80)
    report_lines.append("MBPP CODE TRANSLATIONS - INTEGRITY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {timestamp.isoformat()}")
    report_lines.append(f"Run timestamp: {metadata.get('run_timestamp', 'unknown')}")
    report_lines.append(f"Duration: {metadata.get('duration_seconds', 0):.1f}s")
    report_lines.append("")
    
    # Overall statistics
    report_lines.append("=" * 80)
    report_lines.append("OVERALL STATISTICS")
    report_lines.append("=" * 80)
    report_lines.append(f"Total tasks processed: {stats.get('total', len(results))}")
    report_lines.append(f"Code languages: {', '.join(languages)}")
    report_lines.append(f"Max translation attempts: {metadata.get('max_translation_attempts', MAX_TRANSLATION_ATTEMPTS)}")
    report_lines.append(f"Concurrency: {metadata.get('concurrency', 'unknown')}")
    report_lines.append("")
    
    # Models used
    report_lines.append("=" * 80)
    report_lines.append("MODELS USED")
    report_lines.append("=" * 80)
    report_lines.append(f"Translation model: {metadata.get('translation_model', 'unknown')}")
    report_lines.append(f"Test conversion model: {metadata.get('test_conversion_model', 'unknown')}")
    report_lines.append(f"Error analysis model: {metadata.get('error_analysis_model', 'unknown')}")
    report_lines.append(f"LLM validation model: {metadata.get('validation_model', 'unknown')}")
    report_lines.append("")
    
    # Per-language statistics
    report_lines.append("=" * 80)
    report_lines.append("PER-LANGUAGE CODE TRANSLATION STATISTICS")
    report_lines.append("=" * 80)
    
    total_success = 0
    total_failed = 0
    for lang in languages:
        lang_stats = stats.get('languages', {}).get(lang, {})
        success = lang_stats.get('success', 0)
        failed = lang_stats.get('failed', 0)
        skipped = lang_stats.get('skipped', 0)
        total = success + failed
        rate = (success / total * 100) if total > 0 else 0
        avg_attempts = lang_stats.get('avg_attempts', 0)
        report_lines.append(f"{lang:12} - Success: {success}/{total} ({rate:.1f}%), Skipped: {skipped}, Avg attempts: {avg_attempts:.2f}")
        total_success += success
        total_failed += failed
    
    total_all = total_success + total_failed
    overall_rate = (total_success / total_all * 100) if total_all > 0 else 0
    report_lines.append("")
    report_lines.append(f"Overall code translation: {total_success}/{total_all} ({overall_rate:.1f}%)")
    report_lines.append("")
    
    # By split statistics (if available)
    split_counts = {}
    for r in results:
        split = r.get('source_split', 'unknown')
        if split not in split_counts:
            split_counts[split] = {'total': 0, 'success': {}, 'failed': {}}
        split_counts[split]['total'] += 1
        for lang in languages:
            if lang not in split_counts[split]['success']:
                split_counts[split]['success'][lang] = 0
                split_counts[split]['failed'][lang] = 0
            status = r.get(f'code_{lang}_status', '')
            if status == 'success':
                split_counts[split]['success'][lang] += 1
            elif status == 'failed':
                split_counts[split]['failed'][lang] += 1
    
    if split_counts:
        report_lines.append("=" * 80)
        report_lines.append("BY SPLIT STATISTICS")
        report_lines.append("=" * 80)
        
        for split in ['prompt', 'test', 'validation', 'train', 'unknown']:
            if split in split_counts:
                s = split_counts[split]
                total_success_split = sum(s['success'].values())
                total_slots = s['total'] * len(languages)
                rate = (total_success_split / total_slots * 100) if total_slots > 0 else 0
                report_lines.append(f"{split}: {s['total']} tasks, {total_success_split}/{total_slots} translations ({rate:.1f}%)")
        report_lines.append("")
    
    # Incomplete tasks (tasks with at least one failed language)
    report_lines.append("=" * 80)
    report_lines.append("INCOMPLETE TASKS (at least one language failed)")
    report_lines.append("=" * 80)
    
    incomplete = []
    for r in results:
        failed_langs = []
        success_langs = []
        for lang in languages:
            status = r.get(f'code_{lang}_status', '')
            if status == 'success':
                success_langs.append(lang)
            elif status == 'failed':
                failed_langs.append(lang)
        
        if failed_langs:
            incomplete.append({
                'task_id': r['task_id'],
                'split': r.get('source_split', 'unknown'),
                'success_count': len(success_langs),
                'failed_count': len(failed_langs),
                'failed_langs': failed_langs,
                'prompt': r.get('prompt', '')[:50]
            })
    
    report_lines.append(f"Total incomplete tasks: {len(incomplete)}")
    report_lines.append("")
    
    if incomplete:
        report_lines.append(f"{'Task ID':<10} {'Split':<12} {'Success':<10} {'Failed Languages':<30} Prompt")
        report_lines.append("-" * 100)
        for t in sorted(incomplete, key=lambda x: -x['failed_count'])[:30]:  # Show top 30
            failed_str = ','.join(t['failed_langs'][:3])
            if len(t['failed_langs']) > 3:
                failed_str += f"... (+{len(t['failed_langs'])-3})"
            report_lines.append(f"{t['task_id']:<10} {t['split']:<12} {t['success_count']}/{len(languages):<8} {failed_str:<30} {t['prompt']}...")
        if len(incomplete) > 30:
            report_lines.append(f"... and {len(incomplete) - 30} more incomplete tasks")
    report_lines.append("")
    
    # Data characteristics
    report_lines.append("=" * 80)
    report_lines.append("DATA CHARACTERISTICS")
    report_lines.append("=" * 80)
    
    # Python code lengths
    python_lengths = [len(r.get('python_code', '')) for r in results if r.get('python_code')]
    if python_lengths:
        report_lines.append(f"Python code length: min={min(python_lengths)}, max={max(python_lengths)}, avg={sum(python_lengths)/len(python_lengths):.0f}")
    
    # Per-language code lengths
    for lang in languages:
        lang_lengths = []
        for r in results:
            code = r.get(f'code_{lang}', '')
            if code and r.get(f'code_{lang}_status') == 'success':
                lang_lengths.append(len(code))
        if lang_lengths:
            report_lines.append(f"{lang} code length: min={min(lang_lengths)}, max={max(lang_lengths)}, avg={sum(lang_lengths)/len(lang_lengths):.0f}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Ensure reports directory exists (reports live INSIDE output/reports/, not at output/ root)
    reports_dir = OUTPUT_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Write main report to reports/ folder (latest version)
    main_report_path = reports_dir / "integrity_report.txt"
    with open(main_report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Write timestamped archive report
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    archive_path = reports_dir / f"integrity_report_{timestamp_str}.txt"
    with open(archive_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    return main_report_path, archive_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='MBPP Multi-Language Translation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate to JavaScript only (default)
  python translate_mbpp.py --num-samples 10
  
  # Translate to multiple languages
  python translate_mbpp.py --language javascript,typescript,rust
  
  # Translate to all languages
  python translate_mbpp.py --all
  
  # Process all 427 samples with high concurrency
  python translate_mbpp.py --all --num-samples 427 --concurrency 40
  
  # Skip validation (faster, just translate)
  python translate_mbpp.py --all --skip-validation
  
  # Retry ONLY failed tasks with more attempts (10 instead of 5)
  python translate_mbpp.py --language rust --retry-failed --max-attempts 10
  
  # Force reprocess all with more attempts
  python translate_mbpp.py --all --force --max-attempts 8
        """
    )
    
    parser.add_argument(
        '--num-samples', type=int, default=10,
        help='Number of samples to process (default: 10, max: 427 sanitized)'
    )
    parser.add_argument(
        '--all-splits', action='store_true',
        help='Load from all MBPP splits (prompt+test+validation+train) instead of just test'
    )
    parser.add_argument(
        '--language', type=str, default='javascript',
        help=f'Comma-separated list of languages: {",".join(ALL_LANGUAGES)}'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Process all supported programming languages'
    )
    parser.add_argument(
        '--skip-validation', action='store_true',
        help='Skip test validation (faster, translation only)'
    )
    parser.add_argument(
        '--concurrency', type=int, default=DEFAULT_CONCURRENCY,
        help=f'Number of concurrent workers (default: {DEFAULT_CONCURRENCY})'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='(Deprecated) Now always resumes - use --force to reprocess'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force reprocess all samples (ignore existing successful translations)'
    )
    parser.add_argument(
        '--retry-failed', action='store_true',
        help='Only retry tasks that previously failed (skip successful ones)'
    )
    parser.add_argument(
        '--max-attempts', type=int, default=DEFAULT_MAX_TRANSLATION_ATTEMPTS,
        help=f'Maximum translation attempts per task (default: {DEFAULT_MAX_TRANSLATION_ATTEMPTS})'
    )
    parser.add_argument(
        '--capture-errors', action='store_true',
        help='Save error traces (expected/actual values) for failed tests to help debug'
    )
    
    args = parser.parse_args()
    
    # Update global max attempts from CLI
    global MAX_TRANSLATION_ATTEMPTS
    MAX_TRANSLATION_ATTEMPTS = args.max_attempts
    
    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)
    
    client = Anthropic(api_key=api_key)
    
    # Determine languages to process
    if args.all:
        languages = ALL_LANGUAGES
    else:
        languages = [l.strip().lower() for l in args.language.split(',')]
        for lang in languages:
            if lang not in TRANSLATORS:
                print(f"ERROR: Unknown language '{lang}'")
                print(f"Available languages: {', '.join(ALL_LANGUAGES)}")
                sys.exit(1)
    
    print("=" * 70)
    print("MBPP Code Translations Pipeline")
    print("=" * 70)
    print(f"Translation model: Claude Opus 4.5")
    print(f"Test conversion model: Claude Opus 4.5")
    print(f"Error analysis model: Claude Sonnet 4.5")
    print(f"Validation model: Claude Haiku 4.5")
    print(f"Max translation attempts: {MAX_TRANSLATION_ATTEMPTS}")
    if args.retry_failed:
        print(f"Mode: RETRY FAILED ONLY (will load all samples to find failed tasks)")
    else:
        print(f"Samples to process: {args.num_samples}")
        print(f"Dataset splits: {'All (prompt+test+validation+train)' if args.all_splits else 'Test only'}")
    print(f"Programming languages: {', '.join(languages)}")
    print(f"Concurrency: {args.concurrency} workers")
    print(f"Skip validation: {args.skip_validation}")
    print(f"Force reprocess: {args.force}")
    print(f"Capture errors: {args.capture_errors}")
    print("=" * 70)
    
    # Setup output directories (per-language folders)
    setup_output_dirs(languages)
    
    # Load existing results for resume (always load to avoid re-processing successful ones)
    existing_results = {}
    failed_task_ids = set()
    
    if args.retry_failed:
        # Get failed task IDs to filter
        failed_task_ids = get_failed_task_ids(languages)
        print(f"Found {len(failed_task_ids)} failed tasks to retry: {sorted(failed_task_ids)[:20]}{'...' if len(failed_task_ids) > 20 else ''}")
        # Don't load existing results so we force reprocess these specific tasks
    elif not args.force:
        existing_results = load_existing_results(languages)
        if existing_results:
            print(f"Found {len(existing_results)} existing results (will skip successful translations)")
    
    # Load samples with proper source tracking
    # IMPORTANT: When --retry-failed, we must load ALL samples from ALL splits
    # because failed tasks could be anywhere in the dataset (not just first N samples)
    if args.retry_failed and failed_task_ids:
        print(f"Loading ALL samples from ALL splits to find failed tasks...")
        samples = load_mbpp_samples(500, use_sanitized=True, all_splits=True)  # 500 > 427 to get all
        samples = [s for s in samples if s['task_id'] in failed_task_ids]
        print(f"Found {len(samples)} failed samples to retry")
        
        if not samples:
            print(f"WARNING: Failed task IDs {sorted(failed_task_ids)} not found in dataset!")
            print("This could mean the tasks were from a different dataset config.")
    else:
        samples = load_mbpp_samples(args.num_samples, use_sanitized=True, all_splits=args.all_splits)
    samples_by_id = {s['task_id']: s for s in samples}
    
    # Initialize stats per language
    stats = {
        'total': len(samples),
        'languages': {lang: {
            'translated': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'llm_validated': 0,
            'total_attempts': 0
        } for lang in languages}
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
            languages,
            existing_results,
            skip_validation=args.skip_validation,
            verbose=True,
            capture_errors=args.capture_errors
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
                    
                    for lang in languages:
                        status = result.get(f'code_{lang}_status', '')
                        attempts = result.get(f'code_{lang}_attempts', 0)
                        code = result.get(f'code_{lang}')
                        
                        if status == 'success':
                            stats['languages'][lang]['success'] += 1
                            stats['languages'][lang]['translated'] += 1
                        elif status == 'unvalidated' and code:
                            stats['languages'][lang]['success'] += 1  # Count as success when validation skipped
                            stats['languages'][lang]['translated'] += 1
                        elif status == 'failed':
                            stats['languages'][lang]['failed'] += 1
                            stats['languages'][lang]['translated'] += 1
                        elif status in ('skipped', 'runtime_unavailable'):
                            stats['languages'][lang]['skipped'] += 1
                        
                        stats['languages'][lang]['total_attempts'] += attempts
                    
                    if completed % 10 == 0 or completed == len(samples):
                        elapsed = (datetime.now() - start_time).total_seconds()
                        rate = completed / elapsed if elapsed > 0 else 0
                        print(f"  Progress: {completed}/{len(samples)} ({rate:.1f}/s)")
                        
            except Exception as e:
                thread_print(f"  [Task {task_id}] [FATAL] Fatal exception: {e}")
                results_dict[task_id] = {
                    'task_id': task_id,
                    'prompt': sample['prompt'],
                    'python_code': sample['code'],
                    'test_list': sample['test_list'],
                    'error': str(e)
                }
    
    # Sort results by task_id
    results = [results_dict[s['task_id']] for s in samples if s['task_id'] in results_dict]
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Calculate summary stats
    for lang in languages:
        lang_stats = stats['languages'][lang]
        total_translated = lang_stats['translated']
        if total_translated > 0:
            lang_stats['avg_attempts'] = lang_stats['total_attempts'] / total_translated
        else:
            lang_stats['avg_attempts'] = 0
    
    # Save master JSON
    metadata = {
        'run_timestamp': start_time.isoformat(),
        'duration_seconds': duration,
        'num_samples': args.num_samples,
        'languages_processed': languages,
        'concurrency': args.concurrency,
        'max_translation_attempts': MAX_TRANSLATION_ATTEMPTS,
        'translation_model': 'claude-opus-4-5',
        'test_conversion_model': 'claude-opus-4-5',
        'error_analysis_model': 'claude-sonnet-4-5',
        'validation_model': 'claude-haiku-4-5',
        'skip_validation': args.skip_validation,
        'stats': stats
    }
    
    master_path, master_archive = save_master_json(results, metadata, languages)
    print(f"\n[OK] Master JSON saved to: {master_path}")
    print(f"[OK] Archived to: {master_archive}")
    
    # Save CSV
    csv_path, csv_archive = save_csv(results, samples_by_id, languages)
    print(f"[OK] Master CSV saved to: {csv_path}")
    print(f"[OK] Archived to: {csv_archive}")
    
    # Generate integrity report
    main_report, archive_report = generate_integrity_report(
        results, languages, stats, metadata
    )
    print(f"[OK] Integrity report saved to: {main_report}")
    print(f"[OK] Archived report saved to: {archive_report}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total samples: {stats['total']}")
    print(f"Duration: {duration:.1f}s ({len(samples)/duration:.2f} samples/s)")
    print(f"Concurrency: {args.concurrency} workers")
    
    print("\nPer-language results:")
    for lang in languages:
        lang_stats = stats['languages'][lang]
        success = lang_stats['success']
        failed = lang_stats['failed']
        total = success + failed
        rate = (success / total * 100) if total > 0 else 0
        avg_attempts = lang_stats['avg_attempts']
        print(f"  {lang:12} - Success: {success}/{total} ({rate:.1f}%), Avg attempts: {avg_attempts:.2f}")
    
    print("\n" + "=" * 70)
    print("Output structure:")
    print(f"  output/")
    print(f"  ├── master/")
    print(f"  │   ├── master_results.json")
    print(f"  │   ├── master_translations.csv")
    print(f"  │   └── archive/ (timestamped backups)")
    print(f"  ├── reports/")
    print(f"  │   └── integrity_report_*.txt (timestamped archives)")
    print(f"  ├── integrity_report.txt (latest)")
    for lang in languages:
        print(f"  ├── {lang}/")
        print(f"  │   ├── individual/task_*.json")
        print(f"  │   ├── {lang}_results.json")
        print(f"  │   └── {lang}_translations.csv")


if __name__ == "__main__":
    main()
