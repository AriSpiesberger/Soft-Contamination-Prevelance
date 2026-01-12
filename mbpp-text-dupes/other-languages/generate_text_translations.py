#!/usr/bin/env python3
"""
MBPP Text Translations Generator - Other Languages

Translates MBPP task descriptions to multiple languages:
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Russian (ru)
- Chinese (zh)

Uses Claude Sonnet for high-quality translations.
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
MODEL = "claude-sonnet-4-5"
MAX_RETRIES = 5
RETRY_DELAY = 2.0
DEFAULT_CONCURRENCY = 20
OUTPUT_DIR = Path(__file__).parent / "output"
INDIVIDUAL_DIR = OUTPUT_DIR / "individual"

# Supported languages (non-English)
TEXT_LANGUAGES = {
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'ru': 'Russian',
    'zh': 'Chinese',
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
    """Load samples from the MBPP dataset."""
    if all_splits:
        splits_to_load = ["prompt", "test", "validation", "train"]
        print(f"Loading up to {num_samples} samples from ALL splits...")
    else:
        splits_to_load = ["test"]
        print(f"Loading up to {num_samples} samples from test split...")
    
    dataset = load_dataset("google-research-datasets/mbpp", "full")
    
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
            sample_dict['source_config'] = 'full'
            samples.append(sample_dict)
        if len(samples) >= num_samples:
            break
    
    print(f"Loaded {len(samples)} samples total")
    return samples


def translate_text(client: Anthropic, text: str, target_lang: str, target_name: str) -> tuple[Optional[str], int]:
    """Translate text to target language. Returns (translated_text, attempts)."""
    prompt = f"""Translate this programming task description to {target_name}. Output ONLY the translation, nothing else.

English: {text}

{target_name}:"""

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=512,
                timeout=60.0,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip(), attempt
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    
    return None, MAX_RETRIES


def process_sample(
    client: Anthropic,
    sample: dict,
    existing_results: dict,
    languages: list[str],
    verbose: bool = True
) -> dict:
    """Process a single sample to generate translations."""
    task_id = sample['task_id']
    text = sample['prompt']
    
    # Check existing results
    existing = existing_results.get(task_id, {})
    
    result = {
        'task_id': task_id,
        'source_split': sample.get('source_split', 'unknown'),
        'source_config': sample.get('source_config', 'full'),
        'text': text,
    }
    
    for lang_code in languages:
        text_col = f'text_{lang_code}'
        status_col = f'text_{lang_code}_status'
        attempts_col = f'text_{lang_code}_attempts'
        
        # Check if already successfully processed
        if text_col in existing and existing.get(status_col) == 'success':
            result[text_col] = existing[text_col]
            result[status_col] = existing[status_col]
            result[attempts_col] = existing.get(attempts_col, 1)
            if verbose:
                thread_print(f"  [Task {task_id}] [SKIP] text_{lang_code} (already exists)")
            continue
        
        if verbose:
            thread_print(f"  [Task {task_id}] Translating to {TEXT_LANGUAGES[lang_code]}...")
        
        translated, attempts = translate_text(
            client, text, lang_code, TEXT_LANGUAGES[lang_code]
        )
        
        if translated:
            result[text_col] = translated
            result[status_col] = 'success'
            result[attempts_col] = attempts
            if verbose:
                thread_print(f"  [Task {task_id}] [OK] text_{lang_code}")
        else:
            result[text_col] = None
            result[status_col] = 'failed'
            result[attempts_col] = attempts
            if verbose:
                thread_print(f"  [Task {task_id}] [FAIL] text_{lang_code}")
    
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


def save_csv(results: list, languages: list[str]):
    """Save CSV with all translations."""
    filepath = OUTPUT_DIR / "text_translations.csv"
    
    # Prepare archive path
    archive_dir = OUTPUT_DIR / "archive"
    archive_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"text_translations_{timestamp_str}.csv"
    
    # Build field names
    fieldnames = ['task_id', 'source_split', 'source_config', 'text']
    for lang in languages:
        fieldnames.extend([f'text_{lang}', f'text_{lang}_status', f'text_{lang}_attempts'])
    
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
                }
                for lang in languages:
                    row[f'text_{lang}'] = r.get(f'text_{lang}', '')
                    row[f'text_{lang}_status'] = r.get(f'text_{lang}_status', '')
                    row[f'text_{lang}_attempts'] = r.get(f'text_{lang}_attempts', 0)
                writer.writerow(row)
    
    return filepath, archive_path


def generate_integrity_report(results: list, languages: list[str], stats: dict, metadata: dict):
    """Generate integrity report."""
    report_lines = []
    timestamp = datetime.now()
    
    report_lines.append("=" * 80)
    report_lines.append("MBPP TEXT TRANSLATIONS (OTHER LANGUAGES) - INTEGRITY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {timestamp.isoformat()}")
    report_lines.append(f"Run timestamp: {metadata.get('run_timestamp', 'unknown')}")
    report_lines.append(f"Duration: {metadata.get('duration_seconds', 0):.1f}s")
    report_lines.append(f"Model: {MODEL}")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("OVERALL STATISTICS")
    report_lines.append("=" * 80)
    report_lines.append(f"Total tasks: {stats['total_samples']}")
    report_lines.append(f"Languages: {', '.join(languages)}")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("PER-LANGUAGE STATISTICS")
    report_lines.append("=" * 80)
    
    for lang in languages:
        lang_stats = stats['languages'].get(lang, {})
        success = lang_stats.get('success', 0)
        failed = lang_stats.get('failed', 0)
        total = success + failed
        rate = (success / total * 100) if total > 0 else 0
        report_lines.append(f"{TEXT_LANGUAGES[lang]:12} ({lang}): {success}/{total} ({rate:.1f}%)")
    
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
        description='MBPP Text Translations - Other Languages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate 10 samples to all languages
  python generate_text_translations.py --num-samples 10
  
  # Translate to specific languages only
  python generate_text_translations.py --language es,fr,de
  
  # Resume a previous run
  python generate_text_translations.py --num-samples 100 --resume
        """
    )
    
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--all-splits', action='store_true', help='Load from all splits')
    parser.add_argument('--language', type=str, default=None,
                       help=f'Comma-separated languages: {",".join(TEXT_LANGUAGES.keys())}')
    parser.add_argument('--all', action='store_true', help='Process all languages')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY, help='Concurrent workers')
    parser.add_argument('--resume', action='store_true', help='Resume from previous run')
    parser.add_argument('--force', action='store_true', help='Force reprocess all')
    
    args = parser.parse_args()
    
    # Determine languages
    if args.all or args.language is None:
        languages = list(TEXT_LANGUAGES.keys())
    else:
        languages = [l.strip() for l in args.language.split(',')]
        for lang in languages:
            if lang not in TEXT_LANGUAGES:
                print(f"ERROR: Unknown language '{lang}'. Available: {', '.join(TEXT_LANGUAGES.keys())}")
                sys.exit(1)
    
    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)
    
    client = Anthropic(api_key=api_key)
    
    print("=" * 70)
    print("MBPP Text Translations - Other Languages")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Languages: {', '.join(languages)}")
    print(f"Samples: {args.num_samples}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Resume: {args.resume}")
    print("=" * 70)
    
    setup_output_dirs()
    
    existing_results = load_existing_results() if args.resume and not args.force else {}
    if existing_results:
        print(f"Found {len(existing_results)} existing results")
    
    samples = load_mbpp_samples(args.num_samples, all_splits=args.all_splits)
    
    stats = {
        'total_samples': len(samples),
        'languages': {lang: {'success': 0, 'failed': 0, 'skipped': 0} for lang in languages}
    }
    
    start_time = datetime.now()
    results = []
    completed = 0
    
    print(f"\nProcessing {len(samples)} samples with {args.concurrency} workers...")
    
    def process_with_stats(sample):
        return sample['task_id'], process_sample(client, sample, existing_results, languages)
    
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_sample = {executor.submit(process_with_stats, s): s for s in samples}
        results_dict = {}
        
        for future in as_completed(future_to_sample):
            sample = future_to_sample[future]
            task_id = sample['task_id']
            
            try:
                _, result = future.result()
                results_dict[task_id] = result
                
                with _stats_lock:
                    completed += 1
                    for lang in languages:
                        status = result.get(f'text_{lang}_status', '')
                        if status == 'success':
                            stats['languages'][lang]['success'] += 1
                        elif status == 'failed':
                            stats['languages'][lang]['failed'] += 1
                    
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
        'languages': languages,
        'model': MODEL,
        'concurrency': args.concurrency,
        'stats': stats
    }
    
    master_path, master_archive = save_master_json(results, metadata)
    print(f"\n[OK] Master JSON: {master_path}")
    
    csv_path, csv_archive = save_csv(results, languages)
    print(f"[OK] CSV: {csv_path}")
    
    report_path, report_archive = generate_integrity_report(results, languages, stats, metadata)
    print(f"[OK] Report: {report_path}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for lang in languages:
        s = stats['languages'][lang]
        total = s['success'] + s['failed']
        rate = (s['success'] / total * 100) if total > 0 else 0
        print(f"  {TEXT_LANGUAGES[lang]:12} ({lang}): {s['success']}/{total} ({rate:.1f}%)")


if __name__ == "__main__":
    main()

