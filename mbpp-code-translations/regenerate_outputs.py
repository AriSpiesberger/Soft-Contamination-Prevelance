#!/usr/bin/env python3
"""
Regenerate CSV, master JSON, and create integrity report from individual JSON files.

This script:
1. Reads all individual task JSON files from each language folder
2. Regenerates master_results.json
3. Regenerates master_translations.csv
4. Regenerates per-language JSONs and CSVs
5. Creates an integrity_report.txt with statistics and characteristics
"""

import csv
import json
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"


def get_all_languages():
    """Detect all language folders in output directory."""
    languages = []
    for item in OUTPUT_DIR.iterdir():
        if item.is_dir() and item.name != "master" and (item / "individual").exists():
            languages.append(item.name)
    return sorted(languages)


def load_all_tasks(languages: list[str]):
    """Load all task JSON files from each language's individual directory."""
    tasks_by_id = {}
    
    for lang in languages:
        lang_individual_dir = OUTPUT_DIR / lang / "individual"
        if not lang_individual_dir.exists():
            continue
            
        json_files = sorted(lang_individual_dir.glob("task_*.json"), 
                          key=lambda x: int(x.stem.split("_")[1]))
        
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)
                task_id = data['task_id']
                
                if task_id not in tasks_by_id:
                    tasks_by_id[task_id] = {
                        'task_id': task_id,
                        'source_split': data.get('source_split', 'unknown'),
                        'source_config': data.get('source_config', 'sanitized'),
                        'prompt': data.get('prompt', ''),
                        'python_code': data.get('python_code', ''),
                        'test_list': data.get('test_list', []),
                    }
                
                # Merge language-specific fields
                for key, value in data.items():
                    if key.startswith(f'code_{lang}') or key.startswith(f'tests_{lang}') or key.startswith('text_'):
                        tasks_by_id[task_id][key] = value
    
    # Sort by task_id
    return [tasks_by_id[tid] for tid in sorted(tasks_by_id.keys())]


def save_master_json(results: list, languages: list[str]):
    """Save master JSON with all results."""
    # Calculate stats
    stats = {
        'total_samples': len(results),
        'languages': {}
    }
    
    for lang in languages:
        lang_stats = {'success': 0, 'failed': 0, 'total_attempts': 0}
        for r in results:
            status = r.get(f'code_{lang}_status', '')
            attempts = r.get(f'code_{lang}_attempts', 0)
            if status == 'success':
                lang_stats['success'] += 1
            elif status == 'failed':
                lang_stats['failed'] += 1
            lang_stats['total_attempts'] += attempts
        stats['languages'][lang] = lang_stats
    
    master_data = {
        "metadata": {
            "regenerated_at": datetime.now().isoformat(),
            "total_tasks": len(results),
            "languages": languages,
            "stats": stats
        },
        "results": results
    }
    
    master_dir = OUTPUT_DIR / "master"
    master_dir.mkdir(exist_ok=True)
    
    # Save latest
    filepath = master_dir / "master_results.json"
    with open(filepath, 'w') as f:
        json.dump(master_data, f, indent=2)
    
    # Save timestamped archive
    archive_dir = master_dir / "archive"
    archive_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"master_results_{timestamp_str}.json"
    with open(archive_path, 'w') as f:
        json.dump(master_data, f, indent=2)
    
    return filepath, archive_path, stats


def save_master_csv(results: list, languages: list[str]):
    """Save master CSV with all translations (latest + timestamped archive)."""
    master_dir = OUTPUT_DIR / "master"
    filepath = master_dir / "master_translations.csv"
    
    # Prepare archive path
    archive_dir = master_dir / "archive"
    archive_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"master_translations_{timestamp_str}.csv"
    
    # Detect text languages from results
    text_languages = set()
    for r in results:
        for key in r:
            if key.startswith('text_') and not key.endswith('_status'):
                text_languages.add(key.replace('text_', ''))
    text_languages = sorted(text_languages)
    
    # Build field names
    fieldnames = [
        'task_id',
        'source_split',
        'source_config',
        'text',
    ]
    
    for text_lang in text_languages:
        fieldnames.append(f'text_{text_lang}')
    
    fieldnames.extend(['code_python', 'test_list'])
    
    for lang in languages:
        fieldnames.extend([
            f'code_{lang}',
            f'code_{lang}_status',
            f'code_{lang}_attempts',
            f'code_{lang}_tests_passed',
            f'code_{lang}_tests_failed',
        ])
    
    # Write to both latest and archive
    for out_path in [filepath, archive_path]:
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            for r in results:
                row = {
                    'task_id': r['task_id'],
                    'source_split': r.get('source_split', ''),
                    'source_config': r.get('source_config', ''),
                    'text': r.get('prompt', ''),
                    'code_python': r.get('python_code', ''),
                    'test_list': json.dumps(r.get('test_list', [])),
                }
                
                for text_lang in text_languages:
                    row[f'text_{text_lang}'] = r.get(f'text_{text_lang}', '')
                
                for lang in languages:
                    row[f'code_{lang}'] = r.get(f'code_{lang}', '')
                    row[f'code_{lang}_status'] = r.get(f'code_{lang}_status', '')
                    row[f'code_{lang}_attempts'] = r.get(f'code_{lang}_attempts', 0)
                    row[f'code_{lang}_tests_passed'] = r.get(f'code_{lang}_tests_passed', 0)
                    row[f'code_{lang}_tests_failed'] = r.get(f'code_{lang}_tests_failed', 0)
                
                writer.writerow(row)
    
    return filepath, archive_path


def save_per_language_outputs(results: list, languages: list[str]):
    """Save per-language JSON and CSV files."""
    text_languages = set()
    for r in results:
        for key in r:
            if key.startswith('text_') and not key.endswith('_status'):
                text_languages.add(key.replace('text_', ''))
    text_languages = sorted(text_languages)
    
    for lang in languages:
        lang_dir = OUTPUT_DIR / lang
        lang_dir.mkdir(exist_ok=True)
        
        # Filter results for this language
        lang_results = []
        for r in results:
            if r.get(f'code_{lang}'):
                lang_result = {
                    'task_id': r['task_id'],
                    'source_split': r.get('source_split', ''),
                    'source_config': r.get('source_config', ''),
                    'prompt': r.get('prompt', ''),
                    'python_code': r.get('python_code', ''),
                    'test_list': r.get('test_list', []),
                    f'code_{lang}': r.get(f'code_{lang}'),
                    f'code_{lang}_status': r.get(f'code_{lang}_status', ''),
                    f'code_{lang}_attempts': r.get(f'code_{lang}_attempts', 0),
                    f'code_{lang}_tests_passed': r.get(f'code_{lang}_tests_passed', 0),
                    f'code_{lang}_tests_failed': r.get(f'code_{lang}_tests_failed', 0),
                }
                if f'tests_{lang}' in r:
                    lang_result[f'tests_{lang}'] = r[f'tests_{lang}']
                for text_lang in text_languages:
                    if f'text_{text_lang}' in r:
                        lang_result[f'text_{text_lang}'] = r[f'text_{text_lang}']
                lang_results.append(lang_result)
        
        # Save JSON
        json_path = lang_dir / f"{lang}_results.json"
        with open(json_path, 'w') as f:
            json.dump({
                "metadata": {
                    "regenerated_at": datetime.now().isoformat(),
                    "language": lang,
                    "total_results": len(lang_results)
                },
                "results": lang_results
            }, f, indent=2)
        
        # Save CSV
        csv_path = lang_dir / f"{lang}_translations.csv"
        fieldnames = [
            'task_id', 'source_split', 'source_config', 'text',
            'code_python', 'test_list',
            f'code_{lang}', f'code_{lang}_status', f'code_{lang}_attempts',
            f'code_{lang}_tests_passed', f'code_{lang}_tests_failed',
            f'tests_{lang}'
        ]
        for text_lang in text_languages:
            fieldnames.insert(4, f'text_{text_lang}')
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            for r in lang_results:
                row = {
                    'task_id': r['task_id'],
                    'source_split': r.get('source_split', ''),
                    'source_config': r.get('source_config', ''),
                    'text': r.get('prompt', ''),
                    'code_python': r.get('python_code', ''),
                    'test_list': json.dumps(r.get('test_list', [])),
                    f'code_{lang}': r.get(f'code_{lang}', ''),
                    f'code_{lang}_status': r.get(f'code_{lang}_status', ''),
                    f'code_{lang}_attempts': r.get(f'code_{lang}_attempts', 0),
                    f'code_{lang}_tests_passed': r.get(f'code_{lang}_tests_passed', 0),
                    f'code_{lang}_tests_failed': r.get(f'code_{lang}_tests_failed', 0),
                    f'tests_{lang}': json.dumps(r.get(f'tests_{lang}', [])),
                }
                for text_lang in text_languages:
                    row[f'text_{text_lang}'] = r.get(f'text_{text_lang}', '')
                writer.writerow(row)


def generate_integrity_report(results: list, languages: list[str], stats: dict):
    """Generate integrity report with statistics and characteristics."""
    report_lines = []
    timestamp = datetime.now().isoformat()
    
    report_lines.append("=" * 80)
    report_lines.append("MBPP MULTI-LANGUAGE TRANSLATION - INTEGRITY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {timestamp}")
    report_lines.append("")
    
    # Overall statistics
    report_lines.append("=" * 80)
    report_lines.append("OVERALL STATISTICS")
    report_lines.append("=" * 80)
    report_lines.append(f"Total tasks: {stats['total_samples']}")
    report_lines.append(f"Languages: {', '.join(languages)}")
    report_lines.append("")
    
    # Per-language statistics
    report_lines.append("=" * 80)
    report_lines.append("PER-LANGUAGE STATISTICS")
    report_lines.append("=" * 80)
    
    total_success = 0
    total_failed = 0
    for lang in languages:
        lang_stats = stats['languages'].get(lang, {})
        success = lang_stats.get('success', 0)
        failed = lang_stats.get('failed', 0)
        total = success + failed
        rate = (success / total * 100) if total > 0 else 0
        avg_attempts = lang_stats.get('total_attempts', 0) / total if total > 0 else 0
        report_lines.append(f"{lang:12} - Success: {success}/{total} ({rate:.1f}%), Avg attempts: {avg_attempts:.2f}")
        total_success += success
        total_failed += failed
    
    total_all = total_success + total_failed
    overall_rate = (total_success / total_all * 100) if total_all > 0 else 0
    report_lines.append("")
    report_lines.append(f"Overall: {total_success}/{total_all} ({overall_rate:.1f}%)")
    report_lines.append("")
    
    # By split statistics
    report_lines.append("=" * 80)
    report_lines.append("BY SPLIT STATISTICS")
    report_lines.append("=" * 80)
    
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
        for t in sorted(incomplete, key=lambda x: -x['failed_count']):
            failed_str = ','.join(t['failed_langs'][:3])
            if len(t['failed_langs']) > 3:
                failed_str += f"... (+{len(t['failed_langs'])-3})"
            report_lines.append(f"{t['task_id']:<10} {t['split']:<12} {t['success_count']}/{len(languages):<8} {failed_str:<30} {t['prompt']}...")
    report_lines.append("")
    
    # Label integrity check
    report_lines.append("=" * 80)
    report_lines.append("LABEL INTEGRITY CHECK")
    report_lines.append("=" * 80)
    
    has_split = sum(1 for r in results if r.get('source_split') and r.get('source_split') != 'unknown')
    has_config = sum(1 for r in results if r.get('source_config') and r.get('source_config') != 'unknown')
    report_lines.append(f"Tasks with source_split: {has_split}/{len(results)}")
    report_lines.append(f"Tasks with source_config: {has_config}/{len(results)}")
    
    if has_split == len(results) and has_config == len(results):
        report_lines.append("[OK] All tasks have proper labels")
    else:
        report_lines.append("[WARN] Some tasks missing labels!")
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
    
    # Write report with timestamp in filename
    report_dir = OUTPUT_DIR / "reports"
    report_dir.mkdir(exist_ok=True)
    
    # Main report (overwritten)
    main_report_path = OUTPUT_DIR / "integrity_report.txt"
    with open(main_report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Timestamped report (archived)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = report_dir / f"integrity_report_{timestamp_str}.txt"
    with open(archive_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    return main_report_path, archive_path


def main():
    print("=" * 60)
    print("Regenerating outputs from individual JSON files")
    print("=" * 60)
    
    # Detect languages
    print("\nDetecting languages...")
    languages = get_all_languages()
    if not languages:
        print("No language folders found in output directory!")
        return
    print(f"Found languages: {', '.join(languages)}")
    
    # Load all tasks
    print("\nLoading individual JSON files...")
    results = load_all_tasks(languages)
    print(f"Loaded {len(results)} tasks")
    
    # Save master JSON
    print("\nRegenerating master_results.json...")
    master_path, master_archive, stats = save_master_json(results, languages)
    print(f"[OK] Saved to: {master_path}")
    print(f"[OK] Archived to: {master_archive}")
    
    # Save master CSV
    print("\nRegenerating master_translations.csv...")
    csv_path, csv_archive = save_master_csv(results, languages)
    print(f"[OK] Saved to: {csv_path}")
    print(f"[OK] Archived to: {csv_archive}")
    
    # Save per-language outputs
    print("\nRegenerating per-language outputs...")
    save_per_language_outputs(results, languages)
    for lang in languages:
        print(f"[OK] {lang}/ - {lang}_results.json, {lang}_translations.csv")
    
    # Generate integrity report
    print("\nGenerating integrity_report.txt...")
    main_report, archive_report = generate_integrity_report(results, languages, stats)
    print(f"[OK] Saved to: {main_report}")
    print(f"[OK] Archived to: {archive_report}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tasks: {stats['total_samples']}")
    print(f"Languages: {', '.join(languages)}")
    
    total_success = sum(s.get('success', 0) for s in stats['languages'].values())
    total_all = total_success + sum(s.get('failed', 0) for s in stats['languages'].values())
    if total_all > 0:
        print(f"Overall success: {total_success}/{total_all} ({total_success/total_all*100:.1f}%)")
    else:
        print("Overall success: 0/0 (N/A)")
    
    print("\nOutput files:")
    print(f"  - {master_path}")
    print(f"  - {csv_path}")
    print(f"  - {main_report}")
    print(f"  - output/master/archive/ (timestamped backups)")
    
    print("\nOutput files:")
    print(f"  - {master_path}")
    print(f"  - {csv_path}")
    print(f"  - {main_report}")
    print(f"  - {archive_report}")


if __name__ == "__main__":
    main()

