#!/usr/bin/env python3
"""
Regenerate CSV, master JSON, and integrity report from individual JSON files.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
INDIVIDUAL_DIR = OUTPUT_DIR / "individual"

TEXT_LANGUAGES = {
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'ru': 'Russian',
    'zh': 'Chinese',
}


def load_all_tasks():
    """Load all task JSON files from individual directory."""
    tasks = []
    json_files = sorted(INDIVIDUAL_DIR.glob("task_*.json"), key=lambda x: int(x.stem.split("_")[1]))
    
    for json_file in json_files:
        with open(json_file, encoding='utf-8') as f:
            tasks.append(json.load(f))
    
    return tasks


def detect_languages(results: list) -> list:
    """Detect which languages are present in results."""
    languages = set()
    for r in results:
        for key in r:
            if key.startswith('text_') and key.endswith('_status'):
                lang = key.replace('text_', '').replace('_status', '')
                if lang in TEXT_LANGUAGES:
                    languages.add(lang)
    return sorted(languages)


def save_master_json(results: list, languages: list):
    """Save master JSON."""
    stats = {
        'total_samples': len(results),
        'languages': {lang: {'success': 0, 'failed': 0} for lang in languages}
    }
    
    for r in results:
        for lang in languages:
            status = r.get(f'text_{lang}_status', '')
            if status == 'success':
                stats['languages'][lang]['success'] += 1
            elif status == 'failed':
                stats['languages'][lang]['failed'] += 1
    
    master_data = {
        "metadata": {
            "regenerated_at": datetime.now().isoformat(),
            "total_tasks": len(results),
            "languages": languages,
            "stats": stats
        },
        "results": results
    }
    
    filepath = OUTPUT_DIR / "master_results.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(master_data, f, indent=2, ensure_ascii=False)
    
    archive_dir = OUTPUT_DIR / "archive"
    archive_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"master_results_{timestamp_str}.json"
    with open(archive_path, 'w', encoding='utf-8') as f:
        json.dump(master_data, f, indent=2, ensure_ascii=False)
    
    return filepath, archive_path, stats


def save_csv(results: list, languages: list):
    """Save CSV."""
    filepath = OUTPUT_DIR / "text_translations.csv"
    
    archive_dir = OUTPUT_DIR / "archive"
    archive_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"text_translations_{timestamp_str}.csv"
    
    fieldnames = ['task_id', 'source_split', 'source_config', 'text']
    for lang in languages:
        fieldnames.extend([f'text_{lang}', f'text_{lang}_status', f'text_{lang}_attempts'])
    
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


def generate_integrity_report(results: list, languages: list, stats: dict):
    """Generate integrity report."""
    report_lines = []
    timestamp = datetime.now()
    
    report_lines.append("=" * 80)
    report_lines.append("MBPP TEXT TRANSLATIONS (OTHER LANGUAGES) - INTEGRITY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Regenerated: {timestamp.isoformat()}")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("STATISTICS")
    report_lines.append("=" * 80)
    report_lines.append(f"Total tasks: {stats['total_samples']}")
    
    for lang in languages:
        s = stats['languages'].get(lang, {})
        success = s.get('success', 0)
        failed = s.get('failed', 0)
        total = success + failed
        rate = (success / total * 100) if total > 0 else 0
        report_lines.append(f"{TEXT_LANGUAGES[lang]:12} ({lang}): {success}/{total} ({rate:.1f}%)")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    archive_dir = OUTPUT_DIR / "archive"
    archive_dir.mkdir(exist_ok=True)
    
    main_path = OUTPUT_DIR / "integrity_report.txt"
    with open(main_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"integrity_report_{timestamp_str}.txt"
    with open(archive_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    return main_path, archive_path


def main():
    print("=" * 60)
    print("Regenerating outputs from individual JSON files")
    print("=" * 60)
    
    print("\nLoading individual JSON files...")
    results = load_all_tasks()
    print(f"Loaded {len(results)} tasks")
    
    languages = detect_languages(results)
    print(f"Detected languages: {', '.join(languages)}")
    
    print("\nRegenerating master_results.json...")
    master_path, master_archive, stats = save_master_json(results, languages)
    print(f"[OK] {master_path}")
    
    print("\nRegenerating text_translations.csv...")
    csv_path, csv_archive = save_csv(results, languages)
    print(f"[OK] {csv_path}")
    
    print("\nGenerating integrity_report.txt...")
    report_path, report_archive = generate_integrity_report(results, languages, stats)
    print(f"[OK] {report_path}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for lang in languages:
        s = stats['languages'].get(lang, {})
        success = s.get('success', 0)
        failed = s.get('failed', 0)
        total = success + failed
        rate = (success / total * 100) if total > 0 else 0
        print(f"  {TEXT_LANGUAGES[lang]:12} ({lang}): {success}/{total} ({rate:.1f}%)")


if __name__ == "__main__":
    main()

