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

ENGLISH_VARIATIONS = {
    'en1': 'Simple Paraphrase',
    'en2': 'Technical Paraphrase',
    'en3': 'Simplified Paraphrase',
    'en4': 'Expanded Paraphrase',
    'en5': 'Condensed Paraphrase',
}


def load_all_tasks():
    """Load all task JSON files from individual directory."""
    tasks = []
    json_files = sorted(INDIVIDUAL_DIR.glob("task_*.json"), key=lambda x: int(x.stem.split("_")[1]))
    
    for json_file in json_files:
        with open(json_file, encoding='utf-8') as f:
            tasks.append(json.load(f))
    
    return tasks


def detect_variations(results: list) -> list:
    """Detect which variations are present in results."""
    variations = set()
    for r in results:
        for key in r:
            if key.startswith('text_en') and key.endswith('_status'):
                var = key.replace('text_', '').replace('_status', '')
                if var in ENGLISH_VARIATIONS:
                    variations.add(var)
    return sorted(variations)


def save_master_json(results: list, variations: list):
    """Save master JSON."""
    stats = {
        'total_samples': len(results),
        'variations': {var: {'success': 0, 'failed': 0} for var in variations}
    }
    
    for r in results:
        for var in variations:
            status = r.get(f'text_{var}_status', '')
            if status == 'success':
                stats['variations'][var]['success'] += 1
            elif status == 'failed':
                stats['variations'][var]['failed'] += 1
    
    master_data = {
        "metadata": {
            "regenerated_at": datetime.now().isoformat(),
            "total_tasks": len(results),
            "variations": variations,
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


def save_csv(results: list, variations: list):
    """Save CSV."""
    filepath = OUTPUT_DIR / "english_variations.csv"
    
    archive_dir = OUTPUT_DIR / "archive"
    archive_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"english_variations_{timestamp_str}.csv"
    
    fieldnames = ['task_id', 'source_split', 'source_config', 'text']
    for var in variations:
        fieldnames.extend([f'text_{var}', f'text_{var}_status', f'text_{var}_attempts', f'text_{var}_method'])
    
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
                for var in variations:
                    row[f'text_{var}'] = r.get(f'text_{var}', '')
                    row[f'text_{var}_status'] = r.get(f'text_{var}_status', '')
                    row[f'text_{var}_attempts'] = r.get(f'text_{var}_attempts', 0)
                    row[f'text_{var}_method'] = r.get(f'text_{var}_method', '')
                writer.writerow(row)
    
    return filepath, archive_path


def generate_integrity_report(results: list, variations: list, stats: dict):
    """Generate integrity report."""
    report_lines = []
    timestamp = datetime.now()
    
    report_lines.append("=" * 80)
    report_lines.append("MBPP ENGLISH VARIATIONS - INTEGRITY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Regenerated: {timestamp.isoformat()}")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("STATISTICS")
    report_lines.append("=" * 80)
    report_lines.append(f"Total tasks: {stats['total_samples']}")
    
    for var in variations:
        s = stats['variations'].get(var, {})
        success = s.get('success', 0)
        failed = s.get('failed', 0)
        total = success + failed
        rate = (success / total * 100) if total > 0 else 0
        method = ENGLISH_VARIATIONS.get(var, var)
        report_lines.append(f"{var} ({method}): {success}/{total} ({rate:.1f}%)")
    
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
    
    variations = detect_variations(results)
    print(f"Detected variations: {', '.join(variations)}")
    
    print("\nRegenerating master_results.json...")
    master_path, master_archive, stats = save_master_json(results, variations)
    print(f"[OK] {master_path}")
    
    print("\nRegenerating english_variations.csv...")
    csv_path, csv_archive = save_csv(results, variations)
    print(f"[OK] {csv_path}")
    
    print("\nGenerating integrity_report.txt...")
    report_path, report_archive = generate_integrity_report(results, variations, stats)
    print(f"[OK] {report_path}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for var in variations:
        s = stats['variations'].get(var, {})
        success = s.get('success', 0)
        failed = s.get('failed', 0)
        total = success + failed
        rate = (success / total * 100) if total > 0 else 0
        print(f"  {var} ({ENGLISH_VARIATIONS.get(var, var)}): {success}/{total} ({rate:.1f}%)")


if __name__ == "__main__":
    main()

