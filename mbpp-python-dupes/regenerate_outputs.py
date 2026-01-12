#!/usr/bin/env python3
"""
Regenerate CSV and create integrity report from individual JSON files.

This script:
1. Reads all individual task JSON files
2. Regenerates master_results.json
3. Regenerates mbpp_python_dupes.csv
4. Creates an integrity_report.txt with statistics and characteristics
"""

import csv
import json
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
INDIVIDUAL_DIR = OUTPUT_DIR / "individual"
MAX_DUPLICATES = 5


def load_all_tasks():
    """Load all task JSON files from individual directory."""
    tasks = []
    json_files = sorted(INDIVIDUAL_DIR.glob("task_*.json"), key=lambda x: int(x.stem.split("_")[1]))
    
    for json_file in json_files:
        with open(json_file) as f:
            tasks.append(json.load(f))
    
    return tasks


def save_master_json(results):
    """Save master JSON with all results (latest + timestamped archive)."""
    # Calculate stats
    stats = {
        'total_samples': len(results),
        'duplicates_generated': 0,
        'duplicates_failed': 0,
        'total_slots': len(results) * MAX_DUPLICATES
    }
    
    for r in results:
        for i in range(1, MAX_DUPLICATES + 1):
            status = r.get(f'python_{i}_status', '')
            if status == 'success':
                stats['duplicates_generated'] += 1
            elif status == 'failed':
                stats['duplicates_failed'] += 1
    
    master_data = {
        "metadata": {
            "regenerated_at": datetime.now().isoformat(),
            "total_tasks": len(results),
            "stats": stats
        },
        "results": results
    }
    
    # Save latest
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
    
    return filepath, archive_path, stats


def save_csv(results):
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
        'code_python',
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


def generate_integrity_report(results, stats):
    """Generate integrity report with statistics and characteristics."""
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("MBPP PYTHON SEMANTIC DUPLICATES - INTEGRITY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().isoformat()}")
    report_lines.append("")
    
    # Overall statistics
    report_lines.append("=" * 80)
    report_lines.append("OVERALL STATISTICS")
    report_lines.append("=" * 80)
    report_lines.append(f"Total tasks: {stats['total_samples']}")
    report_lines.append(f"Total duplicate slots: {stats['total_slots']}")
    report_lines.append(f"Successful duplicates: {stats['duplicates_generated']}")
    report_lines.append(f"Failed duplicates: {stats['duplicates_failed']}")
    success_rate = stats['duplicates_generated'] / stats['total_slots'] * 100 if stats['total_slots'] > 0 else 0
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
    
    for split in ['prompt', 'test', 'validation', 'train']:
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
        for t in sorted(incomplete, key=lambda x: x['successes']):
            report_lines.append(f"{t['task_id']:<10} {t['split']:<12} {t['successes']}/5       {t['failures']}/5       {t['prompt']}...")
    report_lines.append("")
    
    # Label check
    report_lines.append("=" * 80)
    report_lines.append("LABEL INTEGRITY CHECK")
    report_lines.append("=" * 80)
    
    has_split = sum(1 for r in results if r.get('source_split'))
    has_config = sum(1 for r in results if r.get('source_config'))
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
    
    # Code lengths
    orig_lengths = [len(r.get('code_python', '')) for r in results]
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
    
    # Comment counts in duplicates
    comment_counts = []
    for r in results:
        for i in range(1, MAX_DUPLICATES + 1):
            code = r.get(f'python_{i}', '')
            if code and r.get(f'python_{i}_status') == 'success':
                comments = sum(1 for line in code.split('\n') if '#' in line)
                comment_counts.append(comments)
    
    if comment_counts:
        report_lines.append(f"Comments per duplicate: min={min(comment_counts)}, max={max(comment_counts)}, avg={sum(comment_counts)/len(comment_counts):.1f}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Ensure archive directory exists
    archive_dir = OUTPUT_DIR / "archive"
    archive_dir.mkdir(exist_ok=True)
    
    # Write main report (overwritten each run)
    filepath = OUTPUT_DIR / "integrity_report.txt"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Write timestamped archive report
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"integrity_report_{timestamp_str}.txt"
    with open(archive_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    return filepath, archive_path


def main():
    print("=" * 60)
    print("Regenerating outputs from individual JSON files")
    print("=" * 60)
    
    # Load all tasks
    print("\nLoading individual JSON files...")
    results = load_all_tasks()
    print(f"Loaded {len(results)} tasks")
    
    # Save master JSON
    print("\nRegenerating master_results.json...")
    master_path, master_archive, stats = save_master_json(results)
    print(f"[OK] Saved to: {master_path}")
    print(f"[OK] Archived to: {master_archive}")
    
    # Save CSV
    print("\nRegenerating mbpp_python_dupes.csv...")
    csv_path, csv_archive = save_csv(results)
    print(f"[OK] Saved to: {csv_path}")
    print(f"[OK] Archived to: {csv_archive}")
    
    # Generate integrity report
    print("\nGenerating integrity_report.txt...")
    report_path, report_archive = generate_integrity_report(results, stats)
    print(f"[OK] Saved to: {report_path}")
    print(f"[OK] Archived to: {report_archive}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tasks: {stats['total_samples']}")
    print(f"Successful duplicates: {stats['duplicates_generated']}/{stats['total_slots']}")
    print(f"Success rate: {stats['duplicates_generated']/stats['total_slots']*100:.1f}%")
    print("\nOutput files:")
    print(f"  - {master_path}")
    print(f"  - {csv_path}")
    print(f"  - {report_path}")
    print(f"  - output/archive/ (timestamped backups)")


if __name__ == "__main__":
    main()

