"""
Export annotations to CSV files with statistics.

Generates:
- {benchmark}_annotations_full.csv: All annotations with full details
- {benchmark}_annotations_summary.csv: Only semantic duplicates (non-unrelated)
- {benchmark}_statistics.json: Aggregate statistics

Usage:
    python export_annotations.py --benchmark zebralogic
    python export_annotations.py --benchmark mbpp
    python export_annotations.py --benchmark codeforces
"""

import argparse
import json
import csv
from pathlib import Path
from collections import Counter
from datetime import datetime

# Paths relative to this script
SCRIPT_DIR = Path(__file__).parent
ANNOTATIONS_DIR = SCRIPT_DIR / "annotations"
EXPORTS_DIR = SCRIPT_DIR / "exports"


def load_annotations(benchmark: str) -> list[dict]:
    """Load all annotation JSON files for a benchmark."""
    annotations_path = ANNOTATIONS_DIR / benchmark
    
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_path}")
    
    records = []
    errors = []
    
    for json_file in annotations_path.glob("*.json"):
        # Skip summary and internal files
        if json_file.stem.startswith('_'):
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract annotation
            ann = data.get('annotation', {}) or {}
            
            # Parse filename for IDs
            parts = json_file.stem.split('__')
            dataset = parts[0] if len(parts) >= 1 else ''
            test_id = parts[1] if len(parts) >= 2 else data.get('test_id', '')
            corpus_id = parts[2] if len(parts) >= 3 else data.get('corpus_id', '')
            
            # Build record
            record = {
                'filename': json_file.stem,
                'dataset': data.get('dataset', dataset),
                'benchmark': data.get('benchmark', benchmark),
                'test_id': data.get('test_id', test_id),
                'corpus_id': data.get('corpus_id', corpus_id),
                'score': data.get('score', 0),
                'weight': data.get('weight', 0),
                'match_type': ann.get('match_type', 'unknown'),
                'is_sd': ann.get('is_sd', False),
                'confidence': ann.get('confidence', 0),
                'reasoning': ann.get('reasoning', ''),
                'test_text': data.get('test_text', ''),
                'corpus_text': data.get('corpus_text', ''),
                'success': data.get('success', False),
                'error': data.get('error', ''),
                # Cost tracking
                'cost_total': data.get('cost', {}).get('total', 0) if data.get('cost') else 0,
                'tokens_input': data.get('usage', {}).get('prompt_tokens', 0) if data.get('usage') else 0,
                'tokens_output': data.get('usage', {}).get('answer_tokens', 0) if data.get('usage') else 0,
                'tokens_thinking': data.get('usage', {}).get('thought_tokens', 0) if data.get('usage') else 0,
            }
            records.append(record)
            
        except Exception as e:
            errors.append({'file': json_file.name, 'error': str(e)})
    
    return records, errors


def compute_statistics(records: list[dict]) -> dict:
    """Compute aggregate statistics from annotations."""
    if not records:
        return {}
    
    # Filter to successful annotations
    successful = [r for r in records if r['success']]
    
    # Count match types
    match_type_counts = Counter(r['match_type'] for r in successful)
    
    # Count semantic duplicates (non-unrelated)
    sd_records = [r for r in successful if r['is_sd'] or r['match_type'] != 'unrelated']
    
    # By dataset breakdown
    by_dataset = {}
    for r in successful:
        ds = r['dataset']
        if ds not in by_dataset:
            by_dataset[ds] = {'total': 0, 'sd': 0, 'match_types': Counter()}
        by_dataset[ds]['total'] += 1
        if r['is_sd'] or r['match_type'] != 'unrelated':
            by_dataset[ds]['sd'] += 1
        by_dataset[ds]['match_types'][r['match_type']] += 1
    
    # Convert counters to dicts for JSON serialization
    for ds in by_dataset:
        by_dataset[ds]['match_types'] = dict(by_dataset[ds]['match_types'])
        by_dataset[ds]['sd_rate'] = by_dataset[ds]['sd'] / max(by_dataset[ds]['total'], 1)
    
    # Confidence stats for SDs
    sd_confidences = [r['confidence'] for r in sd_records if r['confidence']]
    
    # Unique test_ids with at least one SD
    test_ids_with_sd = set(r['test_id'] for r in sd_records)
    
    # Unique test_ids with at least one exact duplicate
    exact_records = [r for r in successful if r['match_type'] == 'exact']
    test_ids_with_exact = set(r['test_id'] for r in exact_records)
    
    # Total unique test_ids in the dataset
    all_test_ids = set(r['test_id'] for r in successful)
    
    # Cost stats
    total_cost = sum(r['cost_total'] for r in records)
    
    stats = {
        'benchmark': records[0]['benchmark'] if records else '',
        'generated_at': datetime.now().isoformat(),
        'total_files': len(records),
        'successful': len(successful),
        'failed': len(records) - len(successful),
        'unique_test_ids': len(all_test_ids),
        'semantic_duplicates': {
            'count': len(sd_records),
            'rate': len(sd_records) / max(len(successful), 1),
            'unique_test_ids_affected': len(test_ids_with_sd),
            'test_id_contamination_rate': len(test_ids_with_sd) / max(len(all_test_ids), 1),
            'avg_confidence': sum(sd_confidences) / max(len(sd_confidences), 1) if sd_confidences else 0,
            'min_confidence': min(sd_confidences) if sd_confidences else 0,
            'max_confidence': max(sd_confidences) if sd_confidences else 0,
        },
        'exact_duplicates': {
            'count': len(exact_records),
            'unique_test_ids_affected': len(test_ids_with_exact),
            'test_id_contamination_rate': len(test_ids_with_exact) / max(len(all_test_ids), 1),
        },
        'match_types': dict(match_type_counts),
        'by_dataset': by_dataset,
        'cost': {
            'total': total_cost,
            'avg_per_annotation': total_cost / max(len(records), 1),
        },
        'similarity_scores': {
            'min': min(r['score'] for r in successful) if successful else 0,
            'max': max(r['score'] for r in successful) if successful else 0,
            'mean': sum(r['score'] for r in successful) / max(len(successful), 1) if successful else 0,
        }
    }
    
    return stats


def export_to_csv(records: list[dict], output_path: Path, include_text: bool = True):
    """Export records to CSV."""
    if not records:
        print(f"  No records to export")
        return
    
    # Define fields
    base_fields = [
        'filename', 'dataset', 'benchmark', 'test_id', 'corpus_id',
        'score', 'match_type', 'is_sd', 'confidence', 'reasoning'
    ]
    
    if include_text:
        base_fields.extend(['test_text', 'corpus_text'])
    
    base_fields.extend(['success', 'cost_total'])
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=base_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    
    print(f"  Saved: {output_path} ({len(records):,} rows)")


def main():
    parser = argparse.ArgumentParser(description="Export annotations to CSV with statistics")
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["mbpp", "codeforces", "zebralogic"],
        help="Benchmark to export"
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Exclude test_text and corpus_text from full CSV (smaller file)"
    )
    args = parser.parse_args()
    
    print(f"=" * 60)
    print(f"EXPORTING ANNOTATIONS: {args.benchmark}")
    print(f"=" * 60)
    
    # Create output directory
    output_dir = EXPORTS_DIR / args.benchmark
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print(f"\nLoading annotations from: {ANNOTATIONS_DIR / args.benchmark}")
    records, errors = load_annotations(args.benchmark)
    print(f"  Loaded: {len(records):,} annotations")
    if errors:
        print(f"  Errors: {len(errors)} files failed to load")
    
    if not records:
        print("No annotations found!")
        return 1
    
    # Compute statistics
    print(f"\nComputing statistics...")
    stats = compute_statistics(records)
    
    # Export full CSV
    print(f"\nExporting full CSV...")
    full_csv_path = output_dir / f"{args.benchmark}_annotations_full.csv"
    export_to_csv(records, full_csv_path, include_text=not args.no_text)
    
    # Export summary CSV (only semantic duplicates)
    print(f"\nExporting summary CSV (semantic duplicates only)...")
    sd_records = [r for r in records if r['success'] and (r['is_sd'] or r['match_type'] != 'unrelated')]
    summary_csv_path = output_dir / f"{args.benchmark}_annotations_summary.csv"
    export_to_csv(sd_records, summary_csv_path, include_text=not args.no_text)
    
    # Export statistics JSON
    print(f"\nExporting statistics...")
    stats_path = output_dir / f"{args.benchmark}_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {stats_path}")
    
    # Print summary
    print(f"\n" + "=" * 60)
    print(f"SUMMARY")
    print(f"=" * 60)
    print(f"  Total annotations: {stats['total_files']:,}")
    print(f"  Successful: {stats['successful']:,}")
    print(f"  Failed: {stats['failed']:,}")
    print(f"  Unique test IDs: {stats['unique_test_ids']:,}")
    print(f"\nSemantic Duplicates:")
    print(f"  Count: {stats['semantic_duplicates']['count']:,} ({100*stats['semantic_duplicates']['rate']:.2f}%)")
    print(f"  Unique test IDs affected: {stats['semantic_duplicates']['unique_test_ids_affected']:,} ({100*stats['semantic_duplicates']['test_id_contamination_rate']:.2f}%)")
    print(f"\nExact Duplicates:")
    print(f"  Count: {stats['exact_duplicates']['count']:,}")
    print(f"  Unique test IDs affected: {stats['exact_duplicates']['unique_test_ids_affected']:,} ({100*stats['exact_duplicates']['test_id_contamination_rate']:.2f}%)")
    print(f"\nCost: ${stats['cost']['total']:.2f}")
    print(f"\nMatch types:")
    for mt, count in sorted(stats['match_types'].items(), key=lambda x: -x[1]):
        print(f"    {mt}: {count:,}")
    print(f"\nOutputs:")
    print(f"  {full_csv_path}")
    print(f"  {summary_csv_path}")
    print(f"  {stats_path}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
