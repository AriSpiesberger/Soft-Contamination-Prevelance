#!/usr/bin/env python3
"""
Check progress and accuracy of ZebraLogic evaluations.

Scans the zebralogic_results directory and shows a table with:
  - wandb_id or folder name
  - answers file (from wandb metadata)
  - number of samples completed
  - current accuracy percentage
"""

import sys
# Avoid conflict with local 'wandb' directory
sys.path = [p for p in sys.path if 'finetuning' not in p]

import json
import re
from pathlib import Path
from tabulate import tabulate

# Re-add finetuning to path for imports
sys.path.insert(0, str(Path(__file__).parent))

WANDB_PROJECT = "semdupes-olmo3"
RESULTS_DIR = Path(__file__).parent / "outputs" / "zebralogic_results"
TOTAL_PUZZLES = 1000  # ZebraLogic has 1000 puzzles total


def is_wandb_run_id(folder_name: str) -> bool:
    """Check if folder name looks like a wandb run ID (8 alphanumeric chars)."""
    return bool(re.match(r'^[a-z0-9]{8}$', folder_name))


def get_wandb_run_info(run_ids):
    """
    Fetch wandb run info for given run IDs.
    Returns dict mapping run_id -> {answers_file, epochs, ...}
    """
    if not run_ids:
        return {}
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs(WANDB_PROJECT)
        
        run_info = {}
        for run in runs:
            if run.id in run_ids:
                answers_path = run.config.get('answers_path', None)
                answers_file = answers_path.split('/')[-1] if answers_path else None
                epochs = run.config.get('epochs', None)
                epochs = int(epochs) if epochs else None
                
                run_info[run.id] = {
                    'answers_file': answers_file,
                    'epochs': epochs,
                    'name': run.name,
                }
        return run_info
    except Exception as e:
        print(f"Warning: Could not fetch wandb info: {e}")
        return {}


def truncate_answers_file(filename, max_len=22):
    """
    Truncate answers filename for display.
    Handles zebralogic files specially:
      'zebralogic-sd-shuffle_and_substitute-shards-000-to-004-of-010-ver2.json'
      -> 'sd-shuff...0-4v2'
    """
    if not filename or filename == '-':
        return '-'
    
    # Remove .jsonl and .json suffixes
    name = filename.removesuffix('.jsonl').removesuffix('.json')

    return name
    
    # Special handling for zebralogic files
    if name.startswith('zebralogic-'):
        name = name[len('zebralogic-'):]
        
        # Shorten common patterns
        name = name.replace('shuffle_and_substitute', 'shuff')
        name = name.replace('shuffle_and_swap', 'shuf-swp')
        name = name.replace('_and_', '-')
        
        # Compress shard info: shards-000-to-004-of-010 -> 0-4
        match = re.search(r'shards-(\d+)-to-(\d+)-of-\d+', name)
        if match:
            start_shard = int(match.group(1))
            end_shard = int(match.group(2))
            name = re.sub(r'-?shards-\d+-to-\d+-of-\d+', f'..{start_shard}-{end_shard}', name)
        
        # Compress version: -ver2 -> v2
        name = re.sub(r'-ver(\d+)', r'v\1', name)
    
    if len(name) <= max_len:
        return name
    
    # Keep start and end with .... in middle
    keep_start = 10
    keep_end = max_len - keep_start - 4  # 4 for "...."
    return name[:keep_start] + "...." + name[-keep_end:]


def parse_progress_file(progress_file: Path) -> dict:
    """Parse a progress JSONL file and return stats."""
    results = []
    with open(progress_file, 'r') as f:
        for line in f:
            try:
                result = json.loads(line.strip())
                results.append(result)
            except json.JSONDecodeError:
                pass
    
    if not results:
        return None
    
    total = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    parsed = sum(1 for r in results if r.get("parsed", False))
    
    # Split by first 500 vs second 500
    # The index field can be like "lgp-test-5x6-16" - we need to track by order in file
    first_half = results[:500] if len(results) >= 500 else results
    second_half = results[500:] if len(results) > 500 else []
    
    first_correct = sum(1 for r in first_half if r.get("correct", False))
    first_total = len(first_half)
    
    second_correct = sum(1 for r in second_half if r.get("correct", False))
    second_total = len(second_half)
    
    return {
        "samples": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "parsed_rate": parsed / total if total > 0 else 0,
        "first_500_samples": first_total,
        "first_500_correct": first_correct,
        "first_500_acc": first_correct / first_total if first_total > 0 else 0,
        "second_500_samples": second_total,
        "second_500_correct": second_correct,
        "second_500_acc": second_correct / second_total if second_total > 0 else 0,
    }


def fmt(val, is_pct=False):
    """Format a value for display."""
    if val is None:
        return '-'
    elif isinstance(val, float):
        if is_pct:
            return f"{val:.1%}"
        return f"{val:.4f}"
    return str(val)


def main():
    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        return
    
    # Collect all results and run IDs
    run_data = []
    run_ids = set()
    
    for folder in sorted(RESULTS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        
        folder_name = folder.name
        
        # Track if this looks like a wandb run ID
        if is_wandb_run_id(folder_name):
            run_ids.add(folder_name)
        
        # Check for finetuned and base progress files
        for progress_type in ["finetuned", "base"]:
            progress_file = folder / f"{progress_type}_progress.jsonl"
            if not progress_file.exists():
                continue
            
            stats = parse_progress_file(progress_file)
            if stats is None:
                continue
            
            # Determine if complete
            is_complete = stats["samples"] >= TOTAL_PUZZLES
            
            run_data.append({
                "folder_name": folder_name,
                "run_id": folder_name if is_wandb_run_id(folder_name) else None,
                "progress_type": progress_type,
                "samples": stats["samples"],
                "accuracy": stats["accuracy"],
                "parsed_rate": stats["parsed_rate"],
                "is_complete": is_complete,
                "first_500_samples": stats["first_500_samples"],
                "first_500_acc": stats["first_500_acc"],
                "second_500_samples": stats["second_500_samples"],
                "second_500_acc": stats["second_500_acc"],
            })
    
    if not run_data:
        print("No zebralogic progress files found.")
        return
    
    # Fetch wandb info for all run IDs
    wandb_info = get_wandb_run_info(run_ids) if run_ids else {}
    
    # Sort: in-progress first, then by accuracy descending
    run_data.sort(key=lambda r: (r["is_complete"], -r["accuracy"]))
    
    # Build table
    headers = [
        'Answers File',
        'Run ID',
        'Ep',
        'N',
        'First 500',
        'Last 500',
        'Δ (L-F)',
        'Total',
        'Status',
    ]
    rows = []
    
    for rd in run_data:
        # Compute delta
        delta = None
        if rd['first_500_acc'] is not None and rd['second_500_acc'] is not None:
            if rd['first_500_samples'] > 0 and rd['second_500_samples'] > 0:
                delta = rd['second_500_acc'] - rd['first_500_acc']
        
        # Get wandb info if available
        wb = wandb_info.get(rd['run_id'], {})
        answers_file = wb.get('answers_file', '-')
        epochs = wb.get('epochs', '-')
        
        # Determine run ID display
        if rd['run_id']:
            run_id_display = rd['run_id']
            if rd['progress_type'] == 'base':
                run_id_display += ' (base)'
        else:
            run_id_display = rd['folder_name']
            if rd['progress_type'] == 'base':
                run_id_display += ' (base)'
        
        status = "✓" if rd['is_complete'] else "..."
        
        row = [
            answers_file,
            run_id_display,
            epochs,
            f"{rd['samples']}/{TOTAL_PUZZLES}",
            fmt(rd['first_500_acc'], is_pct=True),
            fmt(rd['second_500_acc'], is_pct=True),
            fmt(delta, is_pct=True) if delta is not None else '-',
            fmt(rd['accuracy'], is_pct=True),
            status,
        ]
        rows.append(row)
    
    # Print table
    print(f"\n{'='*100}")
    print(f"ZebraLogic Evaluation Progress")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Files found: {len(run_data)} | wandb runs matched: {len(wandb_info)}")
    print(f"{'='*100}\n")
    
    print(tabulate(rows, headers=headers, tablefmt='github'))
    print()
    
    # Summary
    in_progress = sum(1 for r in run_data if not r["is_complete"])
    complete = sum(1 for r in run_data if r["is_complete"])
    print(f"Total: {len(run_data)} evaluations ({complete} complete, {in_progress} in progress)")
    
    if complete > 0:
        complete_rows = [r for r in run_data if r["is_complete"]]
        avg_acc = sum(r["accuracy"] for r in complete_rows) / len(complete_rows)
        best = max(complete_rows, key=lambda r: r["accuracy"])
        print(f"Average accuracy (complete): {avg_acc:.1%}")
        print(f"Best: {best['folder_name']} with {best['accuracy']:.1%}")


if __name__ == "__main__":
    main()
