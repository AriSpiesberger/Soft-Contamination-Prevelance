#!/usr/bin/env python3
"""
Compute first/second 50% accuracy from local eval logs.
Optionally fetches wandb metadata (answers_file) if available.
"""

import sys
# Avoid conflict with local 'wandb' directory
sys.path = [p for p in sys.path if 'finetuning' not in p]

import json
import os
from glob import glob
from tabulate import tabulate
import re
from pathlib import Path

# Re-add finetuning to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from baseline_results import baseline_results, baseline_musr, baseline_zebralogic

WANDB_PROJECT = "semdupes-musr"
EVAL_LOGS_DIR = Path(__file__).parent / "outputs" / "eval_logs"

def load_eval_log(filepath):
    """Load JSONL eval log and return list of question results."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_split_accuracy(results):
    """
    Compute accuracy for first 125 vs last 125 questions (MuSR has 250 questions).
    Returns (first_125_acc, last_125_acc, total_acc).
    """
    n = len(results)
    if n == 0:
        return None, None, None
    
    first_125 = results[:125]
    last_125 = results[-125:] if n >= 125 else results
    
    def avg_acc(questions):
        if not questions:
            return None
        total_correct = 0
        total_attempts = 0
        for q in questions:
            correct_arr = q.get('correct', [])
            total_correct += sum(correct_arr)
            total_attempts += len(correct_arr)
        return total_correct / total_attempts if total_attempts > 0 else None
    
    return avg_acc(first_125), avg_acc(last_125), avg_acc(results)


def parse_run_id_from_filename(filename):
    """Extract wandb run ID from filename like eval_outputs_finetuned_9yqx51io_x8.jsonl"""
    # Pattern: eval_outputs_finetuned_{run_id}_x{n}.jsonl
    match = re.search(r'eval_outputs_finetuned_([a-z0-9]+)_x\d+\.jsonl', filename)
    if match:
        return match.group(1)
    return None


def get_wandb_run_info(run_ids):
    """
    Fetch wandb run info for given run IDs.
    Returns dict mapping run_id -> {answers_file, epochs, ...}
    """
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs(WANDB_PROJECT)
        
        run_info = {}
        for run in runs:
            if run.id in run_ids:
                answers_path = run.config.get('answers_path', None)
                epochs = run.config.get('epochs', None)
                epochs = int(epochs) if epochs else None
                first_half_only = run.config.get('first_half_only', False)
                
                run_info[run.id] = {
                    'answers_path': answers_path,
                    'epochs': epochs,
                    'first_half_only': first_half_only,
                    'name': run.name,
                }
        return run_info
    except Exception as e:
        print(f"Warning: Could not fetch wandb info: {e}")
        return {}


def fmt(val):
    """Format a value for display."""
    if val is None:
        return '-'
    elif isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def main():
    # Find all eval log files
    pattern = os.path.join(EVAL_LOGS_DIR, "eval_outputs_*.jsonl")
    eval_files = sorted(glob(pattern))
    
    if not eval_files:
        print(f"No eval log files found in {EVAL_LOGS_DIR}")
        return
    
    run_data = []
    
    for filepath in eval_files:
        filename = os.path.basename(filepath)
        run_id = parse_run_id_from_filename(filename)
        
        results = load_eval_log(filepath)
        n_questions = len(results)
        first_acc, second_acc, total_acc = compute_split_accuracy(results)
        
        run_data.append({
            'filename': filename,
            'run_id': run_id,
            'n_questions': n_questions,
            'first_125_acc': first_acc,
            'last_125_acc': second_acc,
            'total_acc': total_acc,
        })
    
    # Fetch wandb info for all run IDs
    run_ids = {rd['run_id'] for rd in run_data if rd['run_id']}
    wandb_info = get_wandb_run_info(run_ids) if run_ids else {}
    
    # Build table
    headers = [
        'Answers Path',
        'Epochs',
        'Flags',
        'Run ID',
        'N',
        'First 125',
        'Last 125',
        'Δ (L-F)',
        'Total',
    ]
    rows = []
    
    for rd in run_data:
        delta = None
        if rd['first_125_acc'] is not None and rd['last_125_acc'] is not None:
            delta = rd['last_125_acc'] - rd['first_125_acc']
        
        # Get wandb info if available
        wb = wandb_info.get(rd['run_id'], {})
        answers_path = wb.get('answers_path') or '-'
        epochs = wb.get('epochs')
        first_half_only = wb.get('first_half_only', False)
        flags = '--first-half-only' if first_half_only else ''
        
        row = [
            answers_path,
            fmt(epochs),
            flags,
            rd['run_id'],
            rd['n_questions'],
            fmt(rd['first_125_acc']),
            fmt(rd['last_125_acc']),
            fmt(delta),
            fmt(rd['total_acc']),
        ]
        rows.append(row)
    
    # Print table
    print(f"\n{'='*100}")
    print(f"Local eval logs: {EVAL_LOGS_DIR}")
    print(f"Files found: {len(run_data)} | wandb runs matched: {len(wandb_info)}")
    print(f"{'='*100}\n")
    
    print(tabulate(rows, headers=headers, tablefmt='github'))
    print()
    
    # Baseline performance
    print("\n" + "="*60)
    print("BASELINE PERFORMANCE (pre-finetuning)")
    print("="*60)
    print(f"  MuSR accuracy:       {baseline_musr['accuracy']:.4f}")
    print(f"  ZebraLogic accuracy: {baseline_zebralogic['accuracy']:.4f}")
    print()
    print("  Baseline eval tasks (acc,none):")
    for task, vals in baseline_results.items():
        print(f"    {task:15s}: {vals['acc,none']:.4f}")
    print()
    
    # Summary
    print("Notes:")
    print("- First 125: Accuracy on first 125 questions (by sample_index)")
    print("- Last 125: Accuracy on last 125 questions")
    print("- Δ (L-F): Difference (positive = better on last 125)")


if __name__ == "__main__":
    main()

