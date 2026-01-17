#!/usr/bin/env python3
"""Pull all run data from wandb project semdupes-musr and create a table."""

WANDB_PROJECT = "olmo3-murder-mystery-finetune" #"semdupes-olmo3"

import sys
# Avoid conflict with local 'wandb' directory
sys.path = [p for p in sys.path if 'finetuning' not in p]

import wandb
from tabulate import tabulate
from pathlib import Path

# Re-add finetuning to path to import baseline_results
sys.path.insert(0, str(Path(__file__).parent))
from baseline_results import baseline_results, baseline_musr, baseline_zebralogic

# Baseline eval tasks - we want acc,none for these
BASELINE_TASKS = ['arc_challenge', 'arc_easy', 'boolq', 'hellaswag', 'piqa', 'winogrande']

def fmt(val):
    """Format a value for display."""
    if val is None:
        return '-'
    elif isinstance(val, float):
        return f"{val:.4f}"
    return str(val)

def main():
    api = wandb.Api()
    
    # Get all runs from the project
    runs = api.runs(WANDB_PROJECT)
    
    run_data = []
    
    for run in runs:
        summary = run.summary._json_dict

        answers_path = run.config.get('answers_path', None)
        answers_file = answers_path.split('/')[-1] if answers_path else None
        first_half_only = run.config.get('first_half_only', False)
        if first_half_only and answers_file:
            answers_file = f"{answers_file} --first-half-only"

        epochs = run.config.get('epochs', None)
        epochs = int(epochs) if epochs else None
        
        # Extract baseline eval metrics (acc,none only)
        baseline_metrics = {}
        for task in BASELINE_TASKS:
            # Try different key patterns
            for pattern in [
                f"eval/finetuned/{task}/acc,none",
                f"eval/baseline/{task}/acc,none",
                f"{task}/acc,none",
                f"{task}_acc,none",
            ]:
                if pattern in summary:
                    baseline_metrics[task] = summary[pattern]
                    break
        
        # Extract musr accuracy
        musr_acc = None
        for pattern in ['musr_accuracy', 'musr/accuracy', 'eval/musr_accuracy', 'accuracy']:
            if pattern in summary:
                musr_acc = summary[pattern]
                break
        
        # Extract zebralogic accuracy
        zebralogic_acc = None
        for pattern in ['zebralogic/acc']:
            if pattern in summary:
                zebralogic_acc = summary[pattern]
                break
        
        # Skip runs with no relevant metrics
        if not baseline_metrics and musr_acc is None and zebralogic_acc is None:
            continue
        
        run_data.append({
            'id': run.id,
            'name': run.name,
            'answers_path': answers_path,
            'answers_file': answers_file,
            'baseline': baseline_metrics,
            'musr': musr_acc,
            'zebralogic': zebralogic_acc,
            'epochs': epochs,
        })
    
    if not run_data:
        print("No runs found with relevant metrics in semdupes-musr")
        return
    
    # Build table
    headers = ['Answers file', 'Epochs', 'MuSR', 'ZebraLogic'] + [t.replace('_', ' ') for t in BASELINE_TASKS]
    rows = []
    
    # Add baseline row first
    baseline_row = [
        '** BASELINE **',
        '-',
        fmt(baseline_musr['accuracy']),
        fmt(baseline_zebralogic['accuracy']),
    ]
    for task in BASELINE_TASKS:
        baseline_row.append(fmt(baseline_results[task]['acc,none']))
    rows.append(baseline_row)
    
    # Add separator row
    rows.append(['---'] * len(headers))
    
    for rd in run_data:
        row = [
            rd['answers_file'],
            rd['epochs'],
            fmt(rd['musr']),
            fmt(rd['zebralogic']),
        ]
        for task in BASELINE_TASKS:
            row.append(fmt(rd['baseline'].get(task)))
        rows.append(row)
    
    # Print table
    print(f"\n{'='*120}")
    print(f"wandb project: semdupes-musr | Runs with data: {len(rows) - 2}")  # -2 for baseline and separator
    print(f"{'='*120}\n")
    
    print(tabulate(rows, headers=headers, tablefmt='github'))
    print()

if __name__ == "__main__":
    main()

