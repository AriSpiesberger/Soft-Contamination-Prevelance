#!/usr/bin/env python3
"""
Analyze MBPP evaluation results from multiple runs.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import argparse


def load_results(jsonl_path):
    """Load results from a JSONL file."""
    results = []
    with open(jsonl_path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def summarize_results(results, name):
    """Print summary statistics for a result set."""
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    pass_rate = 100 * correct / total if total > 0 else 0

    # Count error types
    errors = defaultdict(int)
    for r in results:
        if not r['correct'] and r.get('error'):
            # Extract error type
            error_lines = r['error'].strip().split('\n')
            if error_lines:
                last_line = error_lines[-1]
                # Extract just the error type
                if ':' in last_line:
                    error_type = last_line.split(':')[0].strip()
                else:
                    error_type = last_line.strip()
                errors[error_type] += 1

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Pass@1: {pass_rate:.2f}% ({correct}/{total})")

    if errors:
        print(f"\nError breakdown:")
        for error_type, count in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")

    return {
        'total': total,
        'correct': correct,
        'pass_rate': pass_rate,
        'errors': dict(errors)
    }


def compare_runs(base_results, ft_results):
    """Compare base model vs finetuned model results."""
    # Create lookup by task_id
    base_by_id = {r['task_id']: r for r in base_results}
    ft_by_id = {r['task_id']: r for r in ft_results}

    # Find tasks where status changed
    improved = []
    regressed = []

    for task_id in base_by_id:
        if task_id in ft_by_id:
            base_correct = base_by_id[task_id]['correct']
            ft_correct = ft_by_id[task_id]['correct']

            if not base_correct and ft_correct:
                improved.append(task_id)
            elif base_correct and not ft_correct:
                regressed.append(task_id)

    print(f"\n{'='*60}")
    print("Comparison: Base → Finetuned")
    print(f"{'='*60}")
    print(f"Tasks improved: {len(improved)}")
    print(f"Tasks regressed: {len(regressed)}")
    print(f"Net improvement: {len(improved) - len(regressed)}")

    if improved and len(improved) <= 10:
        print(f"\nImproved task IDs: {improved}")
    if regressed and len(regressed) <= 10:
        print(f"Regressed task IDs: {regressed}")

    return improved, regressed


def show_example_failures(results, n=5):
    """Show example failures from results."""
    failures = [r for r in results if not r['correct']]

    print(f"\n{'='*60}")
    print(f"Example Failures (showing {min(n, len(failures))} of {len(failures)})")
    print(f"{'='*60}")

    for i, r in enumerate(failures[:n]):
        print(f"\n--- Task {r['task_id']} ---")
        print(f"Prompt: {r['prompt'][:80]}...")
        print(f"\nGenerated code:")
        print(r['generated_code'][:200])
        if len(r['generated_code']) > 200:
            print("...")
        print(f"\nError: {r.get('error', 'No error message')[:150]}...")


def main():
    parser = argparse.ArgumentParser(description="Analyze MBPP evaluation results")
    parser.add_argument('files', nargs='*', help='Result files to analyze')
    parser.add_argument('--compare', action='store_true', help='Compare first two files (base vs finetuned)')
    parser.add_argument('--failures', type=int, default=0, help='Show N example failures')
    parser.add_argument('--latest', action='store_true', help='Analyze latest base and finetuned results')
    args = parser.parse_args()

    output_dir = Path(__file__).parent / "outputs" / "mbpp_eval"

    # If --latest, find most recent base and finetuned results
    if args.latest:
        all_files = sorted(output_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)

        # Find latest base train/eval
        base_train = next((f for f in all_files if 'base_train' in f.name), None)
        base_eval = next((f for f in all_files if 'base_eval' in f.name), None)

        # Find latest finetuned train/eval
        ft_train = next((f for f in all_files if 'finetuned' in f.name and 'train' in f.name), None)
        ft_eval = next((f for f in all_files if 'finetuned' in f.name and 'eval' in f.name), None)

        print("Latest results found:")
        if base_train:
            print(f"  Base (train): {base_train.name}")
        if base_eval:
            print(f"  Base (eval): {base_eval.name}")
        if ft_train:
            print(f"  Finetuned (train): {ft_train.name}")
        if ft_eval:
            print(f"  Finetuned (eval): {ft_eval.name}")

        # Analyze all
        results = {}
        if base_train:
            results['base_train'] = load_results(base_train)
            summarize_results(results['base_train'], f"Base Model (Train Split) - {base_train.name}")
        if base_eval:
            results['base_eval'] = load_results(base_eval)
            summarize_results(results['base_eval'], f"Base Model (Eval Split) - {base_eval.name}")
        if ft_train:
            results['ft_train'] = load_results(ft_train)
            summarize_results(results['ft_train'], f"Finetuned Model (Train Split) - {ft_train.name}")
        if ft_eval:
            results['ft_eval'] = load_results(ft_eval)
            summarize_results(results['ft_eval'], f"Finetuned Model (Eval Split) - {ft_eval.name}")

        # Compare if both base and finetuned exist
        if 'base_train' in results and 'ft_train' in results:
            compare_runs(results['base_train'], results['ft_train'])
        if 'base_eval' in results and 'ft_eval' in results:
            compare_runs(results['base_eval'], results['ft_eval'])

        return

    # Original file-based analysis
    if not args.files:
        print("Usage: python analyze_results.py [files...] or --latest")
        print(f"\nAvailable result files in {output_dir}:")
        for f in sorted(output_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]:
            print(f"  {f.name}")
        return

    results_list = []
    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            path = output_dir / file_path

        results = load_results(path)
        results_list.append((path.name, results))
        summarize_results(results, path.name)

        if args.failures > 0:
            show_example_failures(results, args.failures)

    # Compare if requested and we have exactly 2 files
    if args.compare and len(results_list) == 2:
        compare_runs(results_list[0][1], results_list[1][1])


if __name__ == "__main__":
    main()
