#!/usr/bin/env python3
"""
Plot ZebraLogic performance by grid size for multiple models.
Compares performance on seen (0-499) vs unseen (500-999) samples.
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def parse_solution(solution_str):
    """Parse solution string from model output."""
    try:
        # Try to find JSON in the output
        if isinstance(solution_str, dict):
            return solution_str

        # Handle list output
        if isinstance(solution_str, list):
            solution_str = solution_str[0] if solution_str else ""

        # Try to parse as JSON
        if isinstance(solution_str, str):
            # Look for JSON content
            start = solution_str.find('{')
            end = solution_str.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = solution_str[start:end]
                parsed = json.loads(json_str)
                # Extract solution from parsed JSON
                if 'solution' in parsed:
                    return parsed['solution']
                return parsed
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    return {}


def normalize_value(val):
    """Normalize values for comparison."""
    if isinstance(val, str):
        return val.lower().strip()
    return str(val).lower().strip()


def convert_gt_to_house_format(gt_solution):
    """
    Convert ground truth from rows format to house-based format.
    Input: {'header': [...], 'rows': [[house, attr1, attr2, ...], ...]}
    Output: {'House 1': {'Attr1': 'val1', ...}, 'House 2': {...}, ...}
    """
    if 'header' not in gt_solution or 'rows' not in gt_solution:
        return {}

    header = list(gt_solution['header'])
    rows = gt_solution['rows']

    house_format = {}
    for row in rows:
        row = list(row)
        house_num = row[0]
        house_key = f"House {house_num}"
        house_format[house_key] = {}

        for i, attr in enumerate(header[1:], start=1):  # Skip 'House' column
            house_format[house_key][attr] = row[i]

    return house_format


def evaluate_solution(pred_solution, gt_solution):
    """
    Evaluate if predicted solution matches ground truth.
    Returns (is_correct, cell_accuracy, total_cells, correct_cells).
    """
    if not isinstance(pred_solution, dict) or not isinstance(gt_solution, dict):
        return False, 0.0, 0, 0

    # Convert ground truth to house format if needed
    if 'header' in gt_solution and 'rows' in gt_solution:
        gt_solution = convert_gt_to_house_format(gt_solution)

    if not gt_solution:
        return False, 0.0, 0, 0

    # Count cells
    total_cells = 0
    correct_cells = 0

    # Iterate through houses in ground truth
    for house_key, gt_house in gt_solution.items():
        if not isinstance(gt_house, dict):
            continue

        if house_key not in pred_solution:
            # Missing house - all cells wrong
            total_cells += len(gt_house)
            continue

        pred_house = pred_solution[house_key]
        if not isinstance(pred_house, dict):
            total_cells += len(gt_house)
            continue

        # Check each attribute
        for attr, gt_val in gt_house.items():
            total_cells += 1

            pred_val = pred_house.get(attr, '')

            if normalize_value(pred_val) == normalize_value(gt_val):
                correct_cells += 1

    is_correct = correct_cells == total_cells and total_cells > 0
    cell_accuracy = (correct_cells / total_cells * 100) if total_cells > 0 else 0.0

    return is_correct, cell_accuracy, total_cells, correct_cells


def load_and_evaluate(result_file, ground_truth_df):
    """Load results and evaluate against ground truth."""
    with open(result_file) as f:
        results = json.load(f)

    # Create ID to ground truth mapping
    gt_map = {}
    for _, row in ground_truth_df.iterrows():
        gt_map[row['id']] = {
            'solution': row['solution'],
            'size': row['size'],
            'index': row['index']
        }

    evaluated = []
    for entry in results:
        puzzle_id = entry['id']
        if puzzle_id not in gt_map:
            print(f"Warning: {puzzle_id} not found in ground truth")
            continue

        gt_info = gt_map[puzzle_id]
        pred_solution = parse_solution(entry['output'])
        gt_solution = gt_info['solution']

        is_correct, cell_acc, total_cells, correct_cells = evaluate_solution(
            pred_solution, gt_solution
        )

        evaluated.append({
            'puzzle_id': puzzle_id,
            'index': gt_info['index'],
            'size': gt_info['size'],
            'is_correct': is_correct,
            'cell_accuracy': cell_acc,
            'total_cells': total_cells,
            'correct_cells': correct_cells
        })

    return pd.DataFrame(evaluated)


def load_model_results(model_prefix, result_dir, ground_truth_df):
    """Load both seen and unseen results for a model."""
    seen_file = result_dir / f"{model_prefix}.0-500.json"
    unseen_file = result_dir / f"{model_prefix}.500-1000.json"

    results = {'seen': None, 'unseen': None}

    if seen_file.exists():
        results['seen'] = load_and_evaluate(seen_file, ground_truth_df)
    else:
        print(f"Warning: {seen_file} not found")

    if unseen_file.exists():
        results['unseen'] = load_and_evaluate(unseen_file, ground_truth_df)
    else:
        print(f"Warning: {unseen_file} not found")

    return results


def compute_accuracy_by_gridsize(df):
    """Compute accuracy grouped by grid size."""
    if df is None or len(df) == 0:
        return {}

    accuracy_by_size = {}
    for size, group in df.groupby('size'):
        n_correct = group['is_correct'].sum()
        n_total = len(group)
        accuracy = (n_correct / n_total * 100) if n_total > 0 else 0
        accuracy_by_size[size] = {
            'accuracy': accuracy,
            'n_correct': n_correct,
            'n_total': n_total
        }

    return accuracy_by_size


def plot_performance_by_gridsize(model_data, output_file, baseline_name="Baseline"):
    """
    Create bar plot of performance by grid size.

    model_data: dict mapping model_name -> {'seen': df, 'unseen': df}
    """
    # Get all unique grid sizes
    all_sizes = set()
    for model_name, data in model_data.items():
        for split in ['seen', 'unseen']:
            if data[split] is not None:
                all_sizes.update(data[split]['size'].unique())

    # Sort grid sizes by total cells (rows * cols)
    def grid_size_key(size_str):
        rows, cols = map(int, size_str.split('*'))
        return rows * cols

    sorted_sizes = sorted(all_sizes, key=grid_size_key)

    # Separate baseline from other models
    model_names = [name for name in model_data.keys() if name != baseline_name]
    n_models = len(model_names)

    # Create figure
    fig, ax = plt.subplots(figsize=(32, 8))

    # Bar widths
    ft_bar_width = 0.12  # Width for fine-tuned model bars
    baseline_bar_width = n_models * 2 * ft_bar_width * 1.3  # Baseline wider than all FT bars combined
    group_spacing = 1.2
    x_positions = np.arange(len(sorted_sizes)) * group_spacing

    # Colors for fine-tuned models (same color for seen/unseen, distinguished by hatching)
    model_colors = ['#4472C4', '#70AD47', '#FFC000', '#ED7D31', '#A5A5A5']  # Blue, Green, Orange, Red, Gray

    # Extend colors if needed
    if n_models > len(model_colors):
        model_colors = plt.cm.tab10(np.linspace(0, 0.9, n_models))

    # Collect accuracy data
    all_acc_data = {}

    # Get baseline data
    baseline_data = model_data.get(baseline_name)
    if baseline_data and baseline_data['seen'] is not None:
        all_acc_data[baseline_name] = {
            'seen': compute_accuracy_by_gridsize(baseline_data['seen']),
            'unseen': compute_accuracy_by_gridsize(baseline_data['unseen']) if baseline_data['unseen'] is not None else {}
        }

    for model_name in model_names:
        data = model_data[model_name]
        all_acc_data[model_name] = {
            'seen': compute_accuracy_by_gridsize(data['seen']) if data['seen'] is not None else {},
            'unseen': compute_accuracy_by_gridsize(data['unseen']) if data['unseen'] is not None else {}
        }

    # Plot baseline first (in background, wide and faded)
    if baseline_name in all_acc_data:
        baseline_values = [all_acc_data[baseline_name]['seen'].get(size, {'accuracy': 0})['accuracy']
                          for size in sorted_sizes]
        bars = ax.bar(x_positions, baseline_values, baseline_bar_width,
                     label=baseline_name, color='#808080', alpha=0.3,
                     edgecolor='black', linewidth=1.5, zorder=1)

    # Helper function to compute Wilson score confidence intervals
    def compute_wilson_interval(n_correct, n_total, confidence_level=0.68):
        """
        Compute Wilson score confidence interval.
        This method naturally respects [0,1] bounds and handles edge cases well.

        Point estimate is the observed proportion (MLE).
        Confidence bounds use Wilson score interval.

        Returns: (observed_proportion, lower_error, upper_error) in percentage
        """
        if n_total == 0:
            return 0, 0, 0

        p = n_correct / n_total  # Observed proportion (MLE)

        # Z-score for desired confidence level
        # 68% ≈ 1.0 sigma, 80% ≈ 1.28 sigma, 90% ≈ 1.645 sigma, 95% ≈ 1.96 sigma
        z = stats.norm.ppf((1 + confidence_level) / 2)

        # Wilson score interval formula
        denominator = 1 + z**2 / n_total
        center = (p + z**2 / (2 * n_total)) / denominator
        margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denominator

        lower = max(0, center - margin)
        upper = min(1, center + margin)

        # Convert to percentages and compute errors from observed proportion
        p_pct = p * 100
        lower_err = max(0, (p - lower) * 100)
        upper_err = max(0, (upper - p) * 100)

        return p_pct, lower_err, upper_err

    # Plot fine-tuned models (foreground, circles with error bars)
    for i, model_name in enumerate(model_names):
        color = model_colors[i]

        # Calculate positions: center the FT markers within the baseline bar
        total_ft_width = n_models * 2 * ft_bar_width
        start_offset = -total_ft_width / 2
        seen_offset = start_offset + i * 2 * ft_bar_width
        unseen_offset = seen_offset + ft_bar_width

        # Seen (hollow circles)
        seen_values = []
        seen_errors_lower = []
        seen_errors_upper = []
        for size in sorted_sizes:
            acc_data = all_acc_data[model_name]['seen'].get(size, {'accuracy': 0, 'n_correct': 0, 'n_total': 1})
            n_correct = acc_data['n_correct']
            n_total = acc_data['n_total']

            mean_pct, lower_err, upper_err = compute_wilson_interval(n_correct, n_total, confidence_level=0.68)

            seen_values.append(mean_pct)
            seen_errors_lower.append(lower_err)
            seen_errors_upper.append(upper_err)

        # Plot hollow circles for seen data
        ax.scatter(x_positions + seen_offset, seen_values,
                  s=120, marker='o', facecolors='none', edgecolors=color,
                  linewidths=2.5, label=f'{model_name}',
                  zorder=3)

        # Add error bars (whiskers) - asymmetric
        ax.errorbar(x_positions + seen_offset, seen_values,
                   yerr=[seen_errors_lower, seen_errors_upper],
                   fmt='none', ecolor='black', elinewidth=1.5, capsize=3, capthick=1.5,
                   zorder=2)

        # Unseen (filled circles)
        unseen_values = []
        unseen_errors_lower = []
        unseen_errors_upper = []
        for size in sorted_sizes:
            acc_data = all_acc_data[model_name]['unseen'].get(size, {'accuracy': 0, 'n_correct': 0, 'n_total': 1})
            n_correct = acc_data['n_correct']
            n_total = acc_data['n_total']

            mean_pct, lower_err, upper_err = compute_wilson_interval(n_correct, n_total, confidence_level=0.68)

            unseen_values.append(mean_pct)
            unseen_errors_lower.append(lower_err)
            unseen_errors_upper.append(upper_err)

        # Plot filled circles for unseen data
        ax.scatter(x_positions + unseen_offset, unseen_values,
                  s=120, marker='o', color=color, edgecolors='black',
                  linewidths=1.0,
                  zorder=3)

        # Add error bars (whiskers) - asymmetric
        ax.errorbar(x_positions + unseen_offset, unseen_values,
                   yerr=[unseen_errors_lower, unseen_errors_upper],
                   fmt='none', ecolor='black', elinewidth=1.5, capsize=3, capthick=1.5,
                   zorder=2)

    # Formatting
    ax.set_xlabel('Puzzle Grid Size (rows × columns)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Puzzle Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_title('ZebraLogic Performance by Grid Size: Baseline vs Fine-Tuned Models\n' +
                'Seen (indices 0-499) vs Unseen (indices 500-999). Error bars: 68% CI (Wilson score)',
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_sizes, fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8, zorder=0)

    # Update legend to explain markers
    handles, labels = ax.get_legend_handles_labels()
    # Add custom handles for marker explanation
    from matplotlib.lines import Line2D
    custom_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='gray', markersize=10, linewidth=2, label='○ = Seen/Training'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=10, linewidth=1, label='● = Unseen/Test')
    ]
    ax.legend(handles=custom_handles + handles, loc='upper right', fontsize=11,
              framealpha=0.95, ncol=2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot ZebraLogic performance by grid size for multiple models"
    )
    parser.add_argument(
        '--result-dir',
        type=Path,
        default=Path('ZeroEval/result_dirs/zebra-grid'),
        help='Directory containing result JSON files'
    )
    parser.add_argument(
        '--ground-truth',
        type=Path,
        default=Path('datasets/zebralogic/original/zebralogic.parquet'),
        help='Ground truth parquet file'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help='Model prefixes to plot (e.g., sdtd-gpt-4.1-mini-2025-04-14-run2)'
    )
    parser.add_argument(
        '--baseline',
        default='sdtd-gpt-4.1-mini-2025-04-14-run2',
        help='Baseline model prefix'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('zebralogic_gridsize_performance.png'),
        help='Output plot file'
    )
    parser.add_argument(
        '--model-names',
        nargs='+',
        help='Display names for models (optional, same order as --models)'
    )

    args = parser.parse_args()

    # Load ground truth
    print("Loading ground truth...")
    gt_df = pd.read_parquet(args.ground_truth)
    print(f"Loaded {len(gt_df)} puzzles")

    # Load and evaluate all models
    model_data = {}
    model_display_names = args.model_names if args.model_names else args.models

    for model_prefix, display_name in zip(args.models, model_display_names):
        print(f"\nLoading results for {model_prefix}...")
        results = load_model_results(model_prefix, args.result_dir, gt_df)

        if results['seen'] is not None:
            n_correct = results['seen']['is_correct'].sum()
            print(f"  Seen: {n_correct}/{len(results['seen'])} correct "
                  f"({n_correct/len(results['seen'])*100:.1f}%)")

        if results['unseen'] is not None:
            n_correct = results['unseen']['is_correct'].sum()
            print(f"  Unseen: {n_correct}/{len(results['unseen'])} correct "
                  f"({n_correct/len(results['unseen'])*100:.1f}%)")

        model_data[display_name] = results

    # Create plot
    print(f"\nCreating plot...")
    baseline_display = model_display_names[args.models.index(args.baseline)] if args.baseline in args.models else "Baseline"
    plot_performance_by_gridsize(model_data, args.output, baseline_name=baseline_display)


if __name__ == '__main__':
    main()
