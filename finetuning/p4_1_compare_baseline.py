#!/usr/bin/env python3
"""Print base vs finetuned accuracy comparison from comparison.json"""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Print base vs finetuned accuracy comparison")
    parser.add_argument("comparison_json", type=str, help="Path to comparison.json file")
    args = parser.parse_args()

    with open(args.comparison_json, "r") as f:
        data = json.load(f)

    print("\n" + "=" * 70)
    print(f"{'Benchmark':<15} {'Base':<20} {'Finetuned':<20}")
    print("=" * 70)

    for benchmark, metrics in data.items():
        # Get acc and stderr
        acc_key = "acc,none"
        stderr_key = "acc_stderr,none"

        if acc_key in metrics and stderr_key in metrics:
            base_acc = metrics[acc_key]["base"]
            base_stderr = metrics[stderr_key]["base"]
            ft_acc = metrics[acc_key]["finetuned"]
            ft_stderr = metrics[stderr_key]["finetuned"]

            base_str = f"{base_acc:.2%} ± {base_stderr:.2%}"
            ft_str = f"{ft_acc:.2%} ± {ft_stderr:.2%}"

            print(f"{benchmark:<15} {base_str:<20} {ft_str:<20}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

