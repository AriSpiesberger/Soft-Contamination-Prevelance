#!/usr/bin/env python3
"""
Config helper for shell scripts.

Reads pipeline YAML configs and outputs values for shell scripts to consume.

Usage:
    python lib/config_helper.py --config configs/default.yaml --get analysis.corpus_dir
    python lib/config_helper.py --config configs/default.yaml --check-skip download
    python lib/config_helper.py --config configs/default.yaml --section embeddings
"""

import argparse
import sys
import yaml
from pathlib import Path


def load_config(config_path):
    """Load YAML config file."""
    config_path = Path(config_path)
    if not config_path.exists():
        # Try relative to configs/ directory
        alt_path = Path(__file__).parent.parent / "configs" / config_path
        if alt_path.exists():
            config_path = alt_path
        else:
            print(f"Error: Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_nested(data, key_path):
    """Get a nested value using dot notation (e.g., 'analysis.corpus_dir')."""
    keys = key_path.split(".")
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def main():
    parser = argparse.ArgumentParser(description="Pipeline config helper")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--get", help="Get a config value by dot-separated key path")
    parser.add_argument("--check-skip", help="Check if a stage should be skipped (exit 0=skip, 1=run)")
    parser.add_argument("--section", help="Print an entire config section as key=value pairs")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.get:
        value = get_nested(config, args.get)
        if value is None:
            print(f"Warning: Key '{args.get}' not found in config", file=sys.stderr)
            print("")
        else:
            print(value)

    elif args.check_skip:
        skip_stages = config.get("skip_stages", {})
        should_skip = skip_stages.get(args.check_skip, False)
        sys.exit(0 if should_skip else 1)

    elif args.section:
        section = config.get(args.section, {})
        if isinstance(section, dict):
            for key, value in section.items():
                print(f"{key}={value}")
        else:
            print(section)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
