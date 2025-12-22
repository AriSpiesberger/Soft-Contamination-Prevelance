"""
Repair a truncated JSON array file.

Typical use case:
- A long-running generator is interrupted while rewriting the output JSON array.
- The file becomes invalid JSON (JSONDecodeError).
- This tool recovers the valid prefix of the array and writes a repaired, valid JSON file.

Example:
  python MuSR/musr_dataset_scripts/repair_json_array.py \
    --input MuSR/datasets/folder_.../halted_dataset_....json \
    --output MuSR/datasets/folder_.../halted_dataset_....repaired.json
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Add parent directory to path so we can import from src
SCRIPT_DIR = Path(__file__).parent.absolute()
MUSR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MUSR_DIR))

from src.utils.json_io import atomic_json_dump, load_json_array_tolerant


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair a JSON array file by recovering its valid prefix.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSON array file.")
    parser.add_argument("--output", type=str, required=True, help="Path to write the repaired JSON file.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output file if it already exists.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {out_path} (use --overwrite to replace)")

    data, info = load_json_array_tolerant(in_path)
    if info.warning:
        print(info.warning)

    atomic_json_dump(data, out_path, indent=2, ensure_ascii=False)
    print(f"Wrote repaired JSON: {out_path}")
    print(f"Recovered items: {info.recovered_items}")
    print(f"Ended cleanly: {info.ended_cleanly}")


if __name__ == "__main__":
    main()


