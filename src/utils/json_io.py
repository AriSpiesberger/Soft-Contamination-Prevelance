"""
Utilities for robust JSON IO on large files produced by long-running generation scripts.

Why this exists:
- Our generators periodically rewrite a single large JSON array file.
- If the process is interrupted during write, the output file can become truncated and
  thus invalid JSON (common on Windows).

This module provides:
- atomic_json_dump: write JSON via temp file + atomic replace (prevents truncation)
- load_json_array_tolerant: load a JSON array; if invalid, recover the valid prefix
  (useful to resume generation and to verify partially-written outputs)
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any, List, Optional, Tuple


@dataclass
class JsonArrayRecoveryInfo:
    recovered_items: int
    ended_cleanly: bool
    warning: Optional[str] = None


def atomic_json_dump(data: Any, path: Path, *, indent: int = 2, ensure_ascii: bool = False) -> None:
    """
    Atomically write JSON to `path` by writing to a temp file in the same directory
    and then replacing the destination.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Same-directory temp file to ensure os.replace is atomic on the same filesystem.
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        # If anything failed before replace, ensure temp is removed.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def load_json_array_tolerant(path: Path) -> Tuple[List[Any], JsonArrayRecoveryInfo]:
    """
    Load a JSON array from `path`.

    If the file is invalid JSON (e.g., truncated mid-write), attempt to recover the
    valid prefix of the top-level array and return that prefix as a list.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON array at top-level, got: {type(data)}")
            return data, JsonArrayRecoveryInfo(recovered_items=len(data), ended_cleanly=True)
        except JSONDecodeError as e:
            # Fall back to prefix recovery.
            f.seek(0)
            recovered, info = _recover_json_array_prefix(f)
            info.warning = (
                f"JSONDecodeError while parsing '{path}': {e}. "
                f"Recovered {info.recovered_items} complete items from the valid prefix."
            )
            return recovered, info


def _recover_json_array_prefix(f) -> Tuple[List[Any], JsonArrayRecoveryInfo]:
    """
    Recover the valid prefix of a JSON array from an open text file handle.

    This uses JSONDecoder.raw_decode iteratively so it can stop cleanly at EOF
    even if the last element is incomplete.
    """
    decoder = json.JSONDecoder()
    buf = ""
    idx = 0
    items: List[Any] = []

    def _need_more() -> bool:
        return idx >= len(buf) - 1024

    def _read_more() -> bool:
        nonlocal buf
        chunk = f.read(1024 * 1024)  # 1MB
        if not chunk:
            return False
        buf += chunk
        return True

    # Prime buffer
    if not _read_more():
        return [], JsonArrayRecoveryInfo(recovered_items=0, ended_cleanly=False, warning="Empty file.")

    # Skip leading whitespace and require '['
    while True:
        while idx < len(buf) and buf[idx].isspace():
            idx += 1
        if idx < len(buf):
            break
        if not _read_more():
            return [], JsonArrayRecoveryInfo(recovered_items=0, ended_cleanly=False, warning="EOF before '['.")

    if buf[idx] != "[":
        return [], JsonArrayRecoveryInfo(
            recovered_items=0,
            ended_cleanly=False,
            warning="Input does not start with a JSON array '['.",
        )
    idx += 1  # consume '['

    ended_cleanly = False
    while True:
        # Skip whitespace + optional commas
        while True:
            while idx < len(buf) and buf[idx].isspace():
                idx += 1
            if idx < len(buf) and buf[idx] == ",":
                idx += 1
                continue
            break

        # Ensure we have some data
        if idx >= len(buf):
            if not _read_more():
                break  # EOF (likely truncated)
            continue

        # End of array
        if buf[idx] == "]":
            ended_cleanly = True
            break

        # Decode next element
        while True:
            try:
                obj, end = decoder.raw_decode(buf, idx)
                items.append(obj)
                idx = end
                break
            except JSONDecodeError:
                # If we can read more, do so; otherwise we're truncated mid-element.
                if not _read_more():
                    return items, JsonArrayRecoveryInfo(recovered_items=len(items), ended_cleanly=False)

        # Keep buffer bounded
        if idx > 2 * 1024 * 1024:
            buf = buf[idx:]
            idx = 0

        # Opportunistically read more to avoid too many small reads
        if _need_more():
            _read_more()

    return items, JsonArrayRecoveryInfo(recovered_items=len(items), ended_cleanly=ended_cleanly)


