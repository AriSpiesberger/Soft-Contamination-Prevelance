#!/usr/bin/env python3
"""
Create a CSV with 5 clean semantic duplicate pairs per test point.

Each pair = (English synonym input, Python semantic output)

Excludes buggy MBPP tasks: 229, 438, 461, 579, 769, 802

Sources:
- mbpp-python-dupes/output/individual/task_*.json (Python code variations)
- mbpp-text-dupes/english-variations/output/individual/task_*.json (English text variations)
"""

import json
import csv
from pathlib import Path

# Paths
PYTHON_DUPES_DIR = Path(__file__).parent / "mbpp-python-dupes" / "output" / "individual"
ENGLISH_VAR_DIR = Path(__file__).parent / "mbpp-text-dupes" / "english-variations" / "output" / "individual"
OUTPUT_CSV = Path(__file__).parent / "semantic_pairs_full.csv"

# Buggy tasks to exclude (from README)
BUGGY_TASKS = {229, 438, 461, 579, 769, 802}

def load_json_safe(path: Path) -> dict:
    """Load JSON file, return empty dict on failure."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def get_all_clean_task_ids() -> list[int]:
    """Get ALL clean task IDs that have complete data in both sources."""
    clean_ids = []
    
    python_files = list(PYTHON_DUPES_DIR.glob("task_*.json"))
    
    for pf in sorted(python_files, key=lambda x: int(x.stem.split('_')[1])):
        task_id = int(pf.stem.split('_')[1])
        
        # Skip buggy tasks
        if task_id in BUGGY_TASKS:
            continue
            
        english_file = ENGLISH_VAR_DIR / f"task_{task_id}.json"
        
        if not english_file.exists():
            continue
            
        python_data = load_json_safe(pf)
        english_data = load_json_safe(english_file)
        
        # Need successful python_1 through python_5
        python_complete = all(
            python_data.get(f"python_{i}_status") == "success"
            for i in range(1, 6)
        )
        
        # Need at least 5 successful text variations (paraphrases)
        text_vars = ["para1", "para2", "para3", "para4", "para5"]
        english_complete = all(
            english_data.get(f"text_{var}_status") == "success"
            for var in text_vars
        )
        
        if python_complete and english_complete:
            clean_ids.append(task_id)
    
    return clean_ids

def create_csv(task_ids: list[int]):
    """Create CSV with 5 semantic pairs per test point."""
    
    fieldnames = [
        "task_id",
        "pair_num",
        "original_text",
        "original_code",
        "english_synonym_input",    # The text variation
        "python_semantic_output",   # The Python code variation
    ]
    
    rows = []
    
    for task_id in task_ids:
        python_path = PYTHON_DUPES_DIR / f"task_{task_id}.json"
        english_path = ENGLISH_VAR_DIR / f"task_{task_id}.json"
        
        python_data = load_json_safe(python_path)
        english_data = load_json_safe(english_path)
        
        original_text = english_data.get("text", python_data.get("prompt", ""))
        original_code = python_data.get("code_python", "")
        
        # Create 5 pairs using para1-5 with python_1-5
        text_vars = ["para1", "para2", "para3", "para4", "para5"]
        
        for i, text_var in enumerate(text_vars, 1):
            row = {
                "task_id": task_id,
                "pair_num": i,
                "original_text": original_text,
                "original_code": original_code,
                "english_synonym_input": english_data.get(f"text_{text_var}", ""),
                "python_semantic_output": python_data.get(f"python_{i}", ""),
            }
            rows.append(row)
    
    # Write CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Created {OUTPUT_CSV}")
    print(f"Excluded buggy tasks: {sorted(BUGGY_TASKS)}")
    print(f"Clean tasks: {len(task_ids)}")
    print(f"Total rows: {len(rows)} (5 pairs × {len(task_ids)} tasks)")

if __name__ == "__main__":
    print("Finding ALL clean task IDs (excluding buggy tasks)...")
    task_ids = get_all_clean_task_ids()
    print(f"Found {len(task_ids)} clean tasks")
    
    if task_ids:
        create_csv(task_ids)
    else:
        print("No complete tasks found!")
