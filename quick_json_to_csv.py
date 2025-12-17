#!/usr/bin/env python3
"""Quick conversion of matches JSON to CSV format"""
import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset

def load_benchmark_texts(name, mode):
    """Load benchmark texts for column in CSV."""
    if name == 'musr':
        ds = load_dataset("TAUR-Lab/MuSR")
        data = []
        for split in ds:
            for idx, item in enumerate(ds[split]):
                inp = item.get('narrative', item.get('question', ''))
                out = item.get('answer', '')
                data.append({'id': f"{split}_{idx}", 'input': inp, 'output': out})

    elif name == 'mbpp':
        try:
            ds = load_dataset("google-research-datasets/mbpp", "full")
        except:
            ds = load_dataset("google-research-datasets/mbpp", "sanitized")

        data = []
        for split in ds:
            for idx, item in enumerate(ds[split]):
                task_id = str(item.get('task_id', f"{split}_{idx}"))
                inp = item.get('text', item.get('prompt', ''))
                out = item.get('code', item.get('canonical_solution', ''))
                data.append({'id': task_id, 'input': inp, 'output': out})
    else:
        raise ValueError(f"Unknown: {name}")

    texts, ids = [], []
    for item in data:
        if mode == 'input':
            texts.append(item['input'])
        elif mode == 'output':
            texts.append(item['output'])
        elif mode in ['input_output', '+']:
            texts.append(f"{item['input']}\n\n{item['output']}")
        else:
            texts.append(f"{item['input']}\n\n{item['output']}")
        ids.append(item['id'])

    return texts, ids


def convert_matches_to_csv(json_path, output_path, benchmark_name, mode):
    """Convert matches JSON to CSV."""
    print(f"Processing {json_path.name}...")

    # Load matches
    with open(json_path) as f:
        matches = json.load(f)

    # Load benchmark texts
    bench_texts, bench_ids = load_benchmark_texts(benchmark_name, mode)

    # Build CSV rows
    rows = []
    for bench_idx, bench_matches in enumerate(matches):
        bench_id = bench_ids[bench_idx] if bench_idx < len(bench_ids) else f'item_{bench_idx}'
        bench_text = bench_texts[bench_idx] if bench_idx < len(bench_texts) else ''

        for match in bench_matches[:100]:  # Top 100
            rows.append({
                'benchmark_id': bench_id,
                'benchmark_text': bench_text[:500],  # Truncate for readability
                'rank': match.get('rank', 0),
                'similarity': match.get('score', match.get('similarity', 0.0)),
                'corpus_file': match.get('file', ''),
                'corpus_source': match.get('source', 'unknown'),
                'corpus_text': match.get('text', '')[:500]  # Already truncated in JSON
            })

    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  ✅ Saved {len(rows)} rows to {output_path.name}")


if __name__ == "__main__":
    input_dir = Path("contamination_analysis/data/top_matches")
    output_dir = Path("contamination_analysis/data/top_matches")

    # Process each matches file
    configs = [
        ('musr_input_matches.json', 'musr', 'input'),
        ('musr_output_matches.json', 'musr', 'output'),
        ('mbpp_input_matches.json', 'mbpp', 'input'),
        ('mbpp_output_matches.json', 'mbpp', 'output'),
        ('mbpp_+_matches.json', 'mbpp', 'input_output'),
    ]

    for json_file, bench_name, mode in configs:
        json_path = input_dir / json_file
        if json_path.exists():
            csv_name = json_file.replace('_matches.json', '_top100.csv').replace('_+', '_input_output')
            csv_path = output_dir / csv_name
            convert_matches_to_csv(json_path, csv_path, bench_name, mode)
        else:
            print(f"Skipping {json_file} (not found)")

    print("\n✅ All CSVs generated!")
