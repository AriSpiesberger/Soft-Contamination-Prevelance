

import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# PART 1: HYDRATION (Text Lookup)
# -----------------------------------------------------------------------------

def load_corpus_text_mapping(corpus_jsonl_path):
    """Build mapping from ID to text."""
    corpus_jsonl_path = Path(corpus_jsonl_path)
    if not corpus_jsonl_path.exists():
        raise ValueError(f"Corpus JSONL file not found: {corpus_jsonl_path}")

    print(f"\nBuilding ID -> text mapping from {corpus_jsonl_path}...")
    id_to_text = {}
    
    with open(corpus_jsonl_path) as f:
        for line in tqdm(f, desc="Indexing corpus texts"):
            try:
                data = json.loads(line)
                # Standardize ID to string to ensure matching works
                id_val = str(data.get('id', ''))
                if id_val:
                    id_to_text[id_val] = data.get('text', '')
            except json.JSONDecodeError:
                continue

    print(f"✓ Built mapping for {len(id_to_text):,} IDs")
    return id_to_text

def hydrate_jsons(results_dir, id_to_text):
    """Add corpus texts to results using direct 'corpus_id' lookup."""
    results_dir = Path(results_dir)
    json_files = list(results_dir.rglob("*top1000.json")) + list(results_dir.rglob("*top_1000.json")) + list(results_dir.rglob("*top100.json"))
    
    print(f"\nFound {len(json_files)} result files to update")
    
    updated_count = 0
    missing_ids = set()

    for json_file in tqdm(json_files, desc="Hydrating results"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            modified = False
            for match in data.get('top_1000', []) or data.get('top_100', []):
                c_id = match.get('corpus_id')
                if c_id:
                    c_id_str = str(c_id)
                    if c_id_str in id_to_text:
                        match['corpus_text'] = id_to_text[c_id_str]
                        modified = True
                    else:
                        if len(missing_ids) < 5: 
                            print(f"⚠️ Missing text for ID: {c_id}")
                        missing_ids.add(c_id_str)

            if modified:
                with open(json_file, 'w') as f:
                    json.dump(data, f, indent=2)
                updated_count += 1

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    print(f"✓ Updated {updated_count} files with corpus texts")
    return json_files

# -----------------------------------------------------------------------------
# PART 2: CSV GENERATION (With BOTH texts)
# -----------------------------------------------------------------------------

def generate_csvs_explicit(results_dir):
    """Read the hydrated JSONs and generate the master CSVs manually."""
    results_dir = Path(results_dir)
    # Identify benchmark directories (exclude system dirs)
    mode_dirs = [d for d in results_dir.iterdir() 
                 if d.is_dir() and d.name not in ['checkpoints', 'logs', 'temp_similarities']]

    print(f"\nGenerating CSVs for {len(mode_dirs)} benchmarks...")

    for mode_dir in mode_dirs:
        print(f"Processing {mode_dir.name}...")
        json_files = list(mode_dir.glob("*top1000.json")) + list(mode_dir.glob("*top_1000.json")) + list(mode_dir.glob("*top100.json"))
        
        if not json_files:
            continue

        all_rows = []
        for jf in tqdm(json_files, desc=f"  Reading {mode_dir.name}"):
            try:
                with open(jf) as f:
                    data = json.load(f)
                
                # Extract Test Metadata
                test_id = data.get('test_id', 'unknown')
                test_text = data.get('test_text', '')  # <-- CAPTURED HERE
                elo_bin = data.get('elo_bin')  # <-- NEW: Codeforces ELO bin
                rating = data.get('rating')    # <-- NEW: Codeforces rating
                
                # Flatten the top 1000/100 list (support both naming conventions)
                for rank, match in enumerate(data.get('top_1000', []) or data.get('top_100', []), 1):
                    row = {
                        'benchmark': mode_dir.name,
                        'test_id': test_id,
                        'elo_bin': elo_bin,           # <-- NEW
                        'rating': rating,             # <-- NEW
                        'test_text': test_text,       # <-- ADDED
                        'rank': rank,
                        'score': match.get('score'),
                        'corpus_id': match.get('corpus_id'),
                        'corpus_text': match.get('corpus_text', '') # Already hydrated
                    }
                    all_rows.append(row)
            except Exception as e:
                print(f"Error reading {jf}: {e}")

        if all_rows:
            df = pd.DataFrame(all_rows)
            # Define specific column order for readability
            cols = ['benchmark', 'test_id', 'elo_bin', 'rating', 'test_text', 'rank', 'score', 'corpus_id', 'corpus_text']
            # Reorder if keys exist, otherwise just use what we have
            df = df[[c for c in cols if c in df.columns]]
            
            # Sort by test_id and rank for the full file
            df_all = df.sort_values(['test_id', 'rank'])
            
            # Save all matches (up to 1000 per test point)
            output_full = mode_dir / "all_top1000_matches.csv"
            df_all.to_csv(output_full, index=False, escapechar='\\', quoting=1)
            
            # Create top 1000 global: highest similarity scores across ALL test points
            df_top1000 = df.nlargest(1000, 'score').sort_values('score', ascending=False)
            output_top = mode_dir / "top_1000_contamination.csv"
            df_top1000.to_csv(output_top, index=False, escapechar='\\', quoting=1)
            
            print(f"  ✅ all_top1000_matches.csv: {len(df_all):,} rows | top_1000_contamination.csv: {len(df_top1000):,} rows")
        else:
            print("  ⚠️ No data found to convert to CSV.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--corpus-jsonl', required=True)
    parser.add_argument('--corpus-parquet-dir', help="Ignored")
    parser.add_argument('--dataset-name', default='dataset')
    args = parser.parse_args()

    # Step 1: Load Text Map
    id_to_text = load_corpus_text_mapping(args.corpus_jsonl)

    # Step 2: Hydrate JSONs
    print("\n" + "="*60 + "\nSTEP 2: Hydrating JSONs\n" + "="*60)
    hydrate_jsons(args.results_dir, id_to_text)

    # Step 3: Generate CSVs (Explicitly!)
    print("\n" + "="*60 + "\nSTEP 3: Generating CSVs\n" + "="*60)
    generate_csvs_explicit(args.results_dir)

    print("\n✅ FINALIZATION COMPLETE!")

if __name__ == "__main__":
    main()
