import json
import csv
import random
from pathlib import Path

annotations_dir = Path(r'c:\Users\arisp\Documents\Research\SDTD_Main\comparison_analysis\annotations\codeforces')
input_csv = Path(r'C:\Users\arisp\Downloads\downloads_clean_full\downloads_clean\codeforces_top100_combined.csv')
output_csv = Path(r'c:\Users\arisp\Documents\Research\SDTD_Main\comparison_analysis\codeforces_annotations_review.csv')

# Load original data for test_text/corpus_text
print("Loading original CSV...")
original_data = {}
with open(input_csv, 'r', encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # The annotation key format is: dataset__test_id__corpus_id
        # Need to match this format
        dataset = row.get('benchmark', row.get('source', row.get('dataset', '')))
        test_id = str(row.get('test_id', ''))
        corpus_id = str(row.get('corpus_id', ''))
        
        # Create multiple key formats to match
        key1 = f"{test_id}_{corpus_id}"
        key2 = f"{dataset}__{test_id}__{corpus_id}"
        
        original_data[key1] = row
        original_data[key2] = row

# Load annotations
print("Loading annotations...")
not_unrelated = []
unrelated = []

for json_file in annotations_dir.glob('*.json'):
    if json_file.stem.startswith('_'):
        continue
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        ann = data.get('annotation', {}) or {}
        key = json_file.stem
        parts = key.split('__')
        
        # Extract IDs - test_id uses underscore in filename but slash in CSV
        test_id_raw = parts[1] if len(parts) >= 2 else ''
        corpus_id = parts[2] if len(parts) >= 3 else ''
        
        # Convert test_id from underscore to slash format (e.g., 1572_F -> 1572/F)
        test_id_csv = test_id_raw.replace('_', '/')
        
        # Get text directly from annotation file (stored when prompt was built)
        test_text = data.get('test_text', '')
        corpus_text = data.get('corpus_text', '')
        
        # Fallback to CSV lookup if not in annotation file
        if not test_text or not corpus_text:
            lookup_key = f"{test_id_csv}_{corpus_id}"
            orig = original_data.get(lookup_key, {})
            test_text = test_text or (orig.get('test_text', '') or '')
            corpus_text = corpus_text or (orig.get('corpus_text', '') or '')
        
        row = {
            'key': key,
            'test_id': test_id_csv,
            'corpus_id': corpus_id,
            'match_type': ann.get('match_type', 'unknown'),
            'is_sd': ann.get('is_sd') or ann.get('is_duplicate', False),
            'confidence': ann.get('confidence', 0),
            'reasoning': (ann.get('reasoning', '') or '')[:500],
            'test_text': (test_text or '')[:1000],
            'corpus_text': (corpus_text or '')[:1000],
        }
        
        if ann.get('match_type') == 'unrelated':
            unrelated.append(row)
        else:
            not_unrelated.append(row)
    except Exception as e:
        pass

# Filter unrelated to only those with text, then sample 700
unrelated_with_text = [r for r in unrelated if r['test_text'] and r['corpus_text']]
print(f"  Unrelated with text: {len(unrelated_with_text)} (from {len(unrelated)} total)")
sample_size = min(700, len(unrelated_with_text))
random.shuffle(unrelated_with_text)
unrelated_sample = unrelated_with_text[:sample_size]

# Combine
all_rows = not_unrelated + unrelated_sample

# Write CSV
print(f"Writing {len(all_rows)} rows...")
fieldnames = ['key', 'test_id', 'corpus_id', 'match_type', 'is_sd', 'confidence', 'reasoning', 'test_text', 'corpus_text']
with open(output_csv, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"\nSaved to {output_csv}")
print(f"  Not unrelated: {len(not_unrelated)}")
print(f"  Unrelated sampled: {len(unrelated_sample)} (from {len(unrelated)})")
print(f"  Total rows: {len(all_rows)}")
