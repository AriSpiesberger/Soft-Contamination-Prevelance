import csv
import json
from pathlib import Path

# Get sample from original CSV
orig_csv = r'C:\Users\arisp\Downloads\downloads_clean_full\downloads_clean\codeforces_top100_combined.csv'
with open(orig_csv, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    row = next(reader)
    orig_key = f"{row.get('test_id', '')}_{row.get('corpus_id', '')}"
    print('Original CSV key format:', orig_key[:100])
    print('test_id:', row.get('test_id'))
    print('corpus_id:', row.get('corpus_id')[:50] if row.get('corpus_id') else None)

# Get sample annotation key parsed
ann_dir = Path(r'c:\Users\arisp\Documents\Research\SDTD_Main\comparison_analysis\annotations\codeforces')
for f in list(ann_dir.glob('*.json'))[:3]:
    if not f.stem.startswith('_'):
        parts = f.stem.split('__')
        print(f'\nAnnotation file: {f.stem}')
        print(f'Parts count: {len(parts)}')
        if len(parts) >= 3:
            test_id = parts[1]
            corpus_id = parts[2]
            ann_key = f'{test_id}_{corpus_id}'
            print(f'Parsed key: {ann_key[:100]}')
