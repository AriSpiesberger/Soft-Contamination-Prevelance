import json
from pathlib import Path
from collections import Counter

d = Path(r'c:\Users\arisp\Documents\Research\SDTD_Main\comparison_analysis\annotations\codeforces')
total = 0
sd_count = 0
match_types = Counter()

for f in d.glob('*.json'):
    if f.stem.startswith('_'):
        continue
    try:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        ann = data.get('annotation', {}) or {}
        total += 1
        if ann.get('is_sd') or ann.get('is_duplicate'):
            sd_count += 1
        mt = ann.get('match_type', 'unknown')
        match_types[mt] += 1
    except Exception as e:
        pass

print(f'Total: {total:,}')
print(f'Semantic duplicates: {sd_count:,} ({100*sd_count/max(total,1):.1f}%)')
print(f'Match types:')
for mt, cnt in match_types.most_common():
    print(f'  {mt}: {cnt:,}')
