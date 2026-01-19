import json
from pathlib import Path
from collections import Counter

# Check MBPP annotations
mbpp_dir = Path(r'c:\Users\arisp\Documents\Research\SDTD_Main\comparison_analysis\annotations\mbpp')
mbpp_stats = {'total': 0, 'sd': 0, 'match_types': Counter()}
for f in mbpp_dir.glob('*.json'):
    if f.stem.startswith('_'): continue
    try:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        ann = data.get('annotation', {}) or {}
        mbpp_stats['total'] += 1
        if ann.get('match_type') != 'unrelated':
            mbpp_stats['sd'] += 1
        mbpp_stats['match_types'][ann.get('match_type', 'unknown')] += 1
    except: pass

# Check Codeforces annotations  
cf_dir = Path(r'c:\Users\arisp\Documents\Research\SDTD_Main\comparison_analysis\annotations\codeforces')
cf_stats = {'total': 0, 'sd': 0, 'match_types': Counter()}
for f in cf_dir.glob('*.json'):
    if f.stem.startswith('_'): continue
    try:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        ann = data.get('annotation', {}) or {}
        cf_stats['total'] += 1
        if ann.get('match_type') != 'unrelated':
            cf_stats['sd'] += 1
        cf_stats['match_types'][ann.get('match_type', 'unknown')] += 1
    except: pass

print('='*60)
print('MBPP ANNOTATIONS')
print('='*60)
print(f'Total: {mbpp_stats["total"]:,}')
sd_pct = 100*mbpp_stats["sd"]/max(mbpp_stats["total"],1)
print(f'Semantic Duplicates: {mbpp_stats["sd"]:,} ({sd_pct:.1f}%)')
print('Match types:')
for mt, cnt in mbpp_stats['match_types'].most_common():
    print(f'  {mt}: {cnt:,}')

print()
print('='*60)
print('CODEFORCES ANNOTATIONS')
print('='*60)
print(f'Total: {cf_stats["total"]:,}')
sd_pct = 100*cf_stats["sd"]/max(cf_stats["total"],1)
print(f'Semantic Duplicates: {cf_stats["sd"]:,} ({sd_pct:.1f}%)')
print('Match types:')
for mt, cnt in cf_stats['match_types'].most_common():
    print(f'  {mt}: {cnt:,}')

print()
print('='*60)
print('COMBINED TOTALS')
print('='*60)
total = mbpp_stats["total"] + cf_stats["total"]
total_sd = mbpp_stats["sd"] + cf_stats["sd"]
print(f'Total annotations: {total:,}')
print(f'Total semantic duplicates: {total_sd:,} ({100*total_sd/max(total,1):.1f}%)')
