import pandas as pd

# MBPP annotations (high SD rate)
mbpp = pd.read_csv(r'C:\Users\arisp\Downloads\mbpp_annotations_full(1).csv')

# Codeforces annotations (low SD rate)
cf = pd.read_csv(r'C:\Users\arisp\Documents\Research\SDTD_Main\comparison_analysis\codeforces_annotations_review.csv')

print('=== MBPP (16.9% SD rate) ===')
print(f'Total rows: {len(mbpp)}')
print(f'Columns: {list(mbpp.columns)}')

if 'similarity' in mbpp.columns:
    print(f'\nSimilarity stats:')
    print(f'  Mean: {mbpp["similarity"].mean():.4f}')
    print(f'  Median: {mbpp["similarity"].median():.4f}')
    print(f'  Min: {mbpp["similarity"].min():.4f}')
    print(f'  Max: {mbpp["similarity"].max():.4f}')
    
    # Compare SD vs non-SD
    sd_mbpp = mbpp[mbpp['match_type'] != 'unrelated']
    non_sd_mbpp = mbpp[mbpp['match_type'] == 'unrelated']
    print(f'  SD similarity mean: {sd_mbpp["similarity"].mean():.4f}')
    print(f'  Non-SD similarity mean: {non_sd_mbpp["similarity"].mean():.4f}')
else:
    print('No similarity column')

print()
print('=== Codeforces ===')
print(f'Total rows: {len(cf)}')
print(f'Columns: {list(cf.columns)}')
sd_cf = cf[cf['match_type'] != 'unrelated']
non_sd_cf = cf[cf['match_type'] == 'unrelated']
print(f'SD (non-unrelated): {len(sd_cf)} ({100*len(sd_cf)/len(cf):.1f}%)')
print(f'Unrelated: {len(non_sd_cf)} ({100*len(non_sd_cf)/len(cf):.1f}%)')
