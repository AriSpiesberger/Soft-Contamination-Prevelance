import pandas as pd

# Load both files
top100 = pd.read_csv(r'C:\Users\arisp\Downloads\codeforces_top100_classified_gptoss_v2.csv')
sample100 = pd.read_csv(r'C:\Users\arisp\Downloads\codeforces_sample100_classified_gptoss_v2.csv')

def analyze(df, name):
    print(f"\n{'='*60}")
    print(f"=== {name} ===")
    print(f"{'='*60}")
    print(f"Total rows: {len(df)}")
    print(f"Unique test_ids: {df['test_id'].nunique()}")
    
    # Semantic duplicates that are NOT "related"
    # (i.e., predicted_is_duplicate=True AND predicted_category != 'related')
    dupes_not_related = df[
        (df['predicted_is_duplicate'] == True) & 
        (df['predicted_category'] != 'related')
    ]
    
    print(f"\n--- Semantic duplicates (excluding 'related' category) ---")
    print(f"Total such pairs: {len(dupes_not_related)}")
    print(f"Category breakdown:")
    print(dupes_not_related['predicted_category'].value_counts())
    
    # Test points with at least 1 such duplicate
    tests_with_dupe = dupes_not_related['test_id'].nunique()
    total_tests = df['test_id'].nunique()
    pct = 100 * tests_with_dupe / total_tests
    
    print(f"\nTest points with at least 1 semantic duplicate (related excluded):")
    print(f"  {tests_with_dupe} / {total_tests} = {pct:.2f}%")
    
    # Also show ALL duplicates for comparison
    all_dupes = df[df['predicted_is_duplicate'] == True]
    print(f"\n--- All semantic duplicates (including 'related') ---")
    print(f"Total pairs: {len(all_dupes)}")
    print(f"Category breakdown:")
    print(all_dupes['predicted_category'].value_counts())
    
    tests_with_any_dupe = all_dupes['test_id'].nunique()
    print(f"\nTest points with at least 1 semantic duplicate (any category):")
    print(f"  {tests_with_any_dupe} / {total_tests} = {100*tests_with_any_dupe/total_tests:.2f}%")

with open('results.txt', 'w') as f:
    import sys
    old_stdout = sys.stdout
    sys.stdout = f
    analyze(top100, "TOP100 (highest similarity)")
    analyze(sample100, "SAMPLE100 (random sample)")
    sys.stdout = old_stdout

print("Results written to results.txt")
