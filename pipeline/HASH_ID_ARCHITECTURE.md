# Hash ID Architecture - Complete Implementation

## Overview

The contamination analysis pipeline now uses **hash IDs** throughout to ensure robust and correct mapping between corpus texts and their embeddings. This eliminates the critical bug where positional indices broke due to sorted embeddings.

## The Problem (Before)

**Root Cause:**
- Stage 3 sorts embeddings by length for GPU efficiency
- Stage 4 stored positional indices (`corpus_idx`)
- Position in sorted parquet ≠ line in original JSONL
- Result: Wrong corpus texts matched to scores

**Example Bug:**
- Test: "Write a function to find palindromes"
- Top match (corpus_idx=1234988): Punjabi religious text
- Stored similarity: 0.7676 (impossibly high for unrelated texts)
- **Actual match:** Programming problem about palindromes

## The Solution (Now)

**Hash ID Architecture:**
Each corpus text has a unique SHA-256 hash ID that follows it through the entire pipeline:

```
JSONL → Embeddings → Analysis → Results
 (id)      (id)       (id)     (corpus_id)
```

## Pipeline Stages - Hash ID Flow

### Stage 1: Download
- Downloads raw data
- No ID handling needed

### Stage 2: Chunk and Sample ✅
**File:** `stages/02_chunk_and_sample.py`

**What it does:**
- Generates unique hash ID for each text using SHA-256
- Stores in JSONL output

**Code:**
```python
p_id = hashlib.sha256(p_clean.encode('utf-8')).hexdigest()
p_data = {
    "id": p_id,  # ✅ Unique hash ID created
    "text": p_clean,
    "source": source_name,
    "token_size": p_token_size
}
```

**Output:** JSONL file with `id` and `text` fields

### Stage 3: Create Embeddings ✅
**File:** `stages/03_create_embeddings_local_multigpu.py`

**What it does:**
- Loads IDs from JSONL along with texts
- Preserves IDs in parquet output
- **Sorts by length** but IDs stay with their embeddings

**Code:**
```python
# Load IDs from JSONL
self.ids.append(data['id'])

# Sort by length (embeddings AND IDs sorted together)
sorted_indices = sorted(range(len(self.lengths)),
                       key=lambda i: self.lengths[i], reverse=True)
self.paragraphs = [self.paragraphs[i] for i in sorted_indices]
self.ids = [self.ids[i] for i in sorted_indices]  # ✅ IDs follow embeddings

# Save to parquet
table = pa.table({
    'id': all_ids,          # ✅ Hash IDs preserved
    'embedding': list(all_embeddings)
})
```

**Output:** Parquet files with `id` and `embedding` columns

### Stage 4: Contamination Analysis ✅ FIXED
**File:** `stages/04_contamination_analysis.py`

**What changed:**
1. Added function to build corpus ID mapping at start
2. Loads IDs from all parquet files in order
3. Stores hash IDs in top-100 results

**Key Functions:**

#### Build ID Mapping (NEW)
```python
def build_corpus_id_mapping(data_dir):
    """Build mapping from global corpus position to hash ID."""
    corpus_ids = []
    con = duckdb.connect(':memory:')

    for pf_path in sorted(data_dir.glob("*.parquet")):
        # Load IDs from parquet
        result = con.execute(f"SELECT id FROM read_parquet('{pf_path}')").fetchall()
        file_ids = [row[0] for row in result]
        corpus_ids.extend(file_ids)

    return corpus_ids
```

#### Store Hash IDs (FIXED)
```python
# BEFORE (Wrong):
topk_matches.append({
    'rank': r,
    'score': float(score),
    'corpus_idx': int(idx),  # ❌ Positional index
})

# AFTER (Correct):
topk_matches.append({
    'rank': r,
    'score': float(score),
    'corpus_id': corpus_ids[idx],  # ✅ Hash ID from mapping
    'corpus_idx': int(idx),  # Keep for backwards compatibility
})
```

**Output:** JSON files with `corpus_id` (hash ID) and `corpus_idx` (for fallback)

### Stage 5: Finalize Results ✅ FIXED
**File:** `stages/05_finalize_results.py`

**What changed:**
- Builds both ID→text and idx→text mappings
- Prefers hash ID lookup, falls back to positional index

**Code:**
```python
def load_corpus_index(corpus_path):
    """Build mappings for both hash IDs and positional indices."""
    id_to_text = {}   # hash_id → text (primary)
    idx_to_text = {}  # position → text (fallback)

    for pf in sorted(parquet_files):
        # Load IDs and texts together
        result = con.execute(f"SELECT id, text FROM read_parquet('{pf}')").fetchall()
        for hash_id, text in result:
            id_to_text[hash_id] = text  # ✅ Hash ID mapping
            idx_to_text[current_idx] = text  # Fallback
            current_idx += 1

    return id_to_text, idx_to_text

def add_texts_to_results(results_dir, id_to_text, idx_to_text):
    """Add corpus texts using hash ID lookup."""
    for match in data.get('top_100', []):
        # Try hash ID first (preferred)
        corpus_id = match.get('corpus_id')
        if corpus_id and corpus_id in id_to_text:
            match['corpus_text'] = id_to_text[corpus_id]  # ✅ ID lookup
        else:
            # Fall back to positional index (for old results)
            corpus_idx = match.get('corpus_idx')
            if corpus_idx in idx_to_text:
                match['corpus_text'] = idx_to_text[corpus_idx]
```

**Output:** JSON files with correct `corpus_text` for each match

## Helper Scripts Updated

### `scripts/complete_aggregates.py` ✅
- Updated CSV generation to include `corpus_id` column
- Maintains backwards compatibility with `corpus_index`

### `scripts/regenerate_csvs.py` ✅
- Handles both `corpus_id` and `corpus_idx` fields
- Generates CSVs with hash ID column

## CSV Output Format

**New CSV columns:**
```
test_id,rank,cosine_similarity,corpus_id,corpus_index,test_text,corpus_text
```

- `corpus_id`: SHA-256 hash ID (primary identifier) ✅ NEW
- `corpus_index`: Positional index (kept for backwards compatibility)
- Both are included for maximum flexibility

## Benefits of Hash ID Architecture

✅ **Robust**: Works regardless of sorting, shuffling, or reordering
✅ **Correct**: IDs always point to the right text
✅ **Fast**: Direct hash lookup O(1) vs linear search
✅ **Debuggable**: IDs are meaningful, not just numbers
✅ **Future-proof**: Can change embedding process without breaking lookups
✅ **Backwards compatible**: Old results still work via positional index fallback

## Verification

To verify hash IDs are working correctly:

```bash
# Check that Stage 4 output contains corpus_id
cat results/*/mbpp_*/100_top100.json | python3 -m json.tool | grep corpus_id

# Should see:
# "corpus_id": "abc123def456..."  ✅ CORRECT
# NOT:
# "corpus_idx": 1234988  ❌ WRONG (only as fallback)

# Verify CSV has hash ID column
head -1 results/*/mbpp_*/all_top100_matches.csv

# Should include:
# ...,corpus_id,corpus_index,... ✅ CORRECT
```

## Migration Path

### For Existing Results
Use the temporary fix script:
```bash
python stages/05_finalize_results_fixed.py \
    --results-dir results/contamination_dolci_100pct_legacy \
    --corpus-jsonl data/dolmino_random_paragraphs.jsonl \
    --corpus-parquet-dir data/embeddings \
    --dataset-name dolci
```

### For New Runs
Just run the normal pipeline - hash IDs are now built-in:
```bash
./cluster/run_04_analysis.sh  # Stage 4 - Uses hash IDs automatically
./cluster/run_05_finalize.sh  # Stage 5 - Lookups via hash IDs
```

## Files Modified

1. **`stages/04_contamination_analysis.py`**
   - Added `build_corpus_id_mapping()` function
   - Modified merger to load and use corpus IDs
   - Stores `corpus_id` in top-100 results

2. **`stages/05_finalize_results.py`**
   - Updated `load_corpus_index()` to return both ID and index mappings
   - Updated `add_texts_to_results()` to prefer hash ID lookup

3. **`scripts/complete_aggregates.py`**
   - Updated CSV generation to include `corpus_id` column

4. **`scripts/regenerate_csvs.py`**
   - Updated CSV generation to include `corpus_id` column

## Summary

**The pipeline now follows this principle:**

> **Never use positional indices for corpus lookup. Always use hash IDs.**

Hash IDs are:
- Created in Stage 2 (chunking)
- Preserved in Stage 3 (embedding)
- Mapped in Stage 4 (analysis)
- Used for lookup in Stage 5 (finalization)

This ensures that corpus texts are **always correctly matched** to their similarity scores, regardless of how the data is sorted or processed.
