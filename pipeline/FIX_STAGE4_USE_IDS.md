# CRITICAL FIX: Stage 4 Must Use Hash IDs, Not Positional Indices

## The Problem

**Current broken behavior:**
- Stage 4 stores `corpus_idx` (positional index in sorted parquet files)
- Parquet files are sorted by length during embedding
- Position 1234988 in parquet ≠ line 1234988 in JSONL
- Corpus texts don't match, scores are wrong

## The Solution

**Use hash IDs instead of positional indices:**
- Each corpus text has a unique hash ID (stored in both JSONL and parquet)
- Stage 4 should store the ID, not the position
- Stage 5 looks up text by ID - always works, regardless of sorting

## Required Changes in Stage 4

### File: `stages/04_contamination_analysis.py`

#### 1. Load IDs Along with Embeddings

**Current code** (~line 250):
```python
def load_single_parquet(pf_path, con, max_rows_per_load=1_500_000):
    """Load a single parquet file and return embeddings."""
    # ...
    result = con.execute(f"SELECT embedding FROM read_parquet('{pf_path}')").fetchall()
    embeddings = np.array([r[0] for r in result])
    return embeddings
```

**Fixed code:**
```python
def load_single_parquet(pf_path, con, max_rows_per_load=1_500_000):
    """Load a single parquet file and return embeddings AND IDs."""
    # ...
    result = con.execute(f"SELECT id, embedding FROM read_parquet('{pf_path}')").fetchall()
    ids = [r[0] for r in result]
    embeddings = np.array([r[1] for r in result])
    return ids, embeddings  # Return both!
```

#### 2. Keep Track of IDs During Processing

**Current code** (~line 700-800):
```python
# Process corpus chunk
corpus_chunk = load_single_parquet(pf_path, con)
corpus_embeddings_gpu = torch.from_numpy(corpus_chunk).to(device, dtype=torch.float16)
```

**Fixed code:**
```python
# Process corpus chunk
corpus_ids, corpus_chunk = load_single_parquet(pf_path, con)
corpus_embeddings_gpu = torch.from_numpy(corpus_chunk).to(device, dtype=torch.float16)
# Keep corpus_ids available for top-K selection
```

#### 3. Store IDs Instead of Indices in Top-100

**Current code** (~line 900-950):
```python
# Find top-100
top_indices = np.argpartition(all_similarities, -100)[-100:]
top_indices = top_indices[np.argsort(all_similarities[top_indices])[::-1]]

top_100 = []
for rank, idx in enumerate(top_indices, 1):
    top_100.append({
        'rank': rank,
        'score': float(all_similarities[idx]),
        'corpus_idx': int(idx)  # ❌ WRONG - positional index
    })
```

**Fixed code:**
```python
# Find top-100
top_indices = np.argpartition(all_similarities, -100)[-100:]
top_indices = top_indices[np.argsort(all_similarities[top_indices])[::-1]]

top_100 = []
for rank, idx in enumerate(top_indices, 1):
    top_100.append({
        'rank': rank,
        'score': float(all_similarities[idx]),
        'corpus_id': corpus_ids[idx]  # ✅ CORRECT - hash ID
    })
```

## Updated Stage 5

With this fix, Stage 5 becomes much simpler:

```python
def add_texts_to_results(results_dir, corpus_jsonl_path):
    """Add corpus texts using ID lookup (simple!)."""
    # Build ID -> text mapping
    id_to_text = {}
    with open(corpus_jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            id_to_text[data['id']] = data['text']

    # Update all JSONs
    for json_file in results_dir.rglob("*top100.json"):
        with open(json_file) as f:
            data = json.load(f)

        for match in data['top_100']:
            corpus_id = match['corpus_id']  # Get the hash ID
            match['corpus_text'] = id_to_text[corpus_id]  # Direct lookup!

        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
```

## Benefits

✅ **Robust**: Works regardless of sorting/shuffling
✅ **Simple**: No complex index mapping needed
✅ **Fast**: Direct hash lookup vs. positional mapping
✅ **Correct**: IDs always point to the right text
✅ **Debuggable**: IDs are meaningful, not just numbers

## Migration Plan

1. **Immediate**: Use the temporary fix script (`05_finalize_results_fixed.py`) to correct existing results
2. **Next run**: Update Stage 4 with the changes above
3. **Future**: All runs will use hash IDs automatically

## Verification

After fixing Stage 4, verify it works:
```bash
# Run a small test
PIPELINE_CONFIG=configs/test.yaml ./cluster/run_04_analysis.sh

# Check that top-100 JSONs contain 'corpus_id' not 'corpus_idx'
cat results/*/mbpp_*/100_top100.json | python3 -m json.tool | grep corpus_id

# Should see: "corpus_id": "abc123def456..."
# NOT: "corpus_idx": 1234988
```

## Summary

**Never use positional indices for corpus lookup!**
**Always use hash IDs!**

This is a fundamental principle: IDs are stable, positions are not.
