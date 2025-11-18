# Similarity Metrics Documentation

This document describes all similarity metrics computed for semantic duplicates (SDs).

## Overview

Each generated SD is compared to its original text using multiple NLP-level similarity metrics. These metrics are stored in a JSON `metrics` field in the output Parquet file. We also compute and store embeddings for both original and SD texts.

## Output Schema

```python
{
    "source_dataset": str,           # Dataset name (gsm8k, codeforces, allenai)
    "sd_level": int,                 # Level (1, 2, etc.)
    "sd_variant": str,               # Variant name (lexical_maximalist, etc.)
    "model_used": str,               # LLM model used for generation
    "original_text": str,            # Original text
    "sd_text": str,                  # Generated semantic duplicate
    "original_embedding": list[float], # Embedding vector for original
    "sd_embedding": list[float],     # Embedding vector for SD
    "embedding_model": str,          # Embedding model name
    "metrics": str,                  # JSON string with all metrics (see below)
    "additional_info": str,          # JSON with dataset-specific fields
    "timestamp": str,                # ISO timestamp
}
```

## Metrics (JSON object)

All metrics are computed between `original_text` and `sd_text`.

### N-gram Overlaps

**Field**: `ngram_overlaps_pct` (list of 5 floats)

Jaccard similarity for n-grams (n=1 to 5), as percentages (0-100).

```python
ngram_overlaps_pct: [45.2, 18.5, 8.1, 3.2, 0.8]
# [unigram, bigram, trigram, 4-gram, 5-gram]
```

**Formula**: Jaccard = |A ∩ B| / |A ∪ B| × 100

**Interpretation**:
- **High overlap (>50%)**: Strong surface-form similarity (not Type C duplicate)
- **Medium overlap (20-50%)**: Partial paraphrase, some lexical change
- **Low overlap (<20%)**: Strong paraphrase, Type C duplicate
- **Expected for our SDs**: Bigram <30%, trigram <10%

**Example**:
- Lexical Maximalist: ~18% bigram (synonym substitution)
- Syntactic Restructuring: ~5% bigram (structure change)
- Abstractive Paraphrase: ~3% bigram (rewrite from scratch)
- Compositional: ~0% bigram (aggressive transformation)

---

### Character N-gram Overlaps

**Field**: `char_ngram_overlaps_pct` (list of 5 floats)

Jaccard similarity for character n-grams (n=1 to 5), as percentages (0-100).

```python
char_ngram_overlaps_pct: [82.8, 43.4, 22.4, 13.4, 9.0]
# [1-char, 2-char, 3-char, 4-char, 5-char]
```

**Purpose**: Detects substring-level copying even when word boundaries change.

**Interpretation**:
- Higher than word n-grams (captures morphological variants)
- Example: "babysitting" vs "babysit" shares many character trigrams
- Useful for detecting copy-paste with minor edits

---

### ROUGE-L F-measure

**Field**: `rouge_l_f` (float, 0-1)

F1-score based on longest common subsequence (LCS).

```python
rouge_l_f: 0.5672
```

**Formula**:
- Precision = LCS_length / SD_words
- Recall = LCS_length / Original_words
- F1 = 2 × (P × R) / (P + R)

**Interpretation**:
- **0.8-1.0**: Very high sequential similarity
- **0.5-0.8**: Moderate similarity, some reordering
- **0.2-0.5**: Significant restructuring
- **<0.2**: Completely different word order

**Comparison to n-grams**: ROUGE-L is order-sensitive but allows gaps (skip-grams implicitly).

---

### Edit Distance (Normalized)

**Field**: `edit_distance_norm` (float, 0-1)

Normalized Levenshtein distance at word level.

```python
edit_distance_norm: 0.8333
```

**Formula**: word_edits / max(len(original_words), len(sd_words))

**Interpretation**:
- **0.0**: Identical texts
- **0.5**: Half the words need to be changed
- **1.0**: Completely different
- **Expected for SDs**: 0.5-0.9 (significant but not complete change)

**Note**: Measures minimum insertions/deletions/substitutions needed to transform one text to another.

---

### TF-IDF Cosine Similarity

**Field**: `tfidf_cosine` (float, 0-1)

Cosine similarity of TF-IDF weighted term vectors.

```python
tfidf_cosine: 0.0000
```

**Formula**:
- TF = term frequency in document
- IDF = log(num_docs / docs_containing_term)
- Cosine = dot(tfidf1, tfidf2) / (norm(tfidf1) × norm(tfidf2))

**Interpretation**:
- Weights rare words more heavily than common words
- More informative than raw word overlap
- **Note**: For pairwise comparison (2 documents), IDF is limited; expect low values

**Why often 0.0**: With only 2 documents in the "corpus", most shared terms have IDF=0, resulting in zero similarity.

---

### Token-level Jaccard

**Field**: `jaccard_token` (float, 0-1)

Set-based Jaccard similarity of word tokens (case-insensitive).

```python
jaccard_token: 0.3636
```

**Formula**: |tokens1 ∩ tokens2| / |tokens1 ∪ tokens2|

**Interpretation**:
- Simpler than n-grams (ignores order)
- **0.8-1.0**: Very similar vocabulary
- **0.5-0.8**: Moderate lexical overlap
- **0.2-0.5**: Significant vocabulary change
- **<0.2**: Mostly different words

**Comparison**: Equivalent to `ngram_overlaps_pct[0]/100` (unigram)

---

### Number Preservation

**Fields**:
- `number_preservation` (bool): True if all numbers preserved
- `number_precision` (float, 0-1): Fraction of SD numbers in original
- `number_recall` (float, 0-1): Fraction of original numbers in SD

```python
number_preservation: True
number_precision: 1.0
number_recall: 1.0
```

**Purpose**: Critical for math/code domains where numerical values must be exact.

**Regex**: `\b\d+\.?\d*\b` (matches integers and decimals)

**Interpretation**:
- **For GSM8K**: Should always be True (preserve 48, $12, 50 minutes)
- **Edge case**: "half" → "50%" may flag as False (semantic equivalence but different tokens)

**Why important**: Changing "48" to "49" completely changes the problem answer.

---

### Embedding Cosine Similarity

**Fields**:
- `cosine_similarity` (float, -1 to 1, typically 0-1)
- `cosine_similarity_model` (str): Embedding model name

```python
cosine_similarity: 0.9437
cosine_similarity_model: "openrouter/qwen/qwen3-embedding-8b"
```

**Purpose**: Semantic similarity at the representation level (not surface form).

**Model**: Qwen3-Embedding-8B (4096-dimensional vectors) via OpenRouter

**Interpretation**:
- **0.95-1.0**: Near-perfect semantic equivalence
- **0.85-0.95**: High semantic similarity (our target for SDs)
- **0.7-0.85**: Moderate similarity
- **<0.7**: Low semantic similarity (not good SDs)

**Expected for Type C duplicates**: High embedding cosine (>0.85) + Low n-gram overlap (<30%)

---

### Length Ratio

**Field**: `length_ratio` (float)

Ratio of word counts: SD_words / Original_words

```python
length_ratio: 1.3548
```

**Interpretation**:
- **1.0**: Same length
- **>1.0**: SD is longer (more verbose)
- **<1.0**: SD is shorter (more concise)
- **Typical range**: 0.8-1.5 for good paraphrases

**Why useful**: Detects if transformation significantly expands/contracts text.

---

## Quality Thresholds for Level 1 SDs

Based on our taxonomy (Type C duplicates):

| Metric | Target Range | Purpose |
|--------|--------------|---------|
| `ngram_overlaps_pct[1]` (bigram) | <30% | Surface form disruption |
| `ngram_overlaps_pct[2]` (trigram) | <10% | Stronger disruption |
| `cosine_similarity` | >0.85 | Semantic preservation |
| `number_preservation` | True | Math problem correctness |
| `rouge_l_f` | 0.2-0.6 | Moderate restructuring |
| `edit_distance_norm` | 0.5-0.9 | Significant change |
| `length_ratio` | 0.8-1.5 | Reasonable length |

## Variant Performance

From empirical testing (GSM8K, N=5):

| Variant | Bigram (%) | Trigram (%) | Embedding Cos | Edit Dist |
|---------|------------|-------------|---------------|-----------|
| Lexical Maximalist | 18.0 | 10.7 | 0.914 | 0.50 |
| Syntactic Restructuring | 5.1 | 0.0 | 0.949 | 0.86 |
| Abstractive Paraphrase | 3.1 | 0.0 | 0.944 | 0.83 |
| Compositional | 0.0 | 0.0 | 0.851 | 0.93 |

**Observations**:
- All variants achieve <30% bigram overlap ✓
- Embedding similarity stays high (0.85-0.95) ✓
- Compositional is most aggressive (0% bigram!)
- Syntactic has highest embedding similarity (structure-only change)

## Usage Examples

### Read and analyze metrics

```python
import polars as pl
import json

df = pl.read_parquet('outputs/gsm8k_level1.parquet')

for row in df.iter_rows(named=True):
    metrics = json.loads(row['metrics'])

    bigram_overlap = metrics['ngram_overlaps_pct'][1]
    embedding_sim = metrics['cosine_similarity']

    print(f"{row['sd_variant']}: {bigram_overlap:.1f}% bigram, {embedding_sim:.3f} cos")
```

### Filter for high-quality SDs

```python
import polars as pl
import json

df = pl.read_parquet('outputs/gsm8k_level1.parquet')

# Filter: bigram <30%, embedding >0.85, numbers preserved
def is_high_quality(row):
    metrics = json.loads(row['metrics'])
    return (
        metrics['ngram_overlaps_pct'][1] < 30.0 and
        metrics['cosine_similarity'] > 0.85 and
        metrics['number_preservation']
    )

high_quality = [r for r in df.iter_rows(named=True) if is_high_quality(r)]
print(f"High quality: {len(high_quality)} / {len(df)}")
```

### Compare variants

```python
import polars as pl
import json
import numpy as np

df = pl.read_parquet('outputs/gsm8k_level1.parquet')

variants = df.group_by('sd_variant').agg([
    pl.len().alias('count')
])

for variant_name in variants['sd_variant']:
    rows = df.filter(pl.col('sd_variant') == variant_name)

    bigrams = [json.loads(r['metrics'])['ngram_overlaps_pct'][1]
               for r in rows.iter_rows(named=True)]

    print(f"{variant_name}: {np.mean(bigrams):.2f}% ± {np.std(bigrams):.2f}%")
```

## Implementation Details

### Metric Computation

All metrics are computed in `sdtd/generate.py`:

- `calculate_ngram_overlap()`: Word n-grams (1-5)
- `calculate_char_ngram_overlap()`: Character n-grams (1-5)
- `calculate_rouge_l()`: LCS-based F-measure
- `calculate_edit_distance_normalized()`: Levenshtein distance
- `calculate_tfidf_cosine()`: TF-IDF weighted similarity
- `check_number_preservation()`: Regex-based number extraction
- `calculate_length_ratio()`: Word count ratio
- `calculate_jaccard_token()`: Token set Jaccard
- `get_embedding()`: OpenRouter API call for embeddings
- `cosine_similarity()`: Vector dot product similarity
- `calculate_all_metrics()`: Orchestrates all metric computation

### Embedding Model

**Model**: `qwen/qwen3-embedding-8b`
**Provider**: OpenRouter (https://openrouter.ai/api/v1/embeddings)
**Dimensions**: 4096
**API**: Direct HTTP call (litellm doesn't support embeddings via OpenRouter)

### Caching

- LLM calls: Cached via litellm disk cache (`.cache/litellm/`)
- Embeddings: No caching currently (TODO: add)

## Future Enhancements

Potential additions:
- **BLEU score**: Standard MT metric
- **Skip-gram overlap**: Detect reordering patterns
- **POS-weighted n-grams**: Separate content vs function words
- **BM25 score**: IR-style ranking
- **Named entity overlap**: For factual text
- **Embedding caching**: Avoid re-computing identical texts
- **Batch embedding calls**: Process multiple texts at once

## References

- **ROUGE**: Lin, C. Y. (2004). "ROUGE: A package for automatic evaluation of summaries"
- **Levenshtein**: Levenshtein, V. I. (1966). "Binary codes capable of correcting deletions, insertions, and reversals"
- **TF-IDF**: Salton, G. & McGill, M. J. (1983). "Introduction to Modern Information Retrieval"
- **Jaccard**: Jaccard, P. (1912). "The distribution of the flora in the alpine zone"
- **Qwen3 Embeddings**: https://openrouter.ai/qwen/qwen3-embedding-8b
