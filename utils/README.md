# Utils

Shared utility functions for embedding and text similarity computation.

## Functions (`utilities.py`)

| Function | Purpose |
|----------|---------|
| `mean_pooling(model_output, attention_mask)` | Mean-pool transformer embeddings, excluding padding tokens. Preserves dtype (float16-safe) |
| `generate_ngrams(text, n, tokenizer)` | Generate n-gram sets from text using a HuggingFace tokenizer |
| `calculate_ngram_jaccard_similarity(text1, text2, n, tokenizer)` | Jaccard similarity between n-gram sets |
| `calculate_ngram_coverage(text1, text2, n, tokenizer)` | Asymmetric n-gram coverage (what fraction of text1's n-grams appear in text2) |
| `calculate_bow_cosine_similarity(text1, text2, tokenizer)` | Bag-of-words cosine similarity |
