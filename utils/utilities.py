import torch
from collections import Counter
import math  

def mean_pooling(model_output, attention_mask):
    """Apply mean pooling to model output correctly, avoid padding tokens.
    
    Args: 
        model_output: the token based embeddings of the mode
        attention_mask: array of 1's and 0's to denote active tokens versus padded tokens
        
    Returns:
        Mean pooled model embeddings without padded tokens
    OPTIMIZED: Preserves dtype and uses more efficient operations"""
    token_embeddings = model_output[0]
    # OPTIMIZED: Keep same dtype as token_embeddings (float16 if model is float16)
    input_mask_expanded = (
        attention_mask.unsqueeze(-1)
        .expand(token_embeddings.size())
        .to(dtype=token_embeddings.dtype)  # Match dtype instead of always float
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def generate_ngrams(text, n, tokenizer):
    """Generate n-grams from text.

    Args:
        text: input text (string)
        n: n-gram size
        tokenizer: the tokenizer we will use 

    Return: 
        set of n grams associated with given text
    """
    tokens = tokenizer.tokenize(text)
    if len(tokens) < n:
        return set()
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return set(ngrams)


def calculate_ngram_jaccard_similarity(
    text1, text2, n, tokenizer
):
    """Calculate Jaccard similarity for n-grams.
    
    Args: 
        text1, text2: both strings of text we compare against
        n: n-grtam size
        tokenizer: size of the tokenizer
        
    Returns: Jacard Similarity |A ∩ B|/ |A ∪ B|"""
    ngrams1 = generate_ngrams(text1, n, tokenizer)
    ngrams2 = generate_ngrams(text2, n, tokenizer)
    
    if not ngrams1 and not ngrams2:
        return 1.0
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    return len(intersection) / len(union)
def calculate_ngram_coverage(text_a, text_b, n, tokenizer):
    """
    Calculate the n-gram coverage of text_a by text_b.

    This asymmetric metric measures the proportion of text_a's
    unique n-grams that are also found in text_b.

    Metric: |N-grams(A) ∩ N-grams(B)| / |N-grams(A)|

    Args:
        text_a (str): The text whose coverage is being measured (the "subset").
        text_b (str): The text providing the coverage (the "superset").
        n (int): The size of the n-grams (e.g., 2 for bigrams).
        tokenizer (callable): A tokenizer function compatible with
                              the (assumed) generate_ngrams function.

    Returns:
        float: The n-gram coverage score (from 0.0 to 1.0).
    """
    # Assumes generate_ngrams returns a set
    ngrams_a = generate_ngrams(text_a, n, tokenizer)
    ngrams_b = generate_ngrams(text_b, n, tokenizer)

    # If text_a has no n-grams, it is vacuously 100% covered.
    # This aligns with the 1.0 return for two empty sets in Jaccard.
    if not ngrams_a:
        return 1.0
    
    # If text_b has no n-grams (but text_a does), coverage is 0.
    if not ngrams_b:
        return 0.0

    # Calculate the intersection
    intersection = ngrams_a.intersection(ngrams_b)

    # Return the ratio of the intersection to the size of the first set
    return len(intersection) / len(ngrams_a)

def calculate_bow_cosine_similarity(text1, text2, tokenizer) -> float:
    """Calculate bag-of-words cosine similarity.
    
    Args: 
        text1, text2: strings of texts we are comparing
        tokenizer: the tokenizer that we like
        
    Returns:
        BOW similarity"""
    tokens1 = tokenizer.tokenize(text1)
    tokens2 = tokenizer.tokenize(text2)
    
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    
    bow1 = Counter(tokens1)
    bow2 = Counter(tokens2)
    all_tokens = set(bow1.keys()).union(set(bow2.keys()))
    
    v1 = [bow1.get(token, 0) for token in all_tokens]
    v2 = [bow2.get(token, 0) for token in all_tokens]
    
    dot_product = sum(v1[i] * v2[i] for i in range(len(v1)))
    magnitude1 = math.sqrt(sum(count**2 for count in v1))
    magnitude2 = math.sqrt(sum(count**2 for count in v2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)