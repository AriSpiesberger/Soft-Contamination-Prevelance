# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def mean_pooling(model_output, token_mask):
    """Apply mean pooling to model output.
    Please keep in mind most embedding models run will pool the representative sentence embeddings at the end, but we pad for batches. This pools appropriately
    
    Args: 
        model_output: embeddings from model
        token_mask: array of 1's and 0's corresponding to real tokens and non-padded tokens. 

    Returns:
        mean pooled output over non-special tokens. 
    """
    token_embeddings = model_output[0]
    input_mask_expanded = (
        token_mask.unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask




def generate_ngrams(text, n, tokenizer):
    """Generate n-grams from text.
    
    Args: 
        text: the text we are concerned with
        n: n_gram size
        tokenizer: tokenizer of your choice
        
    Returns:
        the complete set of n-grams"""
    tokens = tokenizer.tokenize(text)
    if len(tokens) < n:
        return set()
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return set(ngrams)


def calculate_ngram_jaccard_similarity(text1, text2, n, tokenizer):
    """Calculate Jaccard similarity for n-grams."""
    ngrams1 = generate_ngrams(text1, n, tokenizer)
    ngrams2 = generate_ngrams(text2, n, tokenizer)
    
    if not ngrams1 and not ngrams2:
        return 1.0
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    return len(intersection) / len(union)