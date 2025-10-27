

def mean_pooling(model_output, attention_mask):
    """Apply mean pooling to model output correctly, avoid padding tokens.
    
    Args: 
        model_output: the token based embeddings of the mode
        attention_mask: array of 1's and 0's to denote active tokens versus padded tokens
        
    Returns:
        Mean pooled model embeddings without padded tokens"""
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
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