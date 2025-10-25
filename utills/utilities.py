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
