#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Similarity Calculator

This script reads an XLSX or CSV file, identifies an 'original' text column
and three 'regenerated' text columns. It then computes and appends similarity
scores (semantic and n-gram) for each original-regenerated pair.

This script is derived from the logic in 'distribution_comparison.py' but
is adapted for batch processing of a spreadsheet instead of comparing two
specific texts against a background corpus.
"""

import os
import sys
import time
import argparse
from collections import Counter
from typing import List, Set

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# --- Configuration ---

# NOTE: Set this to the model you want to use.
# 'Qwen/Qwen1.5-0.5B' is a modern small model from the Qwen family,
# matching your mention of 'qwen'.
# You can also use sentence-transformer models like 'all-MiniLM-L6-v2'.
MODEL_NAME = 'Qwen/Qwen3-Embedding-8B'

# N-gram orders to calculate
N_GRAM_ORDERS = [3, 4] # Trigrams and Quadgrams

# Column names from your file
COL_ORIGINAL = 'original_story'
COL_REGEN = [
    'regenerated_story_1',
    'regenerated_story_2',
    'regenerated_story_3'
]

# --- Parse file paths from command line ---
def parse_args():
    parser = argparse.ArgumentParser(description="Batch Similarity Calculator")
    parser.add_argument('--input', '-i', required=True,
                        help='Path to input CSV or XLSX file')
    parser.add_argument('--output', '-o', default="murder_mystery_comparison_with_scores.csv",
                        help='Path for output CSV file (default: murder_mystery_comparison_with_scores.csv)')
    return parser.parse_args()

_args = parse_args()
INPUT_FILE_PATH = _args.input
OUTPUT_FILE_PATH = _args.output


# --- Helper Functions (Adapted from distribution_comparison.py) ---

def get_device() -> torch.device:
    """Gets the best available device (CUDA or CPU)."""
    if torch.cuda.is_available():
        print("Using CUDA device.")
        return torch.device("cuda")
    else:
        print("CUDA not available. Using CPU.")
        return torch.device("cpu")

def get_ngrams(text: str, n: int) -> Set[str]:
    """
    Generates a set of n-grams from a text string.
    """
    if not isinstance(text, str) or not text.strip():
        return set()
    words = text.split()
    return set(" ".join(words[i:i+n]) for i in range(len(words)-n+1))

def jaccard_similarity(set1: Set, set2: Set) -> float:
    """
    Calculates the Jaccard similarity between two sets.
    """
    if not set1 and not set2:
        return 1.0  # Two empty sets are identical
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def mean_pooling(model_output, attention_mask):
    """
    Performs mean pooling on the token embeddings using the attention mask.
    (This is standard for getting sentence embeddings from base models)
    """
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

@torch.no_grad()
def get_embeddings(texts: List[str], model, tokenizer, device, batch_size: int = 16):
    """
    Computes embeddings for a list of texts in batches.
    """
    all_embeddings = []
    
    # Use float16 for faster inference if on CUDA
    use_fp16 = device.type == 'cuda'
    
    print(f"Calculating embeddings for {len(texts)} texts...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        # Handle potential None or NaN values passed in
        batch_texts = [str(text) if text else "" for text in batch_texts]
        
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        
        if use_fp16:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                model_output = model(**inputs)
        else:
            model_output = model(**inputs)
            
        pooled_embeddings = mean_pooling(model_output, inputs['attention_mask'])
        all_embeddings.append(pooled_embeddings.to('cpu')) # Move to CPU to save VRAM

    return torch.cat(all_embeddings, dim=0)

def main():
    """Main execution function."""
    
    # --- 1. Setup ---
    start_time = time.time()
    device = get_device()

    try:
        print(f"Loading model: {MODEL_NAME}...")
        # trust_remote_code=True is often required for models like Qwen
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model '{MODEL_NAME}'. Make sure it's a valid model name.")
        print(f"Details: {e}")
        sys.exit(1)

    # --- 2. Load Data ---
    print(f"Loading data from: {INPUT_FILE_PATH}...")
    try:
        if INPUT_FILE_PATH.endswith('.xlsx'):
            df = pd.read_excel(INPUT_FILE_PATH)
        elif INPUT_FILE_PATH.endswith('.csv'):
            df = pd.read_csv(INPUT_FILE_PATH)
        else:
            print(f"Error: Unknown file type for '{INPUT_FILE_PATH}'. Please use .xlsx or .csv.")
            sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_FILE_PATH}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
        
    print(f"Data loaded. Found {len(df)} rows.")

    # --- 3. Validate Columns ---
    all_cols = [COL_ORIGINAL] + COL_REGEN
    missing_cols = [col for col in all_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {', '.join(missing_cols)}")
        print(f"Required: {', '.join(all_cols)}")
        sys.exit(1)

    # --- 4. Prepare Texts for Embedding ---
    
    # Fill NaN/NaT values with empty strings to prevent errors
    df[all_cols] = df[all_cols].fillna("")
    
    num_rows = len(df)
    original_texts = df[COL_ORIGINAL].tolist()
    regen1_texts = df[COL_REGEN[0]].tolist()
    regen2_texts = df[COL_REGEN[1]].tolist()
    regen3_texts = df[COL_REGEN[2]].tolist()
    
    # Concatenate all texts for a single, efficient embedding pass
    all_texts_to_embed = original_texts + regen1_texts + regen2_texts + regen3_texts
    
    # --- 5. Calculate Semantic Similarity (Embeddings) ---
    all_embeddings = get_embeddings(all_texts_to_embed, model, tokenizer, device)
    
    # De-allocate model from VRAM to save memory for subsequent processing if needed
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    # Slice the embeddings back into their respective groups
    original_embeds = all_embeddings[0*num_rows : 1*num_rows]
    regen1_embeds = all_embeddings[1*num_rows : 2*num_rows]
    regen2_embeds = all_embeddings[2*num_rows : 3*num_rows]
    regen3_embeds = all_embeddings[3*num_rows : 4*num_rows]
    
    print("Calculating semantic similarities (Cosine)...")
    # Calculate cosine similarity for each pair (vectorized)
    # --- FIX was here ---
    cos_sim_1 = F.cosine_similarity(original_embeds, regen1_embeds).numpy()
    cos_sim_2 = F.cosine_similarity(original_embeds, regen2_embeds).numpy()
    cos_sim_3 = F.cosine_similarity(original_embeds, regen3_embeds).numpy()
    # --- End Fix ---
    
    # Add to DataFrame
    df[f'semantic_sim_{COL_REGEN[0]}'] = cos_sim_1
    df[f'semantic_sim_{COL_REGEN[1]}'] = cos_sim_2
    df[f'semantic_sim_{COL_REGEN[2]}'] = cos_sim_3

    # --- 6. Calculate Syntactic Similarity (N-grams) ---
    print("Calculating syntactic similarities (N-grams)...")
    
    # This part is not easily vectorized and must be done row-by-row
    
    # Prepare lists to store results
    ngram_results = {
        n: {col: [] for col in COL_REGEN} for n in N_GRAM_ORDERS
    }
    
    for _, row in tqdm(df.iterrows(), total=num_rows):
        # Get n-grams for the original text once per row
        original_ngrams = {
            n: get_ngrams(row[COL_ORIGINAL], n) for n in N_GRAM_ORDERS
        }
        
        # Compare against each regenerated text
        for col_name in COL_REGEN:
            for n in N_GRAM_ORDERS:
                regen_ngrams = get_ngrams(row[col_name], n)
                sim = jaccard_similarity(original_ngrams[n], regen_ngrams)
                ngram_results[n][col_name].append(sim)

    # Add n-gram results to the DataFrame
    for n in N_GRAM_ORDERS:
        for col_name in COL_REGEN:
            df[f'{n}gram_jaccard_{col_name}'] = ngram_results[n][col_name]
            
    # --- 7. Save Output ---
    try:
        df.to_csv(OUTPUT_FILE_PATH, index=False)
        print(f"\nSuccessfully processed {num_rows} rows.")
        print(f"Output saved to: {OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"\nError saving output file: {e}")
        
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
