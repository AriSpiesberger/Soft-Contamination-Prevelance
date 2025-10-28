#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Similarity Analyzer - Comparing Two Semantic Duplicates

This script compares two semantically similar texts against a background corpus
to determine if they are truly duplicates from a distributional perspective.

Uses GPU vectorization for semantic scores and CPU parallelization for lexical scores.

Changes include:
- Added asymmetric n-gram coverage metric (Coverage = |A ∩ B| / |A|)
- Integrated coverage metric into parallel lexical analysis pipeline
- Added pairwise asymmetric scores and percentile rankings to final report
- MODIFIED: Updated batching cost function to N log N scaling.
- MODIFIED: Moved GPU-side computations to float16.
"""

import os
import sys
import json
import math
import shutil
import multiprocessing
from collections import Counter
from typing import List, Set, Dict, Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

#our utils
from utils.utilities import *

# =============================================================================
# CONFIGURATION
# =============================================================================

# The two semantic duplicates to compare
TEXT_1 =  """A system has two toggles, blue and green with multiple settings. The system is arranged in a way such that if both the blue and green are both at there max settings at the same time, then it will fail. The system does not fail if only one of blue or green are at max settings. In our set up, the designers decided that the blue toggle should be at max, and the green setting should be at some other useful setting. Today, both the blue and the green toggles are at max. The system fails. Did the blue toggle cause the system failure?"""

TEXT_2 = """How would a typical person answer each of the following questions about causation? A machine is set up in such a way that it will short circuit if both the black wire and the red wire touch the battery at the same time. The machine will not short circuit if just one of these wires touches the battery. The black wire is designated as the one that is supposed to touch the battery, while the red wire is supposed to remain in some other part of the machine. One day, the black wire and the red wire both end up touching the battery at the same time. There is a short circuit. Did the black wire cause the short circuit? Options: - Yes - No"""

TEXTS_OF_INTEREST = [TEXT_1, TEXT_2]
TEXT_LABELS = ["Duplicate", "original"] #plotting, labeling

BACKGROUND_FILE = r"data/random_paragraphs.jsonl"  # Using paragraphs, random_sentences.jsonl is other option
OUTPUT_DIR = "duplicate_comparison" #our dir here
MODEL_NAME = 'Qwen/Qwen3-Embedding-0.6B'
EMBEDDER_KEY = MODEL_NAME.replace('/', '-') + "_vector"

# Acceleration parameters
TARGET_TOKENS_PER_BATCH = 8000  # Total tokens to process per batch (B * L)
MAX_BATCH_SIZE = 512  # Upper limit to prevent edge cases
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1) #utilization for computational intensive tasks (BOW, n-gram)
LEXICAL_CHUNK_SIZE = 1000 #bundle size per cpu




# =============================================================================
# DATA LOADING AND EMBEDDING
# =============================================================================

def create_dynamic_batches(items_with_counts: List[tuple], target_tokens: int, max_batch_size: int):
    """
    Creates batches with N log N attention cost limits.
    """
    current_batch = []
    
    for token_count, text, original_idx in items_with_counts:
        if current_batch:
            # Max tokens after padding
            potential_max_len = max(
                max([tc for tc, _, _ in current_batch]),
                token_count
            )
            potential_batch_size = len(current_batch) + 1
            
            # --- MODIFIED: N log N cost function ---
            # N log N memory cost model (B * L log L)
            # This is a common cost for efficient attention mechanisms
            safe_max_len = max(2, potential_max_len) # Avoid log(0) or log(1)
            attention_memory_cost = potential_batch_size * (safe_max_len * math.log(safe_max_len, 2))
            
            # --- MODIFIED: N log N budget ---
            # Adjust the threshold to also be N log N, based on the target_tokens budget
            # This scales the budget down from quadratic to N log N
            safe_target_tokens = max(2, target_tokens)
            # The B*L budget (target_tokens) is scaled by log(B*L)
            # This is a heuristic to relate the B*(L log L) cost to the B*L budget
            max_allowed_attention_cost = (safe_target_tokens * math.log(safe_target_tokens, 2)) / 4 # Conservative factor
            # --- End modification ---
            
            if (attention_memory_cost > max_allowed_attention_cost or 
                potential_batch_size >= max_batch_size):
                yield [(text, idx) for _, text, idx in current_batch]
                current_batch = []
        
        current_batch.append((token_count, text, original_idx))
    
    if current_batch:
        yield [(text, idx) for _, text, idx in current_batch]

def load_and_embed_background_data(
    filepath: str, 
    tokenizer, 
    model, 
    device: str,
    min_tokens: int = 0,
    max_tokens: int = float('inf')
) -> List[Dict[str, Any]]:
    """
    Load background paragraphs from JSONL file, FILTER BY TOKEN RANGE,
    and compute/cache embeddings for items missing them.
    
    This function UPDATES the original file (e.g., BACKGROUND_FILE)
    with any new token counts or embeddings it computes.
    
    It returns a list filtered by min_tokens/max_tokens for the current run.
    """
    if not os.path.exists(filepath):
        print(f"Error: Background file not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    
    all_data_from_file = [] # Holds *all* data for rewrite
    filtered_background_data = [] # Holds *filtered* data for return
    to_embed_info = []   # Holds items that need embedding
    
    needs_rewrite = False # Flag to track if file update is needed
    total_lines_scanned = 0
    total_lines_filtered_out = 0
    
    print(f"Loading and filtering background data from {filepath}...")
    print(f"Applying token filter for this run: {min_tokens} <= tokens <= {max_tokens}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Scanning background file")):
            total_lines_scanned += 1
            try:
                data = json.loads(line)
                
                # --- Token Count Calculation ---
                token_count = data.get('token_count')
                if token_count is None:
                    token_count = len(tokenizer.tokenize(data['text']))
                    data['token_count'] = token_count
                    needs_rewrite = True # Need to save the new token count
                
                # Add to the master list for potential rewrite
                all_data_from_file.append(data)
                
                # --- Filtering Logic for this Run ---
                if not (min_tokens <= token_count <= max_tokens):
                    total_lines_filtered_out += 1
                    continue # Skip this item for the *current analysis*
                
                # If we are here, the data is within the token range
                # Add it to the list we will return and use
                filtered_background_data.append(data)
                
                # Check if this model's embedding is present
                if EMBEDDER_KEY not in data:
                    needs_rewrite = True # Need to save the new embedding
                    # Get the index from the *master* list
                    original_index = len(all_data_from_file) - 1
                    to_embed_info.append((token_count, data['text'], original_index))
                
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON at line {i+1}")
    
    print(f"Scan complete. Scanned {total_lines_scanned:,} lines.")
    print(f"Filtered out {total_lines_filtered_out:,} lines (outside token range).")
    print(f"Using {len(filtered_background_data):,} lines for this analysis.")
    
    if not filtered_background_data and total_lines_scanned > 0:
        print(f"Warning: No data loaded from {filepath} after filtering.", file=sys.stderr)
        print("Continuing with no background data, but this is unusual.")
        # We don't exit here, because we might still need to rewrite the file
        
    # --- Start Embedding (if anything is missing) ---
    if to_embed_info:
        print(f"Found {len(to_embed_info)} paragraphs missing required embeddings (within range).")
        print("Computing and caching embeddings...")
        
        # Sort by token length for efficient batching
        to_embed_info.sort(key=lambda x: x[0])
        
        # Create dynamic batches based on token budget
        batches = list(create_dynamic_batches(
            to_embed_info,
            TARGET_TOKENS_PER_BATCH,
            MAX_BATCH_SIZE
        ))
        
        if not batches and to_embed_info: 
             batches = [[(text, idx) for _, text, idx in to_embed_info]]

        if not batches:
             print("No batches to embed.")
        else:
            print(f"Created {len(batches)} dynamic batches.")
            print(f"   Batch sizes range: {min(len(b) for b in batches)} to {max(len(b) for b in batches)}")
        
        with torch.no_grad():
            for batch_data in tqdm(batches, desc="Embedding batches"):
                batch_texts = [text for text, _ in batch_data]
                batch_indices = [idx for _, idx in batch_data]
                max_len = getattr(model.config, 'max_position_embeddings', 1024)
                
                encoded_input = tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=max_len,
                    return_tensors='pt'
                )
                encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
                
                model_output = model(**encoded_input)
                batch_embeddings = mean_pooling(
                    model_output,
                    encoded_input['attention_mask']
                )
                
                cpu_embeddings = batch_embeddings.cpu().numpy().tolist()
                
                for j, original_index in enumerate(batch_indices):
                    # 'original_index' is the index in the *master* 'all_data_from_file' list
                    all_data_from_file[original_index][EMBEDDER_KEY] = cpu_embeddings[j]
        
        print("Embedding complete. Caching results to disk...")
    
    # --- Rewrite the file if needed ---
    if needs_rewrite:
        print(f"Updating {filepath} with new token counts/embeddings...")
        temp_filepath = filepath + ".tmp"
        try:
            with open(temp_filepath, 'w', encoding='utf-8') as f_out:
                for data in all_data_from_file:
                    f_out.write(json.dumps(data) + '\n')
            shutil.move(temp_filepath, filepath)
            print(f"Successfully updated {filepath}")
        except Exception as e:
            print(f"Error writing file: {e}", file=sys.stderr)
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            sys.exit(1)
    else:
        print("No file update needed. All required data was already cached.")
        
    return filtered_background_data

def get_text_token_counts(
    texts: List[str], labels: List[str], tokenizer
) -> Dict[str, int]:
    """Calculate token counts for texts of interest (respecting max_length)."""
    print("Calculating token counts for texts of interest...")
    # This should ideally use model.config.max_position_embeddings
    # We pass the model to main, but not here. Using a safe default.
    max_len = 8192 
    return {
        labels[i]: len(tokenizer.tokenize(
            text,
            truncation=True,
            max_length=max_len
        ))
        for i, text in enumerate(texts)
    }



def get_text_embeddings(
    texts: List[str], tokenizer, model, device: str
) -> torch.Tensor:
    """Compute embeddings for texts of interest."""
    print("Computing embeddings for texts of interest (in float16)...")
    max_len = getattr(model.config, 'max_position_embeddings', 8192)
    with torch.no_grad():
        encoded_input = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,  
            return_tensors='pt'
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        model_output = model(**encoded_input)
        # Embeddings will be float16 since model is float16
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    print("Done.")
    return embeddings


# =============================================================================
# SEMANTIC SIMILARITY (GPU VECTORIZED)
# =============================================================================

def compute_all_semantic_scores(
    text_embeddings: torch.Tensor,
    background_data: List[Dict[str, Any]],
    device: str
) -> List[List[float]]:
    """
    Compute all N*M semantic scores via single matrix multiplication in float16.
    """
    print("Computing all semantic scores (vectorized in float16)...")
    
    # Extract background embeddings (which are Python floats)
    bg_vectors = [entry[EMBEDDER_KEY] for entry in background_data]
    
    # --- MODIFIED: Cast background embeddings to float16 ---
    bg_embeddings_gpu = torch.tensor(bg_vectors, dtype=torch.float16, device=device)
    
    # --- START: ADDED FIX ---
    # Explicitly cast text_embeddings to float16 to match.
    # This undoes any up-casting that mean_pooling might have done.
    text_embeddings_gpu = text_embeddings.to(dtype=torch.float16, device=device)
    # --- END: ADDED FIX ---
    
    # Normalize both matrices
    text_norm = F.normalize(text_embeddings_gpu, p=2, dim=1) # Use the new _gpu var
    bg_norm = F.normalize(bg_embeddings_gpu, p=2, dim=1)
    
    # Single matrix multiplication: (N, D) @ (D, M) -> (N, M)
    print("Performing matrix multiplication on GPU (float16)...")
    all_sim_scores = torch.matmul(text_norm, bg_norm.T)
    
    # Move results to CPU
    print("Moving results to CPU...")
    # .tolist() converts np.float16 -> python float
    all_sim_scores_cpu = all_sim_scores.cpu().numpy().tolist()
    
    print("Semantic score computation complete.")
    return all_sim_scores_cpu

# =============================================================================
# LEXICAL SIMILARITY (CPU PARALLELIZED)
# =============================================================================

# Global variables for worker processes
worker_tokenizer = None
worker_text_list = None


def initialize_worker_lexical(text_list: List[str]):
    """Initialize tokenizer and texts in worker process."""
    global worker_tokenizer, worker_text_list
    worker_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    worker_text_list = text_list


def process_lexical_chunk(
    bg_chunk: List[Dict[str, Any]]
) -> Dict[str, Dict[str, List[float]]]:
    """
    Process a chunk of background data on a single CPU core.
    """
    global worker_tokenizer, worker_text_list
    
    chunk_results = {
        label: {
            "unigram": [],
            "bigram": [],
            "trigram": [],
            "bow_cosine": [],
            "unigram_coverage": [],
            "bigram_coverage": [],
            "trigram_coverage": []
        }
        for label in TEXT_LABELS
    }
    
    for bg_entry in bg_chunk:
        bg_text = bg_entry['text']
        
        for i, text in enumerate(worker_text_list):
            label = TEXT_LABELS[i]
            
            # --- Jaccard Similarity (Symmetric) ---
            chunk_results[label]["unigram"].append(
                calculate_ngram_jaccard_similarity(text, bg_text, 1, worker_tokenizer)
            )
            chunk_results[label]["bigram"].append(
                calculate_ngram_jaccard_similarity(text, bg_text, 2, worker_tokenizer)
            )
            chunk_results[label]["trigram"].append(
                calculate_ngram_jaccard_similarity(text, bg_text, 3, worker_tokenizer)
            )
            chunk_results[label]["bow_cosine"].append(
                calculate_bow_cosine_similarity(text, bg_text, worker_tokenizer)
            )
            
            # --- N-gram Coverage (Asymmetric) ---
            # Calculates Coverage(text, bg_text) -> |text ∩ bg_text| / |text|
            chunk_results[label]["unigram_coverage"].append(
                calculate_ngram_coverage(text, bg_text, 1, worker_tokenizer)
            )
            chunk_results[label]["bigram_coverage"].append(
                calculate_ngram_coverage(text, bg_text, 2, worker_tokenizer)
            )
            chunk_results[label]["trigram_coverage"].append(
                calculate_ngram_coverage(text, bg_text, 3, worker_tokenizer)
            )
    
    return chunk_results


def generate_bg_chunks(background_data: List[Dict[str, Any]], chunk_size: int):
    """Yield chunks of background data for multiprocessing."""
    for i in range(0, len(background_data), chunk_size):
        yield background_data[i:i+chunk_size]


def compute_all_lexical_scores(
    background_data: List[Dict[str, Any]]
) -> Dict[str, Dict[str, List[float]]]:
    """
    Orchestrate parallel computation of all lexical scores.
    """
    print(f"Computing lexical scores (parallelized across {NUM_WORKERS} workers)...")
    
    all_lexical_dists = {
        label: {
            "unigram": [],
            "bigram": [],
            "trigram": [],
            "bow_cosine": [],
            "unigram_coverage": [],
            "bigram_coverage": [],
            "trigram_coverage": []
        }
        for label in TEXT_LABELS
    }
    
    num_chunks = math.ceil(len(background_data) / LEXICAL_CHUNK_SIZE)
    if num_chunks == 0:
        print("No background data to process for lexical scores.")
        return all_lexical_dists

    with multiprocessing.Pool(
        processes=NUM_WORKERS,
        initializer=initialize_worker_lexical,
        initargs=(TEXTS_OF_INTEREST,)
    ) as pool:
        
        data_chunks = generate_bg_chunks(background_data, LEXICAL_CHUNK_SIZE)
        
        pbar = tqdm(
            pool.imap_unordered(process_lexical_chunk, data_chunks),
            total=num_chunks,
            desc="Processing lexical chunks"
        )
        
        for chunk_results in pbar:
            for label in TEXT_LABELS:
                all_lexical_dists[label]["unigram"].extend(
                    chunk_results[label]["unigram"]
                )
                all_lexical_dists[label]["bigram"].extend(
                    chunk_results[label]["bigram"]
                )
                all_lexical_dists[label]["trigram"].extend(
                    chunk_results[label]["trigram"]
                )
                all_lexical_dists[label]["bow_cosine"].extend(
                    chunk_results[label]["bow_cosine"]
                )
                # --- Aggregate Coverage ---
                all_lexical_dists[label]["unigram_coverage"].extend(
                    chunk_results[label]["unigram_coverage"]
                )
                all_lexical_dists[label]["bigram_coverage"].extend(
                    chunk_results[label]["bigram_coverage"]
                )
                all_lexical_dists[label]["trigram_coverage"].extend(
                    chunk_results[label]["trigram_coverage"]
                )
    
    print("Lexical score computation complete.")
    return all_lexical_dists


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comparison_distributions(
    text_a_scores: List[float],
    text_b_scores: List[float],
    direct_similarity: float,
    output_dir: str
):
    """
    Plot overlapping distributions of both texts against background corpus.
    """
    print("Generating comparison distribution plot...")
    try:
        df = pd.DataFrame({
            'score': text_a_scores + text_b_scores,
            'text': [TEXT_LABELS[0]] * len(text_a_scores) + [TEXT_LABELS[1]] * len(text_b_scores)
        })
        
        plt.figure(figsize=(14, 8))
        
        sns.kdeplot(
            data=df,
            x='score',
            hue='text',
            fill=True,
            common_norm=False,
            alpha=0.3,
            linewidth=2.5
        )
        
        # Add vertical line for direct comparison
        plt.axvline(
            x=direct_similarity,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'{TEXT_LABELS[0]} vs {TEXT_LABELS[1]}: {direct_similarity:.4f}'
        )
        
        plt.title(
            f'Semantic Similarity: {TEXT_LABELS[0]} and {TEXT_LABELS[1]} vs Background Corpus',
            fontsize=16,
            fontweight='bold'
        )
        plt.xlabel('Cosine Similarity', fontsize=13)
        plt.ylabel('Density', fontsize=13)
        plt.legend(fontsize=11)
        plt.grid(axis='both', linestyle='--', alpha=0.3)
        plt.xlim(-0.05, 1.0)
        
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, "semantic_comparison.png")
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        
        print(f"Saved comparison plot to: {plot_filename}")
        
    except Exception as e:
        print(f"Error generating comparison plot: {e}", file=sys.stderr)


def plot_bivariate_distribution(
    label: str,
    text: str,
    background_data: List[Dict[str, Any]],
    semantic_scores: List[float],
    token_count: int,
    other_label: str,
    other_token_count: int,
    pairwise_score: float,
    output_dir: str
):
    """
    Generate 2D KDE plot of cosine similarity vs log token count.
    """
    print(f"Generating bivariate plot for {label}...")
    try:
        bg_token_counts = [max(1, d.get('token_count', 1)) for d in background_data]
        
        if not bg_token_counts or not semantic_scores:
             print(f"Skipping bivariate plot for {label}: no background data.")
             return

        df = pd.DataFrame({
            "semantic_score": semantic_scores,
            "token_count": bg_token_counts
        })
        
        # Determine dynamic x-axis limits
        all_x_data = semantic_scores + [pairwise_score]
        data_min = np.min(all_x_data)
        data_max = np.max(all_x_data)
        data_range = data_max - data_min
        padding = data_range * 0.05
        # Ensure padding is reasonable, avoid negative lims if data_min is low
        new_xlim = (max(data_min - padding, -0.05), min(data_max + padding, 1.05))

        
        # Create 2D KDE plot
        g = sns.jointplot(
            data=df,
            x="semantic_score",
            y="token_count",
            kind="kde",
            fill=True,
            cmap="viridis",
            height=10,
            space=0,
            xlim=new_xlim,
            log_scale=(False, True)
        )
        
        g.set_axis_labels(
            'Cosine Similarity',
            'Token Count (Log Scale)',
            fontsize=12
        )
        
        # Truncate text for title
        text_preview = text[:100].replace('\n', ' ') + "..." if len(text) > 100 else text.replace('\n', ' ')
        g.fig.suptitle(
            f'Bivariate Distribution: {label} vs Background\n"{text_preview}"',
            fontsize=14,
            y=1.03
        )
        
        # Plot the other text as a scatter point
        ax = g.ax_joint
        ax.scatter(
            [pairwise_score],
            [other_token_count],
            s=150,
            c='red',
            edgecolor='white',
            linewidth=2,
            zorder=5,
            label=other_label
        )
        ax.text(
            pairwise_score + (data_range * 0.01),
            other_token_count,
            other_label,
            color='red',
            fontsize=11,
            fontweight='bold',
            zorder=5,
            bbox=dict(facecolor='white', alpha=0.7, pad=0.2, edgecolor='none')
        )
        
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f"{label}_bivariate.png")
        g.fig.savefig(plot_filename, dpi=300)
        plt.close(g.fig)
        
        print(f"Saved bivariate plot to: {plot_filename}")
        
    except Exception as e:
        print(f"Error generating bivariate plot for {label}: {e}", file=sys.stderr)





# =============================================================================
# MAIN EXECUTION
# =============================================================================
# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function.
    """
    print("=" * 80)
    print("SEMANTIC DUPLICATE COMPARISON ANALYSIS")
    print("=" * 80)
    print()
    
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # --- MODIFIED: Use float16 only on CUDA ---
    use_fp16 = device == "cuda"
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    print(f"Using device: {device} (FP16: {use_fp16})\n")
    
    # Load model
    try:
        print(f"Loading model: {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype # Load model in specified dtype
        )
        model.to(device)
        model.eval()
        print("Model loaded successfully.\n")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return
    
    # --- START: Modified data loading ---
    # Define filter range
    MIN_TOKEN_LIMIT = 10  
    MAX_TOKEN_LIMIT = 200 # Using the 200 from your last file
    
    # Load, filter, and embed background data simultaneously
    # The returned 'background_data' is ALREADY filtered.
    background_data = load_and_embed_background_data(
        BACKGROUND_FILE, 
        tokenizer, 
        model, 
        device,
        min_tokens=MIN_TOKEN_LIMIT,
        max_tokens=MAX_TOKEN_LIMIT
    )
    
    # Check if filtering left anything
    if not background_data:
        print(f"Error: No background data remaining after filtering (range {MIN_TOKEN_LIMIT}-{MAX_TOKEN_LIMIT} tokens).", file=sys.stderr)
        print("Please check your BACKGROUND_FILE or adjust the token limits.", file=sys.stderr)
        return
        
    print(f"Successfully prepared {len(background_data):,} paragraphs for analysis.\n")
    # --- END: Modified data loading ---
    
    
    # Embed texts of interest
    text_embeddings = get_text_embeddings(TEXTS_OF_INTEREST, tokenizer, model, device)
    text_token_counts = get_text_token_counts(TEXTS_OF_INTEREST, TEXT_LABELS, tokenizer)
    
    # Calculate direct pairwise similarity
    print(f"\nCalculating direct similarity between {TEXT_LABELS[0]} and {TEXT_LABELS[1]}...")
    # --- MODIFIED: Ensure comparison is done in float32 for stability ---
    direct_similarity = F.cosine_similarity(
        text_embeddings[0].to(torch.float32), # Cast to float32
        text_embeddings[1].to(torch.float32), # Cast to float32
        dim=0
    ).item()
    print(f"Direct semantic similarity: {direct_similarity:.6f}\n")
    
    # Compute semantic distributions
    semantic_distributions = compute_all_semantic_scores(
        text_embeddings, background_data, device
    )
    
    # Compute lexical distributions
    lexical_distributions = compute_all_lexical_scores(background_data)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_comparison_distributions(
        semantic_distributions[0],
        semantic_distributions[1],
        direct_similarity,
        OUTPUT_DIR
    )
    
    plot_bivariate_distribution(
        TEXT_LABELS[0],
        TEXTS_OF_INTEREST[0],
        background_data,
        semantic_distributions[0],
        text_token_counts[TEXT_LABELS[0]],
        TEXT_LABELS[1],
        text_token_counts[TEXT_LABELS[1]],
        direct_similarity,
        OUTPUT_DIR
    )
    
    plot_bivariate_distribution(
        TEXT_LABELS[1],
        TEXTS_OF_INTEREST[1],
        background_data,
        semantic_distributions[1],
        text_token_counts[TEXT_LABELS[1]],
        TEXT_LABELS[0],
        text_token_counts[TEXT_LABELS[0]],
        direct_similarity,
        OUTPUT_DIR
    )
    
    # Calculate and report statistics
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)
    
    label_a = TEXT_LABELS[0]
    label_b = TEXT_LABELS[1]
    text_a = TEXTS_OF_INTEREST[0] # Duplicate
    text_b = TEXTS_OF_INTEREST[1] # original
    
    print(f"\nDirect Comparison: {label_a} vs {label_b}")
    print(f"   Token Count {label_a}: {text_token_counts[label_a]}")
    print(f"   Token Count {label_b}: {text_token_counts[label_b]}")
    print(f"   Semantic Similarity: {direct_similarity:.6f}")
    
    # Calculate lexical similarities (Symmetric)
    print("\n   --- Symmetric Lexical Scores ---")
    score_uni = calculate_ngram_jaccard_similarity(
        text_a, text_b, 1, tokenizer
    )
    score_bi = calculate_ngram_jaccard_similarity(
        text_a, text_b, 2, tokenizer
    )
    score_tri = calculate_ngram_jaccard_similarity(
        text_a, text_b, 3, tokenizer
    )
    score_bow = calculate_bow_cosine_similarity(
        text_a, text_b, tokenizer
    )
    
    print(f"   Unigram Jaccard: {score_uni:.6f}")
    print(f"   Bigram Jaccard: {score_bi:.6f}")
    print(f"   Trigram Jaccard: {score_tri:.6f}")
    print(f"   BoW Cosine: {score_bow:.6f}")

    # Calculate lexical coverage (Asymmetric)
    print(f"\n   --- Asymmetric Coverage ({label_a} by {label_b}) ---")
    score_uni_a_by_b = calculate_ngram_coverage(text_a, text_b, 1, tokenizer)
    score_bi_a_by_b = calculate_ngram_coverage(text_a, text_b, 2, tokenizer)
    score_tri_a_by_b = calculate_ngram_coverage(text_a, text_b, 3, tokenizer)
    print(f"   Unigram Cov ({label_a} by {label_b}): {score_uni_a_by_b:.6f}   (|\"{label_a}\" ∩ \"{label_b}\"| / |\"{label_a}\"|)")
    print(f"   Bigram Cov ({label_a} by {label_b}): {score_bi_a_by_b:.6f}")
    print(f"   Trigram Cov ({label_a} by {label_b}): {score_tri_a_by_b:.6f}")

    print(f"\n   --- Asymmetric Coverage ({label_b} by {label_a}) ---")
    score_uni_b_by_a = calculate_ngram_coverage(text_b, text_a, 1, tokenizer)
    score_bi_b_by_a = calculate_ngram_coverage(text_b, text_a, 2, tokenizer)
    score_tri_b_by_a = calculate_ngram_coverage(text_b, text_a, 3, tokenizer)
    print(f"   Unigram Cov ({label_b} by {label_a}): {score_uni_b_by_a:.6f}   (|\"{label_a}\" ∩ \"{label_b}\"| / |\"{label_b}\"|)")
    print(f"   Bigram Cov ({label_b} by {label_a}): {score_bi_b_by_a:.6f}")
    print(f"   Trigram Cov ({label_b} by {label_a}): {score_tri_b_by_a:.6f}")
    
    # Helper for safe percentile calculation
    def safe_percentile(scores: List[float], value: float) -> float:
        if not scores:
            return 0.0
        return percentileofscore(scores, value)

    # Percentile rankings for Text A's view
    print(f"\nPercentile Rankings (from {label_a}'s perspective vs Background):")
    print(f"   Semantic: {safe_percentile(semantic_distributions[0], direct_similarity):.2f}th percentile")
    print(f"   Unigram Jaccard: {safe_percentile(lexical_distributions[label_a]['unigram'], score_uni):.2f}th percentile")
    print(f"   Bigram Jaccard: {safe_percentile(lexical_distributions[label_a]['bigram'], score_bi):.2f}th percentile")
    print(f"   Trigram Jaccard: {safe_percentile(lexical_distributions[label_a]['trigram'], score_tri):.2f}th percentile")
    print(f"   BoW Cosine: {safe_percentile(lexical_distributions[label_a]['bow_cosine'], score_bow):.2f}th percentile")
    print(f"   Unigram Coverage (of {label_a}): {safe_percentile(lexical_distributions[label_a]['unigram_coverage'], score_uni_a_by_b):.2f}th percentile")
    print(f"   Bigram Coverage (of {label_a}): {safe_percentile(lexical_distributions[label_a]['bigram_coverage'], score_bi_a_by_b):.2f}th percentile")
    print(f"   Trigram Coverage (of {label_a}): {safe_percentile(lexical_distributions[label_a]['trigram_coverage'], score_tri_a_by_b):.2f}th percentile")

    
    # Percentile rankings for Text B's view
    print(f"\nPercentile Rankings (from {label_b}'s perspective vs Background):")
    print(f"   Semantic: {safe_percentile(semantic_distributions[1], direct_similarity):.2f}th percentile")
    print(f"   Unigram Jaccard: {safe_percentile(lexical_distributions[label_b]['unigram'], score_uni):.2f}th percentile")
    print(f"   Bigram Jaccard: {safe_percentile(lexical_distributions[label_b]['bigram'], score_bi):.2f}th percentile")
    print(f"   Trigram Jaccard: {safe_percentile(lexical_distributions[label_b]['trigram'], score_tri):.2f}th percentile")
    print(f"   BoW Cosine: {safe_percentile(lexical_distributions[label_b]['bow_cosine'], score_bow):.2f}th percentile")
    print(f"   Unigram Coverage (of {label_b}): {safe_percentile(lexical_distributions[label_b]['unigram_coverage'], score_uni_b_by_a):.2f}th percentile")
    print(f"   Bigram Coverage (of {label_b}): {safe_percentile(lexical_distributions[label_b]['bigram_coverage'], score_bi_b_by_a):.2f}th percentile")
    print(f"   Trtam Coverage (of {label_b}): {safe_percentile(lexical_distributions[label_b]['trigram_coverage'], score_tri_b_by_a):.2f}th percentile")

    
    # Save detailed results
    results = {
        "text_a": {
            "label": label_a,
            "text": text_a,
            "token_count": text_token_counts[label_a],
            "semantic_distribution": semantic_distributions[0],
            "lexical_distributions": lexical_distributions[label_a]
        },
        "text_b": {
            "label": label_b,
            "text": text_b,
            "token_count": text_token_counts[label_b],
            "semantic_distribution": semantic_distributions[1],
            "lexical_distributions": lexical_distributions[label_b]
        },
        "pairwise_comparison": {
            "semantic_similarity": direct_similarity,
            "unigram_jaccard": score_uni,
            "bigram_jaccard": score_bi,
            "trigram_jaccard": score_tri,
            "bow_cosine": score_bow,
            f"unigram_coverage_{label_a}_by_{label_b}": score_uni_a_by_b,
            f"bigram_coverage_{label_a}_by_{label_b}": score_bi_a_by_b,
            f"trigram_coverage_{label_a}_by_{label_b}": score_tri_a_by_b,
            f"unigram_coverage_{label_b}_by_{label_a}": score_uni_b_by_a,
            f"bigram_coverage_{label_b}_by_{label_a}": score_bi_b_by_a,
            f"trigram_coverage_{label_b}_by_{label_a}": score_tri_b_by_a,
        },
        "background_corpus": {
            "file": BACKGROUND_FILE,
            "num_paragraphs_used": len(background_data),
            "token_limit_min": MIN_TOKEN_LIMIT,
            "token_limit_max": MAX_TOKEN_LIMIT
        },
        "model": MODEL_NAME
    }
    
    results_file = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        # A simple numpy-safe encoder
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)
        
        json.dump(results, f, indent=2, cls=NpEncoder)
    
    print(f"\n\nDetailed results saved to: {results_file}")
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()