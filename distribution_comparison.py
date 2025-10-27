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
from utils.utilities import * 

# =============================================================================
# CONFIGURATION
# =============================================================================

# The two semantic duplicates to compare
TEXT_1 = """An electrical device has been designed so that a power failure will occur only if both a green cable and a yellow cable make contact with the power source simultaneously. If only one cable connects to the power source, no power failure happens. According to the device's specifications, the green cable is intended to connect to the power source during normal operation, whereas the yellow cable should stay positioned elsewhere in the system. On a particular occasion, both the green cable and the yellow cable simultaneously contact the power source, resulting in a power failure.
Did the green cable cause the power failure?
Options:
Yes
No"""

TEXT_2 = """How would a typical person answer each of the following questions about causation? A machine is set up in such a way that it will short circuit if both the black wire and the red wire touch the battery at the same time. The machine will not short circuit if just one of these wires touches the battery. The black wire is designated as the one that is supposed to touch the battery, while the red wire is supposed to remain in some other part of the machine. One day, the black wire and the red wire both end up touching the battery at the same time. There is a short circuit. Did the black wire cause the short circuit? Options: - Yes - No"""

TEXTS_OF_INTEREST = [TEXT_1, TEXT_2]
TEXT_LABELS = ["Duplicate", "original"] #plotting, labeling

BACKGROUND_FILE = r"data/random_paragraphs.jsonl"  # Using paragraphs, random_sentences.jsonl is other option
OUTPUT_DIR = "duplicate_comparison" #our dir here
MODEL_NAME = 'Qwen/Qwen3-Embedding-0.6B' 
EMBEDDER_KEY = MODEL_NAME.replace('/', '-') + "_vector"

# Acceleration parameters
TARGET_TOKENS_PER_BATCH = 8000  # Total tokens to process per batch
MAX_BATCH_SIZE = 512  # Upper limit to prevent edge cases
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1) #utilization for computational intensive tasks (BOW, n-gram)
LEXICAL_CHUNK_SIZE = 1000 #bundle size per cpu

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def mean_pooling(model_output, attention_mask):
    """Apply mean pooling to model output."""
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def generate_ngrams(text: str, n: int, tokenizer) -> Set[tuple]:
    """Generate n-grams from text."""
    tokens = tokenizer.tokenize(text)
    if len(tokens) < n:
        return set()
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return set(ngrams)


def calculate_ngram_jaccard_similarity(
    text1: str, text2: str, n: int, tokenizer
) -> float:
    """Calculate Jaccard similarity for n-grams. |A ∩ B| / |A ∪ B|"""
    ngrams1 = generate_ngrams(text1, n, tokenizer)
    ngrams2 = generate_ngrams(text2, n, tokenizer)
    
    if not ngrams1 and not ngrams2:
        return 1.0
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    return len(intersection) / len(union)


def calculate_ngram_coverage(
    text_a: str, text_b: str, n: int, tokenizer
) -> float:
    """
    Calculate asymmetric n-gram coverage of text_a by text_b.
    
    Metric: |N-grams(A) ∩ N-grams(B)| / |N-grams(A)|
    Answers: "What fraction of text A's n-grams are in text B?"
    """
    ngrams_a = generate_ngrams(text_a, n, tokenizer)
    ngrams_b = generate_ngrams(text_b, n, tokenizer)
    
    # If text_a has no n-grams, it is vacuously 100% covered.
    if not ngrams_a:
        return 1.0
    
    # If text_b has no n-grams (but text_a does), coverage is 0.
    if not ngrams_b:
        return 0.0
    
    intersection = ngrams_a.intersection(ngrams_b)
    
    # Return the ratio of the intersection to the size of the *first* set
    return len(intersection) / len(ngrams_a)


def calculate_bow_cosine_similarity(text1: str, text2: str, tokenizer) -> float:
    """Calculate bag-of-words cosine similarity."""
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


# =============================================================================
# DATA LOADING AND EMBEDDING
# =============================================================================

def create_dynamic_batches(items_with_counts: List[tuple], target_tokens: int, max_batch_size: int):
    """
    Creates batches based on token count AND attention complexity to maximize GPU utilization.
    
    Accounts for O(n²) attention cost - longer sequences are exponentially more expensive.
    
    Args:
        items_with_counts: List of (token_count, text, original_index) tuples
        target_tokens: Target total tokens per batch (linear cost baseline)
        max_batch_size: Maximum number of items in a single batch
    
    Yields:
        List of (text, original_index) tuples for each batch
    """
    current_batch = []
    
    for token_count, text, original_idx in items_with_counts:
        # Calculate what the max tokens would be if we add this item
        # (all items in batch get padded to the longest)
        potential_max_tokens = max(
            max([tc for tc, _, _ in current_batch], default=0),
            token_count
        )
        
        # Linear memory cost (VRAM usage)
        potential_memory_cost = potential_max_tokens * (len(current_batch) + 1)
        
        # Quadratic attention cost (compute time)
        # For a batch of size B with sequence length L:
        # Total ops ≈ B × L² (simplified, ignoring model dimension constant)
        potential_attention_cost = (len(current_batch) + 1) * (potential_max_tokens ** 2)
        
        # We use a weighted combination:
        # - Memory cost limits VRAM (hard constraint)
        # - Attention cost limits throughput (soft constraint, but matters for speed)
        # Scale attention cost down so both constraints are comparable
        scaled_attention_cost = potential_attention_cost / potential_max_tokens
        
        # If adding this item would exceed our budget, yield current batch
        if current_batch and (
            potential_memory_cost > target_tokens or          # VRAM constraint
            scaled_attention_cost > target_tokens * 2 or      # Compute constraint (more lenient)
            len(current_batch) >= max_batch_size              # Safety limit
        ):
            yield [(text, idx) for _, text, idx in current_batch]
            current_batch = []
        
        current_batch.append((token_count, text, original_idx))
    
    # Yield remaining items
    if current_batch:
        yield [(text, idx) for _, text, idx in current_batch]


def load_and_embed_background_data(
    filepath: str, tokenizer, model, device: str
) -> List[Dict[str, Any]]:
    """
    Load background paragraphs from JSONL file and compute embeddings if needed.
    """
    if not os.path.exists(filepath):
        print(f"Error: Background file not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    
    background_data = []
    to_embed_info = []
    needs_rewrite = False
    
    print(f"Loading background data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Scanning background file")):
            try:
                data = json.loads(line)
                
                # Calculate token count if not present
                token_count = data.get('token_count')
                if token_count is None:
                    token_count = len(tokenizer.tokenize(data['text']))
                
                data['token_count'] = token_count
                background_data.append(data)
                
                if EMBEDDER_KEY not in data:
                    needs_rewrite = True
                    to_embed_info.append((token_count, data['text'], i))
                    
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON at line {i+1}")
    
    if not background_data:
        print(f"Error: No data loaded from {filepath}", file=sys.stderr)
        sys.exit(1)
    
    if needs_rewrite:
        print(f"Found {len(to_embed_info)} paragraphs missing embeddings.")
        
        # Sort by token length for efficient batching
        print("Sorting by token length for optimal batching...")
        to_embed_info.sort(key=lambda x: x[0])
        
        # Create dynamic batches based on token budget
        print(f"Creating dynamic batches (target: {TARGET_TOKENS_PER_BATCH} tokens/batch)...")
        batches = list(create_dynamic_batches(
            to_embed_info,
            TARGET_TOKENS_PER_BATCH,
            MAX_BATCH_SIZE
        ))
        
        print(f"Created {len(batches)} dynamic batches.")
        print(f"  Batch sizes range: {min(len(b) for b in batches)} to {max(len(b) for b in batches)}")
        
        print("Computing embeddings with dynamic batching...")
        with torch.no_grad():
            for batch_data in tqdm(batches, desc="Embedding batches"):
                batch_texts = [text for text, _ in batch_data]
                batch_indices = [idx for _, idx in batch_data]
                max_len = getattr(model.config, 'max_position_embeddings', 10000)
                # print(max_len) # Removed verbose print
                encoded_input = tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=max_len,  # <-- ADD THIS
                    return_tensors='pt'
                )
                encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
                
                model_output = model(**encoded_input)
                batch_embeddings = mean_pooling(
                    model_output,
                    encoded_input['attention_mask']
                )
                
                cpu_embeddings = batch_embeddings.cpu().numpy().tolist()
                
                for j, original_idx in enumerate(batch_indices):
                    background_data[original_idx][EMBEDDER_KEY] = cpu_embeddings[j]
        
        print("Embedding complete. Updating background file...")
        temp_filepath = filepath + ".tmp"
        try:
            with open(temp_filepath, 'w', encoding='utf-8') as f_out:
                for data in background_data:
                    f_out.write(json.dumps(data) + '\n')
            shutil.move(temp_filepath, filepath)
            print(f"Successfully updated {filepath}")
        except Exception as e:
            print(f"Error writing file: {e}", file=sys.stderr)
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            sys.exit(1)
    else:
        print("All background paragraphs already have embeddings.")
    
    return background_data


def get_text_token_counts(
    texts: List[str], labels: List[str], tokenizer
) -> Dict[str, int]:
    """Calculate token counts for texts of interest."""
    print("Calculating token counts for texts of interest...")
    return {
        labels[i]: len(tokenizer.tokenize(text))
        for i, text in enumerate(texts)
    }


def get_text_embeddings(
    texts: List[str], tokenizer, model, device: str
) -> torch.Tensor:
    """Compute embeddings for texts of interest."""
    print("Computing embeddings for texts of interest...")
    with torch.no_grad():
        encoded_input = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        model_output = model(**encoded_input)
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
    Compute all N*M semantic scores via single matrix multiplication.
    """
    print("Computing all semantic scores (vectorized)...")
    
    # Extract background embeddings
    bg_vectors = [entry[EMBEDDER_KEY] for entry in background_data]
    bg_embeddings_gpu = torch.tensor(bg_vectors, dtype=torch.float32, device=device)
    
    # Normalize both matrices
    text_norm = F.normalize(text_embeddings, p=2, dim=1)
    bg_norm = F.normalize(bg_embeddings_gpu, p=2, dim=1)
    
    # Single matrix multiplication: (N, D) @ (D, M) -> (N, M)
    print("Performing matrix multiplication on GPU...")
    all_sim_scores = torch.matmul(text_norm, bg_norm.T)
    
    # Move results to CPU
    print("Moving results to CPU...")
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
    print(f"Using device: {device}\n")
    
    # Load model
    try:
        print(f"Loading model: {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()
        print("Model loaded successfully.\n")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return
    
    # Load background data
    background_data = load_and_embed_background_data(
        BACKGROUND_FILE, tokenizer, model, device
    )
    print(f"Loaded {len(background_data):,} background paragraphs.\n")
    
    # Embed texts of interest
    text_embeddings = get_text_embeddings(TEXTS_OF_INTEREST, tokenizer, model, device)
    text_token_counts = get_text_token_counts(TEXTS_OF_INTEREST, TEXT_LABELS, tokenizer)
    
    # Calculate direct pairwise similarity
    print(f"\nCalculating direct similarity between {TEXT_LABELS[0]} and {TEXT_LABELS[1]}...")
    direct_similarity = F.cosine_similarity(
        text_embeddings[0],
        text_embeddings[1],
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
    print(f"  Token Count {label_a}: {text_token_counts[label_a]}")
    print(f"  Token Count {label_b}: {text_token_counts[label_b]}")
    print(f"  Semantic Similarity: {direct_similarity:.6f}")
    
    # Calculate lexical similarities (Symmetric)
    print("\n  --- Symmetric Lexical Scores ---")
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
    
    print(f"  Unigram Jaccard: {score_uni:.6f}")
    print(f"  Bigram Jaccard: {score_bi:.6f}")
    print(f"  Trigram Jaccard: {score_tri:.6f}")
    print(f"  BoW Cosine: {score_bow:.6f}")

    # Calculate lexical coverage (Asymmetric)
    print(f"\n  --- Asymmetric Coverage ({label_a} by {label_b}) ---")
    score_uni_a_by_b = calculate_ngram_coverage(text_a, text_b, 1, tokenizer)
    score_bi_a_by_b = calculate_ngram_coverage(text_a, text_b, 2, tokenizer)
    score_tri_a_by_b = calculate_ngram_coverage(text_a, text_b, 3, tokenizer)
    print(f"  Unigram Cov ({label_a} by {label_b}): {score_uni_a_by_b:.6f}  (|\"{label_a}\" ∩ \"{label_b}\"| / |\"{label_a}\"|)")
    print(f"  Bigram Cov ({label_a} by {label_b}): {score_bi_a_by_b:.6f}")
    print(f"  Trigram Cov ({label_a} by {label_b}): {score_tri_a_by_b:.6f}")

    print(f"\n  --- Asymmetric Coverage ({label_b} by {label_a}) ---")
    score_uni_b_by_a = calculate_ngram_coverage(text_b, text_a, 1, tokenizer)
    score_bi_b_by_a = calculate_ngram_coverage(text_b, text_a, 2, tokenizer)
    score_tri_b_by_a = calculate_ngram_coverage(text_b, text_a, 3, tokenizer)
    print(f"  Unigram Cov ({label_b} by {label_a}): {score_uni_b_by_a:.6f}  (|\"{label_a}\" ∩ \"{label_b}\"| / |\"{label_b}\"|)")
    print(f"  Bigram Cov ({label_b} by {label_a}): {score_bi_b_by_a:.6f}")
    print(f"  Trigram Cov ({label_b} by {label_a}): {score_tri_b_by_a:.6f}")
    
    # Percentile rankings for Text A's view
    print(f"\nPercentile Rankings (from {label_a}'s perspective vs Background):")
    print(f"  Semantic: {percentileofscore(semantic_distributions[0], direct_similarity):.2f}th percentile")
    print(f"  Unigram Jaccard: {percentileofscore(lexical_distributions[label_a]['unigram'], score_uni):.2f}th percentile")
    print(f"  Bigram Jaccard: {percentileofscore(lexical_distributions[label_a]['bigram'], score_bi):.2f}th percentile")
    print(f"  Trigram Jaccard: {percentileofscore(lexical_distributions[label_a]['trigram'], score_tri):.2f}th percentile")
    print(f"  BoW Cosine: {percentileofscore(lexical_distributions[label_a]['bow_cosine'], score_bow):.2f}th percentile")
    print(f"  Unigram Coverage (of {label_a}): {percentileofscore(lexical_distributions[label_a]['unigram_coverage'], score_uni_a_by_b):.2f}th percentile")
    print(f"  Bigram Coverage (of {label_a}): {percentileofscore(lexical_distributions[label_a]['bigram_coverage'], score_bi_a_by_b):.2f}th percentile")
    print(f"  Trigram Coverage (of {label_a}): {percentileofscore(lexical_distributions[label_a]['trigram_coverage'], score_tri_a_by_b):.2f}th percentile")

    
    # Percentile rankings for Text B's view
    print(f"\nPercentile Rankings (from {label_b}'s perspective vs Background):")
    print(f"  Semantic: {percentileofscore(semantic_distributions[1], direct_similarity):.2f}th percentile")
    print(f"  Unigram Jaccard: {percentileofscore(lexical_distributions[label_b]['unigram'], score_uni):.2f}th percentile")
    print(f"  Bigram Jaccard: {percentileofscore(lexical_distributions[label_b]['bigram'], score_bi):.2f}th percentile")
    print(f"  Trigram Jaccard: {percentileofscore(lexical_distributions[label_b]['trigram'], score_tri):.2f}th percentile")
    print(f"  BoW Cosine: {percentileofscore(lexical_distributions[label_b]['bow_cosine'], score_bow):.2f}th percentile")
    print(f"  Unigram Coverage (of {label_b}): {percentileofscore(lexical_distributions[label_b]['unigram_coverage'], score_uni_b_by_a):.2f}th percentile")
    print(f"  Bigram Coverage (of {label_b}): {percentileofscore(lexical_distributions[label_b]['bigram_coverage'], score_bi_b_by_a):.2f}th percentile")
    print(f"  Trigram Coverage (of {label_b}): {percentileofscore(lexical_distributions[label_b]['trigram_coverage'], score_tri_b_by_a):.2f}th percentile")

    
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
            "num_paragraphs": len(background_data)
        },
        "model": MODEL_NAME
    }
    
    results_file = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nDetailed results saved to: {results_file}")
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()