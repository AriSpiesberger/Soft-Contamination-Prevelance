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
TARGET_TOKENS_PER_BATCH = 4000  # Total tokens to process per batch (B * L)
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
    (Robust Streaming Version)
    Load background paragraphs from JSONL file, FILTER BY TOKEN RANGE,
    and compute/cache embeddings for items missing them.
    
    This function uses a multi-pass, streaming approach to avoid
    loading the entire file into memory, ensuring stability and
    memory efficiency.
    
    It returns a list filtered by min_tokens/max_tokens for the current run.
    """
    if not os.path.exists(filepath):
        print(f"Error: Background file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading and filtering background data from {filepath}...")
    print(f"Applying token filter for this run: {min_tokens} <= tokens <= {max_tokens}")
    
    # --- Pass 1: Scan, Filter, and Collect Tasks ---
    print("--- Pass 1: Scanning file, collecting tasks ---")
    
    to_embed_tasks = [] # List of (token_count, text, line_index)
    tasks_need_token_count = {} # {line_index: text}
    
    filtered_data_indices = set() # Set of line_indices for our run
    lines_to_rewrite = set() # All lines that need any update
    
    total_lines_scanned = 0
    total_lines_filtered_out = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Pass 1: Scanning")):
            total_lines_scanned += 1
            try:
                data = json.loads(line)
                text = data['text']
                token_count = data.get('token_count')
                
                # 1. Check for token count
                if token_count is None:
                    tasks_need_token_count[i] = text
                    lines_to_rewrite.add(i)
                    # We'll compute this in a moment
                
                # 2. Check for embedding
                if EMBEDDER_KEY not in data:
                    lines_to_rewrite.add(i)
                    # We need token_count to batch properly
                    if token_count is None:
                        # Compute it now if we don't have it
                        token_count = len(tokenizer.tokenize(text))
                        data['token_count'] = token_count # For the next check
                    
                    to_embed_tasks.append((token_count, text, i))

                # 3. Check filter logic for this run
                if token_count is not None: # We must have token_count to filter
                    if min_tokens <= token_count <= max_tokens:
                        filtered_data_indices.add(i)
                    else:
                        total_lines_filtered_out += 1
                        
            except Exception as e:
                print(f"Skipping malformed line {i}: {e}", file=sys.stderr)
    
    print(f"Scan complete. Scanned {total_lines_scanned:,} lines.")
    print(f"Filtered out {total_lines_filtered_out:,} lines (outside token range).")
    print(f"Using {len(filtered_data_indices):,} lines for this analysis.")

    # --- Pre-computation: Token Counts ---
    new_token_counts = {} # {line_index: count}
    if tasks_need_token_count:
        print(f"Computing {len(tasks_need_token_count):,} missing token counts...")
        for i, text in tqdm(tasks_need_token_count.items(), desc="Tokenizing"):
            new_token_counts[i] = len(tokenizer.tokenize(text))

    # --- Pass 2: Compute Embeddings (if needed) ---
    new_embeddings_map = {} # {line_index: embedding_list}
    if to_embed_tasks:
        print(f"--- Pass 2: Computing {len(to_embed_tasks):,} embeddings ---")
        
        to_embed_tasks.sort(key=lambda x: x[0]) # Sort by token_count
        batches = list(create_dynamic_batches(to_embed_tasks, TARGET_TOKENS_PER_BATCH, MAX_BATCH_SIZE))
        
        if not batches and to_embed_tasks: 
            batches = [[(text, idx) for _, text, idx in to_embed_tasks]]

        if not batches:
             print("No batches to embed.")
        else:
            print(f"Created {len(batches)} dynamic batches.")
            print(f"   Batch sizes range: {min(len(b) for b in batches)} to {max(len(b) for b in batches)}")

        with torch.no_grad():
            for batch_data in tqdm(batches, desc="Embedding batches"):
                batch_texts = [text for text, _ in batch_data]
                batch_indices = [idx for _, idx in batch_data] # These are the original line indices
                
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
                    new_embeddings_map[original_index] = cpu_embeddings[j]
                    
        print("Embedding computation complete.")
    else:
        print("--- Pass 2: Skipped (No embeddings to compute) ---")

    # --- Pass 3: Stream-Write New File and Filter for Return ---
    needs_rewrite = bool(lines_to_rewrite)
    filtered_background_data = [] # This is the final list to return
    
    if needs_rewrite:
        print(f"--- Pass 3: Streaming new file to disk and filtering ---")
        temp_filepath = filepath + ".tmp"
        lines_written = 0
        try:
            with open(filepath, 'r', encoding='utf-8') as f_in, \
                 open(temp_filepath, 'w', encoding='utf-8') as f_out:
                
                for i, line in enumerate(tqdm(f_in, desc="Writing new file", total=total_lines_scanned)):
                    try:
                        data = json.loads(line)
                        
                        # Update token_count if it was missing
                        if i in new_token_counts:
                            data['token_count'] = new_token_counts[i]
                        
                        # Update embedding if it was missing
                        if i in new_embeddings_map:
                            
                            data[EMBEDDER_KEY] = new_embeddings_map[i]
                        
                        # Write the (potentially updated) line to the new file
                        f_out.write(json.dumps(data) + '\n')
                        lines_written += 1
                        
                        # Now, check if this line belongs in our return list
                        if i in filtered_data_indices:
                            # Add the *fully updated* data object
                            filtered_background_data.append(data)
                            
                    except Exception as e:
                        print(f"Error processing line {i} during write: {e}", file=sys.stderr)
            
            # Verify write
            if lines_written != total_lines_scanned:
                raise ValueError(f"Incomplete write: {lines_written:,} lines, expected {total_lines_scanned:,}")
            
            print(f"✓ Temp file verified: {lines_written:,} lines")
            
            # Atomic replace with retry
            print("Attempting to replace original file...")
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    os.replace(temp_filepath, filepath)
                    print(f"✓ Successfully updated {filepath}")
                    break
                except (PermissionError, OSError) as e:
                    if attempt < max_retries - 1:
                        print(f"  File locked, retrying in 3 seconds... ({attempt + 1}/{max_retries})")
                        time.sleep(3)
                    else:
                        print(f"\n{'='*60}", file=sys.stderr)
                        print(f"FILE LOCK ERROR (after {max_retries} attempts)", file=sys.stderr)
                        print(f"✓ Good news: Temp file is COMPLETE ({lines_written:,} lines)", file=sys.stderr)
                        print(f"✓ All embeddings are safely saved in: {temp_filepath}", file=sys.stderr)
                        print(f"TO FIX: 1. Close file locks. 2. Manually rename .tmp file.", file=sys.stderr)
                        print(f"{'='*60}\n", file=sys.stderr)
                        sys.exit(1)
                        
        except Exception as e:
            print(f"CRITICAL ERROR DURING FILE UPDATE: {e}", file=sys.stderr)
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            sys.exit(1)
            
    else:
        print("--- Pass 3: Skipped (No file update needed) ---")
        # File is already up to date, just build the filtered list from Pass 1
        # We need to re-read to get the embeddings
        print("Reading cached data for filtered list...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Reading cache", total=total_lines_scanned)):
                if i in filtered_data_indices:
                    try:
                        filtered_background_data.append(json.loads(line))
                    except Exception as e:
                        print(f"Skipping line {i}: {e}")

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
    Calculates trigram and quadgram Jaccard similarities.
    """
    global worker_tokenizer, worker_text_list

    chunk_results = {
        label: {
            "trigram_jaccard": [], # MODIFIED: Only trigram
            "quadgram_jaccard": [], # MODIFIED: Added quadgram
        }
        for label in TEXT_LABELS
    }

    for bg_entry in bg_chunk:
        try:
            bg_text = bg_entry['text']

            for i, text in enumerate(worker_text_list):
                label = TEXT_LABELS[i]

                # --- MODIFIED: Calculate only n=3 and n=4 Jaccard ---
                chunk_results[label]["trigram_jaccard"].append(
                    calculate_ngram_jaccard_similarity(text, bg_text, 3, worker_tokenizer)
                )
                chunk_results[label]["quadgram_jaccard"].append(
                    calculate_ngram_jaccard_similarity(text, bg_text, 4, worker_tokenizer) # Added n=4
                )
                # --- END MODIFICATION ---

        except Exception as e:
            print(f"Error in lexical worker {os.getpid()} processing entry: {bg_entry.get('id', 'Unknown')}. Error: {e}", file=sys.stderr)
            # Add NaNs or default values to maintain list lengths if needed, or just skip
            for label in TEXT_LABELS:
                 chunk_results[label]["trigram_jaccard"].append(float('nan'))
                 chunk_results[label]["quadgram_jaccard"].append(float('nan'))

    return chunk_results


def generate_bg_chunks(background_data: List[Dict[str, Any]], chunk_size: int):
    """Yield chunks of background data for multiprocessing."""
    for i in range(0, len(background_data), chunk_size):
        yield background_data[i:i+chunk_size]


def compute_all_lexical_scores(
    background_data: List[Dict[str, Any]]
) -> Dict[str, Dict[str, List[float]]]:
    """
    Orchestrate parallel computation of all lexical scores (trigram & quadgram Jaccard).
    """
    print(f"Computing lexical scores (parallelized across {NUM_WORKERS} workers)...")

    # --- MODIFIED: Updated dictionary structure ---
    all_lexical_dists = {
        label: {
            "trigram_jaccard": [],
            "quadgram_jaccard": [],
        }
        for label in TEXT_LABELS
    }
    # --- END MODIFICATION ---

    if not background_data:
        print("No background data to process for lexical scores.")
        return all_lexical_dists

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

        # --- MODIFIED: Update aggregation loop ---
        for chunk_results in pbar:
            for label in TEXT_LABELS:
                all_lexical_dists[label]["trigram_jaccard"].extend(
                    chunk_results[label]["trigram_jaccard"]
                )
                all_lexical_dists[label]["quadgram_jaccard"].extend(
                    chunk_results[label]["quadgram_jaccard"]
                )
        # --- END MODIFICATION ---

    # Optional: Clean NaNs if they were added during error handling
    # for label in TEXT_LABELS:
    #     for metric in all_lexical_dists[label]:
    #         all_lexical_dists[label][metric] = [s for s in all_lexical_dists[label][metric] if not math.isnan(s)]

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

def plot_lexical_histogram(
    scores: List[float],
    text_label: str,
    ngram_size: int,
    output_dir: str
):
    """
    Plots a histogram of n-gram Jaccard scores with a logarithmic y-axis.
    """
    print(f"Generating {ngram_size}-gram Jaccard histogram for {text_label}...")

    # Filter out NaNs
    valid_scores = [s for s in scores if not math.isnan(s)]

    if not valid_scores:
        print(f"Skipping {ngram_size}-gram histogram: No valid scores available.", file=sys.stderr)
        return

    try:
        plt.figure(figsize=(12, 7))
        plt.hist(valid_scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7, log=True)

        plt.title(f'{ngram_size}-gram Jaccard Similarity Distribution: {text_label} vs Background', fontsize=16, fontweight='bold')
        plt.xlabel(f'{ngram_size}-gram Jaccard Similarity', fontsize=13)
        plt.ylabel('Count (Log Scale)', fontsize=13)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.xlim(0, 1.0) # Jaccard scores are between 0 and 1

        # Add mean/median lines if desired
        mean_score = np.mean(valid_scores)
        median_score = np.median(valid_scores)
        plt.axvline(mean_score, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_score:.3f}')
        plt.axvline(median_score, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median_score:.3f}')
        plt.legend()


        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f"{text_label}_{ngram_size}gram_jaccard_histogram.png")
        plt.savefig(plot_filename, dpi=300)
        plt.close()

        print(f"Saved {ngram_size}-gram histogram to: {plot_filename}")

    except Exception as e:
        print(f"Error generating {ngram_size}-gram histogram for {text_label}: {e}", file=sys.stderr)
        # Attempt to close plot if it's still open
        try: plt.close()
        except: pass




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
    background_data = load_and_embed_background_data(
        BACKGROUND_FILE,
        tokenizer,
        model,
        device,
        min_tokens=MIN_TOKEN_LIMIT,
        max_tokens=MAX_TOKEN_LIMIT
    )

    # Check if filtering left anything usable
    if not background_data:
        print(f"Error: No background data available after loading/filtering (range {MIN_TOKEN_LIMIT}-{MAX_TOKEN_LIMIT} tokens). Cannot proceed.", file=sys.stderr)
        return

    print(f"Successfully prepared {len(background_data):,} paragraphs for analysis.\n")
    # --- END: Modified data loading ---


    # Embed texts of interest
    text_embeddings = get_text_embeddings(TEXTS_OF_INTEREST, tokenizer, model, device)
    text_token_counts = get_text_token_counts(TEXTS_OF_INTEREST, TEXT_LABELS, tokenizer)

    # Calculate direct pairwise similarity
    print(f"\nCalculating direct similarity between {TEXT_LABELS[0]} and {TEXT_LABELS[1]}...")
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
    if not semantic_distributions or not any(semantic_distributions):
         print("Warning: Semantic distributions are empty. Skipping plots and percentile stats.", file=sys.stderr)
         semantic_distributions = [[], []] # Ensure it's a list of two empty lists


    # Compute lexical distributions (Now only trigram/quadgram Jaccard)
    lexical_distributions = compute_all_lexical_scores(background_data)
    if not lexical_distributions or not any(v for d in lexical_distributions.values() for v in d.values()):
        print("Warning: Lexical distributions are empty. Skipping plots and percentile stats.", file=sys.stderr)
        # Ensure structure exists even if empty
        lexical_distributions = { label: {"trigram_jaccard": [], "quadgram_jaccard": []} for label in TEXT_LABELS }


    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_comparison_distributions(
        semantic_distributions[0],
        semantic_distributions[1],
        direct_similarity,
        OUTPUT_DIR
    )
    plot_bivariate_distribution(
        TEXT_LABELS[0], TEXTS_OF_INTEREST[0], background_data, semantic_distributions[0],
        text_token_counts[TEXT_LABELS[0]], TEXT_LABELS[1], text_token_counts[TEXT_LABELS[1]],
        direct_similarity, OUTPUT_DIR
    )
    plot_bivariate_distribution(
        TEXT_LABELS[1], TEXTS_OF_INTEREST[1], background_data, semantic_distributions[1],
        text_token_counts[TEXT_LABELS[1]], TEXT_LABELS[0], text_token_counts[TEXT_LABELS[0]],
        direct_similarity, OUTPUT_DIR
    )

    # --- START: Call new histogram plots ---
    # Plot histograms for the first text (e.g., "Duplicate") vs Background
    plot_lexical_histogram(
        lexical_distributions[TEXT_LABELS[0]]['trigram_jaccard'],
        TEXT_LABELS[0], # Label for the text being compared
        3, # n-gram size
        OUTPUT_DIR
    )
    plot_lexical_histogram(
        lexical_distributions[TEXT_LABELS[0]]['quadgram_jaccard'],
        TEXT_LABELS[0], # Label for the text being compared
        4, # n-gram size
        OUTPUT_DIR
    )
    # Plot histograms for the second text (e.g., "original") vs Background
    plot_lexical_histogram(
        lexical_distributions[TEXT_LABELS[1]]['trigram_jaccard'],
        TEXT_LABELS[1], # Label for the text being compared
        3, # n-gram size
        OUTPUT_DIR
    )
    plot_lexical_histogram(
        lexical_distributions[TEXT_LABELS[1]]['quadgram_jaccard'],
        TEXT_LABELS[1], # Label for the text being compared
        4, # n-gram size
        OUTPUT_DIR
    )
    # --- END: Call new histogram plots ---


    # Calculate and report statistics
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    label_a = TEXT_LABELS[0]
    label_b = TEXT_LABELS[1]
    text_a = TEXTS_OF_INTEREST[0]
    text_b = TEXTS_OF_INTEREST[1]

    print(f"\nDirect Comparison: {label_a} vs {label_b}")
    print(f"   Token Count {label_a}: {text_token_counts[label_a]}")
    print(f"   Token Count {label_b}: {text_token_counts[label_b]}")
    print(f"   Semantic Similarity: {direct_similarity:.6f}")

    # --- MODIFIED: Calculate only trigram/quadgram Jaccard ---
    print("\n   --- Symmetric Lexical Scores ---")
    score_tri = calculate_ngram_jaccard_similarity(
        text_a, text_b, 3, tokenizer
    )
    score_quad = calculate_ngram_jaccard_similarity( # Added quadgram
        text_a, text_b, 4, tokenizer
    )
    print(f"   Trigram Jaccard: {score_tri:.6f}")
    print(f"   Quadgram Jaccard: {score_quad:.6f}") # Added quadgram
    # --- END MODIFICATION ---

    # Helper for safe percentile calculation
    def safe_percentile(scores: List[float], value: float) -> float:
        # Filter NaNs before calculating percentile
        valid_scores = [s for s in scores if not math.isnan(s)]
        if not valid_scores or math.isnan(value):
            return 0.0 # Or np.nan, or however you want to handle insufficient data
        # Use 'strict' for percentile (value must be > score to count)
        return percentileofscore(valid_scores, value, kind='strict')


    # --- MODIFIED: Percentile rankings ---
    print(f"\nPercentile Rankings (from {label_a}'s perspective vs Background):")
    print(f"   Semantic: {safe_percentile(semantic_distributions[0], direct_similarity):.2f}th percentile")
    print(f"   Trigram Jaccard: {safe_percentile(lexical_distributions[label_a]['trigram_jaccard'], score_tri):.2f}th percentile")
    print(f"   Quadgram Jaccard: {safe_percentile(lexical_distributions[label_a]['quadgram_jaccard'], score_quad):.2f}th percentile") # Added quadgram

    print(f"\nPercentile Rankings (from {label_b}'s perspective vs Background):")
    print(f"   Semantic: {safe_percentile(semantic_distributions[1], direct_similarity):.2f}th percentile")
    print(f"   Trigram Jaccard: {safe_percentile(lexical_distributions[label_b]['trigram_jaccard'], score_tri):.2f}th percentile")
    print(f"   Quadgram Jaccard: {safe_percentile(lexical_distributions[label_b]['quadgram_jaccard'], score_quad):.2f}th percentile") # Added quadgram
    # --- END MODIFICATION ---


    # --- MODIFIED: Save detailed results ---
    results = {
        "text_a": {
            "label": label_a,
            "text": text_a,
            "token_count": text_token_counts[label_a],
            # Store only valid scores (filter NaNs)
            "semantic_distribution": [s for s in semantic_distributions[0] if not math.isnan(s)],
            "lexical_distributions": {k: [s for s in v if not math.isnan(s)] for k, v in lexical_distributions[label_a].items()}
        },
        "text_b": {
            "label": label_b,
            "text": text_b,
            "token_count": text_token_counts[label_b],
            "semantic_distribution": [s for s in semantic_distributions[1] if not math.isnan(s)],
             "lexical_distributions": {k: [s for s in v if not math.isnan(s)] for k, v in lexical_distributions[label_b].items()}
        },
        "pairwise_comparison": {
            "semantic_similarity": direct_similarity,
            "trigram_jaccard": score_tri,
            "quadgram_jaccard": score_quad, # Added quadgram
        },
        "background_corpus": {
            "file": BACKGROUND_FILE,
            "num_paragraphs_used": len(background_data), # This already reflects filtering
            "token_limit_min": MIN_TOKEN_LIMIT,
            "token_limit_max": MAX_TOKEN_LIMIT
        },
        "model": MODEL_NAME
    }
    # --- END MODIFICATION ---

    results_file = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        # A simple numpy-safe encoder
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int_)): # Corrected type
                    return int(obj)
                if isinstance(obj, (np.floating, np.float_)): # Corrected type
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (torch.Tensor)): # Handle potential tensors
                    return obj.tolist()
                if math.isnan(obj): # Handle NaNs explicitly if needed
                    return None # Represent NaN as null in JSON
                return super(NpEncoder, self).default(obj)

        # allow_nan=False ensures JSON standard compliance (no literal NaN/Infinity)
        json.dump(results, f, indent=2, cls=NpEncoder, allow_nan=False)

    print(f"\n\nDetailed results saved to: {results_file}")
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()