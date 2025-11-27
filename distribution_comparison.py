#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Similarity Analyzer - Randomized Sampling Edition
Optimized for NVIDIA A100

Modifications:
- Fixed unpacking error in batch generation.
- Implements Uniform Random Sampling (N=100k) from the background corpus.
- Defers embedding computation until after sampling to save GPU compute.
- Preserves cache write-back for partial updates.
"""
import os
import sys
import json
import math
import multiprocessing
import random
from typing import List, Set, Dict, Any, Tuple
from functools import partial

# --- CRASH FIX: Monkeypatch np.int for compatibility ---
import numpy as np
if not hasattr(np, 'int'):
    np.int = int
# -------------------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import gc
import tiktoken

# =============================================================================
# CONFIGURATION
# =============================================================================
TEXT_1 = """A system has two toggles, blue and green with multiple settings. The system is arranged in a way such that if both the blue and green are both at there max settings at the same time, then it will fail. The system does not fail if only one of blue or green are at max settings. In our set up, the designers decided that the blue toggle should be at max, and the green setting should be at some other useful setting. Today, both the blue and the green toggles are at max. The system fails. Did the blue toggle cause the system failure?"""
TEXT_2 = """How would a typical person answer each of the following questions about causation? A machine is set up in such a way that it will short circuit if both the black wire and the red wire touch the battery at the same time. The machine will not short circuit if just one of these wires touches the battery. The black wire is designated as the one that is supposed to touch the battery, while the red wire is supposed to remain in some other part of the machine. One day, the black wire and the red wire both end up touching the battery at the same time. There is a short circuit. Did the black wire cause the short circuit? Options: - Yes - No"""

TEXTS_OF_INTEREST = [TEXT_1, TEXT_2]
TEXT_LABELS = ["Duplicate", "original"]
BACKGROUND_FILE = "data/full_paragraphs.jsonl"
OUTPUT_DIR = "duplicate_comparison_sampled"

MODEL_NAME = 'nvidia/llama-embed-nemotron-8b'
EMBEDDER_KEY = MODEL_NAME.replace('/', '-') + "_vector"

# --- SAMPLING CONFIGURATION ---
SAMPLE_SIZE = 100_000
RANDOM_SEED = 42

# --- A100 TUNED SETTINGS ---
TARGET_TOKENS_PER_BATCH = 120_000
MAX_BATCH_SIZE = 512
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)
LEXICAL_CHUNK_SIZE = 5000
MAX_SEQ_LENGTH = 384
SEMANTIC_CHUNK_SIZE = 50_000
USE_TORCH_COMPILE_POOLING = True
USE_FLASH_ATTENTION = True

# =============================================================================
# COMPILED MEAN POOLING
# =============================================================================
def _mean_pooling_impl(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

if USE_TORCH_COMPILE_POOLING and torch.cuda.is_available():
    try:
        mean_pooling = torch.compile(_mean_pooling_impl, mode="reduce-overhead")
    except Exception as e:
        print(f"Could not compile mean_pooling: {e}, using uncompiled version")
        mean_pooling = _mean_pooling_impl
else:
    mean_pooling = _mean_pooling_impl


# =============================================================================
# DYNAMIC BATCHING
# =============================================================================
def create_dynamic_batches(items_with_counts: List[tuple], target_tokens: int, max_batch_size: int):
    """Yields batches of (text, original_idx) tuples, grouped by similar token counts."""
    current_batch = []
    current_max_len = 0

    for token_count, text, original_idx in items_with_counts:
        potential_max_len = max(current_max_len, token_count)
        potential_batch_size = len(current_batch) + 1
        padded_estimate = potential_max_len * potential_batch_size

        if (padded_estimate > target_tokens * 1.1 or potential_batch_size >= max_batch_size):
            if current_batch:
                # Yields a list of 2-tuples: (text, idx)
                yield [(text, idx) for _, text, idx in current_batch]
                current_batch = []
                current_max_len = 0

        current_batch.append((token_count, text, original_idx))
        current_max_len = max(current_max_len, token_count)

    if current_batch:
        # Yields a list of 2-tuples: (text, idx)
        yield [(text, idx) for _, text, idx in current_batch]


# =============================================================================
# SAMPLING & LOADING LOGIC
# =============================================================================
def load_and_embed_background_data(filepath, tokenizer, model, device, min_tokens=0, max_tokens=float('inf')):
    """
    Optimized loader with Randomized Sampling.
    """
    if not os.path.exists(filepath):
        sys.exit(f"Error: Background file not found: {filepath}")

    random.seed(RANDOM_SEED)
    print(f"Loading background data from {filepath}...")

    # Fast tokenizer setup
    try:
        fast_tokenizer = tiktoken.get_encoding("cl100k_base")
        use_tiktoken = True
    except:
        fast_tokenizer = None
        use_tiktoken = False

    all_data = [] # Store all parsed entries
    lines_modified = set()
    valid_indices = [] # Indices satisfying length constraints

    # --- Pass 1: Scan, Tokenize, Filter ---
    print("Pass 1: Scanning and tokenizing...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Scanning")):
            if not line.strip():
                all_data.append(None)
                continue

            try:
                data = json.loads(line)
                text = data.get('text', '')

                # Compute token count if missing
                token_count = data.get('token_count') or data.get('token_size')
                if token_count is None:
                    if use_tiktoken:
                        token_count = len(fast_tokenizer.encode(text))
                    else:
                        token_count = len(tokenizer.tokenize(text))
                    data['token_count'] = token_count
                    lines_modified.add(i)
                elif data.get('token_count') is None and data.get('token_size') is not None:
                    data['token_count'] = data['token_size']
                    lines_modified.add(i)

                all_data.append(data)

                # Check constraints for valid pool
                if min_tokens <= token_count <= max_tokens:
                    valid_indices.append(i)

            except Exception as e:
                all_data.append(None)
                continue

    total_lines = len(all_data)
    valid_count = len(valid_indices)
    print(f"Scanned {total_lines:,} lines. Found {valid_count:,} valid candidates (length {min_tokens}-{max_tokens}).")

    # --- Step 2: Random Sampling ---
    if valid_count > SAMPLE_SIZE:
        print(f"Downsampling from {valid_count:,} to {SAMPLE_SIZE:,}...")
        sampled_indices = random.sample(valid_indices, SAMPLE_SIZE)
    else:
        print(f"Available candidates ({valid_count:,}) <= Sample Size ({SAMPLE_SIZE:,}). Taking all.")
        sampled_indices = valid_indices

    # Sort indices for sequential access consistency (optional but cleaner)
    sampled_indices.sort()

    # --- Step 3: Identify Missing Embeddings IN SAMPLE ONLY ---
    to_embed_tasks = [] # (token_count, text, index)
    
    for idx in sampled_indices:
        data = all_data[idx]
        if EMBEDDER_KEY not in data:
            to_embed_tasks.append((data['token_count'], data['text'], idx))
            lines_modified.add(idx)

    # --- Step 4: Compute Embeddings ---
    if to_embed_tasks:
        print(f"Computing {len(to_embed_tasks):,} embeddings (Sampled Subset Only)...")
        to_embed_tasks.sort(key=lambda x: x[0]) # Sort by length for efficient dynamic batching

        batches = list(create_dynamic_batches(to_embed_tasks, TARGET_TOKENS_PER_BATCH, MAX_BATCH_SIZE))

        with torch.no_grad():
            for batch_data in tqdm(batches, desc="Embedding batches"):
                # FIX: Unpack 2 items (text, idx) instead of 3
                batch_texts = [text for text, _ in batch_data]
                batch_indices = [idx for _, idx in batch_data]

                encoded_input = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_SEQ_LENGTH,
                    return_tensors='pt'
                )
                encoded_input = {k: v.to(device, non_blocking=True) for k, v in encoded_input.items()}

                model_output = model(**encoded_input)
                batch_embeddings = mean_pooling(model_output[0], encoded_input['attention_mask'])

                # Transfer to CPU and store in main list
                cpu_embeddings = batch_embeddings.cpu().numpy()

                for j, original_index in enumerate(batch_indices):
                    all_data[original_index][EMBEDDER_KEY] = cpu_embeddings[j].tolist()

                del batch_embeddings, cpu_embeddings, model_output
                torch.cuda.empty_cache()

    # --- Step 5: Write Back (Cache Update) ---
    # We update the file so that next time, these 100k samples don't need re-embedding
    if lines_modified:
        print(f"Pass 2: Updating cache for {len(lines_modified):,} modified lines...")
        temp_filepath = filepath + ".tmp"
        with open(temp_filepath, 'w', encoding='utf-8') as f_out:
            for data in tqdm(all_data, desc="Writing cache"):
                if data is not None:
                    f_out.write(json.dumps(data) + '\n')
        os.replace(temp_filepath, filepath)
        print("Cache updated.")
    else:
        print("No modifications needed, skipping write pass.")

    # --- Step 6: Return Sampled Data ---
    print("Constructing final sampled dataset...")
    final_dataset = [all_data[i] for i in sampled_indices if all_data[i] is not None]
    
    del all_data
    gc.collect()
    
    return final_dataset


# =============================================================================
# OPTIMIZED LEXICAL SIMILARITY
# =============================================================================
def get_ngram_hashes(text: str, n: int, encoder) -> Set[int]:
    try:
        tokens = encoder.encode(text, disallowed_special=())
    except:
        try:
            tokens = encoder.encode(text, add_special_tokens=False)
        except:
            tokens = encoder.encode(text)

    if len(tokens) < n:
        return set()
    return {hash(tuple(tokens[i:i+n])) for i in range(len(tokens) - n + 1)}

def _hash_entry_worker(text: str) -> Tuple[Set[int], Set[int]]:
    encoder = tiktoken.get_encoding("cl100k_base")
    grams_3 = get_ngram_hashes(text, 3, encoder)
    grams_4 = get_ngram_hashes(text, 4, encoder)
    return grams_3, grams_4

def precompute_lexical_features_parallel(background_data: List[Dict], num_workers: int) -> List[Dict]:
    print(f"Pre-computing lexical n-gram sets in parallel ({num_workers} workers)...")
    texts = [entry.get('text', '') for entry in background_data]
    chunk_size = max(100, len(texts) // (num_workers * 10))

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(_hash_entry_worker, texts, chunksize=chunk_size),
            total=len(texts),
            desc="Hashing N-grams"
        ))

    for i, (grams_3, grams_4) in enumerate(results):
        background_data[i]['grams_3'] = grams_3
        background_data[i]['grams_4'] = grams_4

    return background_data

def process_lexical_chunk_optimized(payload):
    bg_chunk, target_sets_3, target_sets_4, labels = payload
    chunk_results = {label: {"trigram_jaccard": [], "quadgram_jaccard": []} for label in labels}

    for bg_entry in bg_chunk:
        bg_set_3 = bg_entry.get('grams_3', set())
        bg_set_4 = bg_entry.get('grams_4', set())

        for i, label in enumerate(labels):
            tgt_3 = target_sets_3[i]
            if not tgt_3 or not bg_set_3:
                score_3 = 0.0
            else:
                intersect = len(tgt_3 & bg_set_3)
                union = len(tgt_3 | bg_set_3)
                score_3 = intersect / union if union > 0 else 0.0
            chunk_results[label]["trigram_jaccard"].append(score_3)

            tgt_4 = target_sets_4[i]
            if not tgt_4 or not bg_set_4:
                score_4 = 0.0
            else:
                intersect = len(tgt_4 & bg_set_4)
                union = len(tgt_4 | bg_set_4)
                score_4 = intersect / union if union > 0 else 0.0
            chunk_results[label]["quadgram_jaccard"].append(score_4)
    return chunk_results

def compute_all_lexical_scores_optimized(background_data, target_sets_3, target_sets_4, labels):
    print(f"Computing lexical scores ({NUM_WORKERS} workers)...")
    all_lexical_dists = {label: {"trigram_jaccard": [], "quadgram_jaccard": []} for label in labels}
    
    if not background_data:
        return all_lexical_dists

    chunk_size = LEXICAL_CHUNK_SIZE
    chunks = [background_data[i:i + chunk_size] for i in range(0, len(background_data), chunk_size)]
    tasks = [(chunk, target_sets_3, target_sets_4, labels) for chunk in chunks]

    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        for chunk_results in tqdm(pool.imap_unordered(process_lexical_chunk_optimized, tasks), total=len(chunks), desc="Lexical Scoring"):
            for label in labels:
                all_lexical_dists[label]["trigram_jaccard"].extend(chunk_results[label]["trigram_jaccard"])
                all_lexical_dists[label]["quadgram_jaccard"].extend(chunk_results[label]["quadgram_jaccard"])

    return all_lexical_dists


# =============================================================================
# STREAMING SEMANTIC SCORE COMPUTATION
# =============================================================================
def compute_all_semantic_scores_streaming(text_embeddings, background_data, device):
    print("Computing semantic scores (streaming)...")
    text_embeddings_gpu = text_embeddings.to(dtype=torch.float16, device=device)
    text_embeddings_norm = F.normalize(text_embeddings_gpu, p=2, dim=1)
    
    num_texts = text_embeddings_norm.shape[0]
    all_scores = [[] for _ in range(num_texts)]
    
    chunk_size = SEMANTIC_CHUNK_SIZE
    
    for i in tqdm(range(0, len(background_data), chunk_size), desc="Semantic similarity"):
        end_idx = min(i + chunk_size, len(background_data))
        chunk_vectors = [background_data[j][EMBEDDER_KEY] for j in range(i, end_idx)]
        
        bg_chunk = torch.tensor(chunk_vectors, dtype=torch.float16, device=device)
        bg_chunk_norm = F.normalize(bg_chunk, p=2, dim=1)
        
        scores = torch.matmul(text_embeddings_norm, bg_chunk_norm.T)
        scores_cpu = scores.cpu()
        
        for t in range(num_texts):
            all_scores[t].extend(scores_cpu[t].tolist())
            
        del bg_chunk, bg_chunk_norm, scores, scores_cpu

    del text_embeddings_gpu, text_embeddings_norm
    torch.cuda.empty_cache()
    return all_scores


# =============================================================================
# EMBEDDING HELPERS & PLOTTING
# =============================================================================
def get_text_embeddings(texts, tokenizer, model, device):
    with torch.no_grad():
        encoded_input = tokenizer(texts, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors='pt')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        model_output = model(**encoded_input)
        embeddings = mean_pooling(model_output[0], encoded_input['attention_mask'])
    return embeddings

def get_text_token_counts(texts, labels, tokenizer):
    return {labels[i]: len(tokenizer.tokenize(text)[:MAX_SEQ_LENGTH]) for i, text in enumerate(texts)}

def plot_comparison_distributions(text_a_scores, text_b_scores, direct_similarity, output_dir):
    try:
        df = pd.DataFrame({
            'score': text_a_scores + text_b_scores,
            'text': [TEXT_LABELS[0]] * len(text_a_scores) + [TEXT_LABELS[1]] * len(text_b_scores)
        })
        plt.figure(figsize=(14, 8))
        sns.kdeplot(data=df, x='score', hue='text', fill=True, common_norm=False, alpha=0.3, linewidth=2.5)
        plt.axvline(x=direct_similarity, color='red', linestyle='--', linewidth=2, label=f'Direct: {direct_similarity:.4f}')
        plt.title(f'Semantic Similarity (Sampled N={len(text_a_scores):,})', fontsize=16)
        plt.xlabel('Cosine Similarity')
        plt.savefig(os.path.join(output_dir, "semantic_comparison.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Plotting error: {e}")

def plot_bivariate_distribution(label, text, background_data, semantic_scores, token_count, other_label, other_token_count, pairwise_score, output_dir):
    try:
        bg_token_counts = [max(1, d.get('token_count', 1)) for d in background_data]
        if not bg_token_counts or not semantic_scores: return
        df = pd.DataFrame({"semantic_score": semantic_scores, "token_count": bg_token_counts})
        
        g = sns.jointplot(data=df, x="semantic_score", y="token_count", kind="kde", fill=True, cmap="viridis", height=10, log_scale=(False, True))
        g.fig.suptitle(f'Bivariate Distribution: {label}', fontsize=14, y=1.03)
        g.ax_joint.scatter([pairwise_score], [other_token_count], s=150, c='red', edgecolor='white', linewidth=2, zorder=5)
        g.fig.savefig(os.path.join(output_dir, f"{label}_bivariate.png"), dpi=300)
        plt.close(g.fig)
    except Exception as e:
        print(f"Bivariate plot error: {e}")

def plot_lexical_histograms(trigram_scores, quadgram_scores, text_label, output_dir):
    try:
        valid_tri = [s for s in trigram_scores if not math.isnan(s) and s > 0]
        valid_quad = [s for s in quadgram_scores if not math.isnan(s) and s > 0]
        if not valid_tri and not valid_quad: return
        
        plt.figure(figsize=(12, 7))
        bins = np.logspace(-5, 0, 51)
        plt.hist(valid_tri, bins=bins, color='dodgerblue', alpha=0.6, log=True, label='Trigrams')
        plt.hist(valid_quad, bins=bins, color='orangered', alpha=0.6, log=True, label='Quadgrams')
        plt.xscale('log')
        plt.title(f'N-gram Jaccard (Log-Log): {text_label}', fontsize=16)
        plt.savefig(os.path.join(output_dir, f"{text_label}_ngram_hist.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Histogram error: {e}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("SEMANTIC ANALYSIS (N=100k RANDOM SAMPLING)")
    print("=" * 80)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cuda": torch.set_float32_matmul_precision("high")

    # Load Model
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code = True)
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype, attn_implementation="flash_attention_2" if device=="cuda" else "sdpa", trust_remote_code = True)
    model.to(device).eval()

    # Load and Sample Background Data
    MIN_TOKEN_LIMIT = 10
    MAX_TOKEN_LIMIT = 300
    
    background_data = load_and_embed_background_data(
        BACKGROUND_FILE, tokenizer, model, device,
        min_tokens=MIN_TOKEN_LIMIT, max_tokens=MAX_TOKEN_LIMIT
    )

    if not background_data:
        print("No data loaded.")
        return

    # Process Features
    background_data = precompute_lexical_features_parallel(background_data, NUM_WORKERS)
    
    try:
        target_encoder = tiktoken.get_encoding("cl100k_base")
    except:
        target_encoder = tokenizer

    target_sets_3 = [get_ngram_hashes(t, 3, target_encoder) for t in TEXTS_OF_INTEREST]
    target_sets_4 = [get_ngram_hashes(t, 4, target_encoder) for t in TEXTS_OF_INTEREST]

    text_embeddings = get_text_embeddings(TEXTS_OF_INTEREST, tokenizer, model, device)
    text_token_counts = get_text_token_counts(TEXTS_OF_INTEREST, TEXT_LABELS, tokenizer)
    
    direct_similarity = F.cosine_similarity(text_embeddings[0].float(), text_embeddings[1].float(), dim=0).item()
    print(f"\nDirect similarity: {direct_similarity:.6f}")

    semantic_distributions = compute_all_semantic_scores_streaming(text_embeddings, background_data, device)
    if not semantic_distributions: semantic_distributions = [[], []]
    
    lexical_distributions = compute_all_lexical_scores_optimized(background_data, target_sets_3, target_sets_4, TEXT_LABELS)

    # Visualization & Stats
    print("\nGenerating outputs...")
    plot_comparison_distributions(semantic_distributions[0], semantic_distributions[1], direct_similarity, OUTPUT_DIR)
    
    plot_bivariate_distribution(TEXT_LABELS[0], TEXTS_OF_INTEREST[0], background_data, semantic_distributions[0], text_token_counts[TEXT_LABELS[0]], TEXT_LABELS[1], text_token_counts[TEXT_LABELS[1]], direct_similarity, OUTPUT_DIR)
    plot_bivariate_distribution(TEXT_LABELS[1], TEXTS_OF_INTEREST[1], background_data, semantic_distributions[1], text_token_counts[TEXT_LABELS[1]], TEXT_LABELS[0], text_token_counts[TEXT_LABELS[0]], direct_similarity, OUTPUT_DIR)

    plot_lexical_histograms(lexical_distributions[TEXT_LABELS[0]]['trigram_jaccard'], lexical_distributions[TEXT_LABELS[0]]['quadgram_jaccard'], TEXT_LABELS[0], OUTPUT_DIR)
    plot_lexical_histograms(lexical_distributions[TEXT_LABELS[1]]['trigram_jaccard'], lexical_distributions[TEXT_LABELS[1]]['quadgram_jaccard'], TEXT_LABELS[1], OUTPUT_DIR)

    def safe_percentile(scores, value):
        valid = [s for s in scores if not math.isnan(s)]
        return percentileofscore(valid, value, kind='strict') if valid else 0.0

    results = {
        "text_a": {"label": TEXT_LABELS[0], "token_count": text_token_counts[TEXT_LABELS[0]]},
        "stats": {
            "sample_size": len(background_data),
            "percentile_rank": safe_percentile(semantic_distributions[0], direct_similarity),
            "direct_similarity": direct_similarity
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()