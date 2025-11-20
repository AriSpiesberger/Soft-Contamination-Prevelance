#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV Distribution Comparison - Comparing Original Stories vs Regenerated Stories

This script reads a CSV file with original_story and regenerated_story_n columns,
and compares each pair against a background corpus to analyze distributional differences.

Reuses functionality from distribution_comparison.py for the actual comparison logic.
"""

import os
import sys
import json
import math
import multiprocessing
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Import functions from the main distribution comparison script
# We'll import specific functions we need
import distribution_comparison as dc
from utils.utilities import calculate_ngram_jaccard_similarity

# =============================================================================
# CONFIGURATION
# =============================================================================

# CSV file with original_story and regenerated_story_n columns
CSV_FILE = "data/comparisons.csv"  # Update with your CSV file path
ORIGINAL_STORY_COLUMN = "original_story"  # Column name for original stories
REGENERATED_STORY_PREFIX = "regenerated_story_"  # Prefix for regenerated story columns

# Background corpus and model settings (from distribution_comparison.py)
BACKGROUND_FILE = r"data/random_paragraphs.jsonl"
OUTPUT_DIR = "csv_comparison_results"
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"

# Token filtering
MIN_TOKEN_LIMIT = 10
MAX_TOKEN_LIMIT = 2000

# =============================================================================
# CSV DATA LOADING
# =============================================================================

def load_csv_data(csv_file: str) -> List[Dict[str, Any]]:
    """
    Load data from CSV file and identify original_story and regenerated_story columns.
    
    Returns:
        List of dictionaries with keys: 'row_index', 'original_story', 'regenerated_stories'
    """
    print(f"Loading CSV file: {csv_file}")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    # Check for original_story column
    if ORIGINAL_STORY_COLUMN not in df.columns:
        raise ValueError(f"Column '{ORIGINAL_STORY_COLUMN}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Find all regenerated_story columns
    regenerated_columns = [col for col in df.columns if col.startswith(REGENERATED_STORY_PREFIX)]
    
    if not regenerated_columns:
        raise ValueError(f"No columns starting with '{REGENERATED_STORY_PREFIX}' found. Available columns: {list(df.columns)}")
    
    print(f"Found {len(regenerated_columns)} regenerated story columns: {regenerated_columns}")
    
    # Process each row
    rows_data = []
    for idx, row in df.iterrows():
        original_story = row[ORIGINAL_STORY_COLUMN]
        
        # Skip rows with empty original_story
        if pd.isna(original_story) or str(original_story).strip() == "":
            print(f"Warning: Row {idx} has empty original_story, skipping...")
            continue
        
        # Collect regenerated stories for this row
        regenerated_stories = []
        for col in regenerated_columns:
            regenerated_text = row[col]
            if pd.isna(regenerated_text) or str(regenerated_text).strip() == "":
                continue
            regenerated_stories.append({
                'name': col,
                'text': str(regenerated_text).strip()
            })
        
        if not regenerated_stories:
            print(f"Warning: Row {idx} has no valid regenerated stories, skipping...")
            continue
        
        rows_data.append({
            'row_index': idx,
            'original_story': str(original_story).strip(),
            'regenerated_stories': regenerated_stories
        })
    
    print(f"Loaded {len(rows_data)} valid rows from CSV.")
    return rows_data

# =============================================================================
# COMPARISON FUNCTION
# =============================================================================

def run_single_comparison(
    original_text: str,
    regenerated_text: str,
    regenerated_label: str,
    background_data: List[Dict[str, Any]],
    tokenizer,
    model,
    device: str,
    output_dir: str,
    comparison_name: str
) -> Dict[str, Any]:
    """
    Run a single comparison between original_story and a regenerated_story against background corpus.
    
    Reuses functions from distribution_comparison.py but with our own texts.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    texts_of_interest = [original_text, regenerated_text]
    text_labels = ["original", regenerated_label]
    
    # Embed texts of interest
    print(f"  Computing embeddings for {comparison_name}...")
    text_embeddings = dc.get_text_embeddings(texts_of_interest, tokenizer, model, device)
    text_token_counts = dc.get_text_token_counts(texts_of_interest, text_labels, tokenizer)
    
    # Calculate direct pairwise similarity
    direct_similarity = F.cosine_similarity(
        text_embeddings[0], 
        text_embeddings[1], 
        dim=0
    ).item()
    
    # Compute semantic distributions against background
    print(f"  Computing semantic scores for {comparison_name}...")
    semantic_distributions = dc.compute_all_semantic_scores(
        text_embeddings, background_data, device
    )
    if not semantic_distributions or not any(semantic_distributions):
        print(f"  Warning: Semantic distributions are empty for {comparison_name}", file=sys.stderr)
        semantic_distributions = [[], []]
    
    # Compute lexical distributions against background
    # Update global variables temporarily for worker processes and plotting functions
    print(f"  Computing lexical scores for {comparison_name}...")
    old_texts = dc.TEXTS_OF_INTEREST
    old_labels = dc.TEXT_LABELS
    dc.TEXTS_OF_INTEREST = texts_of_interest
    dc.TEXT_LABELS = text_labels
    
    lexical_distributions = dc.compute_all_lexical_scores(background_data)
    if not lexical_distributions or not any(v for d in lexical_distributions.values() for v in d.values()):
        print(f"  Warning: Lexical distributions are empty for {comparison_name}", file=sys.stderr)
        lexical_distributions = {label: {"trigram_jaccard": [], "quadgram_jaccard": []} for label in text_labels}
    
    # Calculate lexical scores between texts
    score_tri = calculate_ngram_jaccard_similarity(original_text, regenerated_text, 3, tokenizer)
    score_quad = calculate_ngram_jaccard_similarity(original_text, regenerated_text, 4, tokenizer)
    
    # Helper for safe percentile calculation
    def safe_percentile(scores: List[float], value: float) -> float:
        valid_scores = [s for s in scores if not math.isnan(s)]
        if not valid_scores or math.isnan(value):
            return 0.0
        return percentileofscore(valid_scores, value, kind='strict')
    
    # Calculate percentile rankings
    original_percentiles = {
        "semantic": safe_percentile(semantic_distributions[0], direct_similarity),
        "trigram_jaccard": safe_percentile(lexical_distributions["original"]['trigram_jaccard'], score_tri),
        "quadgram_jaccard": safe_percentile(lexical_distributions["original"]['quadgram_jaccard'], score_quad)
    }
    
    regenerated_percentiles = {
        "semantic": safe_percentile(semantic_distributions[1], direct_similarity),
        "trigram_jaccard": safe_percentile(lexical_distributions[regenerated_label]['trigram_jaccard'], score_tri),
        "quadgram_jaccard": safe_percentile(lexical_distributions[regenerated_label]['quadgram_jaccard'], score_quad)
    }
    
    # Generate visualizations (TEXT_LABELS already set above)
    print(f"  Generating visualizations for {comparison_name}...")
    dc.plot_comparison_distributions(
        semantic_distributions[0],
        semantic_distributions[1],
        direct_similarity,
        output_dir
    )
    dc.plot_bivariate_distribution(
        "original", original_text, background_data, semantic_distributions[0],
        text_token_counts["original"], regenerated_label, text_token_counts[regenerated_label],
        direct_similarity, output_dir
    )
    dc.plot_bivariate_distribution(
        regenerated_label, regenerated_text, background_data, semantic_distributions[1],
        text_token_counts[regenerated_label], "original", text_token_counts["original"],
        direct_similarity, output_dir
    )
    dc.plot_lexical_histograms(
        lexical_distributions["original"]['trigram_jaccard'],
        lexical_distributions["original"]['quadgram_jaccard'],
        "original",
        output_dir
    )
    dc.plot_lexical_histograms(
        lexical_distributions[regenerated_label]['trigram_jaccard'],
        lexical_distributions[regenerated_label]['quadgram_jaccard'],
        regenerated_label,
        output_dir
    )
    
    # Restore global variables now that we're done with lexical computation and plotting
    dc.TEXTS_OF_INTEREST = old_texts
    dc.TEXT_LABELS = old_labels
    
    # Prepare results dictionary
    results = {
        "original_story": {
            "text": original_text,
            "token_count": text_token_counts["original"],
            "semantic_distribution": [s for s in semantic_distributions[0] if not math.isnan(s)],
            "lexical_distributions": {
                k: [s for s in v if not math.isnan(s)] 
                for k, v in lexical_distributions["original"].items()
            },
            "percentile_rankings": original_percentiles
        },
        "regenerated_story": {
            "label": regenerated_label,
            "text": regenerated_text,
            "token_count": text_token_counts[regenerated_label],
            "semantic_distribution": [s for s in semantic_distributions[1] if not math.isnan(s)],
            "lexical_distributions": {
                k: [s for s in v if not math.isnan(s)] 
                for k, v in lexical_distributions[regenerated_label].items()
            },
            "percentile_rankings": regenerated_percentiles
        },
        "pairwise_comparison": {
            "semantic_similarity": direct_similarity,
            "trigram_jaccard": score_tri,
            "quadgram_jaccard": score_quad
        }
    }
    
    # Save results
    results_file = os.path.join(output_dir, f"{comparison_name}_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int_)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float_)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                if math.isnan(obj):
                    return None
                return super(NpEncoder, self).default(obj)
        
        json.dump(results, f, indent=2, cls=NpEncoder, allow_nan=False)
    
    return results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function for CSV-based comparisons.
    """
    print("=" * 80)
    print("CSV DISTRIBUTION COMPARISON ANALYSIS")
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
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype, 
            trust_remote_code=True
        )
        model.to(device)
        model.eval()
        print("Model loaded successfully.\n")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return
    
    # Load background data
    print("Loading background corpus...")
    background_data = dc.load_and_embed_background_data(
        BACKGROUND_FILE,
        tokenizer,
        model,
        device,
        min_tokens=MIN_TOKEN_LIMIT,
        max_tokens=MAX_TOKEN_LIMIT
    )
    
    if not background_data:
        print(f"Error: No background data available after loading/filtering (range {MIN_TOKEN_LIMIT}-{MAX_TOKEN_LIMIT} tokens). Cannot proceed.", file=sys.stderr)
        return
    
    print(f"Successfully prepared {len(background_data):,} paragraphs for analysis.\n")
    
    # Load CSV data
    try:
        csv_data = load_csv_data(CSV_FILE)
    except Exception as e:
        print(f"Error loading CSV file: {e}", file=sys.stderr)
        return
    
    if not csv_data:
        print("Error: No valid rows found in CSV file.", file=sys.stderr)
        return
    
    # Process each row and comparison
    print(f"\n{'='*80}")
    print(f"Processing {len(csv_data)} rows with comparisons...")
    print(f"{'='*80}\n")
    
    all_results = []
    for row_data in csv_data:
        row_idx = row_data['row_index']
        original_story = row_data['original_story']
        regenerated_stories = row_data['regenerated_stories']
        
        print(f"\n{'='*80}")
        print(f"Row {row_idx}: {len(regenerated_stories)} regenerated story(ies)")
        print(f"{'='*80}")
        
        # Process each regenerated story for this row
        for regen_story in regenerated_stories:
            regen_name = regen_story['name']
            regen_text = regen_story['text']
            
            # Create comparison name and output directory
            comparison_name = f"row_{row_idx}_{regen_name}"
            comparison_output_dir = os.path.join(OUTPUT_DIR, comparison_name)
            
            print(f"\n  Comparing: original_story vs {regen_name}...")
            
            try:
                result = run_single_comparison(
                    original_text=original_story,
                    regenerated_text=regen_text,
                    regenerated_label=regen_name,
                    background_data=background_data,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    output_dir=comparison_output_dir,
                    comparison_name=comparison_name
                )
                
                # Add metadata
                result['row_index'] = row_idx
                result['regenerated_story_column'] = regen_name
                all_results.append(result)
                
                print(f"  ✓ Completed: {comparison_name}")
                print(f"    Semantic Similarity: {result['pairwise_comparison']['semantic_similarity']:.6f}")
                print(f"    Trigram Jaccard: {result['pairwise_comparison']['trigram_jaccard']:.6f}")
                print(f"    Quadgram Jaccard: {result['pairwise_comparison']['quadgram_jaccard']:.6f}")
                
            except Exception as e:
                print(f"  ✗ Error processing {comparison_name}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                continue
    
    # Save summary of all comparisons
    print(f"\n{'='*80}")
    print("Creating summary file...")
    print(f"{'='*80}\n")
    
    summary_file = os.path.join(OUTPUT_DIR, "all_comparisons_summary.json")
    summary_data = {
        "csv_file": CSV_FILE,
        "total_comparisons": len(all_results),
        "background_corpus": {
            "file": BACKGROUND_FILE,
            "num_paragraphs_used": len(background_data),
            "token_limit_min": MIN_TOKEN_LIMIT,
            "token_limit_max": MAX_TOKEN_LIMIT
        },
        "model": MODEL_NAME,
        "comparisons": []
    }
    
    # Extract key metrics for summary
    for result in all_results:
        summary_data["comparisons"].append({
            "row_index": result.get("row_index"),
            "regenerated_story_column": result.get("regenerated_story_column"),
            "semantic_similarity": result["pairwise_comparison"]["semantic_similarity"],
            "trigram_jaccard": result["pairwise_comparison"]["trigram_jaccard"],
            "quadgram_jaccard": result["pairwise_comparison"]["quadgram_jaccard"],
            "original_token_count": result["original_story"]["token_count"],
            "regenerated_token_count": result["regenerated_story"]["token_count"],
            "original_percentiles": result["original_story"]["percentile_rankings"],
            "regenerated_percentiles": result["regenerated_story"]["percentile_rankings"]
        })
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int_)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float_)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                if math.isnan(obj):
                    return None
                return super(NpEncoder, self).default(obj)
        
        json.dump(summary_data, f, indent=2, cls=NpEncoder, allow_nan=False)
    
    print(f"\n{'='*80}")
    print("CSV PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total comparisons processed: {len(all_results)}")
    print(f"Summary saved to: {summary_file}")
    print(f"Individual results saved in subdirectories under: {OUTPUT_DIR}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

