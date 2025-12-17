# Contamination Analysis Results

This directory contains the results of benchmark contamination analysis for training data.

## Overview

The analysis computes cosine similarity between benchmark datasets (MuSR, HumanEval) and our training corpus using NVIDIA's llama-embed-nemotron-8b model. For each benchmark item, we find the top 100 most similar items in the training corpus.

### Key Facts
- **MuSR**: 756 benchmark items × 100 top matches each
- **HumanEval**: 164 benchmark items × 100 top matches each
- **Similarity Metric**: Cosine similarity on L2-normalized embeddings
- **Embedding Source**: Local parquet files (not AWS API calls)
- **Model**: nvidia/llama-embed-nemotron-8b

## Directory Structure

```
contamination_analysis/
├── README.md                    # This file
├── data/
│   ├── embeddings_cache/        # Cached benchmark embeddings
│   │   ├── MuSR_*.npy          # MuSR embeddings for input/output/both
│   │   ├── MuSR_*_ids.json     # Benchmark item IDs
│   │   ├── HumanEval_*.npy     # HumanEval embeddings
│   │   └── HumanEval_*_ids.json
│   ├── statistics/              # Similarity statistics
│   │   ├── *_stats.json        # Per-dataset statistics
│   │   ├── *_max_sims.npy      # Maximum similarity scores
│   │   └── summary.json        # Overall summary
│   └── top_matches/             # Top-K matches for each benchmark item
│       ├── MuSR_*_matches.json
│       └── HumanEval_*_matches.json
└── plots/
    ├── musr/                    # MuSR visualization plots
    │   ├── input_distribution.png
    │   ├── output_distribution.png
    │   └── input_output_distribution.png
    ├── humaneval/               # HumanEval visualization plots
    │   ├── input_distribution.png
    │   ├── output_distribution.png
    │   └── input_output_distribution.png
    └── summary/                 # Cross-dataset comparison
        └── joint_comparison.png
```

## Analysis Modes

Three modes of comparison for each benchmark:

1. **input**: Compare benchmark inputs to training corpus
2. **output**: Compare benchmark outputs (answers/solutions) to training corpus
3. **input_output**: Compare concatenated input+output to training corpus

## Running the Analysis

From the project root:

```bash
# Run full analysis (both datasets, all modes)
python analysis_scripts/contamination_analysis.py

# Run specific dataset only
python analysis_scripts/contamination_analysis.py --dataset musr
python analysis_scripts/contamination_analysis.py --dataset humaneval

# Customize parameters
python analysis_scripts/contamination_analysis.py \
    --top-k 100 \
    --embedding-batch-size 16 \
    --compute-batch-size 500000 \
    --data-dir /path/to/embeddings \
    --output-dir contamination_analysis
```

## Output Files

### Statistics Files (`data/statistics/*_stats.json`)
Contains:
- Max/mean/median similarity scores
- Percentiles (p50, p75, p90, p95, p99)
- Contamination rates at different thresholds (0.7, 0.8, 0.9, 0.95, 0.99)

### Matches Files (`data/top_matches/*_matches.json`)
For each benchmark item, contains top 100 matches with:
- Similarity score
- Corpus text snippet (first 500 chars)
- Source metadata
- Global corpus index

### Plots
- **Distribution plots**: Histogram of similarity scores with contamination rate annotations
- **Joint comparison**: Multi-panel comparison across datasets and modes

## Implementation Details

### Two-Phase Architecture
1. **Phase 1**: Load model, embed all benchmarks, cache to disk, unload model
2. **Phase 2**: Stream corpus embeddings from disk, compute similarities with full GPU memory

### Memory Optimization
- Benchmark embeddings: Cached as fp16 numpy arrays
- Corpus streaming: Memory-mapped parquet files, processed in batches
- Top-K tracking: In-place GPU updates to maintain only top 100 matches
- Model unloading: Free VRAM before similarity computation phase

### Corpus Data Source
- Location: `/lambda/nfs/embeddings/embedding_folder`
- Format: Parquet files with pre-computed embeddings
- Embeddings: L2-normalized, fp16 precision
- No API calls: All computation done locally with cached embeddings
