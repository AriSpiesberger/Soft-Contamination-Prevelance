# Contamination Analysis Pipeline

Config-driven pipeline for detecting semantic duplicates between LLM training corpora and benchmark test sets using embedding similarity.

## Overview

This pipeline computes cosine similarity between training corpus chunks and benchmark test items using the `nvidia/llama-embed-nemotron-8b` embedding model (MTEB #2). It processes terabytes of training data across multi-GPU clusters to find the most similar training examples for each benchmark test point.

**Stages:**

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `stages/01_download_dolmo.py` | Download training data from HuggingFace |
| 2 | `stages/02_chunk_and_sample.py` | Split into paragraphs with token-based filtering |
| 3 | `stages/03_create_embeddings.py` | Generate embeddings (multi-GPU, H100/A100) |
| 4 | `stages/04_contamination_analysis.py` | Distributed cosine similarity analysis |
| 5 | `stages/05_finalize_results.py` | Hydrate results with corpus text, generate CSVs |
| 6 | `stages/06_sample_top1pct.py` | Sample from top N% for manual analysis |

## Quick Start

```bash
# Run entire pipeline with default config
./run_all.sh

# Use a specific dataset config
./run_all.sh --config dolci.yaml

# Run specific stages
./run_all.sh --stage 4    # Start from stage 4
./run_all.sh --only 3     # Run only stage 3

# Dry run
./run_all.sh --dry-run
```

## Configuration

The pipeline is driven by YAML config files in `configs/`. Each config specifies the dataset source, chunking parameters, embedding settings, analysis benchmarks, and cluster configuration.

**Available configs:**

| Config | Dataset | Description |
|--------|---------|-------------|
| `default.yaml` | Dolma3 | 1% sample of Dolma3 6T pretraining mix |
| `dolma3_dolmino_1pct.yaml` | Dolmino | 1% sample of Dolmino 100B mix |
| `dolci.yaml` | Dolci-SFT | Dolci instruction-tuning SFT data |
| `dolci_dpo.yaml` | Dolci-DPO | Dolci DPO preference data |
| `dolci_rl.yaml` | Dolci-RL | Dolci RL prompt/solution data |
| `example_custom.yaml` | Template | Copy and modify for custom datasets |

To create a custom run:
```bash
cp configs/example_custom.yaml configs/my_dataset.yaml
# Edit my_dataset.yaml with your dataset details
./run_all.sh --config my_dataset.yaml
```

## Directory Structure

```
pipeline/
├── run_all.sh                        # Main orchestration script
├── configs/                          # Pipeline YAML configs
│   ├── default.yaml                  # Dolma3 default config
│   ├── dolci.yaml                    # Dolci SFT config
│   ├── dolci_dpo.yaml                # Dolci DPO config
│   ├── dolci_rl.yaml                 # Dolci RL config
│   ├── dolma3_dolmino_1pct.yaml      # Dolmino 1% config
│   └── example_custom.yaml           # Template for custom runs
├── stages/                           # Python scripts for each stage
│   ├── 01_download_dolmo.py          # Download from HuggingFace
│   ├── 02_chunk_and_sample.py        # Chunk and sample paragraphs
│   ├── 03_create_embeddings.py       # Generate embeddings (multi-GPU)
│   ├── 04_contamination_analysis.py  # Distributed similarity analysis
│   ├── 05_finalize_results.py        # Hydrate JSONs, generate CSVs
│   └── 06_sample_top1pct.py          # Sample top N% for analysis
├── cluster/                          # GPU cluster job launchers
│   ├── run_03_embeddings_multigpu.sh # Multi-GPU embedding generation
│   ├── run_04_analysis.sh            # Multi-GPU analysis with retries
│   └── run_05_finalize.sh            # Finalize results
├── scripts/                          # Utility scripts
│   ├── regenerate_csvs.py            # Regenerate CSVs from JSON results
│   └── verify_csv_scores.py          # Verify scores by re-embedding
└── lib/                              # Shared utilities
    └── config_helper.py              # Shell script config reader
```

## Environment Variables

```bash
# Override config file
PIPELINE_CONFIG=configs/dolci.yaml ./run_all.sh

# Override Python venv path
PIPELINE_VENV=/path/to/.venv/bin/python ./run_all.sh

# Override analysis parameters
ANALYSIS_CORPUS_DIR=/path/to/embeddings ./run_all.sh
ANALYSIS_WORLD_SIZE=4 ./run_all.sh
```

## Running Individual Stages

Each stage can be run independently via the cluster scripts:

```bash
# Stage 3: Multi-GPU embeddings
./cluster/run_03_embeddings_multigpu.sh

# Stage 4: Contamination analysis (8x A100 with retries)
./cluster/run_04_analysis.sh

# Stage 5: Finalize results
./cluster/run_05_finalize.sh

# Stage 6: Sample top N% (standalone)
python stages/06_sample_top1pct.py \
    --results-dir ./results/contamination \
    --corpus-jsonl ./data/paragraphs.jsonl \
    -b mbpp --percentile 99.9
```

## Outputs

Results are saved per-benchmark under the configured output directory:

```
results/contamination/
├── temp_similarities/         # Per-rank intermediate similarity scores
├── checkpoints/               # Recovery checkpoints
├── mbpp_input/                # Per-benchmark results
│   ├── *_top100.json          # Top-100 matches per test point (with corpus text)
│   ├── all_top1000_matches.csv
│   └── top_1000_contamination.csv
├── musr_murder_mysteries_input_output/
│   └── ...
└── ...
```

## Hardware Requirements

- **Stages 1-2**: CPU only, 32GB+ RAM recommended for chunking
- **Stage 3**: GPU required (H100 or A100 recommended), multi-GPU supported
- **Stage 4**: Multi-GPU strongly recommended (8x A100 40GB used in paper)
- **Stages 5-6**: CPU only
