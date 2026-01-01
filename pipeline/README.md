# SDTD Pipeline

Unified end-to-end pipeline for contamination analysis of Dolma3 data.

## Overview

This pipeline consolidates all production scripts into a single, config-driven workflow:

1. **Download** - Download Dolma3 samples from HuggingFace
2. **Chunk** - Split into sentences/paragraphs with token-based filtering
3. **Embed** - Create embeddings using nvidia/llama-embed-nemotron-8b (H100 cluster)
4. **Analyze** - Run contamination analysis against benchmarks (8x A100 cluster)
5. **Merge** - Combine distributed results
6. **Aggregate** - Generate plots and top-100 lists

## Quick Start

```bash
# Run entire pipeline with default config
./run_all.sh

# Use a custom config
./run_all.sh --config my_dataset.yaml

# Run specific stages
./run_all.sh --stage 4    # Start from stage 4 (analysis)
./run_all.sh --only 6     # Run only stage 6 (aggregates)

# Dry run to see what would execute
./run_all.sh --dry-run
```

## Configuration

The pipeline is controlled by a single YAML config file. Copy `configs/default.yaml` and modify for your dataset:

```yaml
# configs/my_dataset.yaml
pipeline:
  name: "my_custom_run"

download:
  repo_id: "allenai/dolma3_mix-6T-1025"
  sample_percentage: 0.001  # 0.1%

analysis:
  corpus_dir: "/path/to/my/embeddings"
  benchmarks:
    - name: "musr_murder_mysteries"
      mode: "input_output"
    - name: "mbpp"
      mode: "input"
```

Run with your config:
```bash
./run_all.sh --config my_dataset.yaml
# OR
PIPELINE_CONFIG=configs/my_dataset.yaml ./run_all.sh
```

## Directory Structure

```
pipeline/
├── run_all.sh              # Main orchestration script
├── configs/
│   └── default.yaml        # Master config (copy & modify for new runs)
├── stages/                 # Python scripts for each stage
│   ├── 01_download_dolmo.py
│   ├── 02_chunk_and_sample.py
│   ├── 03_create_embeddings.py
│   ├── 04_contamination_analysis.py
│   ├── 05_merge_results.py
│   └── 06_generate_aggregates.py
├── cluster/                # Cluster job launchers
│   ├── run_04_analysis.sh  # 8x A100 parallel analysis
│   ├── run_05_merge.sh     # Parallel merge
│   └── run_06_aggregates.sh
├── lib/                    # Shared utilities
│   ├── config_loader.py    # Config parsing
│   ├── config_helper.py    # Shell script helper
│   └── s3_config.py        # S3 configuration
├── logs/                   # Execution logs
└── results/                # Output results
```

## Running Individual Stages

Each cluster stage can be run independently:

```bash
# Stage 4: Contamination Analysis (8x A100)
./cluster/run_04_analysis.sh

# Stage 5: Merge Results
./cluster/run_05_merge.sh

# Stage 6: Generate Aggregates
./cluster/run_06_aggregates.sh
```

## Environment Variables

Override config values via environment:

```bash
# Override corpus directory
ANALYSIS_CORPUS_DIR=/different/path ./run_all.sh

# Override world size
ANALYSIS_WORLD_SIZE=4 ./run_all.sh

# Override Python venv
PIPELINE_VENV=/path/to/python ./run_all.sh
```

## Skipping Stages

In your config, set stages to skip:

```yaml
skip_stages:
  download: true    # Skip download (already have data)
  chunking: true    # Skip chunking
  embeddings: true  # Skip embedding generation
  analysis: false   # Run analysis
  merge: false      # Run merge
  aggregates: false # Run aggregates
```

## Outputs

Results are saved to `results/contamination/`:

```
results/contamination/
├── temp_similarities/         # Per-rank checkpoint similarities
├── checkpoints/               # Recovery checkpoints
├── merged_rank_*.json         # Merged outputs
├── musr_*/                    # Per-benchmark results
│   ├── aggregate_histogram_linear.png
│   ├── aggregate_histogram_log.png
│   ├── aggregate_cdf.png
│   └── top_100_contamination.csv
└── mbpp_*/
    └── ...
```

## Adding New Data Sources

1. Copy `configs/default.yaml` to `configs/my_source.yaml`
2. Modify the `download`, `embeddings`, and `analysis` sections
3. Run: `./run_all.sh --config my_source.yaml`

## Troubleshooting

**Logs**: Check `logs/stage*_rank_*.log` for detailed output.

**Resume from failure**:
```bash
# Resume analysis from file 50
./cluster/run_04_analysis.sh 50
```

**Check config values**:
```bash
python lib/config_helper.py --config default.yaml --get analysis.corpus_dir
python lib/config_helper.py --config default.yaml --section embeddings
```
