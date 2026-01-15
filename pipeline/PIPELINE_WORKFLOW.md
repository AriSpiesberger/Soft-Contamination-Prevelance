# SDTD Pipeline Workflow

Complete workflow for running contamination analysis on instruction-tuning datasets.

## Pipeline Stages

### Stage 1: Download Data
**Script:** `stages/01_download_dolmo.py`

Downloads the dataset from HuggingFace.

**Config section:** `download`

**Run:** Manual (as needed)

---

### Stage 2: Chunk and Sample
**Script:** `stages/02_chunk_and_sample.py`

Chunks the dataset into conversations/paragraphs and samples for analysis.

**Config section:** `chunking`

**Output:** `./data/<dataset>_conversations.jsonl`

**Run:** Manual (as needed)

---

### Stage 3: Generate Embeddings
**Script:** `stages/03_create_embeddings_local_multigpu.py`
**Cluster script:** `cluster/run_03_embeddings_multigpu.sh`

Generates embeddings for the corpus using multi-GPU setup.

**Config section:** `embeddings`

**Output:** `./data/<dataset>_embeddings_rank_*.parquet`

**Run:**
```bash
PIPELINE_CONFIG=configs/dolci.yaml \
PIPELINE_VENV=/lambda/nfs/embeddings/SDTD_Main/.venv/bin/python \
./pipeline/cluster/run_03_embeddings_multigpu.sh
```

---

### Stage 4: Contamination Analysis
**Script:** `stages/04_contamination_analysis.py`
**Cluster script:** `cluster/run_04_analysis.sh`

Runs contamination analysis comparing benchmarks against the corpus embeddings.

**Config section:** `analysis`

**Outputs:**
- Individual test results: `<benchmark>_<mode>/*_similarities.npy.gz`
- Top-100 matches: `<benchmark>_<mode>/*_top100.json`
- Aggregate stats: `<benchmark>_<mode>/aggregate_stats.json`
- Log histogram: `<benchmark>_<mode>/aggregate_histogram.png`
- Top-K plot: `<benchmark>_<mode>/aggregate_topk.png`

**Run:**
```bash
PIPELINE_CONFIG=configs/dolci.yaml \
PIPELINE_VENV=/lambda/nfs/embeddings/SDTD_Main/.venv/bin/python \
./pipeline/cluster/run_04_analysis.sh
```

---

### Stage 5: Finalize Results
**Script:** `stages/05_finalize_results.py`
**Cluster script:** `cluster/run_05_finalize.sh`

Finalizes all results by:
1. Adding corpus text to top-100 JSON files
2. Generating additional aggregate plots (linear histogram, CDF)
3. Generating CSV files with full text
4. Cleaning up temporary files

**Config sections:** `chunking.output_paragraphs`, `aggregates.dataset_name`

**Outputs:**
- Updated JSONs with corpus text: `<benchmark>_<mode>/*_top100.json`
- Linear histogram: `<benchmark>_<mode>/aggregate_histogram_linear.png`
- CDF plot: `<benchmark>_<mode>/aggregate_cdf.png`
- Full matches CSV: `<benchmark>_<mode>/all_top100_matches.csv`
- Top-100 CSV: `<benchmark>_<mode>/top_100_contamination.csv`

**Cleanup:**
- Removes `temp_similarities/`
- Removes individual `*_similarities.npy.gz` files (keeps aggregates)
- Removes `checkpoints/`

**Run:**
```bash
PIPELINE_CONFIG=configs/dolci.yaml \
PIPELINE_VENV=/lambda/nfs/embeddings/SDTD_Main/.venv/bin/python \
./pipeline/cluster/run_05_finalize.sh
```

---

## Complete Production Workflow

### Initial Setup (One Time)
```bash
# Set environment variables
export PIPELINE_CONFIG=/home/ubuntu/embeddings/SDTD_Main/pipeline/configs/dolci.yaml
export PIPELINE_VENV=/lambda/nfs/embeddings/SDTD_Main/.venv/bin/python

# Download and chunk data (if not already done)
cd /home/ubuntu/embeddings/SDTD_Main/pipeline
python stages/01_download_dolmo.py  # if needed
python stages/02_chunk_and_sample.py
```

### Regular Production Run

**Step 1: Generate Embeddings**
```bash
cd /home/ubuntu/embeddings/SDTD_Main/pipeline
./cluster/run_03_embeddings_multigpu.sh
```

**Step 2: Run Contamination Analysis**
```bash
./cluster/run_04_analysis.sh
```

**Step 3: Finalize Results**
```bash
./cluster/run_05_finalize.sh
```

### Output Location
All final results will be in:
```
./pipeline/results/contamination_<dataset>_<percent>/
├── mbpp_input_output/
│   ├── aggregate_histogram.png (log scale)
│   ├── aggregate_histogram_linear.png (linear scale)
│   ├── aggregate_cdf.png
│   ├── aggregate_topk.png
│   ├── aggregate_stats.json
│   ├── all_top100_matches.csv (all matches with full text)
│   ├── top_100_contamination.csv (top 100 overall)
│   └── *_top100.json (individual test results with corpus text)
└── musr_input_output/
    └── (same as above)
```

**Note:** `musr_input_output` contains results for ALL MuSR splits combined (murder_mysteries, object_placements, team_allocation)

---

## Configuration

Edit `configs/dolci.yaml` to configure:

- **Dataset:** `download.repo_id`
- **Benchmarks:** `analysis.benchmarks` (currently: `musr` and `mbpp`)
  - `musr` - Loads ALL MuSR splits combined (murder_mysteries, object_placements, team_allocation)
  - `mbpp` - MBPP coding benchmark
- **Modes:** Set `mode: "input_output"` for combined input+output analysis
- **GPU settings:** `analysis.cluster` section
- **Output directory:** `analysis.output_dir` (format: `contamination_{dataset}_{percent}`)

---

## CSV Format

The generated CSVs contain the following columns:

| Column | Description |
|--------|-------------|
| `test_id` | Benchmark test ID |
| `rank` | Rank within this test's top-100 |
| `cosine_similarity` | Similarity score (0-1) |
| `corpus_index` | Index in corpus |
| `test_text` | Full benchmark test text (prompt+response for conversations) |
| `corpus_text` | Full matching corpus text |

---

## Notes

- **Stage 5 (old merge):** The old `run_05_merge.sh` is deprecated. Use `run_05_finalize.sh` instead.
- **Stage 6 (old aggregates):** The old `run_06_aggregates.sh` is deprecated. Aggregates are now generated in Stage 5.
- **Stage 7 (corpus text):** Integrated into Stage 5.
- **Temporary files:** Stage 5 automatically cleans up intermediate files after generating final outputs.
- **Resuming:** If Stage 4 or 5 fails midway, you can re-run them. Stage 5 is idempotent and safe to re-run.

---

## Troubleshooting

### Stage 4 runs out of memory
- Reduce `gpu_batch_size` in config
- Reduce `corpus_gpu_chunk` in config

### Stage 5 can't find corpus file
- Check that `chunking.output_paragraphs` in config points to the correct file
- Verify the file exists: `ls ./data/*_conversations.jsonl`

### Missing plots or CSVs
- Re-run Stage 5: `./cluster/run_05_finalize.sh`
- Check logs for errors

---

## Quick Reference

**Run full pipeline:**
```bash
export PIPELINE_CONFIG=configs/dolci.yaml
export PIPELINE_VENV=/lambda/nfs/embeddings/SDTD_Main/.venv/bin/python

./cluster/run_03_embeddings_multigpu.sh  # Generate embeddings
./cluster/run_04_analysis.sh             # Run analysis
./cluster/run_05_finalize.sh             # Finalize results
```

**Check results:**
```bash
ls -lh ./results/contamination/mbpp_input/*.{png,csv,json}
```
