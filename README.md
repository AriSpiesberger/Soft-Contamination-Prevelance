# Soft Contamination Means Benchmarks Test Shallow Generalization

Code repository for the ICML 2026 submission: *Soft Contamination Means Benchmarks Test Shallow Generalization*.

We study **soft contamination** of LLM training data by semantic duplicates of benchmark test data. Using the open-data model OLMo3 as a case study, we show that:

1. **Contamination is widespread** - We find semantic duplicates for 78% of CodeForces and exact duplicates for 50% of ZebraLogic problems in OLMo3's training corpus.
2. **Contamination improves benchmark scores** - Finetuning on semantic duplicates of benchmark data improves performance on those benchmarks.
3. **Shallow generalization** - Performance also improves on truly held-out items from the same benchmark, suggesting within-benchmark-distribution generalization rather than genuine capability growth.

## Repository Structure

| Directory | Description | Paper Section |
|-----------|-------------|---------------|
| [`pipeline/`](pipeline/) | Embedding pipeline for scanning OLMo3 training corpora (Dolma, Dolmino, Dolci) for semantic duplicates using cosine similarity | Sec 3.2 |
| [`comparison_analysis/`](comparison_analysis/) | Annotation and classification of high-cosine-similarity matches as semantic duplicates (using Gemini, GPT, fine-tuned classifiers) | Sec 3.2, 4.2 |
| [`sdtd-llm-generation/`](sdtd-llm-generation/) | Generation of semantic duplicates at multiple levels for MBPP, MuSR, ZebraLogic, and CodeForces | Sec 3.1, Appendix A.3 |
| [`finetuning/`](finetuning/) | Finetuning experiments on exact and semantic duplicates, with evaluation on MBPP, HumanEval, MuSR, ZebraLogic | Sec 4.3, Tables 2-4 |
| [`ecology/`](ecology/) | Ecologically valid contamination experiments - finetuning on realistic contamination rates mixed into Dolci training data | Sec 4.4, Tables 5, 11 |
| [`model_diffusion/`](model_diffusion/) | Tinker-based finetuning and classifier training for semantic duplicate detection | Sec 4.2 |
| [`analysis_scripts/`](analysis_scripts/) | Supporting analysis tools: contamination analysis, similarity computation, benchmark comparisons | Supporting |
| [`utils/`](utils/) | Shared utilities: embedding helpers, n-gram similarity, mean pooling | Supporting |
| [`data/`](data/) | Data files: semantic pairs, Codeforces samples, MBPP splits | Data |
| [`results/`](results/) | Cached analysis results and contamination outputs | Outputs |

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

### Quick Start

```bash
# Install uv (if not already installed)
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create virtual environment and install dependencies
uv venv
uv pip install -e .

# Activate the virtual environment
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\Activate.ps1
```

Or use the setup scripts:
```bash
# macOS/Linux
./setup.sh

# Windows (PowerShell)
.\setup.ps1
```

### Optional Dependencies

```bash
# Development tools (pytest, black, ruff)
uv pip install -e ".[dev]"

# Flash Attention (requires CUDA, primarily Linux)
uv pip install flash-attn
```

**Note:** Individual subdirectories (e.g., `sdtd-llm-generation/`, `finetuning/`) may have their own dependency specifications. See their READMEs for details.

## Key Technologies

- **Embedding model:** `nvidia/llama-embed-nemotron-8b` (MTEB #2 at time of writing)
- **Models tested:** OLMo3-7B, Qwen3-8B, Llama-3.1-8B
- **Finetuning:** LoRA via PEFT/TRL
- **Semantic duplicate generation:** Claude, GPT, Qwen via OpenRouter
- **Benchmarks:** MBPP, CodeForces, MuSR (Murder Mysteries, Team Allocation), ZebraLogic, HumanEval

## Data Preparation Scripts

The following root-level scripts handle data preparation across multiple experiments:

| Script | Purpose |
|--------|---------|
| `create_semantic_pairs_csv.py` | Create CSV with 5 semantic duplicate pairs per MBPP task (English synonym input + Python semantic output) |
| `split_semantic_pairs.py` | Split MBPP semantic pairs into train/test sets for contamination experiments |
| `gathering_codeforces.py` | Sample CodeForces problems uniformly across Elo ratings |
| `distribution_comparison.py` | Compute cosine similarity distributions between texts and a background corpus |

## Reproducing Paper Results

### 1. Scanning for Semantic Duplicates (Sec 3.2)
```bash
# Run the embedding pipeline against OLMo3 training data
# See pipeline/README.md for configuration and cluster setup
cd pipeline
python -m pipeline.run --config configs/dolci.yaml
```

### 2. Generating Semantic Duplicates (Sec 3.1)
```bash
# See sdtd-llm-generation/README.md for full documentation
cd sdtd-llm-generation
python -m sdtd generate --dataset mbpp --level 1
```

### 3. Finetuning Experiments (Sec 4.3)
```bash
# See finetuning/README.md for experiment configurations
cd finetuning
accelerate launch --num_processes 8 train_and_eval.py  # Main MBPP experiments
python p2_finetune_mixed.py                            # MuSR experiments
```

### 4. Ecological Validity Experiments (Sec 4.4)
```bash
# See ecology/README.md for details
cd ecology
python run_experiment_tinker.py --model meta-llama/Llama-3.1-8B
```

## License

TBD

## Citation

```bibtex
@inproceedings{anonymous2026softcontamination,
  title={Soft Contamination Means Benchmarks Test Shallow Generalization},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```
