# Model Diffusion - Semantic Duplicate Classifiers

Fine-tune LLMs to classify whether training-benchmark pairs are semantic duplicates. Uses Tinker API for LoRA training on Qwen3-30B and GPT-OSS-20b.

## Pipeline

### Training

| Script | Purpose |
|--------|---------|
| `train_mbpp_classifier.py` | Train Qwen3-30B on MBPP+Codeforces data, eval on held-out set |
| `train_codeforces_classifier.py` | Train GPT-OSS-20b on MBPP+Codeforces data, eval on Codeforces only |

### Classification (Inference)

| Script | Purpose |
|--------|---------|
| `classify_mbpp.py` | Classify MBPP pairs using trained Qwen3-30B checkpoint |
| `classify_codeforces.py` | Classify Codeforces pairs using trained GPT-OSS-20b checkpoint |

### Shared / Data Preparation

| Script | Purpose |
|--------|---------|
| `shared_utilities.py` | Shared functions: loss computation, evaluation, prompt templates, chat formats |
| `sample_for_classification.py` | Stratified sampling of data for balanced classification |

## Quick Start

```bash
# Set API key
export TINKER_API_KEY=your_key_here

# Train MBPP classifier
python train_mbpp_classifier.py

# Train Codeforces classifier
python train_codeforces_classifier.py

# Run MBPP classification with trained checkpoint
python classify_mbpp.py

# Run Codeforces classification (with custom input)
python classify_codeforces.py --input data/custom.csv --output data/custom_classified.csv
```

## Configuration

- Input/output data: `data/` subdirectory
- Training outputs: `outputs/`
- Environment variable: `TINKER_API_KEY` (required)
