# Contamination Ecology Experiments

Controlled experiments measuring whether models trained on semantically contaminated data perform better on contaminated test points vs clean ones. Uses MuSR murder mystery reasoning tasks with semantic duplicates injected into Dolci training data.

## Pipeline

### 1. Data Preparation

| Script | Purpose |
|--------|---------|
| `sample_dolci.py` | Sample 10k examples from Dolci-Instruct-SFT dataset |
| `create_contaminated_dataset.py` | Insert semantic duplicates of MuSR test points into training data. Creates contaminated + clean splits with metadata |

### 2. Training

| Script | Purpose |
|--------|---------|
| `run_experiment_tinker.py` | **Primary** - Fine-tune via Tinker API with checkpoints at epochs [1,2,3,6,10]. Supports Llama-3.1-8B, Qwen3-8B, and more |
| `run_experiment_qwen.py` | Multi-GPU local training for Qwen3-8B-Base via accelerate |
| `run_experiment_multigpu.py` | Multi-GPU local training for OLMo-3-7B via accelerate |

### 3. Evaluation

| Script | Purpose |
|--------|---------|
| `evaluate_contamination.py` | Evaluate trained checkpoints on contaminated vs clean test splits |
| `eval_base_models.py` | Baseline evaluation of untuned models |
| `eval_base_fast.py` | Multi-GPU parallel baseline evaluation (8 GPUs) |
| `eval_true_detective.py` | Degradation test on True Detective abductive reasoning benchmark |
| `run_benchmark_evals.py` | Orchestrate all benchmark evaluations (ARC, HellaSwag, GSM8K, True Detective) |

### 4. Testing

| Script | Purpose |
|--------|---------|
| `test_tinker_pipeline.py` | Minimal pipeline test (3 test + 10 train examples) to verify Tinker setup |

## Quick Start

```bash
# Install dependencies (from repo root)
uv sync

# Set API key for Tinker-based training
export TINKER_API_KEY=your_key_here

# 1. Prepare data
python sample_dolci.py
python create_contaminated_dataset.py

# 2. Train (via Tinker API)
python run_experiment_tinker.py --model meta-llama/Llama-3.1-8B

# 3. Evaluate
python evaluate_contamination.py --model_path outputs/tinker_contaminated_*/
python run_benchmark_evals.py
```

## Models Tested

- Llama-3.1-8B, Llama-3.2-1B/3B, Llama-3.3-70B
- Qwen3-8B-Base
- OLMo-3-1025-7B

## Key Design

Each experiment trains two models on the same base data:
- **Contaminated**: Dolci + semantic duplicates of MuSR test points
- **Clean**: Dolci only (no test-related content)

Performance difference between contaminated and clean models on contaminated test points measures the "ecology" of soft contamination.
