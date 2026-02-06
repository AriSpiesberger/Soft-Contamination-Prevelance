# Finetuning Experiments

LoRA fine-tuning experiments measuring how different types of training data contamination affect benchmark performance. Tests exact duplicates, semantic duplicates, and high-cosine-similarity matches on MBPP, MuSR, and ZebraLogic benchmarks.

## Directory Structure

```
finetuning/
├── mbpp/                  # MBPP code generation experiments
├── musr/                  # MuSR murder mystery experiments
├── zebralogic/            # ZebraLogic puzzle experiments
├── eval_baseline.py       # Multi-benchmark baseline evaluation
├── mbpp_data/             # MBPP training/test data
├── outputs/               # Checkpoints and evaluation results
├── accelerate_config.yaml # Multi-GPU distributed training config
└── requirements.txt
```

## mbpp/ - MBPP Code Generation

### Data Preparation

| Script | Purpose |
|--------|---------|
| `create_exact_duplicates.py` | Create exact-duplicate training set (5x repetition per MBPP task) |
| `generate_semantic_dupes.py` | Generate paraphrased semantic duplicates via LLM |
| `filter_training_data.py` | Filter MBPP training data to only code-passing examples |
| `filter_semantic_dupes.py` | Filter semantic duplicates to quality samples |

### Training

| Script | Purpose |
|--------|---------|
| `train_and_eval.py` | **Main pipeline** - trains 3 experiments (exact, semantic, cosine_top5) with KL regularization and early stopping, then evaluates |
| `train_mbpp_kl.py` | Standalone MBPP training with KL divergence regularization |
| `train_mbpp_sft.py` | Standard SFT baseline (no KL) on MBPP semantic data |

### Evaluation

| Script | Purpose |
|--------|---------|
| `eval_only.py` | Evaluate baseline + 3 key models via lm-evaluation-harness |
| `eval_mbpp.py` | MBPP pass@1 evaluation (contamination vs generalization splits) |
| `eval_mbpp_fast.py` | Fast batched MBPP evaluation |
| `eval_semantic_pairs.py` | Semantic pair code generation evaluation (pass@k) |

## musr/ - MuSR Murder Mystery

| Script | Purpose |
|--------|---------|
| `generate_answers.py` | Generate ground-truth answers for MuSR murder mysteries |
| `finetune_musr_only.py` | MuSR-only fine-tuning on regenerated stories |
| `finetune_musr_mixed.py` | MuSR + Dolci-Instruct mixed training (5:1 ratio) |
| `eval_musr.py` | MuSR murder mystery evaluation |

## zebralogic/ - ZebraLogic Puzzles

| Script | Purpose |
|--------|---------|
| `eval_zebralogic.py` | ZebraLogic puzzle evaluation |

## Root-Level Scripts

| Script | Purpose |
|--------|---------|
| `eval_baseline.py` | General benchmark evaluation for catastrophic forgetting detection |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run main MBPP experiment pipeline (requires 8 GPUs)
accelerate launch --num_processes 8 mbpp/train_and_eval.py

# Or run individual training
python mbpp/train_mbpp_kl.py --experiment sem_dupes

# Evaluate
python mbpp/eval_only.py
```

## Configuration

- `accelerate_config.yaml` - Multi-GPU distributed training config (8x GPU, bfloat16)
- Training data goes in `mbpp_data/`
- Checkpoints and results go in `outputs/`

## Models

- Base model: `allenai/OLMo-3-7B-Instruct` (MBPP experiments)
- Base model: `allenai/Olmo-3-1025-7B` (MuSR experiments)
- Fine-tuning: QLoRA (4-bit quantization, LoRA rank 32)
