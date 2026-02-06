# Semantic Duplicate Annotation & Classification

Pipeline for annotating, classifying, and analyzing semantic duplicates found by the contamination analysis pipeline.

## Overview

This directory handles the human-in-the-loop validation step: given high-similarity corpus-benchmark pairs from `pipeline/`, classify them as true semantic duplicates or false positives using LLM-based annotation (Gemini) and fine-tuned classifiers.

## Pipeline Steps

### 1. Sample high-similarity pairs
```bash
python sample_for_annotation.py --benchmark mbpp --max-per-test 20 --seed 42
python sample_for_annotation.py --benchmark codeforces --max-per-test 20 --seed 42
```

### 2. Annotate with Gemini
```bash
# Concurrent annotation with budget control
python annotate_semantic_duplicates.py --benchmark mbpp --budget 100 --workers 50

# Or batch annotation (cheaper)
python annotate_batch.py --benchmark mbpp --batch-size 5000
```

### 3. Export annotations
```bash
python export_annotations.py --benchmark mbpp
python export_annotations.py --benchmark codeforces
```

### 4. Validate exports
```bash
python validate_annotations.py --benchmark mbpp
```

### 5. Train classifier (optional)
```bash
# Fine-tune Qwen3-30B on annotation data (requires Tinker API)
export TINKER_API_KEY=your_key_here
python train_mbpp_classifier.py --epochs 3
```

### 6. Classify at scale
```bash
export TINKER_API_KEY=your_key_here
python classify_dolma_codeforces_top100.py
python classify_mbpp_sample100.py
```

### 7. Generate plots
```bash
# Unified publication-ready plots
python generate_all_plots.py

# Per-benchmark plots
python codeforces_v2_duplicate_plots.py
python mbpp_duplicate_plots.py
python zebralogic_duplicate_plots.py
```

## File Descriptions

| Script | Purpose |
|--------|---------|
| `sample_for_annotation.py` | Sample high-cosine pairs from pipeline output |
| `annotate_semantic_duplicates.py` | Concurrent Gemini annotation with cost tracking |
| `annotate_batch.py` | Gemini Batch API annotation (cheaper, async) |
| `export_annotations.py` | Export JSON annotations to aggregate CSVs |
| `validate_annotations.py` | Validate annotation data integrity |
| `train_mbpp_classifier.py` | Fine-tune Qwen3-30B classifier on annotations |
| `classify_dolma_codeforces_top100.py` | Classify CodeForces pairs with fine-tuned model |
| `classify_mbpp_sample100.py` | Classify MBPP pairs with fine-tuned model |
| `generate_all_plots.py` | Unified publication-ready plot generation |
| `codeforces_v2_duplicate_plots.py` | CodeForces-specific analysis plots |
| `mbpp_duplicate_plots.py` | MBPP-specific analysis plots |
| `zebralogic_duplicate_plots.py` | ZebraLogic-specific analysis plots |

## Data Layout

```
data/                          # Classified CSVs and plots (gitignored)
annotations/                   # Individual JSON annotations (gitignored)
training_data/                 # Ground-truth data for classifier training
```

## Environment Variables

- `GOOGLE_API_KEY`: Required for Gemini annotation
- `TINKER_API_KEY`: Required for classifier training and inference
