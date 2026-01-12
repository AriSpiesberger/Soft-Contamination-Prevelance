# Pipeline to classify MuSR sem dupes

This directory contains tools for analyzing potential semantic duplicates between MuSR benchmark samples and training data samples.

## Overview

The analysis uses a **hybrid approach** with Claude Opus 4.5:

1. **Workflow 1 (Direct Comparison)**: Ask Claude to directly compare the MuSR test sample and training corpus sample
2. **Workflow 2 (Structural Analysis)**: Extract MMO (Means-Motive-Opportunity) structures from both texts, then compare structures

### Hybrid Logic

- Always run Workflow 1 first (efficient)
- If Workflow 1 returns `is_semantic_duplicate=1` AND `confidence < 0.7`, run Workflow 2 for verification
- Use Workflow 2 result as final verdict for uncertain cases

This balances **efficiency** (most cases resolved by Workflow 1) with **accuracy** (uncertain positives get deeper analysis).

## Setup

### 1. Install Dependencies

```bash
cd MuSR-top-1000-study
pip install -r requirements.txt
```

### 2. Set API Key

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

Or create a `.env` file in this directory or the parent directory:
```
ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

### Process All Murder Mystery Files

```bash
python analyze_murder_mysteries.py --input all
```

## Output Structure

```
analysis/
└── murder_mysteries/
    ├── dolci_100pct/
    │   ├── individual/
    │   │   ├── musr_murder_mysteries_0_rank1_abc123.json
    │   │   ├── musr_murder_mysteries_0_rank2_def456.json
    │   │   └── ...
    │   ├── consolidated_results.json
    │   ├── consolidated_results.csv
    │   └── semantic_duplicates_only.csv
    ├── dolci_dpo_100pct/
    │   └── ...
    └── _master/
        ├── consolidated_results.json
        └── consolidated_results.csv
```

### Individual JSON Structure

Each comparison produces a JSON file with:

```json
{
  "test_id": "musr_murder_mysteries_70",
  "corpus_id": "abc123...",
  "rank": 1,
  "score": 0.536,
  "test_text": "In the lavish surroundings...",
  "corpus_text": "The detective arrived...",
  "timestamp": "2026-01-12T10:30:00",
  "workflow1": {
    "workflow": 1,
    "success": true,
    "corpus_is_murder_mystery": false,
    "corpus_type_if_not_mm": "creative writing prompt",
    "is_semantic_duplicate": 0,
    "confidence": 0.95,
    "reasoning": "The corpus text is a creative writing prompt, not a murder mystery puzzle"
  },
  "workflow2_triggered": false,
  "final_verdict": {
    "is_semantic_duplicate": 0,
    "confidence": 0.95,
    "source": "workflow1"
  }
}
```

### Consolidated CSV Columns

| Column | Description |
|--------|-------------|
| `test_id` | MuSR sample ID |
| `corpus_id` | Training data sample ID |
| `rank` | Similarity rank (1 = most similar) |
| `score` | Similarity score |
| `is_semantic_duplicate` | Final verdict (0 or 1) |
| `confidence` | Confidence in verdict (0.0-1.0) |
| `verdict_source` | Which workflow produced the verdict |
| `workflow2_triggered` | Whether Workflow 2 was needed |
| `corpus_is_mm` | Whether corpus text is a murder mystery |
| `reasoning` | Brief explanation |
