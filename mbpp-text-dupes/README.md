# MBPP Text Dupes

Generates natural language variations of MBPP task descriptions for semantic duplicate testing.

## Structure

```
mbpp-text-dupes/
├── english-variations/     # English semantic variations (10 per task)
│   ├── generate_english_variations.py
│   ├── regenerate_outputs.py
│   └── output/
│
├── other-languages/        # Translations (6 languages)
│   ├── generate_text_translations.py
│   ├── regenerate_outputs.py
│   └── output/
│
└── README.md
```

## English Variations

Two types of variations, 10 total:

### Subject-Based Variations (sub1-sub5)

Contextualizes the task in different real-world domains while preserving:
- The exact same algorithmic requirement
- The same function name (if mentioned in original)
- The same difficulty/complexity/clarity

| Code | Domain | Example Context |
|------|--------|-----------------|
| **sub1** | Sports/Games | players, teams, scores, matches, tournaments |
| **sub2** | Shopping/Inventory | products, prices, orders, stock, customers |
| **sub3** | School/Grades | students, grades, assignments, exams, courses |
| **sub4** | Food/Nutrition | ingredients, recipes, calories, meals, portions |
| **sub5** | Accounting/Finance | transactions, expenses, revenue, budgets |

### Paraphrase Variations (para1-para5)

Five different phrasings of the same task, **generated in a single batch API call** to ensure maximum diversity:
- Different wording but identical meaning
- Varied sentence structures (active/passive, formal/informal)
- Same technical requirements preserved
- Function names kept unchanged

| Code | Description |
|------|-------------|
| **para1** | Different wording, same meaning |
| **para2** | Alternative phrasing |
| **para3** | Rephrased version |
| **para4** | Different sentence structures |
| **para5** | Semantically equivalent version |

> **Note**: All 5 paraphrases are generated together so the model explicitly diversifies each one.

### Usage

```bash
cd english-variations

# Generate all 10 variations for 10 samples
python generate_english_variations.py --num-samples 10 --all

# Generate only subject-based variations (5)
python generate_english_variations.py --num-samples 100 --subjects

# Generate only paraphrases (5)
python generate_english_variations.py --num-samples 100 --paraphrases

# Generate specific variations
python generate_english_variations.py --variation sub1,sub2,para1

# Process full sanitized dataset (427 samples)
python generate_english_variations.py --num-samples 427 --all-splits --all

# Resume a previous run
python generate_english_variations.py --num-samples 427 --all --resume
```

## Other Languages

Translations to 6 languages:
- **es**: Spanish
- **fr**: French
- **de**: German
- **it**: Italian
- **ru**: Russian
- **zh**: Chinese

### Usage

```bash
cd other-languages

# Translate to all languages
python generate_text_translations.py --num-samples 10 --all

# Translate to specific languages
python generate_text_translations.py --language es,fr,de

# Resume a previous run
python generate_text_translations.py --num-samples 427 --all --resume
```

## Output Structure

Each subfolder generates:
- `output/individual/task_*.json` - Per-task results
- `output/master_results.json` - All results with metadata
- `output/*.csv` - Flat CSV for analysis
- `output/integrity_report.txt` - Statistics and status
- `output/archive/` - Timestamped backups

## Dataset

| Component | Dataset | Samples |
|-----------|---------|---------|
| English variations | MBPP Sanitized | 427 |
| Other languages | MBPP Full | 974 |

**MBPP Sanitized** (used for English):
- Higher quality, cleaner problem descriptions
- Well-defined test cases
- Consistent formatting

**MBPP Full** (used for translations):
- Complete dataset with all samples

## Models

| Component | Model | Reason |
|-----------|-------|--------|
| English variations | Claude Opus 4.5 | High-quality semantic transformations |
| Other languages | Claude Sonnet 4.5 | Efficient translation |

## Configuration

Requires `ANTHROPIC_API_KEY` environment variable.

```bash
export ANTHROPIC_API_KEY="your-key-here"
```
