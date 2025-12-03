# SDTD - Semantic Duplicate Training Data

Generate semantic duplicates of LLM training datasets at multiple levels of abstraction.

## Overview

This project generates **semantic duplicates** (SDs) of textual data in datasets:
- **Level 1**: Linguistic paraphrases (same meaning, different words/syntax/style)
- **Level 2**: Structural duplicates (same problem template, different parameters)

See `docs/SDs-level-1-claude.md` and `docs/SDs-level-2-claude.md` for detailed taxonomies.

## Project Setup

### Prerequisites
- Python 3.11+
- `uv` for dependency management
- OpenRouter API key (provides access to Anthropic, OpenAI, and other models)

### Installation

```bash
# Install dependencies
uv sync

# Copy and configure environment variables
cp .env.example .env
# Edit .env and add your OpenRouter API key:
# OPENROUTER_API_KEY=your_key_here
```

Get your OpenRouter API key from: https://openrouter.ai/

### Dependency Management
This project uses **`uv`** for dependency management. All dependencies are specified in `pyproject.toml`.

## Usage

### Generate Semantic Duplicates

```bash
# Generate Level 1 SDs for GSM8K (test with 10 items)
uv run python -m sdtd generate -d gsm8k -l 1 -n 10

# Generate both levels for Codeforces
uv run python -m sdtd generate -d codeforces -l 1,2 -n 5

# Generate for all datasets
uv run python -m sdtd generate -d all -l 1,2 -o outputs/full_run/

# Show available datasets
uv run python -m sdtd info
```

### Command Options

- `-d, --dataset`: Dataset name (`gsm8k`, `codeforces`, `allenai`, `mbpp`, `humaneval`, `popqa`, or `all`)
- `-l, --level`: Levels to generate (`1`, `2`, or `1,2`)
- `-o, --output-dir`: Output directory (default: `outputs/`)
- `-n, --limit`: Limit number of items (useful for testing)

## Project Structure

```
sdtd/
├── datasets/          # Input datasets (parquet, jsonl)
├── docs/              # Documentation and SD taxonomies
├── outputs/           # Generated semantic duplicates
├── prompts/           # YAML prompt templates
│   ├── level1.yaml    # Level 1 prompts
│   └── level2.yaml    # Level 2 prompts
└── sdtd/              # Python package
    ├── cli.py         # CLI interface
    ├── generate.py    # Generation logic
    ├── datasets.py    # Dataset loaders
    └── utils.py       # Utilities
```

## Datasets

See `docs/DATASETS.md` for details on available datasets:
- **GSM8K**: Math word problems (7,473 train items)
- **Codeforces**: Programming problems (869 train items)
- **AllenAI**: Educational text content
- **MBPP**: Python programming problems (427 train items)
- **HumanEval**: Python code evaluation (164 test items)
- **PopQA**: Question answering (14,000 test items)

## Prompts

Prompts are defined in YAML files under `prompts/`:
- `level1.yaml`: Linguistic paraphrase prompts
- `level2.yaml`: Structural duplicate prompts

Each prompt specifies:
- Model to use via OpenRouter (Claude Sonnet/Haiku, GPT-5.1/5-mini)
- Temperature and generation parameters
- Prompt template with placeholders

See `docs/MODELS.md` for details on model selection and OpenRouter configuration.

## Output Format

Generated SDs are saved as Parquet files with the following schema:
- `sd_level`: Level (1 or 2)
- `sd_variant`: Variant identifier
- `model_used`: Model that generated this SD
- `sd_text`: Generated semantic duplicate
- `timestamp`: Generation timestamp
- `original_*`: All original fields with `original_` prefix

## Features

- **OpenRouter integration**: Access multiple LLM providers with one API key
- **Disk caching**: LiteLLM responses cached to `.cache/litellm/`
- **Multi-model support**: Claude Sonnet 4.5/Haiku 4.5, GPT-5.1/5-mini via OpenRouter
- **Batch processing**: Process multiple datasets and levels
- **Efficient storage**: Parquet format with full metadata
- **Model override**: Test different models via CLI `--model` flag

## Note for AI Agents

AI agents working on this project should read `.cursorrules` for specific guidelines and policies.
