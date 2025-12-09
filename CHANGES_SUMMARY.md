# Project Organization and UV Setup - Summary of Changes

## Overview
This document summarizes the project reorganization and migration to `uv` for package management.

## Changes Made

### 1. Package Management with UV

#### Added Files:
- **`pyproject.toml`**: Modern Python project configuration with:
  - Project metadata and dependencies
  - Development dependencies group
  - Tool configurations for black and ruff
  - Hatchling build backend configuration
  
- **`uv.lock`**: Lock file with resolved dependencies for reproducible installs

- **`.python-version`**: Specifies Python 3.11 for the project

- **`setup.sh`**: Setup script for Unix/Linux/macOS systems

- **`setup.ps1`**: Setup script for Windows PowerShell

#### Updated Files:
- **`README.md`**: Added comprehensive UV setup instructions and updated project structure documentation

### 2. Project Organization

#### New Folders Created:

1. **`analysis_scripts/`**: Contains analysis and comparison scripts
   - `csv_distribution_comparison.py` (moved from root)
   - `dup_compy.py` (moved from root)

2. **`results/`**: Contains output files and visualizations
   - `duplicate_comparison_sampled/` (moved from root)
   - `murder_mystery_comparison_with_scores.csv` (moved from root)

3. **`misc_scripts/`**: Contains miscellaneous scripts
   - `calls.py` (API testing script, moved from root)

#### Files Removed:
- `random_paragraphs.jsonl` (duplicate removed from root, kept in data_creation/)
- `random_sentences.jsonl` (duplicate removed from root, kept in data_creation/)
- `duplicate_comparison/` (empty folder removed)

#### Files Updated:
- **`distribution_comparison.py`**: Updated OUTPUT_DIR path to `results/duplicate_comparison_sampled`
- **`analysis_scripts/csv_distribution_comparison.py`**: Updated paths to reflect new structure

### 3. Project Structure (After Reorganization)

```
SDTD_Main/
├── pyproject.toml              # UV project configuration
├── uv.lock                     # Locked dependencies
├── .python-version             # Python version specification
├── requirements.txt            # Legacy pip requirements (kept for compatibility)
├── README.md                   # Updated with UV instructions
├── setup.sh                    # Unix setup script
├── setup.ps1                   # Windows setup script
│
├── distribution_comparison.py  # Main analysis script
│
├── analysis_scripts/           # Analysis and comparison tools
│   ├── csv_distribution_comparison.py
│   └── dup_compy.py
│
├── data/                       # Data files
│   └── full_paragraphs.jsonl
│
├── data_creation/              # Data preprocessing scripts
│   ├── data_grubber.py
│   ├── data_grubber_dolmo.py
│   ├── data_subsampler.py
│   ├── process_full_sample.py
│   ├── random_paragraphs.jsonl
│   ├── random_sentences.jsonl
│   └── dolma3_sample/
│
├── production/                 # Production pipeline
│   ├── __init__.py
│   ├── production_embeddings.py
│   ├── production_pipeline.py
│   ├── s3_config.py
│   └── README.md
│
├── results/                    # Output files and visualizations
│   ├── duplicate_comparison_sampled/
│   │   ├── *.png (various plots)
│   │   └── results.json
│   └── murder_mystery_comparison_with_scores.csv
│
├── sdtd-llm-generation/        # LLM generation tools (has its own pyproject.toml)
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── README.md
│   ├── sdtd/
│   ├── docs/
│   └── prompts/
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   └── utilities.py
│
└── misc_scripts/               # Miscellaneous scripts
    └── calls.py
```

## Benefits of UV

1. **Speed**: UV is 10-100x faster than pip for package installation
2. **Reliability**: Lock file ensures reproducible installs across environments
3. **Modern**: Built on Rust, follows modern Python packaging standards
4. **Compatibility**: Works alongside existing pip/requirements.txt workflows

## Quick Start Commands

### First Time Setup:
```bash
# Windows PowerShell
.\setup.ps1

# Or manually:
uv venv
uv sync
.venv\Scripts\Activate.ps1
```

### Running Scripts:
```bash
# Run without activating venv
uv run python distribution_comparison.py
uv run python analysis_scripts/csv_distribution_comparison.py

# Or activate venv first
.venv\Scripts\Activate.ps1
python distribution_comparison.py
```

### Managing Dependencies:
```bash
# Add a new dependency
uv add numpy

# Update dependencies
uv lock --upgrade

# Install dev dependencies
uv sync --group dev
```

## Migration Notes

- The `requirements.txt` file is kept for backward compatibility but is no longer the primary dependency source
- All paths in scripts have been updated to reflect the new folder structure
- Flash Attention is documented but not included by default due to CUDA build requirements on Windows

## Testing

The setup has been tested and verified:
- ✅ `uv lock` - Successfully generates lock file
- ✅ `uv sync` - Successfully installs all dependencies
- ✅ Virtual environment creation works
- ✅ All file paths updated correctly

## Next Steps

Users can now:
1. Clone the repository
2. Run `.\setup.ps1` (Windows) or `./setup.sh` (Unix)
3. Start using the project with UV

The project is now properly organized and ready for efficient development with modern Python tooling.

