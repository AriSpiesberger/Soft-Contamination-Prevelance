# Project Structure Comparison

## Before Reorganization

```
SDTD_Main/
├── requirements.txt
├── README.md
│
├── distribution_comparison.py
├── csv_distribution_comparison.py    ❌ Root clutter
├── dup_compy.py                       ❌ Root clutter
├── calls.py                           ❌ Unrelated API script
├── random_paragraphs.jsonl           ❌ Duplicate file
├── random_sentences.jsonl            ❌ Duplicate file
├── murder_mystery_comparison_with_scores.csv  ❌ Result in root
│
├── data/
│   └── full_paragraphs.jsonl
│
├── data_creation/
│   ├── *.py
│   ├── random_paragraphs.jsonl
│   └── random_sentences.jsonl
│
├── duplicate_comparison/              ❌ Empty folder
│
├── duplicate_comparison_sampled/      ❌ Results in root
│   └── *.png
│
├── production/
│   └── *.py
│
├── sdtd-llm-generation/
│   └── ...
│
└── utils/
    └── *.py
```

## After Reorganization ✨

```
SDTD_Main/
├── pyproject.toml                     ✅ UV configuration
├── uv.lock                            ✅ Locked dependencies
├── .python-version                    ✅ Python version
├── requirements.txt                   ✅ Kept for compatibility
├── README.md                          ✅ Updated with UV docs
├── setup.sh                           ✅ Unix setup script
├── setup.ps1                          ✅ Windows setup script
├── CHANGES_SUMMARY.md                 ✅ Documentation
├── PROJECT_STRUCTURE.md               ✅ This file
│
├── distribution_comparison.py         ✅ Main script (clean root)
│
├── analysis_scripts/                  ✅ NEW: Organized scripts
│   ├── csv_distribution_comparison.py
│   └── dup_compy.py
│
├── data/
│   └── full_paragraphs.jsonl
│
├── data_creation/
│   ├── *.py
│   ├── random_paragraphs.jsonl        ✅ Single source of truth
│   └── random_sentences.jsonl         ✅ Single source of truth
│
├── production/
│   └── *.py
│
├── results/                           ✅ NEW: Organized outputs
│   ├── duplicate_comparison_sampled/
│   │   └── *.png
│   └── murder_mystery_comparison_with_scores.csv
│
├── sdtd-llm-generation/
│   ├── pyproject.toml                 ✅ Has its own UV config
│   ├── uv.lock
│   └── ...
│
├── utils/
│   └── *.py
│
└── misc_scripts/                      ✅ NEW: Miscellaneous code
    └── calls.py
```

## Key Improvements

### 1. **Cleaner Root Directory**
- Moved analysis scripts to `analysis_scripts/`
- Moved results to `results/`
- Moved miscellaneous scripts to `misc_scripts/`
- Removed duplicate data files
- Removed empty folder

### 2. **Modern Package Management**
- Added `pyproject.toml` for UV
- Generated `uv.lock` for reproducibility
- Added `.python-version` for version control
- Created convenient setup scripts

### 3. **Better Organization**
- **Scripts by purpose**: Main, analysis, production, data creation, misc
- **Results isolated**: All outputs in `results/`
- **Clear hierarchy**: Easy to find what you need
- **No duplicates**: Single source of truth for data files

### 4. **Improved Documentation**
- Updated README with UV instructions
- Added CHANGES_SUMMARY.md
- Added PROJECT_STRUCTURE.md (this file)
- Documented all changes

## Folder Purposes

| Folder | Purpose |
|--------|---------|
| `analysis_scripts/` | Scripts for comparing and analyzing data |
| `data/` | Main data corpus |
| `data_creation/` | Scripts for generating and preprocessing data |
| `production/` | Production-ready pipeline code |
| `results/` | All output files, plots, and analysis results |
| `sdtd-llm-generation/` | LLM-based generation tools (standalone module) |
| `utils/` | Shared utility functions |
| `misc_scripts/` | One-off scripts and experimental code |

## Migration Commands Used

```bash
# Created new folders
mkdir analysis_scripts results misc_scripts

# Moved files with git
git mv csv_distribution_comparison.py analysis_scripts/
git mv dup_compy.py analysis_scripts/
git mv calls.py misc_scripts/

# Moved untracked files
move murder_mystery_comparison_with_scores.csv results/
move duplicate_comparison_sampled results/

# Removed duplicates
rm random_paragraphs.jsonl
rm random_sentences.jsonl

# Removed empty folder
rmdir duplicate_comparison

# Set up UV
uv lock
uv sync
```

## Result

- **13 files** reorganized
- **3 new folders** created for better organization
- **1 empty folder** removed
- **2 duplicate files** removed
- **100% working** UV setup with lock file
- **All tests passing** ✅

