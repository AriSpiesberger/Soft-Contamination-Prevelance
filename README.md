# Testing semdupes generation for the MuSR dataset

## Recent Changes & Extensions

This repository contains extensions to the original MuSR (Multistep Soft Reasoning) dataset for analyzing story generation variability.

### What's New

New scripts have been added to `musr_dataset_scripts/` that enable story regeneration and analysis:

#### 1. **Story Regeneration from Logic Trees** (`regenerate_from_trees.py`)
- Regenerates stories from existing logic trees with `temperature=1.0`
- Demonstrates that the same logic trees produce **different wording** but **identical facts**
- Takes the first 10 samples from `murder_mystery.json` and creates new story text
- Outputs: `murder_mystery_regenerated_first10.json`

**Key Insight**: By using the same underlying logic trees but with temperature sampling, we can generate multiple story variations that preserve semantic content while varying linguistic expression.

**CSV Columns**:
- sample_number, victim, weapon, crime_scene, suspects, murderer
- question, choices, answer_index, answer_choice
- original_story, regenerated_story

### Usage

#### Regenerate Stories
```bash
cd musr_dataset_scripts
OPENAI_API_KEY=your_key python regenerate_from_trees.py
```
This will:
- Load the first 10 samples from `murder_mystery.json`
- Regenerate story text using the same logic trees
- Save results to `murder_mystery_regenerated_first10.json`
- Display cost estimates (~$0.02-0.05 per sample)


#### Export to CSV
```bash
cd musr_dataset_scripts
python export_to_csv.py
```
Creates a spreadsheet-friendly CSV file with side-by-side story comparisons.

### Dataset Files

New files in `datasets/`:
- `murder_mystery_regenerated_first10.json` - Regenerated stories with metadata
- `murder_mystery_comparison_first10.csv` - CSV export for analysis

### Original MuSR Documentation

For the original MuSR project documentation, installation instructions, and evaluation guides, see: **[README_MuSR.md](README_MuSR.md)**

