# Scanning Semantic Duplicates of the Test Data (SSDTD)

Repository for SDTD experiments analyzing semantic similarity and distributional differences in text data.

See main working doc [here](https://docs.google.com/document/d/1uXhZt0kYXrt7xXtE0zCjLsUzMb1EhJr2VnBvbP4eQsc/edit?userstoinvite=nandischoots@gmail.com&sharingaction=manageaccess&role=writer&tab=t.3nos6mynlvkv).

## Setup with uv

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

### Installation

1. Install uv (if not already installed):
```bash
# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment and install dependencies:
```bash
uv venv
uv pip install -e .
```

3. Activate the virtual environment:
```bash
# On Windows (PowerShell)
.venv\Scripts\Activate.ps1

# On macOS/Linux
source .venv/bin/activate
```

### Running Scripts

```bash
# Run the main distribution comparison script
uv run python distribution_comparison.py

# Run analysis scripts
uv run python analysis_scripts/csv_distribution_comparison.py
uv run python analysis_scripts/dup_compy.py
```

### Optional Dependencies

For development tools:
```bash
uv pip install -e ".[dev]"
```

For GPU acceleration with Flash Attention (requires CUDA and is complex to build on Windows):
```bash
# Manual installation required
uv pip install flash-attn
```

**Note:** Flash Attention requires CUDA toolkit and is primarily supported on Linux. Windows users may need to use WSL2 or skip this optional dependency.

## Project Structure

- `distribution_comparison.py` - Main semantic similarity analysis script
- `analysis_scripts/` - Additional analysis and comparison scripts
- `data/` - Data files and background corpus
- `data_creation/` - Scripts for data preprocessing and sampling
- `production/` - Production pipeline and S3 integration
- `results/` - Output files and visualizations
- `sdtd-llm-generation/` - LLM-based data generation tools
- `utils/` - Utility functions and helpers
- `misc_scripts/` - Miscellaneous scripts

## Legacy Setup (requirements.txt)

If you prefer using pip with requirements.txt:
```bash
pip install -r requirements.txt
```
