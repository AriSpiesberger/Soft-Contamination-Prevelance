#!/bin/bash
# Setup script for Soft Contamination project using uv

echo "Setting up project with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing now..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "Please restart your terminal and run this script again."
    exit 1
fi

# Sync venv to uv.lock (idempotent: creates .venv if missing, installs/updates to locked versions).
# Lock is cross-platform — torch resolves to cu124 wheels on x86_64 and pypi+cu13 wheels on aarch64.
echo "Syncing dependencies from uv.lock..."
uv sync

echo ""
echo "Setup complete. Run commands via 'uv run' (no activation needed), e.g.:"
echo "  uv run accelerate launch --num_processes=8 ecology/run_experiment_mistral_multigpu.py --epochs 10 --eval-every 5 --packing"
