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

# Create virtual environment and install dependencies
echo "Creating virtual environment..."
uv venv

echo "Installing dependencies..."
uv pip install -e .

echo ""
echo "Setup complete! To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
