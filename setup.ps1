# Setup script for SDTD project using uv (Windows PowerShell)

Write-Host "Setting up SDTD project with uv..." -ForegroundColor Green

# Check if uv is installed
if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed. Installing now..." -ForegroundColor Yellow
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    Write-Host "Please restart your terminal and run this script again." -ForegroundColor Yellow
    exit 1
}

# Create virtual environment and install dependencies
Write-Host "Creating virtual environment..." -ForegroundColor Cyan
uv venv

Write-Host "Installing dependencies..." -ForegroundColor Cyan
uv pip install -e .

Write-Host ""
Write-Host "Setup complete! To activate the virtual environment, run:" -ForegroundColor Green
Write-Host "  .venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "To run the main script:" -ForegroundColor Green
Write-Host "  uv run python distribution_comparison.py" -ForegroundColor Yellow

