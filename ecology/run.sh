#!/bin/bash
# Wrapper: ensures deps are synced, then launches the multi-GPU mistral experiment.
# Usage: ./run.sh [extra args forwarded to the python script]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

uv run accelerate launch \
  --num_processes=8 \
  ecology/run_experiment_llama_multigpu.py \
  --epochs 10 \
  --eval-every 5 \
  --packing \
  "$@"
