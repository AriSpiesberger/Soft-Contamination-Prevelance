#!/bin/bash
set -euo pipefail
cd /lambda/nfs/ecological-files/Soft-Contamination-Prevelance/ecology
export PATH="/home/ubuntu/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run --project /lambda/nfs/ecological-files/Soft-Contamination-Prevelance \
  accelerate launch --num_processes=8 \
    run_experiment_qwen3_multigpu.py \
    --epochs 20 \
    --eval-every 5 \
    --per-device-batch-size 8 \
    --effective-batch-size 64 \
    --gradient-checkpointing \
    --no-torch-compile
