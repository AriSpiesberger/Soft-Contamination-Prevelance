#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# KL anchor doubles the forward-pass cost (model + ref). bs=8 with
# grad_accum=2 keeps the effective batch at 16 while halving peak memory.
export PYTORCH_ALLOC_CONF=expandable_segments:True

exec uv run python finetuning/mbpp/train_mbpp_kl.py \
    --skip_quantization \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --kl_beta 0.1 \
    --save_steps 5 \
    --no-wandb
