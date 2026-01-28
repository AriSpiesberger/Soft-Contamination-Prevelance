#!/bin/bash
# Launch MBPP finetuning experiments on 8 GPUs
#
# Usage:
#   ./launch_experiments.sh              # Run all experiments
#   ./launch_experiments.sh semantic     # Run semantic only
#   ./launch_experiments.sh exact        # Run exact only
#   ./launch_experiments.sh cosine       # Run cosine only

set -e

cd "$(dirname "$0")"

EXPERIMENT="${1:-all}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="outputs/experiment_${EXPERIMENT}_${TIMESTAMP}.log"

mkdir -p outputs

echo "=========================================="
echo "MBPP Finetuning Experiments"
echo "=========================================="
echo "Experiment: $EXPERIMENT"
echo "Log file: $LOG_FILE"
echo "Start time: $(date)"
echo "=========================================="

# Check GPUs
echo ""
echo "GPUs available:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Activate venv if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "../.venv" ]; then
    source ../.venv/bin/activate
fi

# Run with accelerate for multi-GPU
echo "Launching training with accelerate (8 GPUs)..."
echo ""

accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes 8 \
    run_mbpp_8gpu.py \
    --experiment "$EXPERIMENT" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Experiment complete!"
echo "End time: $(date)"
echo "Results: outputs/mbpp_experiment_results.json"
echo "Log: $LOG_FILE"
echo "=========================================="
