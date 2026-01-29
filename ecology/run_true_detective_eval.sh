#!/bin/bash
# Run True Detective evaluation on base models and final finetuned models
#
# Usage:
#   ./run_true_detective_eval.sh          # Run all models
#   ./run_true_detective_eval.sh olmo     # Run OLMo only
#   ./run_true_detective_eval.sh qwen     # Run Qwen only

set -e

cd "$(dirname "$0")"

MODEL="${1:-all}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "TRUE DETECTIVE EVALUATION"
echo "=============================================="
echo "Model: $MODEL"
echo "Start time: $(date)"
echo "=============================================="

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo ""
echo "=== STEP 1: Evaluating BASE models ==="
echo ""
python eval_true_detective.py --baseline --model "$MODEL"

echo ""
echo "=== STEP 2: Evaluating FINETUNED models ==="
echo ""
python eval_true_detective.py --model "$MODEL"

echo ""
echo "=============================================="
echo "EVALUATION COMPLETE"
echo "End time: $(date)"
echo "Results in: outputs/true_detective_evals/"
echo "=============================================="
