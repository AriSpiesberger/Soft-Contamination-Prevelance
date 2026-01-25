#!/bin/bash
source .venv/bin/activate

# Generate plot comparing baseline with top fine-tuned models
python plot_zebralogic_gridsize_performance.py \
    --models \
        sdtd-gpt-4.1-mini-2025-04-14-run2 \
        sdtd-gpt-4.1-mini-zebralogic-original-ver2-run1 \
        sdtd-gpt-4.1-mini-zebralogic-original-ver2-2x-run1 \
        sdtd-gpt-4.1-mini-zebralogic-sd-shuffle-and-substitute-and-paraphrase-rea2-h1-run1 \
    --model-names \
        "Baseline (GPT-4.1-mini)" \
        "FT-Original" \
        "FT-Original-2x" \
        "FT-SD-All+Reasoning" \
    --baseline sdtd-gpt-4.1-mini-2025-04-14-run2 \
    --output zebralogic_gridsize_performance.png

echo ""
echo "==========================================="
echo "Plot generated: zebralogic_gridsize_performance.png"
echo "==========================================="
