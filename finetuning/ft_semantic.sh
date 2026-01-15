#!/bin/bash
# Finetune on semantic pairs (english_synonym_input -> python_semantic_output)
# Uses same QLoRA setup as other finetunes

# Full dataset, 3 epochs
python3 p2_finetune_semantic.py \
    -i ../semantic_pairs_full.csv \
    -o outputs/checkpoints/olmo3-semantic-qlora-{wandb_id} \
    -e 3

# First half only, 6 epochs (semantic duplicate experiment)
python3 p2_finetune_semantic.py \
    -i ../semantic_pairs_full.csv \
    -o outputs/checkpoints/olmo3-semantic-qlora-{wandb_id} \
    -e 6 \
    --first_half_only
