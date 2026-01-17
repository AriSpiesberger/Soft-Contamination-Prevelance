#!/bin/bash
# Run mixed MUSR + Dolci finetuning experiments
# Uses 5:1 ratio (5 Dolci samples per MUSR sample)

giga_files=(
level0_murder_mystery_regenerated_samples-250_variants-2.json_opus45.jsonl     
level1_murder_mystery_swapped_samples-250_variants-2.json_opus45.jsonl
level2_murder_mystery_full_innocent_swap_samples-250_variants-2.json_opus45.jsonl
murder_mystery_original_samples_250_variants_2_opus45.jsonl
)

for gigafile in ${giga_files[@]}
do
    # 3 epochs with 5:1 Dolci ratio and saturation tracking
    python3 p2_finetune_mixed.py \
        -a ./datasets/teacher_answers/musr/$gigafile \
        -o outputs/checkpoints/olmo3-mixed-qlora-{wandb_id} \
        --dolci_ratio 5 \
        --eval_saturation \
        -e 3
    
    # 6 epochs on first half only (semantic duplicate experiment)
    python3 p2_finetune_mixed.py \
        -a ./datasets/teacher_answers/musr/$gigafile \
        -o outputs/checkpoints/olmo3-mixed-qlora-{wandb_id} \
        --dolci_ratio 5 \
        --eval_saturation \
        -e 6 \
        --first_half_only
done
