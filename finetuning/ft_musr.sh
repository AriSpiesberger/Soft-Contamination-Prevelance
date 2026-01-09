giga_files=(
level0_murder_mystery_regenerated_samples-250_variants-2.json_opus45.jsonl     
level1_murder_mystery_swapped_samples-250_variants-2.json_opus45.jsonl
level2_murder_mystery_full_innocent_swap_samples-250_variants-2.json_opus45.jsonl
murder_mystery_original_samples_250_variants_2_opus45.jsonl
)

for gigafile in ${giga_files[@]}
do
    python3 p2_finetune_model.py -a ./datasets/teacher_answers/musr/$gigafile -o outputs/checkpoints/olmo3-qlora-{wandb_id} -e 3
    python3 p2_finetune_model.py -a ./datasets/teacher_answers/musr/$gigafile -o outputs/checkpoints/olmo3-qlora-{wandb_id} -e 6 --first_half_only
done
