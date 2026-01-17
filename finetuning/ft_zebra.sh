ft_files=(
zebralogic-original-shards-000-to-004-of-010-ver2.jsonl                   zebralogic-sd-shuffle_and_substitute-shards-000-to-004-of-010-ver2.jsonl
zebralogic-sd-paraphrase-shards-000-to-004-of-010-ver2.jsonl              zebralogic-sd-shuffle_and_substitute_and_paraphrase-shards-000-to-004-of-010-ver2.jsonl
zebralogic-sd-shuffle_and_paraphrase-shards-000-to-004-of-010-ver2.jsonl
)

for ft_file_name in ${ft_files[@]}; do
    full_path=$"./datasets/teacher_answers/zebralogic_v2/$ft_file_name"
    python3 p2_finetune_model.py -a $full_path -o "outputs/checkpoints/olmo3-qlora-{wandb_id}" -e 10
done
