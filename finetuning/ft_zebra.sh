ft_files=(
zebralogic-original-shards-000-to-004-of-010-ver1.jsonl                   zebralogic-sd-shuffle_and_paraphrase-shards-005-to-009-of-010-ver1.jsonl
zebralogic-original-shards-000-to-009-of-010-ver1.jsonl                   zebralogic-sd-shuffle_and_substitute-shards-000-to-004-of-010-ver1.jsonl
zebralogic-original-shards-005-to-009-of-010-ver1.jsonl                   zebralogic-sd-shuffle_and_substitute-shards-000-to-009-of-010-ver1.jsonl
zebralogic-sd-paraphrase-shards-000-to-004-of-010-ver1.jsonl              zebralogic-sd-shuffle_and_substitute-shards-005-to-009-of-010-ver1.jsonl
zebralogic-sd-paraphrase-shards-000-to-009-of-010-ver1.jsonl              zebralogic-sd-shuffle_and_substitute_and_paraphrase-shards-000-to-004-of-010-ver1.jsonl
zebralogic-sd-paraphrase-shards-005-to-009-of-010-ver1.jsonl              zebralogic-sd-shuffle_and_substitute_and_paraphrase-shards-000-to-009-of-010-ver1.jsonl
zebralogic-sd-shuffle_and_paraphrase-shards-000-to-004-of-010-ver1.jsonl  zebralogic-sd-shuffle_and_substitute_and_paraphrase-shards-005-to-009-of-010-ver1.jsonl
zebralogic-sd-shuffle_and_paraphrase-shards-000-to-009-of-010-ver1.jsonl
)

for ft_file_name in ${ft_files[@]}; do
    full_path=$"./datasets/teacher_answers/zebralogic/$ft_file_name"
    python3 p2_finetune_model.py -a $full_path -o "outputs/checkpoints/olmo3-qlora-{wandb_id}" -e 5
done
