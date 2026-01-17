ids=(
t0q78psm
9begodxy
jyc0s66t
3vik4ffz
hwyymney
)
for id in ${ids[@]}; do
    python3 p3_2_eval_musr.py --finetuned --wandb-id $id --fast
done
