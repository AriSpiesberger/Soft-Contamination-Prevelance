WANDB_IDS=(
  "fncwl91t"
  "6cuzb4us"
  "1fwmmqcm"
  "wfbq7fli"
  "dw374ycd"
  "rz796d0t"
  "kxcfvdye"
  "qorae2ss"
  "veebl6gl"
  "nqjujwky"
  "eehpc1k2"
)

for WANDB_ID in ${WANDB_IDS[@]}; do
  python p3_3_eval_zebralogic.py --wandb-id $WANDB_ID --batch-size 8
done
