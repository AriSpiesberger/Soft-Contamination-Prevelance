#!/bin/bash
# Train MBPP semantic pairs and evaluate

# Train for 3 epochs
echo "=== Training 3 epochs ==="
python3 p2_train_mbpp.py -e 3
# Save the wandb id from output

# Train for 6 epochs
echo "=== Training 6 epochs ==="
python3 p2_train_mbpp.py -e 6

# After training, evaluate with:
# python3 p3_eval_mbpp.py --finetuned --wandb-id <3_EPOCH_ID> --test-split train   # contamination
# python3 p3_eval_mbpp.py --finetuned --wandb-id <3_EPOCH_ID> --test-split eval    # generalization
# python3 p3_eval_mbpp.py --finetuned --wandb-id <6_EPOCH_ID> --test-split train   # contamination
# python3 p3_eval_mbpp.py --finetuned --wandb-id <6_EPOCH_ID> --test-split eval    # generalization
