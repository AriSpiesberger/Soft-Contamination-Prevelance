#!/bin/bash
# Full MBPP experiment: base eval + train 3ep + train 6ep + all finetuned evals
set -e

echo "=============================================="
echo "MBPP Full Experiment Pipeline"
echo "=============================================="

cd "$(dirname "$0")"

# Step 1-2: Eval base model (only once)
echo ""
echo "[1/8] Evaluating BASE model on TRAIN split..."
python p3_eval_mbpp.py --test-split train

echo ""
echo "[2/8] Evaluating BASE model on EVAL split..."
python p3_eval_mbpp.py --test-split eval

# Step 3: Train 3 epochs with KL regularization
echo ""
echo "[3/8] Training for 3 EPOCHS (with KL regularization)..."
TRAIN_OUTPUT_3=$(python p2_train_mbpp_kl.py --num_train_epochs 3 --kl_beta 0.1 2>&1 | tee /dev/stderr)
WANDB_ID_3EP=$(echo "$TRAIN_OUTPUT_3" | grep -oP "Wandb run: .* \(id: \K[a-z0-9]+(?=\))" | head -1)

if [ -z "$WANDB_ID_3EP" ]; then
    echo "ERROR: Could not capture wandb_id from 3-epoch training"
    exit 1
fi
echo "3-epoch training complete. Wandb ID: $WANDB_ID_3EP"

# Step 4-5: Eval 3-epoch finetuned model
echo ""
echo "[4/8] Evaluating 3-EPOCH FINETUNED on TRAIN split..."
python p3_eval_mbpp.py --test-split train --finetuned --wandb-id "$WANDB_ID_3EP" --epochs 3

echo ""
echo "[5/8] Evaluating 3-EPOCH FINETUNED on EVAL split..."
python p3_eval_mbpp.py --test-split eval --finetuned --wandb-id "$WANDB_ID_3EP" --epochs 3

# Step 6: Train 6 epochs with KL regularization
echo ""
echo "[6/8] Training for 6 EPOCHS (with KL regularization)..."
TRAIN_OUTPUT_6=$(python p2_train_mbpp_kl.py --num_train_epochs 6 --kl_beta 0.1 2>&1 | tee /dev/stderr)
WANDB_ID_6EP=$(echo "$TRAIN_OUTPUT_6" | grep -oP "Wandb run: .* \(id: \K[a-z0-9]+(?=\))" | head -1)

if [ -z "$WANDB_ID_6EP" ]; then
    echo "ERROR: Could not capture wandb_id from 6-epoch training"
    exit 1
fi
echo "6-epoch training complete. Wandb ID: $WANDB_ID_6EP"

# Step 7-8: Eval 6-epoch finetuned model
echo ""
echo "[7/8] Evaluating 6-EPOCH FINETUNED on TRAIN split..."
python p3_eval_mbpp.py --test-split train --finetuned --wandb-id "$WANDB_ID_6EP" --epochs 6

echo ""
echo "[8/8] Evaluating 6-EPOCH FINETUNED on EVAL split..."
python p3_eval_mbpp.py --test-split eval --finetuned --wandb-id "$WANDB_ID_6EP" --epochs 6

echo ""
echo "=============================================="
echo "EXPERIMENT COMPLETE"
echo "=============================================="
echo ""
echo "Runs in W&B (semdupes-olmo3-mbpp):"
echo "  - eval-mbpp-base-train-*"
echo "  - eval-mbpp-base-eval-*"
echo "  - train-mbpp-*-3ep ($WANDB_ID_3EP)"
echo "  - eval-mbpp-ft-train-* (3ep)"
echo "  - eval-mbpp-ft-eval-* (3ep)"
echo "  - train-mbpp-*-6ep ($WANDB_ID_6EP)"
echo "  - eval-mbpp-ft-train-* (6ep)"
echo "  - eval-mbpp-ft-eval-* (6ep)"
echo ""
echo "=============================================="
