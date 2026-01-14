#!/bin/bash
# =============================================================================
# MBPP Overnight Training + Evaluation Script
# Trains MBPP semantic pairs for 3 and 6 epochs, then evaluates all combos
# =============================================================================
set -e  # Exit on error

cd "$(dirname "$0")"  # Change to finetuning directory

LOG_FILE="overnight_mbpp_$(date +%Y%m%d_%H%M%S).log"

echo "=============================================" | tee -a "$LOG_FILE"
echo "Starting MBPP overnight run at $(date)" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"

# =============================================================================
# PHASE 1: Training (3 epochs)
# =============================================================================
echo "" | tee -a "$LOG_FILE"
echo "[PHASE 1/2] Training 3 epochs..." | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"

# Train and capture wandb id from output
TRAIN_3_OUTPUT=$(python3 p2_train_mbpp.py -e 3 2>&1 | tee -a "$LOG_FILE")
WANDB_ID_3=$(echo "$TRAIN_3_OUTPUT" | grep -oP "Wandb run id: \K\S+" | tail -1)

echo "3-epoch training complete! Wandb ID: $WANDB_ID_3" | tee -a "$LOG_FILE"
echo "Checkpoint: outputs/checkpoints/olmo3-mbpp-qlora-$WANDB_ID_3" | tee -a "$LOG_FILE"

# =============================================================================
# PHASE 2: Training (6 epochs)
# =============================================================================
echo "" | tee -a "$LOG_FILE"
echo "[PHASE 2/2] Training 6 epochs..." | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"

TRAIN_6_OUTPUT=$(python3 p2_train_mbpp.py -e 6 2>&1 | tee -a "$LOG_FILE")
WANDB_ID_6=$(echo "$TRAIN_6_OUTPUT" | grep -oP "Wandb run id: \K\S+" | tail -1)

echo "6-epoch training complete! Wandb ID: $WANDB_ID_6" | tee -a "$LOG_FILE"
echo "Checkpoint: outputs/checkpoints/olmo3-mbpp-qlora-$WANDB_ID_6" | tee -a "$LOG_FILE"

# =============================================================================
# PHASE 3: Evaluations
# =============================================================================
echo "" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
echo "Starting evaluations at $(date)" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"

# --- 3.1: Base model evals (for comparison) ---
echo "" | tee -a "$LOG_FILE"
echo "[EVAL] Base model - Train split (contamination baseline)..." | tee -a "$LOG_FILE"
python3 p3_eval_mbpp.py --test-split train 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "[EVAL] Base model - Eval split (generalization baseline)..." | tee -a "$LOG_FILE"
python3 p3_eval_mbpp.py --test-split eval 2>&1 | tee -a "$LOG_FILE"

# --- 3.2: 3-epoch model evals ---
echo "" | tee -a "$LOG_FILE"
echo "[EVAL] 3-epoch model ($WANDB_ID_3) - Train split (contamination)..." | tee -a "$LOG_FILE"
python3 p3_eval_mbpp.py --finetuned --wandb-id "$WANDB_ID_3" --test-split train 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "[EVAL] 3-epoch model ($WANDB_ID_3) - Eval split (generalization)..." | tee -a "$LOG_FILE"
python3 p3_eval_mbpp.py --finetuned --wandb-id "$WANDB_ID_3" --test-split eval 2>&1 | tee -a "$LOG_FILE"

# --- 3.3: 6-epoch model evals ---
echo "" | tee -a "$LOG_FILE"
echo "[EVAL] 6-epoch model ($WANDB_ID_6) - Train split (contamination)..." | tee -a "$LOG_FILE"
python3 p3_eval_mbpp.py --finetuned --wandb-id "$WANDB_ID_6" --test-split train 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "[EVAL] 6-epoch model ($WANDB_ID_6) - Eval split (generalization)..." | tee -a "$LOG_FILE"
python3 p3_eval_mbpp.py --finetuned --wandb-id "$WANDB_ID_6" --test-split eval 2>&1 | tee -a "$LOG_FILE"

# =============================================================================
# PHASE 4: Semantic Evaluations
# =============================================================================
echo "" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
echo "Starting semantic evaluations at $(date)" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"

# --- 4.1: Base model semantic eval ---
echo "" | tee -a "$LOG_FILE"
echo "[SEMANTIC EVAL] Base model..." | tee -a "$LOG_FILE"
python3 p3_4_eval_semantic.py 2>&1 | tee -a "$LOG_FILE"

# --- 4.2: 3-epoch semantic eval ---
echo "" | tee -a "$LOG_FILE"
echo "[SEMANTIC EVAL] 3-epoch model ($WANDB_ID_3)..." | tee -a "$LOG_FILE"
python3 p3_4_eval_semantic.py --finetuned --finetuned-path "./outputs/checkpoints/olmo3-mbpp-qlora-$WANDB_ID_3" 2>&1 | tee -a "$LOG_FILE"

# --- 4.3: 6-epoch semantic eval ---
echo "" | tee -a "$LOG_FILE"
echo "[SEMANTIC EVAL] 6-epoch model ($WANDB_ID_6)..." | tee -a "$LOG_FILE"
python3 p3_4_eval_semantic.py --finetuned --finetuned-path "./outputs/checkpoints/olmo3-mbpp-qlora-$WANDB_ID_6" 2>&1 | tee -a "$LOG_FILE"

# =============================================================================
# SUMMARY
# =============================================================================
echo "" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
echo "OVERNIGHT RUN COMPLETE at $(date)" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Wandb IDs:" | tee -a "$LOG_FILE"
echo "  3-epoch: $WANDB_ID_3" | tee -a "$LOG_FILE"
echo "  6-epoch: $WANDB_ID_6" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Checkpoints:" | tee -a "$LOG_FILE"
echo "  outputs/checkpoints/olmo3-mbpp-qlora-$WANDB_ID_3" | tee -a "$LOG_FILE"
echo "  outputs/checkpoints/olmo3-mbpp-qlora-$WANDB_ID_6" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Full log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
