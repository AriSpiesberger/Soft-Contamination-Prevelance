#!/usr/bin/env bash
# Evaluate baseline + every per-epoch LoRA adapter from a training run on the
# custom mbpp_split task (3-shot, chat-template, greedy decode, fenced code,
# 207 contam + 211 clean items). Greedy is deterministic so num_runs=1 is the
# right default.
#
# Usage:
#   bash finetuning/mbpp/run_eval.sh                  # most recent run, n=1
#   bash finetuning/mbpp/run_eval.sh <run_dir>        # specific run
#   bash finetuning/mbpp/run_eval.sh <run_dir> 3      # n=3 (sanity)
#   SKIP_BASELINE=1 bash finetuning/mbpp/run_eval.sh  # adapters only
set -euo pipefail
cd "$(dirname "$0")/../.."

# MBPP runs untrusted model code to compute pass@1 — HF requires explicit opt-in.
export HF_ALLOW_CODE_EVAL=1

RUN_DIR="${1:-}"
NUM_RUNS="${2:-1}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"

if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR=$(ls -td finetuning/outputs/checkpoints/olmo3-mbpp-qlora-*/ 2>/dev/null | head -1 || true)
fi
RUN_DIR="${RUN_DIR%/}"

if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
    echo "ERROR: no run_dir found; pass one as the first arg" >&2
    exit 1
fi

echo "==== Eval plan ===="
echo "Run dir : $RUN_DIR"
echo "Num runs: $NUM_RUNS"
echo "Baseline: $([[ $SKIP_BASELINE -eq 1 ]] && echo skip || echo run)"
echo "==================="

if [[ "$SKIP_BASELINE" != "1" ]]; then
    echo
    echo "===== baseline ====="
    uv run python finetuning/mbpp/eval_mbpp_lmeval.py \
        --num_runs "$NUM_RUNS" \
        --tasks mbpp_split \
        --name baseline
fi

# Per-epoch checkpoints, ascending step order
for ckpt in $(ls -d "$RUN_DIR"/checkpoint-*/ 2>/dev/null \
              | awk -F'checkpoint-' '{print $2"\t"$0}' \
              | sort -n \
              | cut -f2); do
    name="$(basename "$ckpt")"
    echo
    echo "===== $name ====="
    uv run python finetuning/mbpp/eval_mbpp_lmeval.py \
        --num_runs "$NUM_RUNS" \
        --tasks mbpp_split \
        --adapter "$ckpt" \
        --name "$name"
done

echo
echo "==== Done ===="
echo "Per-condition summaries:"
ls -d finetuning/outputs/harness_results/*_*/ 2>/dev/null | tail -n 12
