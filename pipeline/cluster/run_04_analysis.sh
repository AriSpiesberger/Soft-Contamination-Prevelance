#!/bin/bash
# =============================================================================
# Pipeline Stage 4: Contamination Analysis
# Runs 8 A100 40GB workers with FP16, TF32, prefetching
# =============================================================================
# This script is called by run_all.sh or can be run standalone.
# Configuration is read from the pipeline config.
# =============================================================================

set -e
set -o pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration (can be overridden by environment variables)
CONFIG_FILE="${PIPELINE_CONFIG:-$PIPELINE_ROOT/configs/default.yaml}"
VENV_PYTHON="${PIPELINE_VENV:-/lambda/nfs/embeddings/SDTD_Main/.venv/bin/python}"
CONFIG_HELPER="$PIPELINE_ROOT/lib/config_helper.py"

# Get config values
get_config() {
    $VENV_PYTHON "$CONFIG_HELPER" --config "$CONFIG_FILE" --get "$1"
}

# Read configuration
DATA_DIR="${ANALYSIS_CORPUS_DIR:-$(get_config 'analysis.corpus_dir')}"
OUTPUT_DIR="${ANALYSIS_OUTPUT_DIR:-$PIPELINE_ROOT/results/contamination}"
WORLD_SIZE="${ANALYSIS_WORLD_SIZE:-$(get_config 'analysis.cluster.world_size')}"
GPU_BATCH_SIZE="${ANALYSIS_GPU_BATCH:-$(get_config 'analysis.cluster.gpu_batch_size')}"
CORPUS_CHUNK="${ANALYSIS_CORPUS_CHUNK:-$(get_config 'analysis.cluster.corpus_gpu_chunk')}"
MAX_RETRIES=3
RESUME_FROM_FILE=${1:-0}

# Go to stages directory
cd "$PIPELINE_ROOT/stages"

echo "=========================================="
echo "Pipeline Stage 4: Contamination Analysis"
echo "FP16 (matches old run) + TF32 + Prefetch"
echo "=========================================="
echo "Config file:  $CONFIG_FILE"
echo "Data dir:     $DATA_DIR"
echo "Output dir:   $OUTPUT_DIR"
echo "World size:   $WORLD_SIZE"
echo "Resume from:  $RESUME_FROM_FILE"
echo "GPU batch:    $GPU_BATCH_SIZE"
echo "Corpus chunk: $CORPUS_CHUNK"
echo ""

# Clean up any existing processes
pkill -f "04_contamination_analysis" || true
sleep 2

# Environment optimizations
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PIPELINE_CONFIG="$CONFIG_FILE"

# Function to run a single worker with retries
run_worker() {
    local rank=$1
    local retries=0

    while [ $retries -lt $MAX_RETRIES ]; do
        echo "[Rank $rank] Starting (attempt $((retries + 1))/$MAX_RETRIES)..."

        CUDA_VISIBLE_DEVICES=$rank $VENV_PYTHON 04_contamination_analysis.py \
            --data-dir "$DATA_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --rank $rank \
            --world-size $WORLD_SIZE \
            --gpu-batch-size $GPU_BATCH_SIZE \
            --corpus-gpu-chunk $CORPUS_CHUNK \
            --resume-from-file $RESUME_FROM_FILE \
            2>&1 | tee -a "$PIPELINE_ROOT/logs/stage4_rank_${rank}.log"

        exit_code=$?

        if [ $exit_code -eq 0 ]; then
            echo "[Rank $rank] Completed successfully!"
            return 0
        else
            retries=$((retries + 1))
            echo "[Rank $rank] Failed with exit code $exit_code, retry $retries/$MAX_RETRIES"
            sleep 5
        fi
    done

    echo "[Rank $rank] Failed after $MAX_RETRIES retries"
    return 1
}

# Create directories
mkdir -p "$PIPELINE_ROOT/logs"
mkdir -p "$OUTPUT_DIR"

# Launch all workers in parallel
echo "Launching $WORLD_SIZE workers..."
pids=()

for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    run_worker $rank &
    pids+=($!)
    sleep 2  # Stagger starts to avoid model loading conflicts
done

echo ""
echo "All workers launched. PIDs: ${pids[*]}"
echo "Logs in: $PIPELINE_ROOT/logs/stage4_rank_*.log"
echo ""
echo "Waiting for completion..."

# Wait for all workers
failed=0
for i in "${!pids[@]}"; do
    wait ${pids[$i]} || failed=$((failed + 1))
done

if [ $failed -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "ALL WORKERS COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    echo "Results in: $OUTPUT_DIR/"
else
    echo ""
    echo "=========================================="
    echo "$failed worker(s) failed"
    echo "=========================================="
    echo "Check logs in: $PIPELINE_ROOT/logs/stage4_rank_*.log"
    echo "You can re-run this script to resume from checkpoints"
    exit 1
fi
