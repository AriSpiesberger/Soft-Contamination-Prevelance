#!/bin/bash
# =============================================================================
# Pipeline Stage 5: Merge Results
# Parallel merge across workers
# =============================================================================

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
CONFIG_FILE="${PIPELINE_CONFIG:-$PIPELINE_ROOT/configs/default.yaml}"
VENV_PYTHON="${PIPELINE_VENV:-/lambda/nfs/embeddings/SDTD_Main/.venv/bin/python}"
CONFIG_HELPER="$PIPELINE_ROOT/lib/config_helper.py"

# Get config values
get_config() {
    $VENV_PYTHON "$CONFIG_HELPER" --config "$CONFIG_FILE" --get "$1"
}

OUTPUT_DIR="${MERGE_OUTPUT_DIR:-$PIPELINE_ROOT/results/contamination}"
WORLD_SIZE="${MERGE_WORLD_SIZE:-$(get_config 'merge.world_size')}"

cd "$PIPELINE_ROOT/stages"

echo "=========================================="
echo "Pipeline Stage 5: Merge Results"
echo "=========================================="
echo "Output dir:  $OUTPUT_DIR"
echo "World size:  $WORLD_SIZE"
echo ""

# Clean up any previous merge outputs
rm -f "$OUTPUT_DIR/merged_rank_*.json"

# Create logs directory
mkdir -p "$PIPELINE_ROOT/logs"

# Launch all workers in parallel
echo "Launching $WORLD_SIZE merge workers..."
pids=()

for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    echo "  Starting rank $rank..."
    $VENV_PYTHON 05_merge_results.py \
        --output-dir "$OUTPUT_DIR" \
        --world-size $WORLD_SIZE \
        --rank $rank \
        2>&1 | tee "$PIPELINE_ROOT/logs/stage5_rank_${rank}.log" &
    pids+=($!)
    sleep 1  # Stagger starts
done

echo ""
echo "All workers launched. PIDs: ${pids[*]}"
echo "Logs in: $PIPELINE_ROOT/logs/stage5_rank_*.log"
echo ""
echo "Waiting for all workers to complete..."

# Wait for all workers
failed=0
for i in "${!pids[@]}"; do
    wait ${pids[$i]} || failed=$((failed + 1))
done

if [ $failed -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "ALL MERGE WORKERS COMPLETED!"
    echo "=========================================="
    echo ""
    echo "Now combining outputs and generating CSVs..."

    # Run coordinator to combine outputs
    $VENV_PYTHON 05_merge_results.py \
        --output-dir "$OUTPUT_DIR" \
        --world-size $WORLD_SIZE

    echo ""
    echo "=========================================="
    echo "MERGE COMPLETE!"
    echo "=========================================="
    echo "Results in: $OUTPUT_DIR/*.csv"
else
    echo ""
    echo "=========================================="
    echo "$failed worker(s) failed"
    echo "=========================================="
    echo "Check logs in: $PIPELINE_ROOT/logs/stage5_rank_*.log"
    exit 1
fi
