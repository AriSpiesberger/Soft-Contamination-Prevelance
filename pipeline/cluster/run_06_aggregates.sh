#!/bin/bash
# =============================================================================
# Pipeline Stage 6: Generate Aggregates
# FULL DATA (NO SAMPLING!) - Aggregate plots and CSVs
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

OUTPUT_DIR="${AGG_OUTPUT_DIR:-$PIPELINE_ROOT/results/contamination}"
WORLD_SIZE="${AGG_WORLD_SIZE:-$(get_config 'aggregates.world_size')}"
SOURCE_WORLD_SIZE="${AGG_SOURCE_WORLD_SIZE:-$(get_config 'aggregates.source_world_size')}"
DATASET_NAME="${DATASET_NAME:-$(get_config 'aggregates.dataset_name')}"
BENCHMARKS_JSON="${BENCHMARKS_JSON:-$(get_config 'analysis.benchmarks')}"

cd "$PIPELINE_ROOT/stages"

echo "=========================================="
echo "Pipeline Stage 6: Generate Aggregates"
echo "=========================================="
echo "Dataset:           $DATASET_NAME"
echo "Output dir:        $OUTPUT_DIR"
echo "Workers:           $WORLD_SIZE"
echo "Source world size: $SOURCE_WORLD_SIZE"
echo ""
echo "Using ALL similarities (NO SAMPLING!)"
echo "Will generate per benchmark:"
echo "  - {Benchmark}__{Dataset}__histogram_linear.png"
echo "  - {Benchmark}__{Dataset}__histogram_log.png"
echo "  - {Benchmark}__{Dataset}__cdf.png"
echo "  - {Benchmark}__{Dataset}__topk.png"
echo "  - {Benchmark}__{Dataset}__top100.csv"
echo ""

# Create logs directory
mkdir -p "$PIPELINE_ROOT/logs"

# Clean up
rm -f "$OUTPUT_DIR/agg_rank_*.json"

# Launch workers
echo "Launching $WORLD_SIZE workers..."
pids=()

for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    $VENV_PYTHON 06_generate_aggregates.py \
        --output-dir "$OUTPUT_DIR" \
        --source-world-size $SOURCE_WORLD_SIZE \
        --world-size $WORLD_SIZE \
        --rank $rank \
        --dataset-name "$DATASET_NAME" \
        --benchmarks "$BENCHMARKS_JSON" \
        2>&1 | tee "$PIPELINE_ROOT/logs/stage6_rank_${rank}.log" &
    pids+=($!)
    sleep 0.5
done

echo "All workers launched. PIDs: ${pids[*]}"
echo ""
echo "Waiting for workers..."

# Wait
failed=0
for i in "${!pids[@]}"; do
    wait ${pids[$i]} || failed=$((failed + 1))
done

if [ $failed -eq 0 ]; then
    echo ""
    echo "All workers done!"
    echo ""
    echo "Generating aggregates..."

    # Run coordinator
    $VENV_PYTHON 06_generate_aggregates.py \
        --output-dir "$OUTPUT_DIR" \
        --source-world-size $SOURCE_WORLD_SIZE \
        --world-size $WORLD_SIZE \
        --dataset-name "$DATASET_NAME" \
        --benchmarks "$BENCHMARKS_JSON"

    echo ""
    echo "=========================================="
    echo "COMPLETE!"
    echo "=========================================="
    echo "Check: $OUTPUT_DIR/{musr,mbpp}_*/"
else
    echo ""
    echo "$failed worker(s) failed"
    echo "Check logs in: $PIPELINE_ROOT/logs/stage6_rank_*.log"
    exit 1
fi
