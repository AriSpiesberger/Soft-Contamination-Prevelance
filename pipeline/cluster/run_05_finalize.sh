#!/bin/bash
# =============================================================================
# Pipeline Stage 5: Finalize Results
# - Add corpus texts to top-100 JSONs
# - Generate aggregate plots and CSVs
# - Clean up temporary files
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

# Get paths from config
OUTPUT_DIR="${OUTPUT_DIR:-$(get_config 'analysis.output_dir')}"
CORPUS_FILE="${CORPUS_FILE:-$(get_config 'chunking.output_paragraphs')}"
DATASET_NAME="${DATASET_NAME:-$(get_config 'aggregates.dataset_name')}"

# If OUTPUT_DIR is relative, make it absolute from PIPELINE_ROOT
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$PIPELINE_ROOT/$OUTPUT_DIR"
fi

# If CORPUS_FILE is relative, make it absolute from PIPELINE_ROOT
if [[ "$CORPUS_FILE" != /* ]]; then
    CORPUS_FILE="$PIPELINE_ROOT/$CORPUS_FILE"
fi

cd "$PIPELINE_ROOT/stages"

echo "=========================================="
echo "Pipeline Stage 5: Finalize Results"
echo "=========================================="
echo "Output dir:   $OUTPUT_DIR"
echo "Corpus file:  $CORPUS_FILE"
echo "Dataset name: $DATASET_NAME"
echo ""

# Check if corpus file exists
if [ ! -f "$CORPUS_FILE" ] && [ ! -d "$CORPUS_FILE" ]; then
    echo "❌ Error: Corpus file not found: $CORPUS_FILE"
    exit 1
fi

# Check if output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ Error: Output directory not found: $OUTPUT_DIR"
    exit 1
fi

# Run Stage 5
$VENV_PYTHON 05_finalize_results.py \
    --results-dir "$OUTPUT_DIR" \
    --corpus "$CORPUS_FILE" \
    --dataset-name "$DATASET_NAME"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ STAGE 5 COMPLETE!"
    echo "=========================================="
    echo "Results finalized in: $OUTPUT_DIR"
else
    echo ""
    echo "=========================================="
    echo "❌ Stage 5 failed with exit code $EXIT_CODE"
    echo "=========================================="
    exit $EXIT_CODE
fi
