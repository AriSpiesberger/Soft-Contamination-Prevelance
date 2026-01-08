#!/bin/bash
# =============================================================================
# SDTD Pipeline - End-to-End Orchestration Script
# =============================================================================
# Runs the complete contamination analysis pipeline:
#   1. Download Dolma3 data from HuggingFace
#   2. Chunk and sample sentences/paragraphs
#   3. Create embeddings (H100 cluster)
#   4. Run contamination analysis (8x A100 cluster)
#   5. Merge distributed results
#   6. Generate aggregate plots and top-100 lists
#
# Usage:
#   ./run_all.sh                           # Run all stages with default config
#   ./run_all.sh --config custom.yaml      # Use custom config
#   ./run_all.sh --stage 4                 # Start from stage 4 (analysis)
#   ./run_all.sh --dry-run                 # Show what would be run
#
# Environment Variables:
#   PIPELINE_CONFIG    - Path to config file (overrides --config)
#   PIPELINE_ROOT      - Pipeline root directory
#   PIPELINE_VENV      - Path to Python venv
# =============================================================================

set -e  # Exit on error
set -o pipefail

# Get script directory (pipeline root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="${PIPELINE_ROOT:-$SCRIPT_DIR}"

# Default config
CONFIG_FILE="${PIPELINE_CONFIG:-$PIPELINE_ROOT/configs/default.yaml}"
VENV_PYTHON="${PIPELINE_VENV:-/lambda/nfs/embeddings/SDTD_Main/.venv/bin/python}"
CONFIG_HELPER="$PIPELINE_ROOT/lib/config_helper.py"

# Stage control
START_STAGE=1
ONLY_STAGE=""
DRY_RUN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_stage() {
    echo ""
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}Stage $1: $2${NC}"
    echo -e "${GREEN}=========================================${NC}"
}

# Get config value using helper
get_config() {
    $VENV_PYTHON "$CONFIG_HELPER" --config "$CONFIG_FILE" --get "$1"
}

# Check if stage should be skipped
is_stage_skipped() {
    $VENV_PYTHON "$CONFIG_HELPER" --config "$CONFIG_FILE" --check-skip "$1"
    return $?
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --stage)
            START_STAGE="$2"
            shift 2
            ;;
        --only)
            ONLY_STAGE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config FILE    Path to config YAML (default: configs/default.yaml)"
            echo "  --stage N        Start from stage N (1-6)"
            echo "  --only N         Run only stage N"
            echo "  --dry-run        Show what would be run without executing"
            echo "  --help           Show this help"
            echo ""
            echo "Stages:"
            echo "  1 - Download Dolma3 data"
            echo "  2 - Chunk and sample data"
            echo "  3 - Create embeddings"
            echo "  4 - Contamination analysis"
            echo "  5 - Finalize and Merge Results"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Validate Setup
# =============================================================================

if [ ! -f "$CONFIG_FILE" ]; then
    # Try relative to configs dir
    if [ -f "$PIPELINE_ROOT/configs/$CONFIG_FILE" ]; then
        CONFIG_FILE="$PIPELINE_ROOT/configs/$CONFIG_FILE"
    else
        log_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi
fi

if [ ! -f "$VENV_PYTHON" ]; then
    log_error "Python venv not found: $VENV_PYTHON"
    log_info "Please set PIPELINE_VENV environment variable"
    exit 1
fi

# =============================================================================
# Print Configuration
# =============================================================================

echo ""
echo "=========================================="
echo "SDTD Pipeline - Contamination Analysis"
echo "=========================================="
echo "Config:        $CONFIG_FILE"
echo "Pipeline root: $PIPELINE_ROOT"
echo "Python:        $VENV_PYTHON"
echo "Start stage:   $START_STAGE"
if [ -n "$ONLY_STAGE" ]; then
    echo "Only stage:    $ONLY_STAGE"
fi
echo "Dry run:       $DRY_RUN"
echo ""

PIPELINE_NAME=$(get_config "pipeline.name")
echo "Pipeline name: $PIPELINE_NAME"
echo "=========================================="
echo ""

if $DRY_RUN; then
    log_warn "DRY RUN - No commands will be executed"
    echo ""
fi

# =============================================================================
# Stage 1: Download Dolma3 Data
# =============================================================================

run_stage_1() {
    log_stage 1 "Download Dolma3 Data"

    if is_stage_skipped "download"; then
        log_warn "Stage 1 skipped (skip_stages.download = true)"
        return 0
    fi

    local REPO_ID=$(get_config "download.repo_id")
    local SAMPLE_PCT=$(get_config "download.sample_percentage")
    local OUTPUT_DIR=$(get_config "download.output_dir")

    log_info "Repository: $REPO_ID"
    log_info "Sample %: $SAMPLE_PCT"
    log_info "Output: $OUTPUT_DIR"

    if $DRY_RUN; then
        echo "Would run: python stages/01_download_dolmo.py"
        return 0
    fi

    cd "$PIPELINE_ROOT"

    # The download script has hardcoded values - we'll need to modify it
    # For now, just run it directly
    log_info "Starting download..."
    $VENV_PYTHON stages/01_download_dolmo.py

    log_success "Stage 1 complete"
}

# =============================================================================
# Stage 2: Chunk and Sample Data
# =============================================================================

run_stage_2() {
    log_stage 2 "Chunk and Sample Data"

    if is_stage_skipped "chunking"; then
        log_warn "Stage 2 skipped (skip_stages.chunking = true)"
        return 0
    fi

    local INPUT_DIR=$(get_config "chunking.input_dir")
    local OUTPUT_SENTENCES=$(get_config "chunking.output_sentences")
    local OUTPUT_PARAGRAPHS=$(get_config "chunking.output_paragraphs")

    log_info "Input: $INPUT_DIR"
    log_info "Output sentences: $OUTPUT_SENTENCES"
    log_info "Output paragraphs: $OUTPUT_PARAGRAPHS"

    if $DRY_RUN; then
        echo "Would run: python stages/02_chunk_and_sample.py"
        return 0
    fi

    cd "$PIPELINE_ROOT"
    $VENV_PYTHON stages/02_chunk_and_sample.py

    log_success "Stage 2 complete"
}

# =============================================================================
# Stage 3: Create Embeddings (H100 Cluster)
# =============================================================================

run_stage_3() {
    log_stage 3 "Create Embeddings"

    if is_stage_skipped "embeddings"; then
        log_warn "Stage 3 skipped (skip_stages.embeddings = true)"
        return 0
    fi

    # New unified script handles both local and multigpu
    log_info "Running unified embedding script..."

    if $DRY_RUN; then
        echo "Would run: python stages/03_create_embeddings.py"
        return 0
    fi

    cd "$PIPELINE_ROOT"
    
    # Run the script - it now defaults to single GPU if no args provided,
    # or can be controlled via config/args if needed.
    # We pass the config file as env var which the script reads.
    export PIPELINE_CONFIG="$CONFIG_FILE"
    $VENV_PYTHON stages/03_create_embeddings.py

    log_success "Stage 3 complete"
}

# =============================================================================
# Stage 4: Contamination Analysis (8x A100 Cluster)
# =============================================================================

run_stage_4() {
    log_stage 4 "Contamination Analysis"

    if is_stage_skipped "analysis"; then
        log_warn "Stage 4 skipped (skip_stages.analysis = true)"
        return 0
    fi

    local CORPUS_DIR=$(get_config "analysis.corpus_dir")
    local OUTPUT_DIR=$(get_config "analysis.output_dir")
    local WORLD_SIZE=$(get_config "analysis.cluster.world_size")
    local GPU_BATCH=$(get_config "analysis.cluster.gpu_batch_size")
    local CORPUS_CHUNK=$(get_config "analysis.cluster.corpus_gpu_chunk")

    log_info "Corpus: $CORPUS_DIR"
    log_info "Output: $OUTPUT_DIR"
    log_info "World size: $WORLD_SIZE"
    log_info "GPU batch: $GPU_BATCH"
    log_info "Corpus chunk: $CORPUS_CHUNK"

    if $DRY_RUN; then
        echo "Would run: cluster/run_04_analysis.sh"
        return 0
    fi

    cd "$PIPELINE_ROOT"

    # Use the cluster script
    chmod +x cluster/run_04_analysis.sh
    ./cluster/run_04_analysis.sh

    log_success "Stage 4 complete"
}

# =============================================================================
# Stage 5: Merge Results
# =============================================================================

run_stage_5() {
    log_stage 5 "Finalize and Merge Results"

    if is_stage_skipped "merge"; then
        log_warn "Stage 5 skipped (skip_stages.merge = true)"
        return 0
    fi

    # Read finalize config from YAML
    RESULTS_DIR=$(read_yaml "finalize.input_dir")
    CORPUS_JSONL=$(read_yaml "finalize.corpus_file")
    DATASET_NAME=$(read_yaml "pipeline.dataset_short_name")

    # Resolve relative paths
    if [[ ! "$RESULTS_DIR" = /* ]]; then
        RESULTS_DIR="$PIPELINE_ROOT/$RESULTS_DIR"
    fi
    if [[ ! "$CORPUS_JSONL" = /* ]]; then
        CORPUS_JSONL="$PIPELINE_ROOT/$CORPUS_JSONL"
    fi

    log_info "Results dir: $RESULTS_DIR"
    log_info "Corpus file: $CORPUS_JSONL"

    if $DRY_RUN; then
        echo "Would run: python stages/05_finalize_results_fixed.py --results-dir $RESULTS_DIR --corpus-jsonl $CORPUS_JSONL --dataset-name $DATASET_NAME"
        return 0
    fi

    cd "$PIPELINE_ROOT"
    
    # Use the fixed finalizer script with proper arguments
    $VENV_PYTHON stages/05_finalize_results_fixed.py \
        --results-dir "$RESULTS_DIR" \
        --corpus-jsonl "$CORPUS_JSONL" \
        --dataset-name "$DATASET_NAME"

    log_success "Stage 5 complete"
}

# =============================================================================
# Run Pipeline
# =============================================================================

# Create logs directory
mkdir -p "$PIPELINE_ROOT/logs"

# Record start time
START_TIME=$(date +%s)

# Run stages based on arguments
if [ -n "$ONLY_STAGE" ]; then
    # Run only specified stage
    case $ONLY_STAGE in
        1) run_stage_1 ;;
        2) run_stage_2 ;;
        3) run_stage_3 ;;
        4) run_stage_4 ;;
        5) run_stage_5 ;;
        6) run_stage_6 ;;
        *)
            log_error "Invalid stage: $ONLY_STAGE (must be 1-6)"
            exit 1
            ;;
    esac
else
    # Run from start stage to end
    [ $START_STAGE -le 1 ] && run_stage_1
    [ $START_STAGE -le 2 ] && run_stage_2
    [ $START_STAGE -le 3 ] && run_stage_3
    [ $START_STAGE -le 4 ] && run_stage_4
    [ $START_STAGE -le 5 ] && run_stage_5
fi

# Calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=========================================="
log_success "PIPELINE COMPLETE"
echo "=========================================="
echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Results in: $(get_config 'analysis.output_dir')"
echo "=========================================="
