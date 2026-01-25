#!/bin/bash
# Helper script to evaluate ZebraLogic puzzles with Claude Sonnet 4.5
set -e

# Default values
MODEL="claude-4.5-sonnet"
WORKERS=8
MAX_TOKENS=8192
TEMPERATURE=0.0

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source .env file if it exists
if [ -f "$SCRIPT_DIR/../.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/../.env" | xargs)
    echo "Loaded environment from ../.env"
elif [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
    echo "Loaded environment from .env"
fi

# Usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Evaluate ZebraLogic puzzles with Claude Sonnet 4.5

Options:
    -i INPUT        Input parquet file with index column (required)
    -o OUTPUT       Output parquet file (required)
    -m MODEL        Model to use (default: $MODEL)
    -w WORKERS      Number of parallel workers (default: $WORKERS)
    --start N       Start index (inclusive, default: 0)
    --end N         End index (exclusive, default: all)
    --retry-failed  Only re-run puzzles that were incorrect
    --max-tokens N  Max tokens for response (default: $MAX_TOKENS)
    --temperature T Sampling temperature (default: $TEMPERATURE)
    -h, --help      Show this help message

Examples:
    # Evaluate all puzzles in merged dataset
    $0 -i datasets/zebralogic/sd-with-reasoning/zebralogic-sd-shuffle_and_substitute_and_paraphrase.parquet \\
       -o results/eval-full.parquet

    # Test on first 10 puzzles (index 0-9)
    $0 -i datasets/zebralogic/sd-with-reasoning/zebralogic-sd-shuffle_and_substitute_and_paraphrase.parquet \\
       -o results/eval-test.parquet \\
       --start 0 --end 10

    # Evaluate puzzles 100-199
    $0 -i datasets/zebralogic/sd-with-reasoning/zebralogic-sd-shuffle_and_substitute_and_paraphrase.parquet \\
       -o results/eval-batch-1.parquet \\
       --start 100 --end 200

    # Retry only failed puzzles
    $0 -i datasets/zebralogic/sd-with-reasoning/zebralogic-sd-shuffle_and_substitute_and_paraphrase.parquet \\
       -o results/eval-full.parquet \\
       --retry-failed

EOF
    exit 0
}

# Parse arguments
INPUT=""
OUTPUT=""
START_ARG=""
END_ARG=""
RETRY_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -i)
            INPUT="$2"
            shift 2
            ;;
        -o)
            OUTPUT="$2"
            shift 2
            ;;
        -m)
            MODEL="$2"
            shift 2
            ;;
        -w)
            WORKERS="$2"
            shift 2
            ;;
        --start)
            START_ARG="--start $2"
            shift 2
            ;;
        --end)
            END_ARG="--end $2"
            shift 2
            ;;
        --retry-failed)
            RETRY_ARG="--retry-failed"
            shift
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check required arguments
if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
    echo "Error: -i and -o are required"
    usage
fi

# Check if input file exists
if [ ! -f "$INPUT" ]; then
    echo "Error: Input file not found: $INPUT"
    exit 1
fi

# Create output directory if needed
OUTPUT_DIR="$(dirname "$OUTPUT")"
mkdir -p "$OUTPUT_DIR"

# Run evaluation
echo "=========================================="
echo "ZebraLogic Evaluation"
echo "=========================================="
echo "Input:       $INPUT"
echo "Output:      $OUTPUT"
echo "Model:       $MODEL"
echo "Workers:     $WORKERS"
echo "Max tokens:  $MAX_TOKENS"
echo "Temperature: $TEMPERATURE"
if [ -n "$START_ARG" ]; then
    echo "Start:       ${START_ARG#--start }"
fi
if [ -n "$END_ARG" ]; then
    echo "End:         ${END_ARG#--end }"
fi
if [ -n "$RETRY_ARG" ]; then
    echo "Mode:        Retry failed only"
fi
echo "=========================================="
echo ""

# Use the virtual environment Python
PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Error: Virtual environment not found at $SCRIPT_DIR/.venv"
    echo "Please create it first with: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Run the evaluation script
$PYTHON "$SCRIPT_DIR/evaluate_zebralogic.py" \
    -i "$INPUT" \
    -o "$OUTPUT" \
    -m "$MODEL" \
    -w "$WORKERS" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    $START_ARG \
    $END_ARG \
    $RETRY_ARG

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT"
echo "Log file: ${OUTPUT%.parquet}.log"
echo "=========================================="
