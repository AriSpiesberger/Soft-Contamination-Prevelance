#!/bin/bash
set -e

#######################################################################
# ZebraLogic Batch Evaluation Script
#
# Run evaluations for multiple models or multiple dataset segments.
#######################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/run_zebralogic_eval.sh"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
    cat << EOF
Usage: $0 [MODE] [OPTIONS]

Run batch ZebraLogic evaluations across multiple models or dataset segments.

MODES:
    seen-unseen         Evaluate both seen (0-499) and unseen (500-999) data
    all-shards          Evaluate all 10 shards (0-9) separately
    custom              Run custom configuration defined in script

REQUIRED (for all modes):
    -m, --model MODEL           Model identifier
    -p, --pretty-name NAME      Base name for outputs (will be suffixed)

OPTIONAL:
    -e, --engine ENGINE         Inference engine: openai, anthropic, google, etc. (default: openai)
    -o, --output-folder PATH    Output folder (default: result_dirs/zebra-grid)
    -b, --batch-size SIZE       Batch size (default: 8)
    --dry-run                   Print commands without executing
    -h, --help                  Show this help

EXAMPLES:
    # Run seen vs unseen evaluation
    $0 seen-unseen -m openai/gpt-4o-mini -p gpt4o-mini

    # Run with Anthropic Claude model
    $0 seen-unseen -m claude-3-5-sonnet-20241022 -p claude-sonnet --engine anthropic

    # Evaluate all 10 shards separately
    $0 all-shards -m openai/ft:gpt-4o-mini:org:model:id -p my-model

    # Compare multiple models on seen data (edit script first)
    $0 custom

OUTPUT:
    seen-unseen mode:
        - {name}-seen.json (items 0-499)
        - {name}-unseen.json (items 500-999)

    all-shards mode:
        - {name}-shard-0.json through {name}-shard-9.json

EOF
    exit 1
}

# Check if eval script exists
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo -e "${RED}Error: Evaluation script not found: $EVAL_SCRIPT${NC}"
    exit 1
fi

# Parse mode
MODE="$1"
shift || usage

# Default values
MODEL_NAME=""
MODEL_PRETTY_NAME=""
ENGINE="openai"
OUTPUT_FOLDER="result_dirs/zebra-grid"
BATCH_SIZE=8
DRY_RUN=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -p|--pretty-name)
            MODEL_PRETTY_NAME="$2"
            shift 2
            ;;
        -e|--engine)
            ENGINE="$2"
            shift 2
            ;;
        -o|--output-folder)
            OUTPUT_FOLDER="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Function to run eval
run_eval() {
    local name="$1"
    local args="$2"

    echo -e "${BLUE}=== Running: $name ===${NC}"

    local cmd="$EVAL_SCRIPT -m $MODEL_NAME -p $name -e $ENGINE -o $OUTPUT_FOLDER -b $BATCH_SIZE $args"

    if [ $DRY_RUN -eq 1 ]; then
        echo -e "${YELLOW}Would run:${NC} $cmd"
    else
        $cmd
    fi
    echo ""
}

# Execute based on mode
case $MODE in
    seen-unseen)
        if [ -z "$MODEL_NAME" ] || [ -z "$MODEL_PRETTY_NAME" ]; then
            echo -e "${RED}Error: --model and --pretty-name required for seen-unseen mode${NC}"
            usage
        fi

        echo -e "${GREEN}Running seen-unseen evaluation${NC}"
        echo -e "${GREEN}Model:${NC} $MODEL_NAME"
        echo -e "${GREEN}Base name:${NC} $MODEL_PRETTY_NAME"
        echo ""

        if [ $DRY_RUN -eq 0 ]; then
            echo -e "${BLUE}Running 2 evaluations...${NC}"
        fi

        run_eval "${MODEL_PRETTY_NAME}-seen" "--seen"
        run_eval "${MODEL_PRETTY_NAME}-unseen" "--unseen"

        echo -e "${GREEN}=== Seen-Unseen Evaluation Complete ===${NC}"
        echo -e "Seen data:   $OUTPUT_FOLDER/${MODEL_PRETTY_NAME}-seen.0-500.json"
        echo -e "Unseen data: $OUTPUT_FOLDER/${MODEL_PRETTY_NAME}-unseen.500-1000.json"
        ;;

    all-shards)
        if [ -z "$MODEL_NAME" ] || [ -z "$MODEL_PRETTY_NAME" ]; then
            echo -e "${RED}Error: --model and --pretty-name required for all-shards mode${NC}"
            usage
        fi

        echo -e "${GREEN}Running all-shards evaluation (10 shards)${NC}"
        echo -e "${GREEN}Model:${NC} $MODEL_NAME"
        echo -e "${GREEN}Base name:${NC} $MODEL_PRETTY_NAME"
        echo ""

        if [ $DRY_RUN -eq 0 ]; then
            echo -e "${BLUE}Running 10 evaluations...${NC}"
        fi

        for shard in {0..9}; do
            run_eval "${MODEL_PRETTY_NAME}-shard-${shard}" "-s $shard"
        done

        echo -e "${GREEN}=== All-Shards Evaluation Complete ===${NC}"
        for shard in {0..9}; do
            start=$((shard * 100))
            end=$(((shard + 1) * 100))
            echo -e "Shard $shard: $OUTPUT_FOLDER/${MODEL_PRETTY_NAME}-shard-${shard}.${start}-${end}.json"
        done
        ;;

    custom)
        echo -e "${BLUE}=== Custom Batch Evaluation ===${NC}"
        echo ""
        echo -e "${YELLOW}Edit this section of the script to define your custom evaluations${NC}"
        echo ""

        # ============================================================
        # CUSTOM CONFIGURATION - Edit this section
        # ============================================================

        # Example 1: Compare multiple models on seen data
        # MODELS=(
        #     "openai/gpt-4o-mini:baseline-model"
        #     "openai/ft:gpt-4o-mini:org:v1:id1:finetuned-v1"
        #     "openai/ft:gpt-4o-mini:org:v2:id2:finetuned-v2"
        # )
        #
        # for model_spec in "${MODELS[@]}"; do
        #     IFS=':' read -r model name <<< "$model_spec"
        #     run_eval "$name" "-m $model --seen"
        # done

        # Example 2: Run same model on all shards
        # MODEL="openai/gpt-4o-mini"
        # for shard in {0..9}; do
        #     run_eval "gpt4o-mini-shard-$shard" "-m $MODEL -s $shard"
        # done

        # Example 3: Custom ranges for specific research questions
        # run_eval "model-first-quartile" "-m $MODEL --start 0 --end 250"
        # run_eval "model-second-quartile" "-m $MODEL --start 250 --end 500"
        # run_eval "model-third-quartile" "-m $MODEL --start 500 --end 750"
        # run_eval "model-fourth-quartile" "-m $MODEL --start 750 --end 1000"

        echo -e "${RED}No custom configuration defined. Edit the script to add your evaluations.${NC}"
        exit 1
        ;;

    *)
        echo -e "${RED}Error: Unknown mode: $MODE${NC}"
        usage
        ;;
esac
