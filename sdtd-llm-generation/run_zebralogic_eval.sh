#!/bin/bash
set -e

#######################################################################
# ZebraLogic Evaluation Script
#
# Run OpenAI model evaluation on ZebraLogic dataset with flexible
# control over which dataset segment to evaluate.
#######################################################################

# Default values
MODEL_NAME=""
MODEL_PRETTY_NAME=""
START_INDEX=-1
END_INDEX=-1
SHARD_NUM=-1
BATCH_SIZE=8
TEMPERATURE=0
TOP_P=1.0
MAX_TOKENS=4096
OUTPUT_FOLDER="result_dirs/zebra-grid"
ENGINE="openai"
ENV_FILE="../.env"
DRY_RUN=0

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZEROEVAL_DIR="${SCRIPT_DIR}/ZeroEval"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run ZebraLogic evaluation on OpenAI models with flexible dataset segment control.

REQUIRED:
    -m, --model MODEL_NAME          OpenAI model identifier (e.g., openai/gpt-4o-mini)
    -p, --pretty-name NAME          Pretty name for output files

DATASET SELECTION (choose one):
    -s, --shard NUM                 Evaluate specific parquet shard (0-9)
                                    Shard 0 = items 0-99, Shard 1 = 100-199, etc.
    --start START --end END         Custom index range [START, END)
    --all                           Evaluate all 1000 samples (0-1000)
    --seen                          Evaluate "seen" data: shards 0-4 (items 0-499)
    --unseen                        Evaluate "unseen" data: shards 5-9 (items 500-999)

OPTIONAL:
    -o, --output-folder PATH        Output folder (default: result_dirs/zebra-grid)
    -b, --batch-size SIZE           Batch size for API calls (default: 8)
    -t, --temperature TEMP          Sampling temperature (default: 0)
    --top-p VALUE                   Top-p sampling (default: 1.0)
    --max-tokens NUM                Max output tokens (default: 4096)
    -e, --engine ENGINE             Inference engine (default: openai)
    --env-file PATH                 Path to .env file (default: ../.env)
    --dry-run                       Print command without executing
    -h, --help                      Show this help message

EXAMPLES:
    # Evaluate gpt-4o-mini on shard 0 (items 0-99)
    $0 -m openai/gpt-4o-mini -p gpt4o-mini-test -s 0

    # Evaluate fine-tuned model on shard 5 (items 500-599)
    $0 -m openai/ft:gpt-4o-mini:org:model:id -p my-model -s 5

    # Evaluate on "seen" data (items 0-499)
    $0 -m openai/gpt-4o-mini -p gpt4o-mini-seen --seen

    # Evaluate on "unseen" data (items 500-999)
    $0 -m openai/gpt-4o-mini -p gpt4o-mini-unseen --unseen

    # Evaluate on custom range (items 100-300)
    $0 -m openai/gpt-4o-mini -p gpt4o-mini-custom --start 100 --end 300

    # Evaluate entire dataset (all 1000 samples)
    $0 -m openai/gpt-4o-mini -p gpt4o-mini-full --all

SHARD MAPPING:
    Shard 0:  items 0-99     (--start 0 --end 100)
    Shard 1:  items 100-199  (--start 100 --end 200)
    Shard 2:  items 200-299  (--start 200 --end 300)
    Shard 3:  items 300-399  (--start 300 --end 400)
    Shard 4:  items 400-499  (--start 400 --end 500)
    Shard 5:  items 500-599  (--start 500 --end 600)
    Shard 6:  items 600-699  (--start 600 --end 700)
    Shard 7:  items 700-799  (--start 700 --end 800)
    Shard 8:  items 800-899  (--start 800 --end 900)
    Shard 9:  items 900-999  (--start 900 --end 1000)

EOF
    exit 1
}

# Parse command line arguments
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
        -s|--shard)
            SHARD_NUM="$2"
            shift 2
            ;;
        --start)
            START_INDEX="$2"
            shift 2
            ;;
        --end)
            END_INDEX="$2"
            shift 2
            ;;
        --all)
            START_INDEX=0
            END_INDEX=1000
            shift
            ;;
        --seen)
            START_INDEX=0
            END_INDEX=500
            shift
            ;;
        --unseen)
            START_INDEX=500
            END_INDEX=1000
            shift
            ;;
        -o|--output-folder)
            OUTPUT_FOLDER="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -t|--temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        -e|--engine)
            ENGINE="$2"
            shift 2
            ;;
        --env-file)
            ENV_FILE="$2"
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

# Validate required parameters
if [ -z "$MODEL_NAME" ]; then
    echo -e "${RED}Error: Model name (-m) is required${NC}"
    usage
fi

if [ -z "$MODEL_PRETTY_NAME" ]; then
    echo -e "${RED}Error: Pretty name (-p) is required${NC}"
    usage
fi

# Convert shard number to indices if provided
if [ $SHARD_NUM -ge 0 ]; then
    if [ $START_INDEX -ge 0 ] || [ $END_INDEX -ge 0 ]; then
        echo -e "${RED}Error: Cannot use both --shard and --start/--end${NC}"
        exit 1
    fi
    if [ $SHARD_NUM -lt 0 ] || [ $SHARD_NUM -gt 9 ]; then
        echo -e "${RED}Error: Shard number must be between 0 and 9${NC}"
        exit 1
    fi
    START_INDEX=$((SHARD_NUM * 100))
    END_INDEX=$(((SHARD_NUM + 1) * 100))
fi

# Validate that indices are set
if [ $START_INDEX -lt 0 ] || [ $END_INDEX -lt 0 ]; then
    echo -e "${RED}Error: Must specify dataset range using --shard, --start/--end, --all, --seen, or --unseen${NC}"
    usage
fi

# Validate indices
if [ $START_INDEX -lt 0 ] || [ $START_INDEX -ge 1000 ]; then
    echo -e "${RED}Error: Start index must be between 0 and 999${NC}"
    exit 1
fi
if [ $END_INDEX -le $START_INDEX ] || [ $END_INDEX -gt 1000 ]; then
    echo -e "${RED}Error: End index must be greater than start index and at most 1000${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}Error: Environment file not found: $ENV_FILE${NC}"
    exit 1
fi

# Check if OPENAI_API_KEY is in .env
if ! grep -q "OPENAI_API_KEY" "$ENV_FILE"; then
    echo -e "${RED}Error: OPENAI_API_KEY not found in $ENV_FILE${NC}"
    exit 1
fi

# Print configuration
echo -e "${BLUE}=== ZebraLogic Evaluation Configuration ===${NC}"
echo -e "${GREEN}Model:${NC}           $MODEL_NAME"
echo -e "${GREEN}Pretty Name:${NC}     $MODEL_PRETTY_NAME"
echo -e "${GREEN}Dataset Range:${NC}   [$START_INDEX, $END_INDEX) ($(($END_INDEX - $START_INDEX)) samples)"
if [ $SHARD_NUM -ge 0 ]; then
    echo -e "${GREEN}Shard:${NC}           $SHARD_NUM"
fi
echo -e "${GREEN}Output Folder:${NC}   $OUTPUT_FOLDER"
echo -e "${GREEN}Batch Size:${NC}      $BATCH_SIZE"
echo -e "${GREEN}Temperature:${NC}     $TEMPERATURE"
if [ "$ENGINE" != "anthropic" ]; then
    echo -e "${GREEN}Top-P:${NC}           $TOP_P"
else
    echo -e "${GREEN}Top-P:${NC}           (skipped for Anthropic)"
fi
echo -e "${GREEN}Max Tokens:${NC}      $MAX_TOKENS"
echo -e "${GREEN}Engine:${NC}          $ENGINE"
echo -e "${BLUE}===========================================${NC}"
echo ""

# Estimate time
SAMPLES_COUNT=$(($END_INDEX - $START_INDEX))
EST_TIME_MIN=$((SAMPLES_COUNT * 15 / 60))  # ~15 seconds per sample
EST_TIME_MAX=$((SAMPLES_COUNT * 25 / 60))  # ~25 seconds per sample
echo -e "${YELLOW}Estimated time: $EST_TIME_MIN-$EST_TIME_MAX minutes${NC}"
echo ""

# Build the command
CMD="bash -c 'export \$(cat $ENV_FILE | grep -v \"^#\" | xargs) && uv run src/unified_infer.py"
CMD="$CMD --engine $ENGINE"
CMD="$CMD --model_name $MODEL_NAME"
CMD="$CMD --model_pretty_name $MODEL_PRETTY_NAME"
CMD="$CMD --data_name zebra-grid"
CMD="$CMD --start_index $START_INDEX"
CMD="$CMD --end_index $END_INDEX"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --temperature $TEMPERATURE"
# Anthropic doesn't allow both temperature and top_p
if [ "$ENGINE" != "anthropic" ]; then
    CMD="$CMD --top_p $TOP_P"
fi
CMD="$CMD --max_tokens $MAX_TOKENS"
CMD="$CMD --output_folder $OUTPUT_FOLDER/"
CMD="$CMD'"

if [ $DRY_RUN -eq 1 ]; then
    echo -e "${YELLOW}Dry run - command that would be executed:${NC}"
    echo "$CMD"
    exit 0
fi

# Run the command
echo -e "${GREEN}Starting evaluation...${NC}"
echo ""

# Change to ZeroEval directory
cd "$ZEROEVAL_DIR"
eval "$CMD"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=== Evaluation Complete ===${NC}"

    # Find output file
    if [ $START_INDEX -eq 0 ] && [ $END_INDEX -eq 1000 ]; then
        OUTPUT_FILE="$OUTPUT_FOLDER/$MODEL_PRETTY_NAME.json"
    else
        OUTPUT_FILE="$OUTPUT_FOLDER/$MODEL_PRETTY_NAME.$START_INDEX-$END_INDEX.json"
    fi

    if [ -f "$OUTPUT_FILE" ]; then
        echo -e "${GREEN}Output saved to:${NC} $OUTPUT_FILE"

        # Show file size
        FILE_SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
        echo -e "${GREEN}File size:${NC} $FILE_SIZE"

        # Count items in output
        ITEM_COUNT=$(cat "$OUTPUT_FILE" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "?")
        echo -e "${GREEN}Items processed:${NC} $ITEM_COUNT"
    fi

    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Run evaluation: cd ZeroEval && python src/evaluation/zebra_grid_eval.py"
    echo "  2. View results: cat result_dirs/zebra-grid.summary.md"
else
    echo ""
    echo -e "${RED}=== Evaluation Failed ===${NC}"
    exit 1
fi
