#!/bin/bash
set -e

#######################################################################
# Parallel ZebraLogic Evaluation Script
#
# Runs multiple evaluation processes in parallel for 8x+ speedup
#######################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run parallel ZebraLogic evaluation for dramatic speedup (5 hours → 40 minutes).

REQUIRED:
    -m, --model MODEL           Model identifier
    -p, --pretty-name NAME      Pretty name for output

DATASET SELECTION (choose one):
    --seen                      Evaluate "seen" data (0-499)
    --unseen                    Evaluate "unseen" data (500-999)
    --start START --end END     Custom range
    --all                       All 1000 samples

OPTIONAL:
    -e, --engine ENGINE         Engine to use: openai, anthropic, google, etc. (default: openai)
    -n, --num-parallel NUM      Number of parallel processes (default: 8)
    -o, --output-folder PATH    Output folder (default: result_dirs/zebra-grid)
    -t, --temperature TEMP      Temperature (default: 0)
    --env-file PATH             Path to .env file (default: .env)
    --dry-run                   Show what would run without executing
    -h, --help                  Show this help

EXAMPLES:
    # Run 8 parallel processes on seen data (500 samples)
    $0 -m openai/gpt-4o-mini -p gpt4o-mini-seen --seen

    # Run with Anthropic Claude model
    $0 -m claude-3-5-sonnet-20241022 -p claude-sonnet-3.5 --engine anthropic --seen

    # Run 16 parallel processes for faster execution
    $0 -m openai/gpt-4o-mini -p gpt4o-mini-seen --seen -n 16

    # Custom range with 4 parallel processes
    $0 -m openai/gpt-4o-mini -p test --start 0 --end 200 -n 4

PERFORMANCE:
    Serial (batch_size=8):     ~5 hours for 500 samples
    4 parallel processes:      ~1.25 hours (4x faster)
    8 parallel processes:      ~40 minutes (8x faster)
    16 parallel processes:     ~20 minutes (16x faster)

EOF
    exit 1
}

# Defaults
MODEL_NAME=""
MODEL_PRETTY_NAME=""
START_INDEX=-1
END_INDEX=-1
ENGINE="openai"
NUM_PARALLEL=8
OUTPUT_FOLDER="result_dirs/zebra-grid"
TEMPERATURE=0
ENV_FILE=".env"
DRY_RUN=0

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZEROEVAL_DIR="${SCRIPT_DIR}/ZeroEval"

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
        --start)
            START_INDEX="$2"
            shift 2
            ;;
        --end)
            END_INDEX="$2"
            shift 2
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
        --all)
            START_INDEX=0
            END_INDEX=1000
            shift
            ;;
        -n|--num-parallel)
            NUM_PARALLEL="$2"
            shift 2
            ;;
        -o|--output-folder)
            OUTPUT_FOLDER="$2"
            shift 2
            ;;
        -t|--temperature)
            TEMPERATURE="$2"
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

# Validate
if [ -z "$MODEL_NAME" ] || [ -z "$MODEL_PRETTY_NAME" ]; then
    echo -e "${RED}Error: --model and --pretty-name are required${NC}"
    usage
fi

if [ $START_INDEX -lt 0 ] || [ $END_INDEX -lt 0 ]; then
    echo -e "${RED}Error: Must specify dataset range${NC}"
    usage
fi

if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}Error: Environment file not found: $ENV_FILE${NC}"
    exit 1
fi

# Calculate sharding
TOTAL_SAMPLES=$((END_INDEX - START_INDEX))
SHARD_SIZE=$((TOTAL_SAMPLES / NUM_PARALLEL))

if [ $SHARD_SIZE -lt 1 ]; then
    echo -e "${RED}Error: Too many parallel processes for sample count${NC}"
    echo "Samples: $TOTAL_SAMPLES, Parallel: $NUM_PARALLEL"
    echo "Reduce --num-parallel or increase sample range"
    exit 1
fi

# Print configuration
echo -e "${BLUE}=== Parallel ZebraLogic Evaluation ===${NC}"
echo -e "${GREEN}Engine:${NC}          $ENGINE"
echo -e "${GREEN}Model:${NC}           $MODEL_NAME"
echo -e "${GREEN}Pretty Name:${NC}     $MODEL_PRETTY_NAME"
echo -e "${GREEN}Dataset Range:${NC}   [$START_INDEX, $END_INDEX) ($TOTAL_SAMPLES samples)"
echo -e "${GREEN}Parallel Procs:${NC}  $NUM_PARALLEL"
echo -e "${GREEN}Per Process:${NC}     ~$SHARD_SIZE samples"
echo -e "${GREEN}Temperature:${NC}     $TEMPERATURE"
if [ "$ENGINE" != "anthropic" ]; then
    echo -e "${GREEN}Top-P:${NC}           1.0"
else
    echo -e "${GREEN}Top-P:${NC}           (skipped for Anthropic)"
fi
echo -e "${BLUE}========================================${NC}"
echo ""

# Estimate time
SERIAL_TIME=$((TOTAL_SAMPLES * 20 / 60))  # ~20 sec per sample
PARALLEL_TIME=$((SERIAL_TIME / NUM_PARALLEL))
echo -e "${YELLOW}Estimated time:${NC}"
echo -e "  Serial: ~$SERIAL_TIME minutes"
echo -e "  Parallel ($NUM_PARALLEL procs): ~$PARALLEL_TIME minutes"
echo -e "  Speedup: ${NUM_PARALLEL}x faster!"
echo ""

if [ $DRY_RUN -eq 1 ]; then
    echo -e "${YELLOW}Dry run - would launch $NUM_PARALLEL processes:${NC}"
    for i in $(seq 0 $((NUM_PARALLEL - 1))); do
        SHARD_START=$((START_INDEX + i * SHARD_SIZE))
        SHARD_END=$((START_INDEX + (i + 1) * SHARD_SIZE))
        if [ $i -eq $((NUM_PARALLEL - 1)) ]; then
            SHARD_END=$END_INDEX
        fi
        echo "  Process $i: samples $SHARD_START to $SHARD_END"
    done
    exit 0
fi

# Create temp directory
TMP_DIR="$OUTPUT_FOLDER/tmp_${MODEL_PRETTY_NAME}"
mkdir -p "$TMP_DIR"

echo -e "${GREEN}Launching $NUM_PARALLEL parallel processes...${NC}"
echo ""

# Launch parallel processes
cd "$ZEROEVAL_DIR"

for i in $(seq 0 $((NUM_PARALLEL - 1))); do
    SHARD_START=$((START_INDEX + i * SHARD_SIZE))
    SHARD_END=$((START_INDEX + (i + 1) * SHARD_SIZE))

    # Last shard gets remainder
    if [ $i -eq $((NUM_PARALLEL - 1)) ]; then
        SHARD_END=$END_INDEX
    fi

    SHARD_SAMPLES=$((SHARD_END - SHARD_START))
    echo -e "${BLUE}[Process $i]${NC} Samples $SHARD_START-$SHARD_END ($SHARD_SAMPLES samples)"

    # Build command with conditional top_p (Anthropic doesn't allow both temperature and top_p)
    TOP_P_ARG=""
    if [ "$ENGINE" != "anthropic" ]; then
        TOP_P_ARG="--top_p 1.0"
    fi

    # Run in background
    bash -c "
        export \$(cat ../$ENV_FILE | grep -v '^#' | xargs)
        uv run src/unified_infer.py \
            --engine '$ENGINE' \
            --model_name '$MODEL_NAME' \
            --model_pretty_name '${MODEL_PRETTY_NAME}' \
            --data_name zebra-grid \
            --start_index $SHARD_START \
            --end_index $SHARD_END \
            --batch_size 8 \
            --temperature $TEMPERATURE \
            $TOP_P_ARG \
            --max_tokens 4096 \
            --output_folder '$TMP_DIR/' \
            2>&1 | sed 's/^/[P$i] /'
    " &
done

echo ""
echo -e "${YELLOW}All processes launched! Waiting for completion...${NC}"
echo -e "${YELLOW}Monitor progress in another terminal:${NC}"
echo -e "  watch -n 5 'ls -lh $TMP_DIR/*.json 2>/dev/null | wc -l'"
echo ""

# Wait for all background processes
wait

echo ""
echo -e "${GREEN}=== All processes complete! ===${NC}"

# Check results
SHARD_FILES=$(ls $TMP_DIR/*.json 2>/dev/null | wc -l)
echo -e "${GREEN}Shard files created: $SHARD_FILES${NC}"

if [ $SHARD_FILES -ne $NUM_PARALLEL ]; then
    echo -e "${RED}Warning: Expected $NUM_PARALLEL files, found $SHARD_FILES${NC}"
    echo "Some shards may have failed. Check the output above."
fi

# Merge results
echo ""
echo -e "${BLUE}Merging results (range $START_INDEX-$END_INDEX only)...${NC}"

# Use Python to selectively merge only shards within the current range (with deduplication)
python3 << EOF
import json
import os
import glob

tmp_dir = "$TMP_DIR"
prefix = "$MODEL_PRETTY_NAME"
start_index = $START_INDEX
end_index = $END_INDEX

# Collect items with deduplication by session_id
items_by_id = {}
shard_count = 0

for file in os.listdir(tmp_dir):
    if file.startswith(prefix + ".") and file.endswith(".json"):
        parts = file.replace(".json", "").split(".")
        if len(parts) >= 2:
            range_part = parts[-1]
            if "-" in range_part:
                try:
                    shard_start, shard_end = map(int, range_part.split("-"))
                    # Only include shards that fall within our range
                    if shard_start >= start_index and shard_end <= end_index:
                        shard_count += 1
                        file_path = os.path.join(tmp_dir, file)
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            for item in data:
                                session_id = item.get("session_id")
                                if session_id:
                                    items_by_id[session_id] = item
                except ValueError:
                    pass

print(f"Found {shard_count} shards within range [{start_index}, {end_index})")

# Sort by session_id for consistent ordering
merged_data = sorted(items_by_id.values(), key=lambda x: x.get("session_id", ""))

print(f"Total unique items: {len(merged_data)}")

# Save to final location
output_file = "$OUTPUT_FOLDER/${MODEL_PRETTY_NAME}.${START_INDEX}-${END_INDEX}.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(merged_data, f, indent=2)

print(f"Saved to: {output_file}")
EOF

FINAL_FILE="$OUTPUT_FOLDER/${MODEL_PRETTY_NAME}.${START_INDEX}-${END_INDEX}.json"

echo -e "${GREEN}✅ Merged results saved to: $FINAL_FILE${NC}"

# Show stats
if [ -f "$FINAL_FILE" ]; then
    FILE_SIZE=$(ls -lh "$FINAL_FILE" | awk '{print $5}')
    ITEM_COUNT=$(python3 -c "import json; print(len(json.load(open('$FINAL_FILE'))))" 2>/dev/null || echo "?")

    echo ""
    echo -e "${BLUE}=== Results Summary ===${NC}"
    echo -e "${GREEN}File:${NC}   $FINAL_FILE"
    echo -e "${GREEN}Size:${NC}   $FILE_SIZE"
    echo -e "${GREEN}Items:${NC}  $ITEM_COUNT / $TOTAL_SAMPLES"

    # Show temp directory location
    echo ""
    echo -e "${BLUE}Temporary shard files kept in: $TMP_DIR${NC}"
fi

echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Run evaluation: cd ZeroEval && python src/evaluation/zebra_grid_eval.py"
echo "  2. View results: cat result_dirs/zebra-grid.summary.md"
