#!/bin/bash
# =============================================================================
# Multi-GPU Local Embedding Generation
# Runs 8 workers in parallel, then merges the parquet files
# =============================================================================

set -e
set -o pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
CONFIG_FILE="${PIPELINE_CONFIG:-$PIPELINE_ROOT/configs/dolci.yaml}"
VENV_PYTHON="${PIPELINE_VENV:-/lambda/nfs/embeddings/SDTD_Main/.venv/bin/python}"
CONFIG_HELPER="$PIPELINE_ROOT/lib/config_helper.py"
WORLD_SIZE=8

# Get config values
get_config() {
    $VENV_PYTHON "$CONFIG_HELPER" --config "$CONFIG_FILE" --get "$1"
}

OUTPUT_FILE=$(get_config 'embeddings.local.output_file')

# Resolve output path
if [[ ! "$OUTPUT_FILE" = /* ]]; then
    OUTPUT_FILE="$PIPELINE_ROOT/$OUTPUT_FILE"
fi

echo "=========================================="
echo "Multi-GPU Local Embedding Generation"
echo "=========================================="
echo "Config:     $CONFIG_FILE"
echo "Output:     $OUTPUT_FILE"
echo "GPUs:       $WORLD_SIZE"
echo "=========================================="
echo ""

# Kill any existing embedding processes
echo "Cleaning up any existing processes..."
pkill -f "03_create_embeddings" || true
sleep 2

# Create logs directory
mkdir -p "$PIPELINE_ROOT/logs"

# Export config for workers
export PIPELINE_CONFIG="$CONFIG_FILE"

# Launch workers
echo "Launching $WORLD_SIZE workers..."
cd "$PIPELINE_ROOT/stages"

pids=()
for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    echo "  Starting worker on GPU $rank..."
    PIPELINE_CONFIG="$CONFIG_FILE" $VENV_PYTHON 03_create_embeddings_local_multigpu.py \
        --rank $rank \
        --world-size $WORLD_SIZE \
        > "$PIPELINE_ROOT/logs/embed_rank_${rank}.log" 2>&1 &
    pids+=($!)
    sleep 1  # Stagger starts to avoid model loading conflicts
done

echo ""
echo "All workers launched. PIDs: ${pids[*]}"
echo "Monitor progress: tail -f $PIPELINE_ROOT/logs/embed_rank_0.log"
echo ""
echo "Waiting for completion..."

# Wait for all workers
failed=0
for i in "${!pids[@]}"; do
    wait ${pids[$i]} || {
        echo "Worker $i (PID ${pids[$i]}) failed!"
        failed=$((failed + 1))
    }
done

if [ $failed -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "ERROR: $failed worker(s) failed"
    echo "=========================================="
    echo "Check logs: $PIPELINE_ROOT/logs/embed_rank_*.log"
    exit 1
fi

echo ""
echo "=========================================="
echo "All workers completed successfully!"
echo "=========================================="

# Merge parquet files
echo ""
echo "Merging parquet files..."

$VENV_PYTHON << 'MERGE_PYTHON'
import sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

output_file = Path(sys.argv[1])
world_size = int(sys.argv[2])

print(f"Output file: {output_file}")
print(f"World size: {world_size}")

# Find all rank files
rank_files = []
for rank in range(world_size):
    rank_file = output_file.parent / f"{output_file.stem}_rank_{rank}.parquet"
    if rank_file.exists():
        rank_files.append(rank_file)
        print(f"  Found: {rank_file.name}")
    else:
        print(f"  WARNING: Missing {rank_file.name}")

if len(rank_files) == 0:
    print("ERROR: No rank files found!")
    sys.exit(1)

print(f"\nMerging {len(rank_files)} files...")
tables = [pq.read_table(str(f)) for f in rank_files]
merged = pa.concat_tables(tables)

print(f"Writing merged file: {output_file}")
pq.write_table(merged, str(output_file), compression='snappy')

file_size_mb = output_file.stat().st_size / (1024**2)
print(f"\n✓ Merge complete!")
print(f"  Total rows: {len(merged):,}")
print(f"  File size: {file_size_mb:.2f} MB")

# Clean up rank files
print(f"\nCleaning up rank files...")
for rank_file in rank_files:
    rank_file.unlink()
    print(f"  Deleted: {rank_file.name}")

print("\n✓ All done!")
MERGE_PYTHON

$VENV_PYTHON - "$OUTPUT_FILE" "$WORLD_SIZE" << 'MERGE_PYTHON'
import sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

output_file = Path(sys.argv[1])
world_size = int(sys.argv[2])

print(f"Output file: {output_file}")
print(f"World size: {world_size}")

# Find all rank files
rank_files = []
for rank in range(world_size):
    rank_file = output_file.parent / f"{output_file.stem}_rank_{rank}.parquet"
    if rank_file.exists():
        rank_files.append(rank_file)
        print(f"  Found: {rank_file.name}")
    else:
        print(f"  WARNING: Missing {rank_file.name}")

if len(rank_files) == 0:
    print("ERROR: No rank files found!")
    sys.exit(1)

print(f"\nMerging {len(rank_files)} files...")
tables = [pq.read_table(str(f)) for f in rank_files]
merged = pa.concat_tables(tables)

print(f"Writing merged file: {output_file}")
pq.write_table(merged, str(output_file), compression='snappy')

file_size_mb = output_file.stat().st_size / (1024**2)
print(f"\n✓ Merge complete!")
print(f"  Total rows: {len(merged):,}")
print(f"  File size: {file_size_mb:.2f} MB")

# Clean up rank files
print(f"\nCleaning up rank files...")
for rank_file in rank_files:
    rank_file.unlink()
    print(f"  Deleted: {rank_file.name}")

print("\n✓ All done!")
MERGE_PYTHON

echo ""
echo "=========================================="
echo "COMPLETE!"
echo "=========================================="
echo "Output: $OUTPUT_FILE"
