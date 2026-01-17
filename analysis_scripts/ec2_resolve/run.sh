#!/bin/bash
# Resolve contamination matches on EC2
#
# Upload this folder to EC2:
#   scp -r ec2_resolve ubuntu@<ec2-ip>:~/
#
# Then SSH in and run:
#   cd ~/ec2_resolve
#   chmod +x run.sh
#   ./run.sh

set -e

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo ""
echo "=== Running resolver ==="
python resolve_matches.py \
    --bucket dolmo-3-sampling \
    --matches-prefix contamination_analysis_fast \
    --corpus-prefix embeddings/output \
    --output-dir resolved_matches \
    --benchmarks musr mbpp \
    --modes input output \
    --max-workers 16 \
    --cache-dir /tmp/corpus_cache

echo ""
echo "=== Done! ==="
echo "Results in: resolved_matches/"
ls -lh resolved_matches/
