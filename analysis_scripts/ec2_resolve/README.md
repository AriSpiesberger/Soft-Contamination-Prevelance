# EC2 Contamination Match Resolver

Minimal package to resolve contamination match IDs to actual corpus text.

## Files

- `resolve_matches.py` - Main script
- `requirements.txt` - Python dependencies
- `run.sh` - One-liner to run everything

## Quick Start

```bash
# 1. Upload to EC2
scp -r ec2_resolve ubuntu@<ec2-ip>:~/

# 2. SSH in
ssh ubuntu@<ec2-ip>

# 3. Run
cd ~/ec2_resolve
chmod +x run.sh
./run.sh
```

## Custom Options

```bash
python resolve_matches.py \
    --bucket dolmo-3-sampling \
    --matches-prefix contamination_analysis_fast \
    --corpus-prefix embeddings/output \
    --output-dir resolved_matches \
    --benchmarks musr mbpp \
    --modes input output \
    --max-workers 16
```

## Output

- `resolved_matches/musr_input_resolved.csv` - CSV with benchmark + corpus text
- `resolved_matches/musr_input_resolved.json` - Full JSON with all details
- Same for `musr_output`, `mbpp_input`, `mbpp_output`

## Notes

- Adjust `--corpus-prefix` if your parquet files are in a different S3 path
- Uses `/tmp/corpus_cache` to cache downloaded parquet files
- Increase `--max-workers` for faster processing (default 16)
