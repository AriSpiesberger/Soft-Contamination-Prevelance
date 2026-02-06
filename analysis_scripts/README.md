# Analysis Scripts

Scripts for analyzing semantic contamination in training corpora. Includes embedding-based similarity search, semantic duplicate classification via Claude API, and distribution comparison.

## Core Scripts

| Script | Purpose |
|--------|---------|
| `benchmark_contamination_analysis.py` | Comprehensive contamination analysis comparing benchmark datasets (MuSR, HumanEval) against S3-hosted training corpus embeddings |
| `resolve_contamination_matches.py` | Resolve contamination match IDs to actual corpus text. Supports parallel file loading and S3 |
| `semantic_duplicate_batch.py` | Identify semantic duplicates using Claude Batch API (50% cost reduction vs streaming) |
| `semantic_duplicate_analysis.py` | Identify semantic duplicates using Claude API (streaming, per-item) |
| `csv_distribution_comparison.py` | Compare original vs regenerated stories using semantic + lexical metrics |
| `dup_compy.py` | Batch similarity calculator for spreadsheet data (semantic + n-gram scores) |

## Comparisons & Benchmarks

Scripts in `comparisons_benchmarks/` for large-scale corpus analysis:

| Script | Purpose |
|--------|---------|
| `fast_run.py` | Fast contamination analysis with checkpointing and async I/O |
| `full_contamination_analysis.py` | Complete contamination analysis pipeline |
| `deduplicate_instruct.py` | Two-stage deduplication: MinHash (Jaccard) + embedding-based (cosine) |
| `compress_jsonl.py` | JSONL compression utility |

## EC2 Resolve

`ec2_resolve/` is a standalone package for running match resolution on EC2 to avoid S3 egress costs. See `ec2_resolve/README.md`.

## Environment Variables

- `ANTHROPIC_API_KEY` - Required for semantic_duplicate_analysis.py and semantic_duplicate_batch.py
- AWS credentials - Required for S3 access in benchmark_contamination_analysis.py
