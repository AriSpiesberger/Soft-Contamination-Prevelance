#!/usr/bin/env python3
"""
Semantic Duplicate Analysis using Claude Batch API

Analyzes potential semantic duplicates between test tasks and corpus tasks.
Uses the Anthropic Batch API for 50% cost reduction.

Workflow:
1. Split input CSV into chunks of 100 rows
2. Create JSONL batch request file
3. Submit batch to API
4. Poll for completion
5. Parse results into output CSV
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime
import anthropic


SYSTEM_PROMPT = """You are an expert at analyzing coding problems and determining if two problem descriptions are semantic duplicates - i.e., the same task just phrased differently.

Rules for determining semantic duplicates:
1. Focus ONLY on the task description, ignore any code solutions provided
2. Direct mathematical equivalence counts as a semantic duplicate (e.g., "sum all numbers 1..n" and "sum all numbers n, n-1, n-2, ..., 0" are equivalent)
3. If the corpus_text task is strictly stronger (asks for additional output, or where test is a test "does X have property?" and corpus finds the solution "find X with property"), count it as a duplicate since it can trivially reduce to the test task
4. This is asymmetric - a stronger corpus task that subsumes the test task counts, but not vice versa

Be calibrated with confidence scores - use lower confidence for ambiguous cases, tricky phrasing, or when you don't fully understand a task."""


USER_PROMPT_TEMPLATE = """Analyze these coding task pairs for semantic duplicates. Input CSV:

test_id,corpus_index,test_text,corpus_text
{csv_content}

For each row, determine:
- is_sd: 1 if semantic duplicate, 0 if not
- confidence: 0.0 to 1.0 (calibrated confidence)

Output ONLY a CSV with columns: test_id,corpus_index,is_sd,confidence
No headers, no explanation, just the data rows."""


def prepare_batch_requests(
    input_csv: str,
    output_jsonl: str,
    batch_size: int = 100,
    score_threshold: Optional[float] = None,
    max_rows: Optional[int] = None,
    model: str = "claude-sonnet-4-20250514"
):
    """
    Prepare batch request JSONL file from input CSV.

    Args:
        input_csv: Path to contamination results CSV
        output_jsonl: Path to write JSONL batch requests
        batch_size: Rows per API request
        score_threshold: Only include rows with score >= threshold
        max_rows: Limit total rows
        model: Model to use
    """
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)

    # Add corpus_index
    df['corpus_index'] = df.groupby('test_id').cumcount()

    # Apply filters
    if score_threshold:
        df = df[df['score'] >= score_threshold]
        print(f"Filtered to score >= {score_threshold}: {len(df)} rows")

    if max_rows:
        df = df.head(max_rows)
        print(f"Limited to {max_rows} rows")

    print(f"Total rows to process: {len(df)}")

    # Create batch requests
    requests = []
    for batch_idx, start in enumerate(range(0, len(df), batch_size)):
        batch_df = df.iloc[start:start + batch_size]

        # Create CSV content for this batch
        csv_lines = []
        for _, row in batch_df.iterrows():
            # Escape CSV properly
            test_text = str(row['test_text']).replace('"', '""')
            corpus_text = str(row['corpus_text']).replace('"', '""')
            csv_lines.append(f'{row["test_id"]},{row["corpus_index"]},"{test_text}","{corpus_text}"')

        csv_content = "\n".join(csv_lines)

        # Create batch request
        request = {
            "custom_id": f"batch_{batch_idx:05d}",
            "params": {
                "model": model,
                "max_tokens": 4096,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(csv_content=csv_content)}
                ]
            }
        }
        requests.append(request)

    # Write JSONL
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')

    print(f"Created {len(requests)} batch requests in {output_jsonl}")

    # Estimate cost
    est_input_tokens = len(df) * 320  # ~320 tokens per row average
    est_output_tokens = len(df) * 15   # ~15 tokens per output row

    # Batch API is 50% off
    sonnet_cost = (est_input_tokens / 1e6 * 3 + est_output_tokens / 1e6 * 15) * 0.5
    print(f"Estimated cost (Sonnet 4, batch): ${sonnet_cost:.2f}")

    return len(requests)


def submit_batch(jsonl_path: str) -> str:
    """Submit batch request and return batch ID."""
    client = anthropic.Anthropic()

    print(f"Submitting batch from {jsonl_path}...")

    with open(jsonl_path, 'rb') as f:
        batch = client.batches.create(requests=f)

    print(f"Batch submitted: {batch.id}")
    print(f"Status: {batch.processing_status}")

    return batch.id


def check_batch_status(batch_id: str) -> dict:
    """Check batch status."""
    client = anthropic.Anthropic()
    batch = client.batches.retrieve(batch_id)

    return {
        'id': batch.id,
        'status': batch.processing_status,
        'created_at': batch.created_at,
        'ended_at': batch.ended_at,
        'request_counts': batch.request_counts
    }


def poll_batch_completion(batch_id: str, poll_interval: int = 60) -> dict:
    """Poll until batch completes."""
    client = anthropic.Anthropic()

    print(f"Polling batch {batch_id}...")

    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {status} | "
              f"Succeeded: {counts.succeeded}/{counts.processing + counts.succeeded + counts.errored}")

        if status == "ended":
            print("Batch completed!")
            return {
                'id': batch.id,
                'status': status,
                'results_url': batch.results_url,
                'request_counts': counts
            }

        time.sleep(poll_interval)


def download_and_parse_results(batch_id: str, output_csv: str):
    """Download batch results and parse into CSV."""
    client = anthropic.Anthropic()

    print(f"Downloading results for batch {batch_id}...")

    # Stream results
    all_results = []

    for result in client.batches.results(batch_id):
        if result.result.type == "succeeded":
            response_text = result.result.message.content[0].text

            # Parse CSV response
            for line in response_text.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('test_id'):  # Skip empty or header
                    continue
                try:
                    parts = line.split(',')
                    if len(parts) >= 4:
                        all_results.append({
                            'test_id': int(parts[0]),
                            'corpus_index': int(parts[1]),
                            'is_sd': int(parts[2]),
                            'confidence': float(parts[3])
                        })
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line: {line}")
        else:
            print(f"Request {result.custom_id} failed: {result.result.error}")

    # Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(all_results)} results to {output_csv}")

    return len(all_results)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Semantic duplicate analysis using Claude Batch API")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Prepare command
    prep = subparsers.add_parser('prepare', help='Prepare batch request JSONL')
    prep.add_argument('--input', '-i', required=True, help='Input CSV')
    prep.add_argument('--output', '-o', required=True, help='Output JSONL path')
    prep.add_argument('--batch-size', '-b', type=int, default=100)
    prep.add_argument('--score-threshold', '-s', type=float, default=None)
    prep.add_argument('--max-rows', '-m', type=int, default=None)
    prep.add_argument('--model', default='claude-sonnet-4-20250514')

    # Submit command
    submit = subparsers.add_parser('submit', help='Submit batch to API')
    submit.add_argument('--jsonl', '-j', required=True, help='JSONL file to submit')

    # Status command
    status = subparsers.add_parser('status', help='Check batch status')
    status.add_argument('--batch-id', '-b', required=True)

    # Poll command
    poll = subparsers.add_parser('poll', help='Poll until batch completes')
    poll.add_argument('--batch-id', '-b', required=True)
    poll.add_argument('--interval', '-i', type=int, default=60)

    # Download command
    download = subparsers.add_parser('download', help='Download and parse results')
    download.add_argument('--batch-id', '-b', required=True)
    download.add_argument('--output', '-o', required=True, help='Output CSV path')

    # Full pipeline command
    run = subparsers.add_parser('run', help='Run full pipeline (prepare, submit, poll, download)')
    run.add_argument('--input', '-i', required=True, help='Input CSV')
    run.add_argument('--output', '-o', required=True, help='Output CSV path')
    run.add_argument('--batch-size', '-b', type=int, default=100)
    run.add_argument('--score-threshold', '-s', type=float, default=None)
    run.add_argument('--max-rows', '-m', type=int, default=None)
    run.add_argument('--model', default='claude-sonnet-4-20250514')

    args = parser.parse_args()

    if args.command == 'prepare':
        prepare_batch_requests(
            args.input, args.output, args.batch_size,
            args.score_threshold, args.max_rows, args.model
        )

    elif args.command == 'submit':
        batch_id = submit_batch(args.jsonl)
        print(f"\nBatch ID: {batch_id}")
        print("Use this ID to check status or download results")

    elif args.command == 'status':
        status = check_batch_status(args.batch_id)
        print(json.dumps(status, indent=2, default=str))

    elif args.command == 'poll':
        result = poll_batch_completion(args.batch_id, args.interval)
        print(json.dumps(result, indent=2, default=str))

    elif args.command == 'download':
        download_and_parse_results(args.batch_id, args.output)

    elif args.command == 'run':
        # Full pipeline
        jsonl_path = args.output.replace('.csv', '_requests.jsonl')

        prepare_batch_requests(
            args.input, jsonl_path, args.batch_size,
            args.score_threshold, args.max_rows, args.model
        )

        batch_id = submit_batch(jsonl_path)

        print("\nPolling for completion (this may take a while)...")
        poll_batch_completion(batch_id)

        download_and_parse_results(batch_id, args.output)


if __name__ == "__main__":
    main()
