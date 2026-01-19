"""
Annotate semantic duplicates using Gemini Batch API.

Benefits:
- 50% cost reduction vs real-time API
- No rate limiting issues
- Handles large volumes (up to 2GB JSONL input)
- Target turnaround: 24 hours (often faster)

Workflow:
1. Prepare JSONL file with all requests
2. Upload via File API
3. Submit batch job
4. Poll for completion
5. Download results and create CSV

Usage:
    python annotate_batch.py --csv all_mbpp_samples.csv --benchmark mbpp
    python annotate_batch.py --check <batch_job_name>  # Check status
    python annotate_batch.py --download <batch_job_name>  # Download results
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

# Load .env file if present (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on manual env var

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent / "data"
BATCH_DIR = Path(__file__).parent / "batch_jobs"
OUTPUT_DIR = Path(__file__).parent / "annotations"

MODEL_ID = "models/gemini-2.0-flash"  # Batch API supports gemini-2.0-flash

# Prompts (same as real-time version)
MBPP_PROMPT_TEMPLATE = """You are an expert programmer analyzing potential semantic duplicates between coding tasks.

## Task
Determine if the following two coding tasks are semantic duplicates - meaning they describe the same programming task, just potentially phrased differently.

## Test Task (from benchmark):
{test_text}

## Corpus Task (from training data):
{corpus_text}

## Guidelines:
1. **Focus on the TASK, not the solution** - ignore any code or solutions that may be present
2. **Mathematical equivalence counts as duplicate** - e.g., "sum 1 to n" and "sum n, n-1, ..., 1" are equivalent
3. **Corpus subsumes test = duplicate** - if the corpus task is strictly harder (asks for more), but solving it would trivially solve the test task, mark as duplicate
4. **Be calibrated** - use confidence primarily for ambiguous cases, tricky phrasing, or when you're uncertain

## Match Types:
- "exact": Nearly identical wording
- "equivalent": Different phrasing, same underlying task
- "subset": Test task is a subset of corpus task (corpus is harder but solves test)
- "superset": Corpus task is a subset of test task (test is harder) - NOT a duplicate
- "unrelated": Different tasks entirely

Respond with a JSON object containing:
- is_sd: boolean (true if semantic duplicate)
- confidence: float 0-1 (calibrated confidence)
- match_type: string (one of: exact, equivalent, subset, superset, unrelated)
- reasoning: string (brief explanation)"""


CODEFORCES_PROMPT_TEMPLATE = """You are an expert competitive programmer analyzing potential semantic duplicates between programming problems.

## Task
Determine if the following two competitive programming problems are semantic duplicates - meaning exposure to the corpus problem during training would effectively leak how to solve the test problem.

## Test Problem (from benchmark):
{test_text}

## Corpus Problem (from training data):
{corpus_text}

## Guidelines:
- **What makes problems duplicates:** Same computational task after removing story, same key insight needed
- **What does NOT make duplicates:** Sharing common techniques (BFS, DP), similar I/O format, same category

## Match Types:
- "exact": Nearly identical problem statements
- "equivalent": Different framing but identical algorithmic core
- "subset": Test is a special case of corpus
- "superset": Corpus is a special case of test - NOT a duplicate
- "unrelated": Different problems, or corpus data is incomplete

Respond with a JSON object containing:
- is_sd: boolean (true if semantic duplicate)
- confidence: float 0-1 (calibrated confidence)
- match_type: string (one of: exact, equivalent, subset, superset, unrelated)
- reasoning: string (brief explanation)"""

PROMPTS = {
    "mbpp": MBPP_PROMPT_TEMPLATE,
    "codeforces": CODEFORCES_PROMPT_TEMPLATE,
}


# =============================================================================
# BATCH JOB MANAGEMENT
# =============================================================================

def create_client() -> genai.Client:
    """Create Gemini client."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    return genai.Client(api_key=api_key)


def load_existing_annotations(benchmark: str) -> set:
    """Load set of already-completed annotation keys."""
    out_dir = OUTPUT_DIR / benchmark
    completed = set()
    
    if not out_dir.exists():
        return completed
    
    for json_file in out_dir.glob("*.json"):
        if not json_file.stem.startswith("_"):
            completed.add(json_file.stem)
    
    return completed


def prepare_batch_requests(
    csv_path: str,
    benchmark: str,
    skip_completed: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Prepare batch requests from CSV.
    
    Returns:
        Tuple of (requests list, metadata list)
    """
    prompt_template = PROMPTS[benchmark]
    
    # Load existing annotations to skip
    completed = set()
    if skip_completed:
        completed = load_existing_annotations(benchmark)
        print(f"Found {len(completed):,} existing annotations to skip")
    
    requests = []
    metadata = []
    
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Map column names
            if "source" in row and "dataset" not in row:
                row["dataset"] = row["source"]
            if "similarity" in row and "score" not in row:
                row["score"] = row["similarity"]
            
            test_id = row["test_id"]
            corpus_id = row["corpus_id"]
            dataset = row["dataset"]
            
            # Create unique key
            import re
            safe_test_id = re.sub(r'[^\w\-]', '_', str(test_id))
            safe_corpus_id = re.sub(r'[^\w\-]', '_', str(corpus_id))
            safe_dataset = re.sub(r'[^\w\-]', '_', str(dataset))
            key = f"{safe_dataset}__{safe_test_id}__{safe_corpus_id}"
            
            # Skip if already completed
            if key in completed:
                continue
            
            # Build prompt
            prompt = prompt_template.format(
                test_text=row["test_text"],
                corpus_text=row["corpus_text"],
            )
            
            # Batch request format - model must be full path
            request = {
                "custom_id": key,
                "request": {
                    "model": MODEL_ID,
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "temperature": 1.0,
                    },
                },
            }
            
            requests.append(request)
            metadata.append({
                "key": key,
                "test_id": test_id,
                "corpus_id": corpus_id,
                "dataset": dataset,
                "score": row.get("score", ""),
                "test_text": row["test_text"][:500],  # Truncate for metadata
                "corpus_text": row["corpus_text"][:500],
            })
    
    return requests, metadata


def write_jsonl(requests: list[dict], output_path: Path) -> None:
    """Write requests to JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")


def submit_batch_job(
    client: genai.Client,
    jsonl_path: Path,
    benchmark: str,
) -> str:
    """
    Upload JSONL and submit batch job.
    
    Returns batch job name.
    """
    print(f"Uploading {jsonl_path}...")
    
    # Upload file
    uploaded_file = client.files.upload(
        file=jsonl_path,
        config={"mime_type": "application/jsonl"},
    )
    print(f"Uploaded: {uploaded_file.name}")
    
    # Create batch job
    print("Creating batch job...")
    batch_job = client.batches.create(
        model=MODEL_ID,
        src=uploaded_file.name,
        config=types.CreateBatchJobConfig(
            display_name=f"sd-annotation-{benchmark}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        ),
    )
    
    print(f"Batch job created: {batch_job.name}")
    print(f"State: {batch_job.state}")
    
    # Save job info
    job_info = {
        "name": batch_job.name,
        "benchmark": benchmark,
        "created": datetime.now().isoformat(),
        "input_file": str(jsonl_path),
        "uploaded_file": uploaded_file.name,
    }
    
    job_info_path = BATCH_DIR / f"{Path(batch_job.name).stem}_info.json"
    with open(job_info_path, "w") as f:
        json.dump(job_info, f, indent=2)
    
    return batch_job.name


def check_batch_status(client: genai.Client, batch_name: str) -> dict:
    """Check batch job status."""
    batch_job = client.batches.get(name=batch_name)
    
    return {
        "name": batch_job.name,
        "state": str(batch_job.state),
        "create_time": str(batch_job.create_time) if batch_job.create_time else None,
        "update_time": str(batch_job.update_time) if batch_job.update_time else None,
    }


def download_results(
    client: genai.Client,
    batch_name: str,
    benchmark: str,
    output_csv: str = None,
) -> None:
    """Download batch results and create CSV."""
    batch_job = client.batches.get(name=batch_name)
    
    if str(batch_job.state) != "JOB_STATE_SUCCEEDED":
        print(f"Batch job not complete. State: {batch_job.state}")
        return
    
    # Get output file
    output_file = batch_job.dest
    if not output_file:
        print("No output file available")
        return
    
    print(f"Downloading results from {output_file}...")
    
    # Download content
    result_content = client.files.download(name=output_file)
    
    # Parse JSONL results
    results = []
    for line in result_content.decode("utf-8").strip().split("\n"):
        if line:
            results.append(json.loads(line))
    
    print(f"Downloaded {len(results):,} results")
    
    # Save individual JSON files (for compatibility with existing structure)
    out_dir = OUTPUT_DIR / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_rows = []
    
    for result in results:
        custom_id = result.get("custom_id", "unknown")
        response = result.get("response", {})
        
        # Extract annotation from response
        annotation = None
        error = None
        
        try:
            if "candidates" in response:
                text = response["candidates"][0]["content"]["parts"][0]["text"]
                annotation = json.loads(text)
        except Exception as e:
            error = str(e)
        
        # Save JSON
        output_data = {
            "custom_id": custom_id,
            "annotation": annotation,
            "error": error,
            "raw_response": response,
        }
        
        json_path = out_dir / f"{custom_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        
        # Build CSV row
        if annotation:
            csv_rows.append({
                "key": custom_id,
                "is_sd": annotation.get("is_sd", False),
                "confidence": annotation.get("confidence", 0),
                "match_type": annotation.get("match_type", ""),
                "reasoning": annotation.get("reasoning", ""),
            })
    
    # Write CSV
    if output_csv is None:
        output_csv = out_dir / f"results_{benchmark}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    if csv_rows:
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["key", "is_sd", "confidence", "match_type", "reasoning"])
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"CSV saved to {output_csv}")
        
        # Summary stats
        n_sd = sum(1 for r in csv_rows if r["is_sd"])
        print(f"\nSummary:")
        print(f"  Total: {len(csv_rows):,}")
        print(f"  Semantic duplicates: {n_sd:,} ({100*n_sd/len(csv_rows):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Batch annotation using Gemini Batch API")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit a new batch job")
    submit_parser.add_argument("--csv", required=True, help="Input CSV file")
    submit_parser.add_argument("--benchmark", choices=["mbpp", "codeforces"], required=True)
    submit_parser.add_argument("--skip-completed", action="store_true", default=True,
                               help="Skip already-completed annotations")
    submit_parser.add_argument("--batch-size", type=int, default=5000,
                               help="Max requests per batch job (default: 5000)")
    
    # Check command
    check_parser = subparsers.add_parser("check", help="Check batch job status")
    check_parser.add_argument("--name", required=True, help="Batch job name")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download batch results")
    download_parser.add_argument("--name", required=True, help="Batch job name")
    download_parser.add_argument("--benchmark", choices=["mbpp", "codeforces"], required=True)
    download_parser.add_argument("--output-csv", help="Output CSV path")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List batch jobs")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    
    client = create_client()
    
    if args.command == "submit":
        # Prepare requests
        print(f"Loading {args.csv}...")
        requests, metadata = prepare_batch_requests(
            args.csv,
            args.benchmark,
            skip_completed=args.skip_completed,
        )
        
        if not requests:
            print("No new requests to submit (all already completed)")
            return
        
        print(f"Prepared {len(requests):,} requests")
        
        # Split into batches
        batch_size = args.batch_size
        num_batches = (len(requests) + batch_size - 1) // batch_size
        print(f"Splitting into {num_batches} batch(es) of up to {batch_size:,} each")
        
        batch_names = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(requests))
            batch_requests = requests[start_idx:end_idx]
            batch_metadata = metadata[start_idx:end_idx]
            
            print(f"\n--- Batch {i+1}/{num_batches} ({len(batch_requests):,} requests) ---")
            
            # Write JSONL
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            jsonl_path = BATCH_DIR / f"batch_{args.benchmark}_{timestamp}_part{i+1}.jsonl"
            write_jsonl(batch_requests, jsonl_path)
            print(f"Wrote {jsonl_path}")
            
            # Save metadata
            meta_path = BATCH_DIR / f"batch_{args.benchmark}_{timestamp}_part{i+1}_meta.json"
            with open(meta_path, "w") as f:
                json.dump(batch_metadata, f)
            
            # Submit
            try:
                batch_name = submit_batch_job(client, jsonl_path, args.benchmark)
                batch_names.append(batch_name)
                print(f"Submitted: {batch_name}")
            except Exception as e:
                print(f"Error submitting batch {i+1}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Submitted {len(batch_names)} batch job(s):")
        for name in batch_names:
            print(f"  - {name}")
        print("Use 'python annotate_batch.py check --name <name>' to check status")
        
    elif args.command == "check":
        status = check_batch_status(client, args.name)
        print(json.dumps(status, indent=2))
        
    elif args.command == "download":
        download_results(client, args.name, args.benchmark, args.output_csv)
        
    elif args.command == "list":
        # List local job info files
        for info_file in BATCH_DIR.glob("*_info.json"):
            with open(info_file) as f:
                info = json.load(f)
            status = check_batch_status(client, info["name"])
            print(f"{info['name']}: {status['state']}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
