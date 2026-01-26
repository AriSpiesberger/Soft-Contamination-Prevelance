#!/usr/bin/env python3
"""
Semantic Duplicate Analysis using Claude API

Analyzes potential semantic duplicates between test tasks and corpus tasks
from contamination analysis results. Processes in batches and saves results.
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import anthropic


SYSTEM_PROMPT = """You are an expert at analyzing coding problems and determining if two problem descriptions are semantic duplicates - i.e., the same task just phrased differently.

Rules for determining semantic duplicates:
1. Focus ONLY on the task description, ignore any code solutions provided
2. Direct mathematical equivalence counts as a semantic duplicate (e.g., "sum all numbers 1..n" and "sum all numbers n, n-1, n-2, ..., 0" are equivalent)
3. If the corpus_text task is strictly stronger (asks for additional output, or where test is a test "does X have property?" and corpus finds the solution "find X with property"), count it as a duplicate since it can trivially reduce to the test task
4. This is asymmetric - a stronger corpus task that subsumes the test task counts, but not vice versa

You will receive multiple pairs to analyze. For each pair, determine:
- is_sd: 1 if semantic duplicate, 0 if not
- confidence: 0.0 to 1.0 (your calibrated confidence, 1.0 = certain, 0.5 = unsure)

Be calibrated - use lower confidence for ambiguous cases, tricky phrasing, hidden complexity, or when you don't fully understand a task."""


def create_batch_prompt(rows: list[dict]) -> str:
    """Create a prompt for analyzing a batch of task pairs."""
    prompt = "Analyze the following pairs of coding tasks. For each, determine if they are semantic duplicates.\n\n"

    for i, row in enumerate(rows):
        prompt += f"--- PAIR {i+1} ---\n"
        prompt += f"TEST_ID: {row['test_id']}\n"
        prompt += f"CORPUS_INDEX: {row['corpus_index']}\n"
        prompt += f"TEST_TEXT:\n{row['test_text']}\n\n"
        prompt += f"CORPUS_TEXT:\n{row['corpus_text']}\n\n"

    prompt += """---

Respond with a JSON array containing one object per pair, in order:
[
  {"test_id": <id>, "corpus_index": <idx>, "is_sd": 0 or 1, "confidence": <float 0-1>},
  ...
]

Only output the JSON array, no other text."""

    return prompt


def call_claude_api(client: anthropic.Anthropic, prompt: str, max_retries: int = 3) -> Optional[str]:
    """Call Claude API with retry logic."""
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except anthropic.RateLimitError:
            wait_time = 60 * (attempt + 1)
            print(f"Rate limited, waiting {wait_time}s...")
            time.sleep(wait_time)
        except anthropic.APIError as e:
            print(f"API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                raise
    return None


def parse_response(response: str, expected_count: int) -> Optional[list[dict]]:
    """Parse Claude's JSON response."""
    try:
        # Try to extract JSON from response
        response = response.strip()
        if response.startswith("```"):
            # Remove markdown code blocks
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        results = json.loads(response)

        if len(results) != expected_count:
            print(f"Warning: expected {expected_count} results, got {len(results)}")

        return results
    except json.JSONDecodeError as e:
        print(f"Failed to parse response: {e}")
        print(f"Response was: {response[:500]}...")
        return None


def analyze_semantic_duplicates(
    input_csv: str,
    output_csv: str,
    batch_size: int = 10,
    max_rows: Optional[int] = None,
    resume: bool = True
):
    """
    Analyze semantic duplicates in contamination results.

    Args:
        input_csv: Path to input CSV with contamination results
        output_csv: Path to save output CSV
        batch_size: Number of rows per API call
        max_rows: Limit rows to process (None = all)
        resume: Resume from existing output file if present
    """
    # Load input data
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    if max_rows:
        df = df.head(max_rows)

    print(f"Total rows to process: {len(df)}")

    # Add corpus_index column (row index within each test_id)
    df['corpus_index'] = df.groupby('test_id').cumcount()

    # Check for existing results to resume
    processed_keys = set()
    existing_results = []

    if resume and Path(output_csv).exists():
        print(f"Found existing output file, resuming...")
        existing_df = pd.read_csv(output_csv)
        processed_keys = set(zip(existing_df['test_id'], existing_df['corpus_index']))
        existing_results = existing_df.to_dict('records')
        print(f"Already processed: {len(processed_keys)} rows")

    # Filter to unprocessed rows
    df['key'] = list(zip(df['test_id'], df['corpus_index']))
    df_to_process = df[~df['key'].isin(processed_keys)].drop(columns=['key'])

    if len(df_to_process) == 0:
        print("All rows already processed!")
        return

    print(f"Rows remaining: {len(df_to_process)}")

    # Initialize API client
    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

    # Process in batches
    results = existing_results.copy()

    for batch_start in tqdm(range(0, len(df_to_process), batch_size), desc="Processing batches"):
        batch_df = df_to_process.iloc[batch_start:batch_start + batch_size]

        # Prepare batch data
        batch_rows = []
        for _, row in batch_df.iterrows():
            batch_rows.append({
                'test_id': row['test_id'],
                'corpus_index': row['corpus_index'],
                'test_text': row['test_text'],
                'corpus_text': row['corpus_text']
            })

        # Call API
        prompt = create_batch_prompt(batch_rows)
        response = call_claude_api(client, prompt)

        if response is None:
            print(f"Failed to get response for batch starting at {batch_start}")
            continue

        # Parse response
        batch_results = parse_response(response, len(batch_rows))

        if batch_results is None:
            # Retry with single items
            print("Batch parse failed, retrying individually...")
            for row in batch_rows:
                single_prompt = create_batch_prompt([row])
                single_response = call_claude_api(client, single_prompt)
                if single_response:
                    single_result = parse_response(single_response, 1)
                    if single_result:
                        result = single_result[0]
                        result['test_text'] = row['test_text']
                        results.append(result)
        else:
            # Add test_text to results and extend
            for i, result in enumerate(batch_results):
                result['test_text'] = batch_rows[i]['test_text']
                results.append(result)

        # Save checkpoint every 10 batches
        if (batch_start // batch_size + 1) % 10 == 0:
            print(f"Saving checkpoint ({len(results)} rows)...")
            save_results(results, output_csv)

        # Small delay to avoid rate limits
        time.sleep(0.5)

    # Save final results
    save_results(results, output_csv)
    print(f"Analysis complete. Results saved to {output_csv}")


def save_results(results: list[dict], output_csv: str):
    """Save results to CSV."""
    df = pd.DataFrame(results)
    # Ensure column order
    cols = ['test_id', 'corpus_index', 'test_text', 'is_sd', 'confidence']
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze semantic duplicates in contamination results")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    parser.add_argument("--output", "-o", required=True, help="Output CSV file")
    parser.add_argument("--batch-size", "-b", type=int, default=10, help="Batch size for API calls")
    parser.add_argument("--max-rows", "-m", type=int, default=None, help="Max rows to process")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, don't resume")

    args = parser.parse_args()

    analyze_semantic_duplicates(
        input_csv=args.input,
        output_csv=args.output,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        resume=not args.no_resume
    )
