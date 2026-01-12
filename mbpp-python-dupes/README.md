# MBPP Python Semantic Duplicates Generator

Generate up to 5 semantically different Python solutions for each MBPP (Mostly Basic Python Problems) benchmark problem using Claude Opus 4.5.

## Purpose

This tool creates **semantic duplicates** - different implementations of the same algorithm that solve the same problem but with:
- Different code structure and algorithms
- Unique line-by-line comments explaining each step
- Validated correctness (all solutions pass original tests)

## Features

- **Recursive Generation**: Each new solution is generated with knowledge of all previous solutions to ensure structural differences
- **Line-by-Line Comments**: Every solution has detailed comments on the line above each code line
- **Automatic Validation**: Each solution is tested against MBPP test cases
- **Similarity Detection**: Rejects solutions that are too similar (>85% similarity) to previous ones
- **Retry with Feedback**: Failed validations trigger retries with error context
- **Parallel Processing**: Process multiple samples concurrently
- **Resume Capability**: Continue interrupted runs without reprocessing

## Quick Start

### Prerequisites

- Python 3.11+
- Anthropic API key
- WSL (Windows Subsystem for Linux) or Linux environment

### Installation

```bash
cd MuSR/mbpp-python-dupes
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
```

### Basic Usage

```bash
# Generate duplicates for 10 samples (default)
python generate_python_dupes.py

# Process more samples with higher concurrency
python generate_python_dupes.py --num-samples 50 --concurrency 20

# Full dataset (974 samples from all splits)
python generate_python_dupes.py --num-samples 974 --all-splits --concurrency 40

# Sanitized dataset (427 samples, higher quality)
python generate_python_dupes.py --num-samples 427 --sanitized --all-splits --concurrency 40

# Resume a previous run
python generate_python_dupes.py --num-samples 100 --resume
```

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--num-samples N` | Number of MBPP samples to process | 10 |
| `--concurrency N` | Parallel workers | 10 |
| `--sanitized` | Use sanitized dataset (427 samples) instead of full (974) | false |
| `--all-splits` | Load from all splits (prompt+test+validation+train) | false |
| `--resume` | Skip already-successful samples | false |
| `--force` | Reprocess all (ignore resume) | false |

## Output

Results are saved in multiple formats:

```
output/
├── master_results.json     # Complete results with metadata
├── mbpp_python_dupes.csv   # Flat CSV for analysis
├── integrity_report.txt    # Statistics and status report
└── individual/
    ├── task_11.json        # Per-task results
    ├── task_12.json
    └── ...
```

### CSV Schema

```
task_id, prompt, code_python, test_list,
python_1, python_1_status, python_1_attempts,
python_2, python_2_status, python_2_attempts,
python_3, python_3_status, python_3_attempts,
python_4, python_4_status, python_4_attempts,
python_5, python_5_status, python_5_attempts
```

### Status Values

- `success`: Solution generated and validated successfully
- `failed`: Generation or validation failed after all retries

## Algorithm

For each MBPP sample:

1. **Load** the original Python solution and test cases
2. **Generate python_1**: Prompt Claude with original solution, requesting a different implementation with comments
3. **Validate**: Run the solution against all test cases
4. **Check Similarity**: Ensure the new solution is <85% similar to previous ones
5. **Generate python_2**: Include both original and python_1 in prompt, request different approach
6. **Repeat** until python_5 or maximum attempts exhausted

### Prompt Strategy

The generation prompt:
- Shows the task description and original solution
- Lists ALL previous solutions with instruction to be structurally different
- Requires comments on the line above each code line
- Encourages use of different data structures, iteration patterns, and built-ins
- On similarity rejection, explicitly requests more algorithmic variation

## Configuration

Key parameters (in `generate_python_dupes.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAX_DUPLICATES` | 5 | Number of duplicates to generate per sample |
| `MAX_GENERATION_ATTEMPTS` | 3 | Retries per duplicate on validation failure |
| `MAX_SIMILARITY_RETRIES` | 2 | Retries if solution is too similar |
| `SIMILARITY_THRESHOLD` | 0.85 | Maximum allowed code similarity |
| `MODEL` | claude-opus-4-5 | Claude model for generation |

## Example Output

For a simple "find minimum" task, you might get:

**python_1** (iterative with explicit comparison):
```python
# Initialize the minimum value with the first element
min_val = nums[0]
# Loop through remaining elements to find smaller values
for num in nums[1:]:
    # Update minimum if current element is smaller
    if num < min_val:
        min_val = num
# Return the smallest value found
return min_val
```

**python_2** (using built-in):
```python
# Use Python's built-in min function for efficiency
# This leverages optimized C implementation
return min(nums)
```

**python_3** (sorting approach):
```python
# Create a sorted copy of the input list
sorted_nums = sorted(nums)
# The minimum is always at the first position after sorting
return sorted_nums[0]
```

## Model Used

| Task | Model | Rationale |
|------|-------|-----------|
| Code generation | `claude-opus-4-5` | Best code generation quality |

## Dataset

| Config | Samples | Description |
|--------|---------|-------------|
| Full (default) | 974 | Complete MBPP dataset |
| Sanitized | 427 | Higher quality, cleaner subset |

Split distribution (Full config):
- prompt: 10 samples (task_ids: 1-10)
- test: 500 samples (task_ids: 11-510)
- validation: 90 samples (task_ids: 511-600)
- train: 374 samples (task_ids: 601-974)

## Current Results

Latest run on sanitized dataset (427 samples):

| Metric | Value |
|--------|-------|
| Total tasks | 427 |
| Total duplicate slots | 2,135 |
| Successful duplicates | 2,106 |
| Failed duplicates | 29 |
| **Success rate** | **98.64%** |

Per-slot breakdown:
- python_1: 421/427 (98.6%)
- python_2: 421/427 (98.6%)
- python_3: 421/427 (98.6%)
- python_4: 422/427 (98.8%)
- python_5: 421/427 (98.6%)

> **Note**: 6 tasks have failures due to dataset quality issues (buggy original code, non-deterministic test ordering).

## License

MIT

