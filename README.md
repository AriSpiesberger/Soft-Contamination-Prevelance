# MuSR - MBPP Semantic Duplicates

Datasets for MBPP semantic duplicate research.

## Datasets

| Directory | Content | Description |
|-----------|-------------|-------------|
| `mbpp-python-dupes/output/` | 5 Python solutions per task | Alternative implementations with comments |
| `mbpp-code-translations/output/master` | 6 languages per task | JS, TS, Rust, Go, Java, Ruby translations |
| `mbpp-text-dupes/english-variations/output/` | 10 English variations per task | 5 subject-based + 5 paraphrases |

## File Formats

Each dataset contains:
- `individual/task_*.json` - Per-task results
- `master_results.json` - Complete dataset
- `*.csv` - Flat CSV for analysis

## Buggy Tasks (Exclude for Python Sem Dupes)

Exclude these task IDs due to original MBPP code/test issues:

| Task ID | Split | Issue |
|---------|-------|-------|
| 229 | test | Stability issue - doesn't preserve relative order |
| 438 | test | Logic bug - incorrect bidirectional check |
| 461 | test | Early return bug - only checks first char |
| 579 | validation | Non-deterministic - set ordering in tests |
| 769 | train | Non-deterministic - set ordering in tests |
| 802 | train | Wrong test case - expects 1 rotation for [3,2,1] |

**Exclude list**: `229, 438, 461, 579, 769, 802`

## Dataset Sizes

- **Python dupes**: 427 tasks × 5 solutions = 2,135 total (98.6% success, 100% if buggy MBPP samples exlcuded)
- **Code translations**: 427 tasks × 6 languages = 2,562 total (100% success)
- **English variations**: 427 tasks × 10 variations = 4,270 total
