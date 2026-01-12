# MBPP Code Translations Pipeline

Translate MBPP (Mostly Basic Python Problems) benchmark solutions from Python to **6 programming languages** using Claude AI.

> **Note:** Text/natural language translations have been moved to [`../mbpp-text-dupes/`](../mbpp-text-dupes/).

## Features

- **6 Programming Languages**: JavaScript, TypeScript, Rust, Go, Java, Ruby
- **Automatic Validation**: Each translation is tested against the original Python tests
- **LLM-Based Test Conversion**: Uses Claude Opus to accurately convert Python tests to target language
- **Retry with Error Feedback**: Failed translations are retried with full error history (up to 10 attempts)
- **LLM Fallback Validation**: Semantically equivalent results are validated by Claude Haiku
- **Concurrent Processing**: Process multiple samples in parallel (up to 40+ workers)
- **Auto-Resume**: Automatically continues from previous runs
- **Retry-Failed Mode**: Only retry previously failed tasks to reach 100% completion
- **External Crate Support**: Rust validator uses Cargo for `regex` and `num-bigint` support

## Quick Start

### Prerequisites

- Python 3.10+
- Anthropic API key
- Language runtimes for validation (see [WSL_SETUP.md](WSL_SETUP.md))

### Installation

```bash
cd MuSR/mbpp-code-translations
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
```

### Basic Usage

```bash
# Translate 10 samples to JavaScript (default)
python translate_mbpp.py

# Translate to a specific language
python translate_mbpp.py --language typescript --num-samples 50

# Translate to multiple languages
python translate_mbpp.py --language javascript,typescript,rust

# Translate to ALL languages
python translate_mbpp.py --all

# Full dataset with high concurrency
python translate_mbpp.py --all --num-samples 500 --all-splits --concurrency 16

# Retry only failed tasks with more attempts
python translate_mbpp.py --language rust --retry-failed --max-attempts 10

# Debug failing tasks with error capture
python translate_mbpp.py --language java --retry-failed --capture-errors
```

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--num-samples N` | Number of MBPP samples to process | 10 |
| `--language L1,L2` | Comma-separated programming languages | javascript |
| `--all` | Process all 6 programming languages | false |
| `--all-splits` | Load from all MBPP splits (prompt, test, validation, train) | false |
| `--skip-validation` | Skip test validation (faster) | false |
| `--concurrency N` | Parallel workers | 10 |
| `--force` | Reprocess all (ignore existing results) | false |
| `--retry-failed` | Only retry previously failed tasks | false |
| `--max-attempts N` | Max translation attempts per task | 5 |
| `--capture-errors` | Save detailed error traces in output | false |

## Supported Languages

| Language | Extension | Runtime | External Dependencies | Status |
|----------|-----------|---------|----------------------|--------|
| JavaScript | `.js` | Node.js | None | ✅ 100% |
| TypeScript | `.ts` | ts-node | None | ✅ 100% |
| Rust | `.rs` | cargo | regex, num-bigint | ✅ 100% |
| Go | `.go` | go | None | ✅ ~98% |
| Java | `.java` | javac + java | None | ✅ ~98% |
| Ruby | `.rb` | ruby | None | ✅ 100% |

> **Note:** Rust requires Cargo (not just rustc) to support external crates like `regex` and `num-bigint`. See [WSL_SETUP.md](WSL_SETUP.md) for installation.

## Output Structure

```
output/
├── master/
│   ├── master_results.json     # Complete results with metadata
│   ├── master_translations.csv # Flat CSV for analysis
│   └── archive/                # Timestamped backups
├── reports/
│   ├── integrity_report.txt    # Latest report
│   └── integrity_report_*.txt  # Timestamped archives
└── {language}/
    ├── individual/
    │   └── task_*.json         # Per-task results
    ├── {language}_results.json
    └── {language}_translations.csv
```

### CSV Schema

```
task_id, source_split, source_config, text, code_python, test_list,
code_javascript, code_javascript_status, code_javascript_attempts, ...
code_typescript, code_typescript_status, code_typescript_attempts, ...
code_rust, code_rust_status, code_rust_attempts, ...
code_go, code_go_status, code_go_attempts, ...
code_java, code_java_status, code_java_attempts, ...
code_ruby, code_ruby_status, code_ruby_attempts, ...
```

## Models Used

| Task | Model | Rationale |
|------|-------|-----------|
| Code translation | `claude-opus-4-5` | Best code generation quality |
| Test conversion | `claude-opus-4-5` | Accurate test syntax conversion |
| Error analysis | `claude-sonnet-4-5` | Detailed error feedback for retries |
| LLM validation | `claude-haiku-4-5` | Fast semantic comparison |

## Architecture

```
translate_mbpp.py           # Main orchestrator
├── translators/            # Language-specific translators
│   ├── base.py             # Base class with retry logic & code extraction
│   ├── javascript.py
│   ├── typescript.py
│   ├── rust.py
│   ├── go.py
│   ├── java.py
│   └── ruby.py
└── validators/             # Test runners
    ├── base.py             # Base class with LLM test conversion
    ├── javascript.py       # Node.js runner
    ├── typescript.py       # ts-node runner
    ├── rust.py             # Cargo build + run (supports external crates)
    ├── go.py               # go run
    ├── java.py             # javac + java
    └── ruby.py             # ruby interpreter
```

## Retry Mechanism

The pipeline uses a sophisticated retry mechanism with full history:

1. **Attempt 1**: Translate Python → Target Language (Opus)
2. **Validate**: Run tests using LLM-converted test code
3. **If failed**: Claude Sonnet analyzes the error
4. **Attempt 2-N**: Re-translate with FULL error history (all previous attempts + their errors)
5. **If still failed**: Claude Haiku checks for semantic equivalence

On retry, the model sees:
- All previous code attempts
- All previous error analyses
- Instructions to try a DIFFERENT approach if same approach keeps failing

This typically achieves **95-100% success rates** depending on language.

## Regenerating Outputs

To rebuild master JSON, CSV, and integrity report from individual files:

```bash
python regenerate_outputs.py
```

## Documentation

- [WSL_SETUP.md](WSL_SETUP.md) - Environment setup guide (required for validation)
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development documentation and changelog

## Related Projects

- [`../mbpp-python-dupes/`](../mbpp-python-dupes/) - Python solution duplicates
- [`../mbpp-text-dupes/`](../mbpp-text-dupes/) - Text/natural language translations

## License

MIT
