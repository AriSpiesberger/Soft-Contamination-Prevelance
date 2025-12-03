# Datasets

This document describes the datasets available in the `datasets/` directory.

## Overview

| Dataset | Type | Rows | Files | Description |
|---------|------|------|-------|-------------|
| GSM8K | Math Problems | 8,792 total | 2 parquet | Grade school math word problems |
| Codeforces | Coding Problems | 1,337 total | 2 parquet | Competitive programming problems |
| AllenAI MathCoder | Educational Text | Unknown | 1 jsonl | Educational math text content |
| MBPP | Python Problems | 427 train | HuggingFace | Mostly Basic Python Problems |
| HumanEval | Code Evaluation | 164 test | HuggingFace | Python code evaluation tasks |
| PopQA | Question Answering | 14,000 test | HuggingFace | Popular entity question answering |
| BigBenchHard | Multi-domain Reasoning | 6,511 items | HuggingFace | 23 challenging BIG-Bench tasks |
| ZebraLogic | Logic Puzzles | 1,000 puzzles | HuggingFace | Logic grid puzzles (Zebra Puzzles) |
| AGIEval | Exam Questions | 8,062 questions | HuggingFace | Human cognition & problem-solving exams |

## Dataset Details

### 1. GSM8K (OpenAI)

Math word problems with natural language solutions.

#### Files
- `openai_gsm8k_train-00000-of-00001.parquet` (7,473 rows)
- `openai_gsm8k_test-00000-of-00001.parquet` (1,319 rows)

#### Columns
- `question` (String): The math word problem
- `answer` (String): The solution with step-by-step reasoning

#### Schema
```
Schema([('question', String), ('answer', String)])
```

---

### 2. Codeforces (Open-R1)

Competitive programming problems from Codeforces contests.

#### Files
- `open-r1_codeforces_train-00000-of-00011.parquet` (869 rows)
- `open-r1_codeforces_test-00000-of-00001.parquet` (468 rows)

#### Columns (27 total)

**Metadata:**
- `id` (String): Unique problem identifier
- `aliases` (List[String]): Alternative identifiers
- `contest_id` (String): Contest identifier
- `contest_name` (String): Name of the contest
- `contest_type` (String): Type of contest
- `contest_start` (Int64): Contest start timestamp
- `contest_start_year` (Int64): Year of contest
- `index` (String): Problem index in contest

**Problem Constraints:**
- `time_limit` (Float64): Time limit in seconds
- `memory_limit` (Float64): Memory limit in MB
- `executable` (Boolean): Whether code execution is required

**Problem Description:**
- `title` (String): Problem title
- `description` (String): Problem description
- `input_format` (String): Input specification
- `output_format` (String): Output specification
- `interaction_format` (String): Interactive problem format (if applicable)
- `note` (String): Additional notes

**Test Cases:**
- `examples` (List[Struct{'input': String, 'output': String}]): Example test cases
- `official_tests` (List[Struct{'input': String, 'output': String}]): Official test cases
- `official_tests_complete` (Boolean): Whether all official tests are included
- `testset_size` (Int64): Number of test cases
- `generated_tests` (Int64): Number of generated tests

**Additional Info:**
- `editorial` (String): Editorial/solution explanation
- `rating` (Int64): Problem difficulty rating
- `tags` (List[String]): Problem tags/categories
- `input_mode` (String): Input reading mode
- `generated_checker` (String): Checker code

#### Schema
```
Schema([
  ('id', String),
  ('aliases', List(String)),
  ('contest_id', String),
  ('contest_name', String),
  ('contest_type', String),
  ('contest_start', Int64),
  ('contest_start_year', Int64),
  ('index', String),
  ('time_limit', Float64),
  ('memory_limit', Float64),
  ('title', String),
  ('description', String),
  ('input_format', String),
  ('output_format', String),
  ('interaction_format', String),
  ('note', String),
  ('examples', List(Struct({'input': String, 'output': String}))),
  ('editorial', String),
  ('rating', Int64),
  ('tags', List(String)),
  ('testset_size', Int64),
  ('official_tests', List(Struct({'input': String, 'output': String}))),
  ('official_tests_complete', Boolean),
  ('input_mode', String),
  ('generated_checker', String),
  ('executable', Boolean),
  ('generated_tests', Int64)
])
```

---

### 3. AllenAI Dolmino MathCoder2

Educational math text content from the SMolInstruct dataset.

#### Files
- `allenai_dolmino-mix-1124_mathcoder2-synthmath_book_math.0000.0008.jsonl`

#### Fields (9 total)

**Metadata:**
- `Language=` (String): Language code (e.g., "en")
- `License=` (String): License information (e.g., "hf apache-2.0")
- `Meta_Link=` (String): URL to source dataset
- `id` (String): Unique identifier

**Content Classification:**
- `audience` (String): Target audience (e.g., "grade_school_students")
- `format` (String): Content format (e.g., "educational_piece")
- `seed_data` (String): Source data type (e.g., "auto_math_text")

**Content:**
- `text` (String): Main educational content
- `text_token_length` (Int): Token count of the text

#### Example Record
```json
{
  "Language=": "en",
  "License=": "hf apache-2.0",
  "Meta_Link=": "https://huggingface.co/datasets/osunlp/SMolInstruct",
  "audience": "grade_school_students",
  "format": "educational_piece",
  "seed_data": "auto_math_text",
  "text": "Here's an extract from a webpage...",
  "text_token_length": 421,
  "id": "book_math.0000.0008_000017"
}
```

---

### 4. MBPP (Google Research)

Mostly Basic Python Problems - a collection of entry-level Python programming problems.

#### Source
- HuggingFace: `google-research-datasets/mbpp` (sanitized split, train)

#### Columns
- `task_id` (Int): Unique task identifier
- `prompt` (String): The problem description
- `code` (String): Reference solution code
- `test_imports` (List[String]): Required imports for testing
- `test_list` (List[String]): List of test cases

#### Purpose
Entry-level Python programming tasks suitable for paraphrasing and parameter substitution.

---

### 5. HumanEval (OpenAI)

Python code evaluation dataset with function signatures and docstrings.

#### Source
- HuggingFace: `openai/openai_humaneval` (test split)

#### Columns
- `task_id` (String): Unique task identifier (e.g., "HumanEval/0")
- `prompt` (String): Function signature with docstring and examples
- `canonical_solution` (String): Reference implementation
- `test` (String): Test code to validate solutions
- `entry_point` (String): Function name to test

#### Purpose
Code generation benchmarking with complete function specifications.

---

### 6. PopQA (Popular Question Answering)

Large-scale question answering dataset about popular entities.

#### Source
- HuggingFace: `lytang/popqa` (test split)

#### Columns
- `id` (String): Unique question identifier
- `question` (String): The question being asked
- `answer` (String): The correct answer
- `subject` (String): The entity the question is about
- `property` (String): The property being queried

#### Purpose
Knowledge-based question answering suitable for entity substitution and paraphrasing.
