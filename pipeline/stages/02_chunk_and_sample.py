# -*- coding: utf-8 -*-
"""
Stage 02: Chunk and Sample

Extracts text chunks from downloaded data files, filters by token count,
and produces a stratified random sample preserving corpus category proportions.

Supports modes: paragraph, conversation, dpo, rl
Supports formats: .jsonl, .jsonl.zst, .json.gz, .jsonl.gz, .parquet
"""

import contextlib
import gzip
import hashlib
import io
import json
import multiprocessing
import os
import random
import re
import sys
import unicodedata
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import duckdb
import tiktoken
import yaml
import zstandard as zstd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIPELINE_ROOT = Path(__file__).parent.parent
CONFIG_FILE = os.environ.get(
    "PIPELINE_CONFIG", PIPELINE_ROOT / "configs" / "default.yaml"
)

SUPPORTED_EXTENSIONS = (".jsonl", ".jsonl.zst", ".json.gz", ".jsonl.gz", ".parquet")
VALID_MODES = ("paragraph", "conversation", "dpo", "rl")

KNOWN_CATEGORY_PREFIXES = (
    "common_crawl", "wiki_to_rcqa", "olmocr_science_pdfs",
    "dolma", "wiki", "olmocr",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ChunkingConfig:
    """All settings for the chunking stage."""
    mode: str = "paragraph"
    data_dir: str = ""
    output_file: str = ""
    dataset_short_name: str = "dataset"

    # Sampling
    sample_percentage: float = 0.01
    sample_size: Optional[int] = None

    # Token filtering
    min_tokens: int = 50
    max_tokens: int = 512
    tokenizer_encoding: str = "cl100k_base"

    # Processing
    num_workers: int = 1
    chunk_size: int = 1000

    # Conversation mode
    user_role: str = "user"
    assistant_role: str = "assistant"
    content_field: str = "content"

    # DPO mode
    dpo_extract_mode: str = "both"
    dpo_chosen_field: str = "chosen"
    dpo_rejected_field: str = "rejected"

    # RL mode
    rl_prompt_field: str = "prompt"
    rl_solution_field: str = "solution"

    # Quality filtering
    quality_filter: bool = True


def load_config_yaml(path=None):
    """Load the raw YAML config dict."""
    config_path = Path(path or CONFIG_FILE)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_path(path_str):
    """Resolve *path_str* relative to the pipeline root when not absolute."""
    p = Path(path_str)
    return str(p if p.is_absolute() else PIPELINE_ROOT / p)


def load_chunking_config(yaml_path=None):
    """Parse YAML into a validated :class:`ChunkingConfig`."""
    raw = load_config_yaml(yaml_path)
    ch = raw.get("chunking", {})
    pipe = raw.get("pipeline", {})
    conv = ch.get("conversation", {})
    dpo = ch.get("dpo", {})
    rl = ch.get("rl", {})

    mode = ch.get("mode", "paragraph")
    assert mode in VALID_MODES, f"Invalid mode '{mode}', expected one of {VALID_MODES}"

    data_dir = resolve_path(ch.get("input_dir", "./data/dolma3_sample"))
    dataset_short_name = pipe.get("dataset_short_name", pipe.get("name", "dataset"))

    sample_pct = ch.get("paragraph_sample_percentage", 0.01)
    sample_size = ch.get("paragraph_sample_size", None)

    # Build standard output filename: conversations_{dataset}_{pct}.jsonl
    pct_str = "fixed" if sample_size else f"{int(sample_pct * 100)}pct"
    output_file = resolve_path(
        str(Path(data_dir).parent / f"conversations_{dataset_short_name}_{pct_str}.jsonl")
    )

    min_tok = ch.get("min_paragraph_tokens", 50)
    max_tok = ch.get("max_paragraph_tokens", 512)
    assert 0 < min_tok < max_tok, (
        f"Token bounds invalid: min={min_tok}, max={max_tok}"
    )

    cfg = ChunkingConfig(
        mode=mode,
        data_dir=data_dir,
        output_file=output_file,
        dataset_short_name=dataset_short_name,
        sample_percentage=sample_pct,
        sample_size=sample_size,
        min_tokens=min_tok,
        max_tokens=max_tok,
        tokenizer_encoding=ch.get("tokenizer_encoding", "cl100k_base"),
        num_workers=max(1, ch.get("num_workers", multiprocessing.cpu_count() - 1)),
        chunk_size=ch.get("chunk_size", 1000),
        user_role=conv.get("user_role", "user"),
        assistant_role=conv.get("assistant_role", "assistant"),
        content_field=conv.get("content_field", "content"),
        dpo_extract_mode=dpo.get("extract_mode", "both"),
        dpo_chosen_field=dpo.get("chosen_field", "chosen"),
        dpo_rejected_field=dpo.get("rejected_field", "rejected"),
        rl_prompt_field=rl.get("prompt_field", "prompt"),
        rl_solution_field=rl.get("solution_field", "solution"),
    )
    assert cfg.dpo_extract_mode in ("both", "chosen", "rejected"), (
        f"Invalid dpo_extract_mode: {cfg.dpo_extract_mode}"
    )
    return cfg


# Module-level config — lazy so that test imports don't require a real YAML.
_CFG: Optional[ChunkingConfig] = None


def get_cfg() -> ChunkingConfig:
    """Return (and cache) the module-level chunking config."""
    global _CFG
    if _CFG is None:
        _CFG = load_chunking_config()
    return _CFG


# ---------------------------------------------------------------------------
# Text sanitization
# ---------------------------------------------------------------------------

# ASCII / C1 control chars to strip (keep \t \n \r)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Collapse runs of U+FFFD (replacement character)
_REPLACEMENT_RE = re.compile(r"\ufffd{2,}")


def sanitize_text(text):
    """
    Clean a text string for safe downstream processing.

    * Removes surrogates that survived JSON decoding.
    * Strips null bytes and ASCII / C1 control characters (keeps tab, LF, CR).
    * Collapses consecutive U+FFFD into a single instance.
    * NFC-normalises the result.

    Returns ``""`` for non-string / None input.
    """
    if not isinstance(text, str):
        return ""

    # Round-trip to kill surrogates
    text = text.encode("utf-8", errors="surrogatepass").decode(
        "utf-8", errors="replace"
    )
    text = _CONTROL_RE.sub("", text)
    text = _REPLACEMENT_RE.sub("\ufffd", text)
    text = unicodedata.normalize("NFC", text)
    return text


# ---------------------------------------------------------------------------
# Quality / garbage detection
# ---------------------------------------------------------------------------


def compression_ratio(text):
    """
    Compute the zlib compression ratio of *text*.

    Returns ``compressed_size / raw_size`` (a value in roughly 0.01–1.0).

    * **Very low** (< 0.05) → extremely repetitive (e.g. same phrase 100×).
    * **Very high** (> 0.95) → random/encrypted bytes, incompressible junk.
    * **Normal text** → 0.15–0.65, regardless of language or content type.

    Returns 0.0 for empty input.
    """
    raw = text.encode("utf-8") if isinstance(text, str) else b""
    if not raw:
        return 0.0
    compressed = zlib.compress(raw, level=1)   # level=1 is fastest
    return len(compressed) / len(raw)


def alpha_ratio(text):
    """Fraction of non-whitespace characters that are alphabetic."""
    non_ws = [c for c in text if not c.isspace()]
    return sum(c.isalpha() for c in non_ws) / len(non_ws) if non_ws else 0.0


def is_garbage(text):
    """
    Return ``True`` only for **total** garbage — text that has no
    informational value whatsoever.

    Uses two principled signals:

    1. **Compression ratio** (zlib) — a zero-model entropy proxy.
       Catches both extremes: mindless repetition and random byte soup.
       Only applied to texts >= 200 bytes (zlib header overhead makes
       short texts look artificially incompressible).
    2. **Alpha ratio** — fraction of non-whitespace chars that are letters.
       Catches binary/symbol junk that zlib might still compress normally.

    Thresholds are intentionally very permissive.  "Somewhat garbage" passes.
    """
    if not text or not text.strip():
        return True

    ar = alpha_ratio(text)

    # Almost no letters at all — pure symbol / number soup
    if ar < 0.10:
        return True

    # Compression ratio is only meaningful above ~200 bytes;
    # below that the zlib header dominates and everything looks "random".
    raw_len = len(text.encode("utf-8"))
    if raw_len >= 200:
        cr = compression_ratio(text)

        # Insanely repetitive — same short phrase looped dozens of times
        if cr < 0.05:
            return True

        # Incompressible random bytes / encrypted / binary data
        if cr > 0.95:
            return True

    return False


# ---------------------------------------------------------------------------
# Content-aware document chunking
# ---------------------------------------------------------------------------

# Keywords that begin a code statement (at the start of a stripped line)
_CODE_KEYWORDS = (
    "def ", "class ", "function ", "import ", "from ", "#include",
    "package ", "public ", "private ", "protected ", "static ",
    "var ", "let ", "const ", "return ", "async def ", "async function ",
    "yield ", "raise ", "throw ", "elif ", "else:", "try:", "except ",
    "catch ", "finally:", "with ", "SELECT ", "INSERT ", "CREATE ",
    "#!/", ">>>",
)

# Line endings typical of code
_CODE_ENDINGS = ("{", "}", ");", "};", "],", "),")

# Line prefixes typical of code comments
_CODE_COMMENT_PREFIXES = ("//", "/*", "* ", "*/", "#!")

# LaTeX / math keywords searched in the full block text
_MATH_KEYWORDS = (
    "\\begin{", "\\end{", "\\frac", "\\int", "\\sum", "\\prod",
    "\\lim", "\\infty", "\\partial", "\\nabla", "\\sqrt", "\\over",
    "\\left", "\\right", "\\text{", "\\mathrm", "\\mathbb", "\\mathcal",
    "\\proof", "\\theorem", "\\lemma", "\\corollary", "\\proposition",
    "\\equation", "\\align", "\\alpha", "\\beta", "\\gamma", "\\theta",
    "\\delta", "\\epsilon", "\\lambda", "\\sigma", "\\phi", "\\omega",
    "\\forall", "\\exists", "\\in ", "\\subset", "\\cup", "\\cap",
    "\\rightarrow", "\\Rightarrow", "\\leq", "\\geq", "\\neq",
)


def classify_block(text):
    """
    Classify a text block as ``'code'``, ``'math'``, or ``'prose'``.

    Uses lightweight heuristics (keyword density, indentation patterns,
    LaTeX marker counts) — good enough for chunking decisions, not a
    language detector.

    Math is checked **before** code because LaTeX braces ``{``/``}`` are
    a common false positive for the code heuristic.
    """
    lines = text.split("\n")
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        return "prose"

    # --- Math (check first — LaTeX {/} looks like code) ---
    math_hits = sum(text.count(kw) for kw in _MATH_KEYWORDS)
    if "$$" in text:
        math_hits += 2
    if math_hits >= 3:
        return "math"

    # --- Code ---
    n = len(non_empty)
    code_score = 0.0

    for line in non_empty:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        if any(stripped.startswith(kw) for kw in _CODE_KEYWORDS):
            code_score += 1
        elif stripped.endswith(_CODE_ENDINGS):
            code_score += 0.5
        elif any(stripped.startswith(cp) for cp in _CODE_COMMENT_PREFIXES):
            code_score += 1
        # Indented >=4 spaces, but not a markdown list / blockquote
        elif indent >= 4 and stripped[0:1] not in ("-", "*", ">", "|", "+"):
            code_score += 0.5

    if code_score / n > 0.3:
        return "code"

    return "prose"


def split_code_block(text):
    """
    Split a code block at top-level definition boundaries.

    Looks for unindented ``def`` / ``class`` / ``function`` lines and splits
    just before them.  Falls back to double-newline splitting if no such
    boundaries are found.
    """
    top_level_starts = (
        "def ", "class ", "function ", "async def ", "async function ",
        "public ", "private ", "protected ",
    )
    lines = text.split("\n")
    chunks = []
    current = []

    for line in lines:
        is_boundary = (
            not line.startswith((" ", "\t"))
            and any(line.startswith(kw) for kw in top_level_starts)
        )
        if is_boundary and current:
            block = "\n".join(current).strip()
            if block:
                chunks.append(block)
            current = [line]
        else:
            current.append(line)

    if current:
        block = "\n".join(current).strip()
        if block:
            chunks.append(block)

    if len(chunks) > 1:
        return chunks

    # Fallback: split on double-newlines
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return parts if parts else [text.strip()]


def split_math_block(text):
    """
    Split math / LaTeX content on double-newlines **without** breaking
    ``\\begin{…}…\\end{…}`` environments.

    Paragraphs inside an open environment are merged back together.
    """
    raw_parts = text.split("\n\n")
    chunks = []
    pending = []
    depth = 0

    for part in raw_parts:
        pending.append(part)
        depth += part.count("\\begin{") - part.count("\\end{")
        depth = max(0, depth)  # don't go negative on malformed input

        if depth == 0:
            merged = "\n\n".join(pending).strip()
            if merged:
                chunks.append(merged)
            pending = []

    # Flush anything left inside an unclosed environment
    if pending:
        merged = "\n\n".join(pending).strip()
        if merged:
            chunks.append(merged)

    return chunks if chunks else [text.strip()]


def _split_respecting_environments(text):
    """
    Split *text* into paragraphs on blank lines, but **never** inside a
    ``\\begin{…}…\\end{…}`` LaTeX environment.

    Tracks nesting depth so that blank lines inside environments are
    preserved rather than treated as paragraph breaks.
    """
    paragraphs = []
    current = []
    depth = 0

    for line in text.split("\n"):
        depth += line.count("\\begin{") - line.count("\\end{")
        depth = max(0, depth)

        if line.strip() == "" and depth == 0:
            # Blank line outside any environment — paragraph break
            if current:
                block = "\n".join(current).strip()
                if block:
                    paragraphs.append(block)
                current = []
        else:
            current.append(line)

    if current:
        block = "\n".join(current).strip()
        if block:
            paragraphs.append(block)

    return paragraphs


def chunk_document(text):
    """
    Split a document into semantically coherent chunks.

    Returns a list of ``(chunk_text, content_type)`` tuples where
    *content_type* is ``'code'``, ``'math'``, or ``'prose'``.

    1. Splits on blank lines **respecting** ``\\begin…\\end`` environments
       (blank lines inside LaTeX environments do not cause splits).
    2. Classifies each segment as code / math / prose.
    3. Merges adjacent code segments back together (blank lines inside
       functions are normal in code).
    4. Re-splits merged code blocks at top-level ``def`` / ``class``
       boundaries.  Prose and math segments are kept as-is.
    """
    segments = _split_respecting_environments(text)

    # Classify each segment and merge adjacent code blocks
    groups = []  # [(type, [seg_text, …]), …]
    for seg in segments:
        btype = classify_block(seg)
        if groups and groups[-1][0] == btype and btype == "code":
            groups[-1][1].append(seg)
        else:
            groups.append((btype, [seg]))

    # Type-appropriate re-splitting, preserving the content_type label
    chunks = []
    for btype, paras in groups:
        if btype == "code":
            for c in split_code_block("\n\n".join(paras)):
                chunks.append((c, "code"))
        else:
            for p in paras:
                chunks.append((p, btype))

    return chunks


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def open_data_file(file_path):
    """
    Context manager — opens a data file with the right decompression.

    Supports ``.jsonl``, ``.jsonl.zst`` / ``.json.zst``, ``.json.gz`` / ``.jsonl.gz``.
    Yields a **text-mode** file object with ``errors='replace'``.
    """
    if file_path.endswith(".zst"):
        fh = open(file_path, "rb")
        try:
            dctx = zstd.ZstdDecompressor()
            reader = dctx.stream_reader(fh)
            wrapper = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            yield wrapper
        finally:
            fh.close()
    elif file_path.endswith(".gz"):
        with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as f:
            yield f
    else:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            yield f


def read_parquet_as_jsonl(file_path):
    """Stream rows from a Parquet file via DuckDB, yielding JSON strings."""
    safe_path = file_path.replace("'", "''")
    con = duckdb.connect()
    try:
        cols = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{safe_path}')"
        ).fetchall()
        col_names = [c[0] for c in cols]

        batch_size = 10_000
        offset = 0
        while True:
            rows = con.execute(
                f"SELECT * FROM read_parquet('{safe_path}') "
                f"LIMIT {batch_size} OFFSET {offset}"
            ).fetchall()
            if not rows:
                break
            for row in rows:
                yield json.dumps(dict(zip(col_names, row)), default=str)
            offset += batch_size
    finally:
        con.close()


def find_data_files(data_dir):
    """Return a sorted list of supported data files under *data_dir*."""
    assert os.path.isdir(data_dir), f"Data directory does not exist: {data_dir}"
    found = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith(SUPPORTED_EXTENSIONS):
                found.append(os.path.join(root, fname))
    found.sort()
    assert found, (
        f"No data files ({', '.join(SUPPORTED_EXTENSIONS)}) found in {data_dir}"
    )
    return found


def read_data_chunks(files, chunk_size):
    """
    Generator — reads lines from *files* and yields them in chunks of
    ``(line_string, source_filename)`` tuples.
    """
    chunk = []
    for file_path in files:
        source = os.path.basename(file_path)
        try:
            if file_path.endswith(".parquet"):
                for line in read_parquet_as_jsonl(file_path):
                    chunk.append((line, source))
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
            else:
                with open_data_file(file_path) as f:
                    for line in f:
                        chunk.append((line, source))
                        if len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)
    if chunk:
        yield chunk


# ---------------------------------------------------------------------------
# Category helpers
# ---------------------------------------------------------------------------


def extract_category(file_path):
    """
    Extract the data-source category from a file path.

    Returns the base category prefix (e.g. ``common_crawl``, ``wiki_to_rcqa``).
    Falls back to ``'unknown'``.
    """
    for part in Path(file_path).parts:
        if part.startswith(KNOWN_CATEGORY_PREFIXES):
            return part.split("-")[0] if "-" in part else part
    return "unknown"


def build_category_inventory(files):
    """Group file paths by category.  Returns ``{category: [paths]}``."""
    inventory = {}
    for fp in files:
        cat = extract_category(fp)
        inventory.setdefault(cat, []).append(fp)
    return inventory


# ---------------------------------------------------------------------------
# Text extraction  (explicit parameters — no hidden config dependency)
# ---------------------------------------------------------------------------


def extract_conversation_text(conversation_data, user_role="user",
                              assistant_role="assistant", content_field="content"):
    """
    Pull the first user prompt + first assistant response from conversation data.

    *conversation_data* may be a list of ``{role, content}`` dicts **or** a single
    dict with common field names (``prompt``/``response``, ``user``/``assistant``, etc.).

    Returns ``"Prompt: …\\n\\nResponse: …"`` or ``None``.
    """
    try:
        if isinstance(conversation_data, list):
            user_content = assistant_content = None
            for msg in conversation_data:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                content = msg.get(content_field, "")
                if role == user_role and user_content is None:
                    user_content = content
                elif role == assistant_role and assistant_content is None:
                    assistant_content = content
            if user_content and assistant_content:
                return f"Prompt: {user_content}\n\nResponse: {assistant_content}"

        elif isinstance(conversation_data, dict):
            user_content = (
                conversation_data.get("prompt")
                or conversation_data.get("user")
                or conversation_data.get("question")
            )
            assistant_content = (
                conversation_data.get("response")
                or conversation_data.get("assistant")
                or conversation_data.get("answer")
            )
            if user_content and assistant_content:
                return f"Prompt: {user_content}\n\nResponse: {assistant_content}"
    except Exception:
        pass
    return None


def extract_dpo_conversations(data, extract_mode="both",
                              chosen_field="chosen", rejected_field="rejected",
                              **conv_kw):
    """
    Extract conversations from DPO preference data.

    Returns list of ``(text, label)`` where label is ``'chosen'`` or ``'rejected'``.
    *conv_kw* is forwarded to :func:`extract_conversation_text`.
    """
    results = []
    try:
        if extract_mode in ("both", "chosen"):
            chosen = data.get(chosen_field, [])
            if chosen:
                text = extract_conversation_text(chosen, **conv_kw)
                if text:
                    results.append((text, "chosen"))
        if extract_mode in ("both", "rejected"):
            rejected = data.get(rejected_field, [])
            if rejected:
                text = extract_conversation_text(rejected, **conv_kw)
                if text:
                    results.append((text, "rejected"))
    except Exception:
        pass
    return results


def extract_rl_conversation(data, prompt_field="prompt", solution_field="solution"):
    """
    Extract prompt + solution from RL-format data.

    Returns ``"Prompt: …\\n\\nResponse: …"`` or ``None``.
    """
    try:
        prompt = data.get(prompt_field, "")
        solution = data.get(solution_field, "")
        if prompt and solution:
            return f"Prompt: {prompt}\n\nResponse: {solution}"
        if prompt:
            return f"Prompt: {prompt}"
    except Exception:
        pass
    return None


def parse_record(data, cfg):
    """
    Parse a single JSON record into ``[(text, label, content_type), ...]``.

    * *label* is ``None`` except in DPO mode (``'chosen'`` / ``'rejected'``).
    * *content_type* is ``'code'``, ``'math'``, ``'prose'``, or
      ``'conversation'`` (for conversation / DPO / RL modes).
    """
    conv_kw = dict(
        user_role=cfg.user_role,
        assistant_role=cfg.assistant_role,
        content_field=cfg.content_field,
    )

    if cfg.mode == "conversation":
        conversation = (
            data if isinstance(data, list)
            else data.get("messages") or data.get("conversation") or data
        )
        text = extract_conversation_text(conversation, **conv_kw)
        return [(text, None, "conversation")] if text else []

    if cfg.mode == "dpo":
        dpo_results = extract_dpo_conversations(
            data,
            extract_mode=cfg.dpo_extract_mode,
            chosen_field=cfg.dpo_chosen_field,
            rejected_field=cfg.dpo_rejected_field,
            **conv_kw,
        )
        return [(text, label, "conversation") for text, label in dpo_results]

    if cfg.mode == "rl":
        text = extract_rl_conversation(
            data,
            prompt_field=cfg.rl_prompt_field,
            solution_field=cfg.rl_solution_field,
        )
        return [(text, None, "conversation")] if text else []

    # paragraph mode (default) — chunk_document returns (text, content_type)
    raw = data.get("text")
    if not raw or not isinstance(raw, str):
        return []
    return [(p, None, ctype) for p, ctype in chunk_document(raw)]


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------

_worker_tokenizer = None


def _init_worker():
    """Initialise the tiktoken encoder in each worker process."""
    global _worker_tokenizer
    _worker_tokenizer = tiktoken.get_encoding(get_cfg().tokenizer_encoding)


def process_line_chunk(chunk):
    """
    Worker function — process a chunk of ``(line, source_name)`` pairs.

    Returns ``(list[dict], total_filtered_tokens)``.
    """
    global _worker_tokenizer
    assert _worker_tokenizer is not None, "Worker tokenizer not initialised"
    cfg = get_cfg()

    paragraphs = []
    garbage = []
    total_tokens = 0

    for line, source_name in chunk:
        try:
            data = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue

        for raw_text, label, content_type in parse_record(data, cfg):
            text = sanitize_text(raw_text).strip()
            if not text:
                continue

            n_tokens = len(_worker_tokenizer.encode(text, disallowed_special=()))
            if not (cfg.min_tokens <= n_tokens <= cfg.max_tokens):
                continue

            entry = {
                "id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text": text,
                "source": source_name,
                "token_size": n_tokens,
                "content_type": content_type,
            }
            if label:
                entry["dpo_label"] = label

            # Route garbage to a separate bin
            if cfg.quality_filter and is_garbage(text):
                garbage.append(entry)
            else:
                total_tokens += n_tokens
                paragraphs.append(entry)

    return paragraphs, garbage, total_tokens


# ---------------------------------------------------------------------------
# Reservoir sampler
# ---------------------------------------------------------------------------


class ReservoirSampler:
    """Vitter's Algorithm R — uniform reservoir sampling."""

    def __init__(self, capacity):
        assert capacity > 0, "Reservoir capacity must be positive"
        self.capacity = capacity
        self.reservoir = []
        self.seen = 0

    def add(self, item):
        self.seen += 1
        if len(self.reservoir) < self.capacity:
            self.reservoir.append(item)
        else:
            j = random.randint(0, self.seen - 1)
            if j < self.capacity:
                self.reservoir[j] = item

    def get_sample(self):
        return list(self.reservoir)

    def __len__(self):
        return self.seen


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_stratified_processing(files, cfg):
    """
    Process all files with a multiprocessing pool, accumulating results
    into per-category reservoir samplers.

    Returns ``(category_samplers, all_garbage, category_tokens)``.
    """
    inventory = build_category_inventory(files)
    print(f"Found {len(inventory)} categories:")
    for cat, cat_files in sorted(inventory.items()):
        print(f"  - {cat}: {len(cat_files)} files")

    target_cap = cfg.sample_size if cfg.sample_size else 10_000_000
    samplers = {cat: ReservoirSampler(target_cap) for cat in inventory}
    tokens = {cat: 0 for cat in inventory}
    all_garbage = []

    print(f"\nProcessing with {cfg.num_workers} workers...")
    with multiprocessing.Pool(cfg.num_workers, initializer=_init_worker) as pool:
        for cat, cat_files in inventory.items():
            print(f"\n--- {cat} ({len(cat_files)} files) ---")
            chunks = read_data_chunks(cat_files, cfg.chunk_size)
            pbar = tqdm(
                pool.imap_unordered(process_line_chunk, chunks),
                desc=f"  {cat}",
                unit="chunk",
            )
            for batch, batch_garbage, batch_tokens in pbar:
                for entry in batch:
                    entry["category"] = cat
                    samplers[cat].add(entry)
                for entry in batch_garbage:
                    entry["category"] = cat
                    all_garbage.append(entry)
                tokens[cat] += batch_tokens
                pbar.set_postfix_str(f"seen={len(samplers[cat]):,}")

    return samplers, all_garbage, tokens


def combine_stratified_samples(samplers, tokens, cfg):
    """
    Draw a proportionally-stratified sample from per-category reservoirs.

    Returns the final list of sample dicts.
    """
    total_seen = sum(len(s) for s in samplers.values())
    assert total_seen > 0, "No paragraphs extracted — check data and config"

    if cfg.sample_size:
        target = cfg.sample_size
    else:
        target = int(total_seen * cfg.sample_percentage)
        print(
            f"Computed target: {target:,} "
            f"({cfg.sample_percentage * 100}% of {total_seen:,})"
        )

    final = []
    print("\nCategory breakdown:")
    for cat, sampler in sorted(samplers.items()):
        weight = len(sampler) / total_seen
        n = min(int(target * weight), len(sampler.get_sample()))
        if n > 0:
            sampled = random.sample(sampler.get_sample(), n)
            final.extend(sampled)
        print(
            f"  {cat}: {len(sampler):,} seen, "
            f"weight={weight:.4f}, sampled={n:,}, "
            f"tokens={tokens.get(cat, 0):,}"
        )

    assert final, "Final sample is empty after stratified combination"
    return final


def save_results(sample, output_path):
    """Write the sample to a JSONL file."""
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in sample:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Sanity-check: re-read first line to make sure it round-trips
    with open(output_path, "r", encoding="utf-8") as f:
        first = f.readline()
        assert first.strip(), "Output file is empty after write"
        parsed = json.loads(first)
        assert "id" in parsed and "text" in parsed, (
            "First output record missing required fields"
        )

    print(f"Saved {len(sample):,} entries to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cfg = get_cfg()

    print(f"{'=' * 72}")
    print("Stage 02 — Chunk & Sample")
    print(f"{'=' * 72}")
    print(f"Mode:       {cfg.mode}")
    print(f"Data dir:   {cfg.data_dir}")
    print(f"Output:     {cfg.output_file}")
    print(f"Tokens:     {cfg.min_tokens}–{cfg.max_tokens}")
    if cfg.sample_size:
        print(f"Sample:     {cfg.sample_size:,} (fixed)")
    else:
        print(f"Sample:     {cfg.sample_percentage * 100}%")

    files = find_data_files(cfg.data_dir)
    print(f"\nFound {len(files)} data files")

    samplers, all_garbage, tokens = run_stratified_processing(files, cfg)

    total_tokens = sum(tokens.values())
    total_seen = sum(len(s) for s in samplers.values())
    print(f"\nTotal paragraphs seen: {total_seen:,}")
    print(f"Total filtered tokens: {total_tokens:,}")
    print(f"Total garbage chunks:  {len(all_garbage):,}")

    sample = combine_stratified_samples(samplers, tokens, cfg)
    print(f"Final sample size:     {len(sample):,}")

    save_results(sample, cfg.output_file)

    # Save garbage to a sidecar file
    if all_garbage:
        garbage_path = cfg.output_file.replace(".jsonl", "_garbage.jsonl")
        save_results(all_garbage, garbage_path)
        print(f"Garbage saved to:      {garbage_path}")

    print("Done.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
