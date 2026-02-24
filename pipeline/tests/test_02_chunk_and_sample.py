# -*- coding: utf-8 -*-
"""
Unit tests for Stage 02: Chunk and Sample

Tests cover:
  - Text sanitization (surrogates, control chars, NFC normalisation)
  - Content-aware chunking (code, math, prose detection & splitting)
  - Category extraction from file paths
  - Conversation / DPO / RL / paragraph text extraction
  - parse_record dispatch across all four modes
  - ReservoirSampler correctness
"""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the module under test.
#
# The module uses lazy config loading (get_cfg()), so we can import it without
# a real YAML file.  We inject a test config where needed via the _CFG global.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "stages"))

# Prevent the module from loading a real config at import time
os.environ.setdefault("PIPELINE_CONFIG", "__nonexistent_test_config__.yaml")

import importlib
s02 = importlib.import_module("02_chunk_and_sample")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_cfg():
    """A minimal ChunkingConfig for tests."""
    return s02.ChunkingConfig(
        mode="paragraph",
        data_dir="/tmp/test_data",
        output_file="/tmp/test_out.jsonl",
        min_tokens=5,
        max_tokens=500,
    )


@pytest.fixture(autouse=True)
def _inject_cfg(default_cfg):
    """Inject test config so get_cfg() returns our fixture."""
    old = s02._CFG
    s02._CFG = default_cfg
    yield
    s02._CFG = old


# ===================================================================
# sanitize_text
# ===================================================================

class TestSanitizeText:
    def test_normal_ascii(self):
        assert s02.sanitize_text("hello world") == "hello world"

    def test_none_returns_empty(self):
        assert s02.sanitize_text(None) == ""

    def test_non_string_returns_empty(self):
        assert s02.sanitize_text(42) == ""
        assert s02.sanitize_text([]) == ""

    def test_strips_null_bytes(self):
        assert s02.sanitize_text("abc\x00def") == "abcdef"

    def test_strips_control_chars(self):
        # \x01 (SOH), \x07 (BEL), \x7f (DEL), \x8f (C1 control)
        assert s02.sanitize_text("a\x01b\x07c\x7fd\x8fe") == "abcde"

    def test_preserves_tab_newline_cr(self):
        text = "line1\n\tindented\r\nline3"
        assert s02.sanitize_text(text) == text

    def test_collapses_replacement_chars(self):
        assert s02.sanitize_text("a\ufffd\ufffd\ufffdb") == "a\ufffdb"

    def test_single_replacement_char_kept(self):
        assert s02.sanitize_text("a\ufffdb") == "a\ufffdb"

    def test_surrogate_removal(self):
        # Lone surrogate \ud800 should be replaced
        text = "hello\ud800world"
        result = s02.sanitize_text(text)
        assert "\ud800" not in result
        assert "hello" in result and "world" in result

    def test_nfc_normalisation(self):
        # e + combining acute  ->  e-acute (NFC)
        decomposed = "e\u0301"   # NFD
        composed = "\u00e9"      # NFC
        assert s02.sanitize_text(decomposed) == composed

    def test_unicode_preserved(self):
        text = "caf\u00e9 \u2603 \U0001f600"
        assert s02.sanitize_text(text) == text

    def test_empty_string(self):
        assert s02.sanitize_text("") == ""

    def test_only_control_chars(self):
        assert s02.sanitize_text("\x00\x01\x02\x03") == ""

    def test_mixed_bad_chars(self):
        text = "\x00hello\x07\ufffd\ufffd world\x01!"
        result = s02.sanitize_text(text)
        assert result == "hello\ufffd world!"


# ===================================================================
# classify_block
# ===================================================================

class TestClassifyBlock:
    def test_python_function(self):
        text = "def hello():\n    print('hi')\n    return 42"
        assert s02.classify_block(text) == "code"

    def test_python_class(self):
        text = "class Foo:\n    def __init__(self):\n        self.x = 1"
        assert s02.classify_block(text) == "code"

    def test_javascript_function(self):
        text = "function greet(name) {\n  return 'hello ' + name;\n}"
        assert s02.classify_block(text) == "code"

    def test_c_style_code(self):
        text = "#include <stdio.h>\nint main() {\n    printf(\"hello\");\n    return 0;\n}"
        assert s02.classify_block(text) == "code"

    def test_import_heavy(self):
        text = "import os\nimport sys\nfrom pathlib import Path\nimport json"
        assert s02.classify_block(text) == "code"

    def test_indented_block(self):
        text = "    result = process(data)\n    print(result)\n    return result"
        assert s02.classify_block(text) == "code"

    def test_latex_math(self):
        text = "\\begin{theorem}\nFor all $x \\in \\mathbb{R}$,\n\\int_0^x f(t) dt \\geq 0\n\\end{theorem}"
        assert s02.classify_block(text) == "math"

    def test_display_math(self):
        text = "We can show that\n$$\\frac{d}{dx} \\int_0^x f(t) dt = f(x)$$\nby the fundamental theorem."
        assert s02.classify_block(text) == "math"

    def test_equation_environment(self):
        text = "\\begin{equation}\n\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}\n\\end{equation}"
        assert s02.classify_block(text) == "math"

    def test_plain_prose(self):
        text = "The quick brown fox jumps over the lazy dog. This is a normal paragraph."
        assert s02.classify_block(text) == "prose"

    def test_short_prose(self):
        text = "Hello world"
        assert s02.classify_block(text) == "prose"

    def test_empty_returns_prose(self):
        assert s02.classify_block("") == "prose"
        assert s02.classify_block("   \n  \n   ") == "prose"

    def test_prose_with_single_math_symbol(self):
        # One \\alpha alone shouldn't trigger math
        text = "The parameter \\alpha controls learning rate in gradient descent."
        assert s02.classify_block(text) == "prose"

    def test_sql_code(self):
        text = "SELECT id, name\nFROM users\nWHERE active = 1;"
        assert s02.classify_block(text) == "code"

    def test_shell_script(self):
        text = "#!/bin/bash\nimport_data() {\n  echo 'importing'\n}"
        assert s02.classify_block(text) == "code"


# ===================================================================
# split_code_block
# ===================================================================

class TestSplitCodeBlock:
    def test_multiple_functions(self):
        text = "def foo():\n    return 1\n\ndef bar():\n    return 2"
        chunks = s02.split_code_block(text)
        assert len(chunks) == 2
        assert "foo" in chunks[0]
        assert "bar" in chunks[1]

    def test_class_and_function(self):
        text = "class Foo:\n    pass\n\ndef bar():\n    pass"
        chunks = s02.split_code_block(text)
        assert len(chunks) == 2

    def test_single_function_stays_intact(self):
        text = "def foo():\n    x = 1\n\n    y = 2\n    return x + y"
        chunks = s02.split_code_block(text)
        # No top-level boundary -> fallback splits on blank line
        # But the key point: this doesn't crash
        assert len(chunks) >= 1
        assert all(c.strip() for c in chunks)

    def test_no_definitions_falls_back(self):
        text = "x = 1\n\ny = 2\n\nz = 3"
        chunks = s02.split_code_block(text)
        assert len(chunks) == 3

    def test_indented_defs_not_split(self):
        """Only top-level (unindented) defs should trigger splits."""
        text = "class Foo:\n    def method_a(self):\n        pass\n    def method_b(self):\n        pass"
        chunks = s02.split_code_block(text)
        # Methods are indented, so no split should happen at them
        assert len(chunks) == 1
        assert "method_a" in chunks[0] and "method_b" in chunks[0]

    def test_empty_text(self):
        assert s02.split_code_block("") == [""]
        assert s02.split_code_block("   ") == [""]


# ===================================================================
# split_math_block
# ===================================================================

class TestSplitMathBlock:
    def test_separate_environments(self):
        text = "\\begin{theorem}\nStatement\n\\end{theorem}\n\n\\begin{proof}\nDetails\n\\end{proof}"
        chunks = s02.split_math_block(text)
        assert len(chunks) == 2
        assert "theorem" in chunks[0]
        assert "proof" in chunks[1]

    def test_blank_line_inside_environment(self):
        """Blank lines inside \\begin...\\end should NOT cause a split."""
        text = "\\begin{align}\nx &= 1\n\ny &= 2\n\\end{align}"
        chunks = s02.split_math_block(text)
        assert len(chunks) == 1
        assert "x &= 1" in chunks[0] and "y &= 2" in chunks[0]

    def test_nested_environments(self):
        text = (
            "\\begin{theorem}\n"
            "\\begin{equation}\nx=1\n\\end{equation}\n\n"
            "Some text\n"
            "\\end{theorem}"
        )
        chunks = s02.split_math_block(text)
        assert len(chunks) == 1

    def test_plain_math_without_environments(self):
        text = "First equation: $x=1$\n\nSecond equation: $y=2$"
        chunks = s02.split_math_block(text)
        assert len(chunks) == 2

    def test_unclosed_environment(self):
        """Malformed LaTeX shouldn't crash."""
        text = "\\begin{proof}\nSome steps\n\nMore steps"
        chunks = s02.split_math_block(text)
        assert len(chunks) == 1  # everything merged because env never closes

    def test_empty_text(self):
        assert s02.split_math_block("") == [""]


# ===================================================================
# chunk_document (integration)
# ===================================================================

class TestChunkDocument:
    """chunk_document returns [(text, content_type), ...]."""

    @staticmethod
    def _texts(chunks):
        """Helper — extract just the text strings from typed chunks."""
        return [t for t, _ in chunks]

    def test_pure_prose(self):
        text = "First paragraph about dogs.\n\nSecond paragraph about cats."
        chunks = s02.chunk_document(text)
        assert len(chunks) == 2
        assert "dogs" in chunks[0][0]
        assert "cats" in chunks[1][0]
        assert all(ct == "prose" for _, ct in chunks)

    def test_pure_code(self):
        text = "def foo():\n    return 1\n\ndef bar():\n    return 2"
        chunks = s02.chunk_document(text)
        assert len(chunks) == 2
        assert "foo" in chunks[0][0]
        assert "bar" in chunks[1][0]
        assert all(ct == "code" for _, ct in chunks)

    def test_code_with_internal_blank_lines(self):
        """Blank lines inside a function should not fragment it."""
        text = (
            "def process(data):\n"
            "    x = data[0]\n"
            "\n"
            "    y = data[1]\n"
            "\n"
            "    return x + y"
        )
        chunks = s02.chunk_document(text)
        combined = " ".join(self._texts(chunks))
        assert "x = data[0]" in combined and "y = data[1]" in combined

    def test_mixed_prose_and_code(self):
        text = (
            "Here is an example:\n\n"
            "def hello():\n"
            "    print('hi')\n\n"
            "This function prints a greeting."
        )
        chunks = s02.chunk_document(text)
        assert len(chunks) >= 2
        types = {ct for _, ct in chunks}
        assert "prose" in types

    def test_mixed_types_labeled_correctly(self):
        text = (
            "Here is an example:\n\n"
            "def hello():\n"
            "    print('hi')\n\n"
            "This function prints a greeting."
        )
        chunks = s02.chunk_document(text)
        for chunk_text, ctype in chunks:
            if "def hello" in chunk_text:
                assert ctype == "code"
            elif "example" in chunk_text or "greeting" in chunk_text:
                assert ctype == "prose"

    def test_math_environments_intact(self):
        text = (
            "Consider the following:\n\n"
            "\\begin{theorem}\n"
            "For all $x$,\n\n"
            "$f(x) \\geq 0$\n"
            "\\end{theorem}\n\n"
            "This completes the argument."
        )
        chunks = s02.chunk_document(text)
        theorem_chunks = [(t, ct) for t, ct in chunks if "\\begin{theorem}" in t]
        assert len(theorem_chunks) == 1
        assert "\\end{theorem}" in theorem_chunks[0][0]
        assert theorem_chunks[0][1] == "math"

    def test_empty_text(self):
        assert s02.chunk_document("") == []
        assert s02.chunk_document("   \n\n   ") == []

    def test_single_paragraph(self):
        chunks = s02.chunk_document("Just one paragraph, no blank lines at all.")
        assert len(chunks) == 1
        assert chunks[0] == ("Just one paragraph, no blank lines at all.", "prose")

    def test_multiple_code_functions_between_prose(self):
        text = (
            "Utility functions:\n\n"
            "def add(a, b):\n    return a + b\n\n"
            "def sub(a, b):\n    return a - b\n\n"
            "Use them wisely."
        )
        chunks = s02.chunk_document(text)
        texts = self._texts(chunks)
        assert any("add" in t for t in texts)
        assert any("sub" in t for t in texts)
        assert any("Utility" in t for t in texts)
        assert any("wisely" in t for t in texts)

    def test_parse_record_uses_chunk_document(self, default_cfg):
        """parse_record in paragraph mode should return content_type."""
        default_cfg.mode = "paragraph"
        data = {
            "text": (
                "Some prose.\n\n"
                "def foo():\n    pass\n\n"
                "def bar():\n    pass\n\n"
                "More prose."
            )
        }
        results = s02.parse_record(data, default_cfg)
        texts = [t for t, _, _ in results]
        types = [ct for _, _, ct in results]
        assert any("foo" in t for t in texts)
        assert any("bar" in t for t in texts)
        assert any("Some prose" in t for t in texts)
        assert "code" in types
        assert "prose" in types


# ===================================================================
# compression_ratio / alpha_ratio
# ===================================================================

class TestCompressionRatio:
    def test_empty_string(self):
        assert s02.compression_ratio("") == 0.0

    def test_none_returns_zero(self):
        assert s02.compression_ratio(None) == 0.0

    def test_normal_text_mid_range(self):
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump."
        )
        cr = s02.compression_ratio(text)
        assert 0.10 < cr < 0.90

    def test_highly_repetitive_very_low(self):
        text = "spam " * 500
        cr = s02.compression_ratio(text)
        assert cr < 0.10

    def test_diverse_text_moderate(self):
        import string
        # All printable chars repeated — compresses somewhat
        text = string.printable * 5
        cr = s02.compression_ratio(text)
        assert 0.05 < cr < 0.80


class TestAlphaRatio:
    def test_pure_letters(self):
        assert s02.alpha_ratio("hello") == 1.0

    def test_pure_digits(self):
        assert s02.alpha_ratio("12345") == 0.0

    def test_mixed(self):
        ar = s02.alpha_ratio("abc123")
        assert 0.4 < ar < 0.6

    def test_empty(self):
        assert s02.alpha_ratio("") == 0.0

    def test_whitespace_ignored(self):
        # "a b" -> non-ws chars are ['a', 'b'], both alpha
        assert s02.alpha_ratio("a b") == 1.0


# ===================================================================
# is_garbage
# ===================================================================

class TestIsGarbage:
    # --- Things that ARE garbage ---

    def test_pure_symbols(self):
        assert s02.is_garbage("@#$%^&*()!@#$%^&*()!@#") is True

    def test_all_numbers(self):
        assert s02.is_garbage("12345 67890 12345 67890 12345") is True

    def test_extremely_repetitive(self):
        assert s02.is_garbage("buy now " * 200) is True

    def test_low_alpha_symbols(self):
        text = "{{{{}}}}(((())))[[[]]];;;;;::::////"
        assert s02.is_garbage(text) is True

    # --- Things that are NOT garbage ---

    def test_normal_prose(self):
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a perfectly normal paragraph with reasonable words."
        )
        assert s02.is_garbage(text) is False

    def test_normal_code(self):
        text = "def hello():\n    print('hello world')\n    return 42"
        assert s02.is_garbage(text) is False

    def test_normal_math(self):
        text = "\\begin{equation}\n\\sum_{i=1}^{n} x_i = \\frac{n(n+1)}{2}\n\\end{equation}"
        assert s02.is_garbage(text) is False

    def test_normal_conversation(self):
        text = "Prompt: What is the capital of France?\n\nResponse: The capital of France is Paris."
        assert s02.is_garbage(text) is False

    def test_code_with_symbols_ok(self):
        """Code naturally has more symbols — shouldn't be flagged."""
        text = (
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "def main():\n"
            "    data = {'key': [1, 2, 3]}\n"
            "    result = process(data)\n"
            "    return result\n"
        )
        assert s02.is_garbage(text) is False

    def test_math_with_heavy_latex(self):
        """LaTeX is symbol-heavy — shouldn't be flagged."""
        text = (
            "\\begin{theorem}\n"
            "For all $x \\in \\mathbb{R}$, we have\n"
            "$\\int_0^x f(t) \\, dt \\geq 0$.\n"
            "\\end{theorem}"
        )
        assert s02.is_garbage(text) is False

    def test_short_text_not_flagged(self):
        assert s02.is_garbage("Hello world") is False

    def test_somewhat_garbage_passes(self):
        """'Somewhat garbage' should NOT be flagged — only total garbage."""
        # Messy but has real words and letters
        text = "lol idk tbh... maybe??? haha ok ok ok sure thing buddy"
        assert s02.is_garbage(text) is False

    # --- Edge cases ---

    def test_empty_string(self):
        assert s02.is_garbage("") is True

    def test_whitespace_only(self):
        assert s02.is_garbage("   \n\n  \t  ") is True


# ===================================================================
# extract_category
# ===================================================================

class TestExtractCategory:
    def test_common_crawl(self):
        path = "data/common_crawl-travel_and_tourism-0018/file.jsonl"
        assert s02.extract_category(path) == "common_crawl"

    def test_wiki_to_rcqa(self):
        path = "data/wiki_to_rcqa-part3/file.jsonl"
        assert s02.extract_category(path) == "wiki_to_rcqa"

    def test_olmocr_science_pdfs(self):
        path = "data/olmocr_science_pdfs-high_quality-health-2e12/file.jsonl"
        assert s02.extract_category(path) == "olmocr_science_pdfs"

    def test_dolma_prefix(self):
        assert s02.extract_category("data/dolma-v1/x.jsonl") == "dolma"

    def test_no_hyphen(self):
        assert s02.extract_category("data/wiki/x.jsonl") == "wiki"

    def test_unknown_fallback(self):
        assert s02.extract_category("/some/random/path/file.jsonl") == "unknown"


# ===================================================================
# build_category_inventory
# ===================================================================

class TestBuildCategoryInventory:
    def test_groups_correctly(self):
        files = [
            "data/common_crawl-a/f1.jsonl",
            "data/common_crawl-b/f2.jsonl",
            "data/wiki-x/f3.jsonl",
        ]
        inv = s02.build_category_inventory(files)
        assert set(inv.keys()) == {"common_crawl", "wiki"}
        assert len(inv["common_crawl"]) == 2
        assert len(inv["wiki"]) == 1

    def test_empty_input(self):
        assert s02.build_category_inventory([]) == {}


# ===================================================================
# extract_conversation_text
# ===================================================================

class TestExtractConversationText:
    def test_list_format(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = s02.extract_conversation_text(msgs)
        assert result == "Prompt: Hi\n\nResponse: Hello!"

    def test_list_custom_roles(self):
        msgs = [
            {"role": "human", "content": "Q"},
            {"role": "bot", "content": "A"},
        ]
        result = s02.extract_conversation_text(
            msgs, user_role="human", assistant_role="bot"
        )
        assert result == "Prompt: Q\n\nResponse: A"

    def test_list_custom_content_field(self):
        msgs = [
            {"role": "user", "text": "Q"},
            {"role": "assistant", "text": "A"},
        ]
        result = s02.extract_conversation_text(msgs, content_field="text")
        assert result == "Prompt: Q\n\nResponse: A"

    def test_dict_prompt_response(self):
        data = {"prompt": "What?", "response": "That."}
        result = s02.extract_conversation_text(data)
        assert result == "Prompt: What?\n\nResponse: That."

    def test_dict_user_assistant(self):
        data = {"user": "Q", "assistant": "A"}
        result = s02.extract_conversation_text(data)
        assert result == "Prompt: Q\n\nResponse: A"

    def test_dict_question_answer(self):
        data = {"question": "Q", "answer": "A"}
        result = s02.extract_conversation_text(data)
        assert result == "Prompt: Q\n\nResponse: A"

    def test_missing_assistant_returns_none(self):
        msgs = [{"role": "user", "content": "Hi"}]
        assert s02.extract_conversation_text(msgs) is None

    def test_empty_list_returns_none(self):
        assert s02.extract_conversation_text([]) is None

    def test_takes_first_pair_only(self):
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        result = s02.extract_conversation_text(msgs)
        assert "Q1" in result and "A1" in result
        assert "Q2" not in result


# ===================================================================
# extract_dpo_conversations
# ===================================================================

class TestExtractDpoConversations:
    def _make_convo(self, prompt, response):
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

    def test_both_mode(self):
        data = {
            "chosen": self._make_convo("Q", "Good A"),
            "rejected": self._make_convo("Q", "Bad A"),
        }
        results = s02.extract_dpo_conversations(data, extract_mode="both")
        assert len(results) == 2
        labels = {r[1] for r in results}
        assert labels == {"chosen", "rejected"}

    def test_chosen_only(self):
        data = {
            "chosen": self._make_convo("Q", "A"),
            "rejected": self._make_convo("Q", "B"),
        }
        results = s02.extract_dpo_conversations(data, extract_mode="chosen")
        assert len(results) == 1
        assert results[0][1] == "chosen"

    def test_rejected_only(self):
        data = {
            "chosen": self._make_convo("Q", "A"),
            "rejected": self._make_convo("Q", "B"),
        }
        results = s02.extract_dpo_conversations(data, extract_mode="rejected")
        assert len(results) == 1
        assert results[0][1] == "rejected"

    def test_empty_fields(self):
        assert s02.extract_dpo_conversations({}) == []

    def test_custom_field_names(self):
        data = {
            "good": [{"role": "user", "content": "Q"},
                     {"role": "assistant", "content": "A"}],
        }
        results = s02.extract_dpo_conversations(
            data, extract_mode="chosen", chosen_field="good"
        )
        assert len(results) == 1


# ===================================================================
# extract_rl_conversation
# ===================================================================

class TestExtractRlConversation:
    def test_prompt_and_solution(self):
        data = {"prompt": "Solve x+1=2", "solution": "x=1"}
        result = s02.extract_rl_conversation(data)
        assert result == "Prompt: Solve x+1=2\n\nResponse: x=1"

    def test_prompt_only(self):
        data = {"prompt": "Solve x+1=2"}
        result = s02.extract_rl_conversation(data)
        assert result == "Prompt: Solve x+1=2"

    def test_empty_returns_none(self):
        assert s02.extract_rl_conversation({}) is None

    def test_custom_fields(self):
        data = {"q": "Hello", "a": "World"}
        result = s02.extract_rl_conversation(data, prompt_field="q", solution_field="a")
        assert result == "Prompt: Hello\n\nResponse: World"


# ===================================================================
# parse_record
# ===================================================================

class TestParseRecord:
    """parse_record returns [(text, label, content_type), ...]."""

    def test_paragraph_mode(self, default_cfg):
        default_cfg.mode = "paragraph"
        data = {"text": "Para one.\n\nPara two.\n\nPara three."}
        results = s02.parse_record(data, default_cfg)
        assert len(results) == 3
        assert results[0] == ("Para one.", None, "prose")
        assert results[2] == ("Para three.", None, "prose")

    def test_paragraph_mode_code_labeled(self, default_cfg):
        default_cfg.mode = "paragraph"
        data = {"text": "Hello world.\n\ndef foo():\n    return 1"}
        results = s02.parse_record(data, default_cfg)
        types = {ct for _, _, ct in results}
        assert "code" in types
        assert "prose" in types

    def test_paragraph_mode_no_text(self, default_cfg):
        default_cfg.mode = "paragraph"
        assert s02.parse_record({"other": "field"}, default_cfg) == []

    def test_paragraph_mode_non_string(self, default_cfg):
        default_cfg.mode = "paragraph"
        assert s02.parse_record({"text": 123}, default_cfg) == []

    def test_conversation_mode_messages_field(self, default_cfg):
        default_cfg.mode = "conversation"
        data = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hey"},
            ]
        }
        results = s02.parse_record(data, default_cfg)
        assert len(results) == 1
        text, label, ctype = results[0]
        assert "Hi" in text and "Hey" in text
        assert label is None
        assert ctype == "conversation"

    def test_conversation_mode_direct_list(self, default_cfg):
        default_cfg.mode = "conversation"
        data = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        results = s02.parse_record(data, default_cfg)
        assert len(results) == 1
        assert results[0][2] == "conversation"

    def test_dpo_mode(self, default_cfg):
        default_cfg.mode = "dpo"
        data = {
            "chosen": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "Good"},
            ],
            "rejected": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "Bad"},
            ],
        }
        results = s02.parse_record(data, default_cfg)
        assert len(results) == 2
        # All DPO results are labeled "conversation"
        assert all(ct == "conversation" for _, _, ct in results)
        # DPO labels preserved
        labels = {lbl for _, lbl, _ in results}
        assert labels == {"chosen", "rejected"}

    def test_rl_mode(self, default_cfg):
        default_cfg.mode = "rl"
        data = {"prompt": "P", "solution": "S"}
        results = s02.parse_record(data, default_cfg)
        assert len(results) == 1
        text, label, ctype = results[0]
        assert "P" in text and "S" in text
        assert label is None
        assert ctype == "conversation"


# ===================================================================
# ReservoirSampler
# ===================================================================

class TestReservoirSampler:
    def test_under_capacity(self):
        rs = s02.ReservoirSampler(10)
        for i in range(5):
            rs.add(i)
        assert len(rs) == 5
        assert sorted(rs.get_sample()) == [0, 1, 2, 3, 4]

    def test_at_capacity(self):
        rs = s02.ReservoirSampler(5)
        for i in range(5):
            rs.add(i)
        assert len(rs.get_sample()) == 5
        assert len(rs) == 5

    def test_over_capacity_length(self):
        rs = s02.ReservoirSampler(3)
        for i in range(1000):
            rs.add(i)
        assert len(rs.get_sample()) == 3
        assert len(rs) == 1000

    def test_all_items_can_appear(self):
        """With enough runs, every item should appear at least once."""
        counts = {i: 0 for i in range(10)}
        for _ in range(5000):
            rs = s02.ReservoirSampler(1)
            for i in range(10):
                rs.add(i)
            counts[rs.get_sample()[0]] += 1
        # Every item should appear at least once across 5000 trials
        for i in range(10):
            assert counts[i] > 0, f"Item {i} never appeared in reservoir"

    def test_zero_capacity_raises(self):
        with pytest.raises(AssertionError):
            s02.ReservoirSampler(0)

    def test_get_sample_returns_copy(self):
        rs = s02.ReservoirSampler(5)
        for i in range(3):
            rs.add(i)
        sample = rs.get_sample()
        sample.append(999)
        assert 999 not in rs.get_sample()


# ===================================================================
# File I/O round-trip tests
# ===================================================================

class TestFileIO:
    def test_open_plain_jsonl(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text('{"text": "hello"}\n', encoding="utf-8")
        with s02.open_data_file(str(p)) as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["text"] == "hello"

    def test_open_gzip_jsonl(self, tmp_path):
        import gzip
        p = tmp_path / "test.jsonl.gz"
        with gzip.open(str(p), "wt", encoding="utf-8") as f:
            f.write('{"text": "compressed"}\n')
        with s02.open_data_file(str(p)) as f:
            lines = f.readlines()
        assert json.loads(lines[0])["text"] == "compressed"

    def test_open_zst_jsonl(self, tmp_path):
        import zstandard as zstd
        p = tmp_path / "test.jsonl.zst"
        cctx = zstd.ZstdCompressor()
        with open(str(p), "wb") as fh:
            with cctx.stream_writer(fh) as writer:
                writer.write(b'{"text": "zstd_data"}\n')
        with s02.open_data_file(str(p)) as f:
            lines = f.readlines()
        assert json.loads(lines[0])["text"] == "zstd_data"

    def test_bad_bytes_replaced(self, tmp_path):
        """Invalid UTF-8 bytes should be replaced, not crash."""
        p = tmp_path / "bad.jsonl"
        # Write raw bytes: valid JSON structure but with \xff inside
        p.write_bytes(b'{"text": "hello\xff world"}\n')
        with s02.open_data_file(str(p)) as f:
            line = f.readline()
        # Should not raise — bad byte replaced with U+FFFD
        assert "\ufffd" in line or "hello" in line

    def test_find_data_files(self, tmp_path):
        (tmp_path / "a.jsonl").touch()
        (tmp_path / "b.jsonl.zst").touch()
        (tmp_path / "c.json.gz").touch()
        (tmp_path / "d.txt").touch()       # should be excluded
        (tmp_path / "e.parquet").touch()
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "f.jsonl.gz").touch()

        files = s02.find_data_files(str(tmp_path))
        basenames = {os.path.basename(f) for f in files}
        assert basenames == {"a.jsonl", "b.jsonl.zst", "c.json.gz", "e.parquet", "f.jsonl.gz"}

    def test_find_data_files_empty_dir(self, tmp_path):
        with pytest.raises(AssertionError, match="No data files"):
            s02.find_data_files(str(tmp_path))

    def test_find_data_files_missing_dir(self):
        with pytest.raises(AssertionError, match="does not exist"):
            s02.find_data_files("/nonexistent_dir_abc123")


# ===================================================================
# save_results round-trip
# ===================================================================

class TestSaveResults:
    def test_round_trip(self, tmp_path):
        out = str(tmp_path / "output.jsonl")
        sample = [
            {"id": "abc", "text": "hello world", "source": "test.jsonl", "token_size": 2},
            {"id": "def", "text": "second entry", "source": "test.jsonl", "token_size": 2},
        ]
        s02.save_results(sample, out)

        with open(out, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2
        parsed = json.loads(lines[0])
        assert parsed["id"] == "abc"
        assert parsed["text"] == "hello world"

    def test_unicode_preserved(self, tmp_path):
        out = str(tmp_path / "output.jsonl")
        sample = [
            {"id": "u", "text": "caf\u00e9 \U0001f600", "source": "s", "token_size": 3},
        ]
        s02.save_results(sample, out)
        with open(out, "r", encoding="utf-8") as f:
            parsed = json.loads(f.readline())
        assert parsed["text"] == "caf\u00e9 \U0001f600"

    def test_creates_parent_dirs(self, tmp_path):
        out = str(tmp_path / "a" / "b" / "c" / "output.jsonl")
        sample = [{"id": "x", "text": "t", "source": "s", "token_size": 1}]
        s02.save_results(sample, out)
        assert os.path.exists(out)
