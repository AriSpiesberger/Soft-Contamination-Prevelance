"""
Base translator class for code translation.
"""

import re
import time
from abc import ABC, abstractmethod
from typing import Optional
from anthropic import Anthropic


class BaseTranslator(ABC):
    """Base class for all code translators."""
    
    # Class-level configuration
    LANGUAGE_NAME: str = ""  # e.g., "JavaScript", "Rust"
    FILE_EXTENSION: str = ""  # e.g., ".js", ".rs"
    MODEL: str = "claude-opus-4-5"
    MAX_RETRIES: int = 5
    RETRY_DELAY: float = 2.0
    
    def __init__(self, client: Anthropic, verbose: bool = True):
        self.client = client
        self.verbose = verbose
    
    @abstractmethod
    def get_language_specific_requirements(self) -> str:
        """Return language-specific translation requirements."""
        pass
    
    def get_initial_prompt(
        self,
        task_description: str,
        python_code: str,
        test_list: list[str]
    ) -> str:
        """Generate the initial translation prompt."""
        test_context = "\n".join(test_list[:3])
        lang_requirements = self.get_language_specific_requirements()
        
        return f"""You are an expert programmer. Translate the following Python code to {self.LANGUAGE_NAME}.

TASK DESCRIPTION:
{task_description}

PYTHON CODE:
```python
{python_code}
```

PYTHON TEST EXAMPLES (for context on expected behavior):
```python
{test_context}
```

REQUIREMENTS:
1. Create a {self.LANGUAGE_NAME} function with the EXACT SAME NAME as the Python function (including underscores - do NOT convert to camelCase)
2. The function should have equivalent behavior
{lang_requirements}
4. Do NOT use any external packages unless absolutely necessary
5. Handle edge cases the same way as the Python version
6. If the Python uses tuples, return arrays/lists in {self.LANGUAGE_NAME}
7. If the Python uses sets, use appropriate equivalents

CRITICAL: The function name MUST be EXACTLY the same as in Python. For example, if Python has `square_nums`, {self.LANGUAGE_NAME} must also have `square_nums`.

OUTPUT ONLY THE {self.LANGUAGE_NAME.upper()} CODE, nothing else. No markdown code blocks, just the raw code."""

    def get_retry_prompt(
        self,
        task_description: str,
        python_code: str,
        attempt_history: list[dict],
        test_list: list[str] = None
    ) -> str:
        """
        Generate the retry prompt with FULL error feedback from ALL previous attempts.
        
        Args:
            task_description: The task description
            python_code: Original Python code
            attempt_history: List of dicts with 'code' and 'error' for each previous attempt
            test_list: Optional test cases for context
        """
        test_context = ""
        if test_list:
            test_context = f"""
PYTHON TEST EXAMPLES (for context on expected behavior):
```python
{chr(10).join(test_list[:3])}
```
"""
        
        # Build history of all previous attempts
        history_text = ""
        for i, attempt in enumerate(attempt_history, 1):
            history_text += f"""
--- ATTEMPT {i} ---
Code:
```{self.LANGUAGE_NAME.lower()}
{attempt['code']}
```

Error/Analysis:
{attempt['error']}
"""
        
        return f"""Your previous {self.LANGUAGE_NAME} translations had errors. You have made {len(attempt_history)} attempt(s) so far, all failed.

TASK DESCRIPTION:
{task_description}

ORIGINAL PYTHON CODE:
```python
{python_code}
```
{test_context}
=== HISTORY OF ALL PREVIOUS ATTEMPTS ===
{history_text}
=== END OF HISTORY ===

CRITICAL: Analyze ALL the previous attempts above. Each attempt failed for a reason.
- Look at the patterns: what's consistently wrong?
- The error analysis provides specific feedback - USE IT.
- Do NOT repeat the same mistakes.
- If regex/pattern matching fails, the issue is likely in escaping or matching logic.
- If tests fail with wrong values, trace through the logic step by step.

Please provide a NEW {self.LANGUAGE_NAME} translation that:
1. Fixes ALL the issues identified in previous attempts
2. Uses a DIFFERENT approach if the same approach keeps failing
3. Keeps the EXACT SAME function name as the Python original

OUTPUT ONLY THE CORRECTED {self.LANGUAGE_NAME.upper()} CODE, nothing else. No markdown code blocks."""

    def clean_code_response(self, code: str) -> str:
        """
        Clean up LLM response to extract only the code.
        
        Handles:
        - Markdown code blocks (```language ... ```)
        - Prose/commentary mixed with code
        - Clean code without any wrapping
        """
        code = code.strip()
        
        # Strategy 1: Extract from markdown code blocks if present anywhere
        # This handles cases where model outputs prose + code block
        code_block_pattern = r'```(?:' + self.LANGUAGE_NAME.lower() + r'|' + self.FILE_EXTENSION[1:] + r')?\s*\n?(.*?)```'
        code_blocks = re.findall(code_block_pattern, code, re.DOTALL | re.IGNORECASE)
        
        if code_blocks:
            # Return the longest code block (most likely the actual code)
            extracted = max(code_blocks, key=len).strip()
            if extracted:
                return extracted
        
        # Strategy 2: Remove leading markdown fence if at start
        patterns = [
            f"```{self.LANGUAGE_NAME.lower()}",
            f"```{self.FILE_EXTENSION[1:]}",
            "```"
        ]
        for pattern in patterns:
            if code.startswith(pattern):
                code = code[len(pattern):]
                break
        
        if code.endswith("```"):
            code = code[:-3]
        
        code = code.strip()
        
        # Strategy 3: Detect if response starts with prose and extract code portion
        # Prose indicators: starts with capital letter followed by sentence-like content
        if self._starts_with_prose(code):
            extracted = self._extract_code_from_prose(code)
            if extracted:
                return extracted
        
        return code
    
    def _starts_with_prose(self, text: str) -> bool:
        """Detect if text starts with English prose rather than code."""
        if not text:
            return False
        
        # Common prose starters that indicate commentary
        prose_starters = [
            "looking at", "i notice", "i see", "the issue", "the problem",
            "here's", "here is", "let me", "based on", "analyzing",
            "the error", "the code", "this is", "after reviewing",
            "i'll", "i will", "note that", "notice that", "as we can see",
            "the previous", "my approach", "to fix", "to solve"
        ]
        
        first_line = text.split('\n')[0].lower().strip()
        
        for starter in prose_starters:
            if first_line.startswith(starter):
                return True
        
        return False
    
    def _extract_code_from_prose(self, text: str) -> str:
        """Extract actual code from a response that contains prose."""
        lines = text.split('\n')
        
        # Language-specific code indicators
        code_indicators = self._get_code_indicators()
        
        # Find where code likely starts
        code_start_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
            
            # Check for code indicators
            for indicator in code_indicators:
                if stripped.startswith(indicator) or indicator in stripped:
                    code_start_idx = i
                    break
            
            if code_start_idx is not None:
                break
        
        if code_start_idx is not None:
            # Extract from code start to end, but stop at obvious prose endings
            code_lines = []
            for line in lines[code_start_idx:]:
                stripped = line.strip().lower()
                
                # Stop if we hit prose again (e.g., "This should fix...")
                if stripped and self._is_prose_line(stripped):
                    break
                
                code_lines.append(line)
            
            return '\n'.join(code_lines).strip()
        
        return ""
    
    def _get_code_indicators(self) -> list[str]:
        """Get language-specific indicators that a line is code."""
        # Base indicators that work across languages
        indicators = [
            "import ", "from ", "use ", "package ",
            "pub fn ", "fn ", "func ", "function ",
            "def ", "class ", "struct ", "enum ",
            "const ", "let ", "var ", "static ",
            "public ", "private ", "protected ",
            "module ", "export ", "require(",
            "#include", "#define", "//", "/*",
        ]
        return indicators
    
    def _is_prose_line(self, line: str) -> bool:
        """Check if a line is likely prose rather than code."""
        # Prose patterns
        prose_patterns = [
            r"^(this|that|the|these|those|i|we|you|it|they)\s+(should|will|would|can|could|is|are|was|were|have|has)\b",
            r"^(note|notice|looking|based|analyzing|here|after)\b",
            r"^(to fix|to solve|the issue|the problem|the error)\b",
        ]
        
        for pattern in prose_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        
        return False
    
    def translate(
        self,
        task_description: str,
        python_code: str,
        test_list: list[str],
        attempt_history: list[dict] = None
    ) -> str:
        """
        Translate Python code to target language. Single attempt.
        
        Args:
            task_description: The task description
            python_code: Original Python code
            test_list: List of Python test assertions
            attempt_history: Optional list of previous attempts, each with 'code' and 'error'
        """
        
        if attempt_history and len(attempt_history) > 0:
            prompt = self.get_retry_prompt(
                task_description,
                python_code,
                attempt_history,
                test_list
            )
        else:
            prompt = self.get_initial_prompt(task_description, python_code, test_list)
        
        response = self.client.messages.create(
            model=self.MODEL,
            max_tokens=2048,
            timeout=240.0,  # 240 second timeout for code translation
            messages=[{"role": "user", "content": prompt}]
        )
        
        code = response.content[0].text.strip()
        return self.clean_code_response(code)
    
    def translate_with_api_retries(
        self,
        task_description: str,
        python_code: str,
        test_list: list[str],
        attempt_history: list[dict] = None,
        api_retries: int = 3
    ) -> str:
        """Translate with retry for API failures only."""
        last_error = None
        
        for attempt in range(1, api_retries + 1):
            try:
                return self.translate(
                    task_description,
                    python_code,
                    test_list,
                    attempt_history
                )
            except Exception as e:
                last_error = e
                if attempt < api_retries:
                    time.sleep(self.RETRY_DELAY)
        
        raise last_error


class TranslationResult:
    """Result of a translation attempt."""
    
    def __init__(
        self,
        code: str,
        status: str,  # 'success', 'success_llm_validated', 'failed'
        attempts: int,
        error_history: list[str] = None,
        test_results: dict = None
    ):
        self.code = code
        self.status = status
        self.attempts = attempts
        self.error_history = error_history or []
        self.test_results = test_results or {}
    
    def to_dict(self) -> dict:
        return {
            'code': self.code,
            'status': self.status,
            'attempts': self.attempts,
            'error_history': self.error_history,
            'tests': self.test_results
        }

