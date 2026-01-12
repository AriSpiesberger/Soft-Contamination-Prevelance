"""
Base validator class for running code tests.
"""

import os
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from anthropic import Anthropic


@dataclass
class TestOutput:
    """Output of a single test."""
    python_test: str
    passed: bool
    llm_validated: bool = False
    llm_explanation: str = ""
    expected_value: str = ""
    actual_value: str = ""
    error: str = ""


@dataclass  
class ValidationResult:
    """Result of validating translated code."""
    total: int = 0
    passed: int = 0
    passed_llm_validated: int = 0
    failed: int = 0
    outputs: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    compilation_error: str = ""
    
    @property
    def all_passed(self) -> bool:
        return self.failed == 0
    
    @property
    def error_message(self) -> str:
        """Get a summary error message for retry feedback."""
        if self.compilation_error:
            return f"Compilation error: {self.compilation_error}"
        if self.errors:
            error_summaries = []
            for e in self.errors[:3]:  # First 3 errors
                error_summaries.append(
                    f"Test: {e.get('test', 'unknown')}\n"
                    f"Expected: {e.get('expected', 'N/A')}\n"
                    f"Got: {e.get('actual', 'N/A')}"
                )
            return "\n\n".join(error_summaries)
        return "Unknown error"
    
    def to_dict(self) -> dict:
        return {
            'total': self.total,
            'passed': self.passed,
            'passed_llm_validated': self.passed_llm_validated,
            'failed': self.failed,
            'outputs': self.outputs,
            'errors': self.errors,
            'compilation_error': self.compilation_error
        }


class BaseValidator(ABC):
    """Base class for all code validators."""
    
    LANGUAGE_NAME: str = ""
    FILE_EXTENSION: str = ""
    TIMEOUT: int = 30  # Default timeout in seconds
    
    def __init__(self, client: Optional[Anthropic] = None, verbose: bool = True):
        self.client = client
        self.verbose = verbose
    
    @abstractmethod
    def convert_test(self, python_test: str, func_name: str) -> str:
        """Convert a Python test assertion to target language test code."""
        pass
    
    @abstractmethod
    def run_single_test(self, code: str, test_code: str) -> tuple[bool, str, str, str]:
        """
        Run a single test.
        Returns: (passed, output, expected_value, actual_value)
        """
        pass
    
    def check_runtime_available(self) -> bool:
        """Check if the runtime is available."""
        return True  # Override in subclasses
    
    def extract_function_name(self, python_code: str) -> str:
        """Extract the main function name from Python code."""
        match = re.search(r'def\s+(\w+)\s*\(', python_code)
        return match.group(1) if match else "unknown"
    
    def python_to_js_expr(self, expr: str) -> str:
        """Convert a Python expression to JavaScript-like syntax (shared helper)."""
        result = expr
        
        # set() -> []
        result = re.sub(r'^set\(', '[', result)
        if result.startswith('[') and result.endswith(')'):
            result = result[:-1] + ']'
        
        # tuple wrapped in parens
        result = re.sub(r'(?<!\w)\((\[[^\]]+\])\)', r'\1', result)
        
        # tuple literals (a, b, c) -> [a, b, c]
        def convert_tuple_to_array(match):
            content = match.group(1)
            if ',' in content:
                return '[' + content.rstrip(',') + ']'
            return match.group(0)
        
        result = re.sub(r'(?<!\w)\(([^()]+,[^()]*)\)', convert_tuple_to_array, result)
        
        # Booleans and None
        result = result.replace('True', 'true').replace('False', 'false')
        result = result.replace('None', 'null')
        
        # Power operator
        result = re.sub(r'(\w+)\s*\*\*\s*(\w+)', r'Math.pow(\1, \2)', result)
        
        # Integer division
        result = re.sub(r'(\w+)\s*//\s*(\w+)', r'Math.floor(\1 / \2)', result)
        
        return result
    
    def clean_generated_test(self, test_code: str) -> str:
        """
        Clean up LLM-generated test code by removing package/import/main wrappers.
        
        The LLM sometimes ignores instructions and outputs full programs instead of snippets.
        This strips out the wrapper so the test can be inserted into our own main function.
        """
        lines = test_code.strip().split('\n')
        cleaned_lines = []
        in_main_func = False
        brace_depth = 0
        skip_until_main = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip package declarations
            if stripped.startswith('package '):
                skip_until_main = True
                continue
            
            # Skip import statements
            if stripped.startswith('import ') or stripped == 'import (':
                skip_until_main = True
                continue
            if skip_until_main and (stripped == ')' or stripped.startswith('"')):
                continue
            
            # Skip function redefinitions (the function is already in the code we're testing)
            if stripped.startswith('func ') and not stripped.startswith('func main'):
                # Skip this function definition entirely
                brace_count = line.count('{') - line.count('}')
                if brace_count > 0:
                    # Multi-line function, skip until balanced
                    skip_depth = brace_count
                    for next_line in lines[lines.index(line)+1:]:
                        skip_depth += next_line.count('{') - next_line.count('}')
                        if skip_depth <= 0:
                            break
                continue
            
            # Handle func main() { ... }
            if 'func main()' in stripped:
                in_main_func = True
                # Check if opening brace is on this line
                if '{' in stripped:
                    brace_depth = 1
                    # Get anything after the opening brace
                    after_brace = stripped.split('{', 1)[1].strip()
                    if after_brace and after_brace != '}':
                        cleaned_lines.append(after_brace)
                continue
            
            if in_main_func:
                brace_depth += line.count('{') - line.count('}')
                if brace_depth <= 0:
                    # End of main function
                    # Get anything before the closing brace
                    if '}' in line:
                        before_brace = line.rsplit('}', 1)[0].strip()
                        if before_brace:
                            cleaned_lines.append(before_brace)
                    in_main_func = False
                    continue
                cleaned_lines.append(line)
            elif not skip_until_main:
                cleaned_lines.append(line)
            
            # Reset skip flag once we see actual code
            if stripped and not stripped.startswith('//') and skip_until_main:
                if not stripped.startswith('import') and not stripped.startswith('"') and stripped != ')':
                    skip_until_main = False
                    cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines).strip()
        
        # If we ended up with nothing, return original
        if not result:
            return test_code
        
        return result
    
    def llm_convert_test(self, python_test: str, func_name: str, translated_code: str) -> str:
        """
        Use Claude Opus to convert a Python test assertion to target language test code.
        This is more robust than regex-based conversion for complex types.
        """
        if not self.client:
            # Fallback to regex-based conversion
            return self.convert_test(python_test, func_name)
        
        prompt = f"""Convert this Python test assertion to {self.LANGUAGE_NAME} code that tests the function.

PYTHON TEST:
{python_test}

{self.LANGUAGE_NAME.upper()} FUNCTION TO TEST:
```
{translated_code[:1500]}
```

REQUIREMENTS:
1. Call the function DIRECTLY by name: {func_name}(...) - do NOT use any class prefix
2. Compare the result with the expected value
3. Print "PASS" if the test passes
4. Print "FAIL" followed by "  Expected: <value>" and "  Got: <value>" if it fails (note the 2-space indent)
5. Handle type conversions properly:
   - Python lists → {self.LANGUAGE_NAME} arrays/slices/lists (use native array syntax, not utility functions)
   - Python tuples → {self.LANGUAGE_NAME} arrays/lists
   - Python True/False → {self.LANGUAGE_NAME} true/false or equivalent
   - Python None → {self.LANGUAGE_NAME} null/nil/None equivalent
6. For Java: use primitive arrays like `new int[]{{1,2,3}}` not `Arrays.asList()`
7. For Go: use slice literals like `[]int{{1,2,3}}` or `[][]int{{...}}`

CRITICAL: Output ONLY the test code snippet - NO package declaration, NO imports, NO func main wrapper.
Just the raw test code that will be placed inside a main function.

Example of GOOD output:
result := myFunc(1, 2)
if result == expected {{
    fmt.Println("PASS")
}}

Example of BAD output (do NOT do this):
package main
import "fmt"
func main() {{
    // test code here
}}"""

        try:
            response = self.client.messages.create(
                model="claude-opus-4-5",
                max_tokens=4096,  # Increased for tests with large expected outputs (e.g., 64 combinations)
                timeout=120.0,  # 120 second timeout for test conversion
                messages=[{"role": "user", "content": prompt}]
            )
            test_code = response.content[0].text.strip()
            
            # Remove any markdown code blocks if present
            if test_code.startswith("```"):
                lines = test_code.split("\n")
                test_code = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            
            # Clean up any package/main wrappers the LLM might have added despite instructions
            test_code = self.clean_generated_test(test_code)
            
            return test_code
        except Exception as e:
            if self.verbose:
                print(f"    [WARN] LLM test conversion failed: {e}, using regex fallback")
            # Fallback to regex-based conversion
            return self.convert_test(python_test, func_name)
    
    def llm_validate_result(
        self,
        task_description: str,
        python_test: str,
        expected_value: str,
        actual_value: str
    ) -> tuple[bool, str]:
        """Use Claude Haiku to validate if expected and actual values are semantically equivalent."""
        if not self.client:
            return False, "No LLM client available"
        
        prompt = f"""You are validating test results for a programming task.

TASK DESCRIPTION:
{task_description}

ORIGINAL PYTHON TEST:
{python_test}

EXPECTED VALUE: {expected_value}
ACTUAL VALUE: {actual_value}

Determine if the ACTUAL value is semantically equivalent to the EXPECTED value.

Consider EQUIVALENT:
- Same elements in different order (for sets/unordered collections)
- Same numeric values with different formatting (1.0 vs 1)
- Equivalent boolean representations

Consider NOT EQUIVALENT:
- Different values entirely
- Missing or extra elements

Respond with ONLY:
EQUIVALENT: <brief explanation>
or
NOT_EQUIVALENT: <brief explanation>"""

        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=256,
                timeout=60.0,  # 60 second timeout for LLM validation
                messages=[{"role": "user", "content": prompt}]
            )
            result_text = response.content[0].text.strip()
            
            if result_text.startswith("EQUIVALENT:"):
                return True, result_text[11:].strip()
            return False, result_text.replace("NOT_EQUIVALENT:", "").strip()
        except Exception as e:
            return False, f"LLM validation error: {e}"
    
    def validate(
        self,
        code: str,
        python_tests: list[str],
        func_name: str,
        task_description: str = "",
        use_llm_conversion: bool = True
    ) -> ValidationResult:
        """
        Run all tests for the translated code.
        
        Args:
            code: The translated code to test
            python_tests: List of Python test assertions
            func_name: Name of the function being tested
            task_description: Description of the task (for LLM validation)
            use_llm_conversion: If True, use Opus to convert tests (more robust)
        """
        result = ValidationResult(total=len(python_tests))
        
        for test in python_tests:
            try:
                # Use LLM-based conversion for robust type handling
                if use_llm_conversion and self.client:
                    test_code = self.llm_convert_test(test, func_name, code)
                else:
                    test_code = self.convert_test(test, func_name)
                
                passed, output, expected, actual = self.run_single_test(code, test_code)
                
                llm_validated = False
                llm_explanation = ""
                
                # Try LLM validation for failed tests (semantic equivalence check)
                if not passed and self.client and expected and actual:
                    llm_validated, llm_explanation = self.llm_validate_result(
                        task_description, test, expected, actual
                    )
                
                test_output = {
                    'python_test': test,
                    'generated_test_code': test_code,  # Save the LLM-generated test
                    'passed': passed,
                    'llm_validated': llm_validated,
                    'llm_explanation': llm_explanation if llm_validated else "",
                    'expected_value': expected,
                    'actual_value': actual
                }
                result.outputs.append(test_output)
                
                if passed:
                    result.passed += 1
                elif llm_validated:
                    result.passed_llm_validated += 1
                else:
                    result.failed += 1
                    result.errors.append({
                        'test': test,
                        'expected': expected,
                        'actual': actual,
                        'output': output
                    })
                    
            except Exception as e:
                result.failed += 1
                result.errors.append({
                    'test': test,
                    'error': str(e)
                })
                result.outputs.append({
                    'python_test': test,
                    'passed': False,
                    'error': str(e)
                })
        
        return result

