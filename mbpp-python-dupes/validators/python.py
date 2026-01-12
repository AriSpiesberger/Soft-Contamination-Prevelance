"""
Python validator for executing MBPP tests.
"""

import subprocess
import tempfile
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from typing import Optional


def get_python_command() -> str:
    """Get the appropriate Python command for this system."""
    # Try python3 first, then python
    for cmd in ['python3', 'python']:
        if shutil.which(cmd):
            return cmd
    return sys.executable  # Fallback to current interpreter


@dataclass
class ValidationResult:
    """Result of validating Python code."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    outputs: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    
    @property
    def all_passed(self) -> bool:
        return self.failed == 0 and self.total > 0
    
    @property
    def error_message(self) -> str:
        """Get a summary error message for retry feedback."""
        if self.errors:
            error_summaries = []
            for e in self.errors[:3]:
                error_summaries.append(
                    f"Test: {e.get('test', 'unknown')}\n"
                    f"Error: {e.get('error', 'N/A')}"
                )
            return "\n\n".join(error_summaries)
        return "Unknown error"
    
    def to_dict(self) -> dict:
        return {
            'total': self.total,
            'passed': self.passed,
            'failed': self.failed,
            'outputs': self.outputs,
            'errors': self.errors
        }


class PythonValidator:
    """Validator for Python code using subprocess execution."""
    
    TIMEOUT = 30  # seconds
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.python_cmd = get_python_command()
    
    def extract_function_name(self, python_code: str) -> str:
        """Extract the main function name from Python code."""
        match = re.search(r'def\s+(\w+)\s*\(', python_code)
        return match.group(1) if match else "unknown"
    
    def run_single_test(self, code: str, test: str) -> tuple[bool, str]:
        """
        Run a single Python test.
        Returns: (passed, error_message)
        """
        # Build full test code
        full_code = f"""
{code}

# Run test
{test}
print("TEST_PASSED")
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                [self.python_cmd, temp_path],
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT
            )
            
            output = result.stdout + result.stderr
            passed = 'TEST_PASSED' in result.stdout and result.returncode == 0
            
            if not passed:
                error_msg = result.stderr.strip() if result.stderr else output
                return False, error_msg
            
            return True, ""
            
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT: Test took too long to execute"
        except FileNotFoundError:
            return False, f"ERROR: Python interpreter '{self.python_cmd}' not found"
        except Exception as e:
            return False, f"Execution error: {str(e)}"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def validate(self, code: str, test_list: list[str]) -> ValidationResult:
        """Run all tests for the Python code."""
        result = ValidationResult(total=len(test_list))
        
        for test in test_list:
            try:
                passed, error = self.run_single_test(code, test)
                
                test_output = {
                    'test': test,
                    'passed': passed,
                    'error': error if not passed else ""
                }
                result.outputs.append(test_output)
                
                if passed:
                    result.passed += 1
                else:
                    result.failed += 1
                    result.errors.append({
                        'test': test,
                        'error': error
                    })
                    
            except Exception as e:
                result.failed += 1
                result.errors.append({
                    'test': test,
                    'error': str(e)
                })
                result.outputs.append({
                    'test': test,
                    'passed': False,
                    'error': str(e)
                })
        
        return result

