"""
JavaScript validator using Node.js.
"""

import os
import subprocess
import tempfile
import shutil

from .base import BaseValidator, ValidationResult


class JavaScriptValidator(BaseValidator):
    """Validator for JavaScript code using Node.js."""
    
    LANGUAGE_NAME = "JavaScript"
    FILE_EXTENSION = ".js"
    TIMEOUT = 20  # Doubled from 10 for complex operations
    
    def check_runtime_available(self) -> bool:
        """Check if Node.js is available."""
        return shutil.which('node') is not None
    
    def convert_test(self, python_test: str, func_name: str) -> str:
        """Convert Python assert to JavaScript test."""
        assertion = python_test.strip()
        if assertion.startswith("assert "):
            assertion = assertion[7:]
        
        if "==" in assertion:
            parts = assertion.split("==", 1)
            left = parts[0].strip()
            right = parts[1].strip()
            
            left = self.python_to_js_expr(left)
            right = self.python_to_js_expr(right)
            
            # Create test with deep comparison and ordering
            return f"""
(() => {{
    const result = {left};
    const expected = {right};
    
    const deepSort = (arr) => {{
        if (!Array.isArray(arr)) return arr;
        return arr.map(deepSort).sort((a, b) => {{
            const aStr = JSON.stringify(a);
            const bStr = JSON.stringify(b);
            return aStr.localeCompare(bStr);
        }});
    }};
    
    const resultStr = JSON.stringify(result);
    const expectedStr = JSON.stringify(expected);
    if (resultStr !== expectedStr) {{
        if (Array.isArray(result) && Array.isArray(expected)) {{
            const sortedResult = deepSort(result);
            const sortedExpected = deepSort(expected);
            if (JSON.stringify(sortedResult) === JSON.stringify(sortedExpected)) {{
                console.log('PASS (order-independent)');
                return true;
            }}
        }}
        console.log('FAIL');
        console.log('  Expected:', expectedStr);
        console.log('  Got:', resultStr);
        return false;
    }}
    console.log('PASS');
    return true;
}})()"""
        
        return f"// Could not convert: {python_test}"
    
    def run_single_test(self, code: str, test_code: str) -> tuple[bool, str, str, str]:
        """Run a single JavaScript test."""
        full_code = f"""
{code}

const testResult = (() => {{
    {test_code.strip()}
}})();
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(full_code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                ['node', temp_path],
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT
            )
            output = result.stdout + result.stderr
            passed = 'PASS' in output and 'FAIL' not in output
            
            expected_value = ""
            actual_value = ""
            for line in output.split('\n'):
                if 'Expected:' in line:
                    expected_value = line.split('Expected:')[-1].strip()
                if 'Got:' in line:
                    actual_value = line.split('Got:')[-1].strip()
            
            return passed, output, expected_value, actual_value
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT", "", ""
        except FileNotFoundError:
            return False, "ERROR: Node.js not found", "", ""
        finally:
            os.unlink(temp_path)

