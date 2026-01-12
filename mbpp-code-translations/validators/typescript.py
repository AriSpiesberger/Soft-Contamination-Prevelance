"""
TypeScript validator using ts-node.
"""

import os
import subprocess
import tempfile
import shutil

from .base import BaseValidator


class TypeScriptValidator(BaseValidator):
    """Validator for TypeScript code using ts-node."""
    
    LANGUAGE_NAME = "TypeScript"
    FILE_EXTENSION = ".ts"
    TIMEOUT = 30  # TypeScript needs more time for type checking
    
    def check_runtime_available(self) -> bool:
        """Check if ts-node is available."""
        # Try npx ts-node first, then global ts-node
        try:
            result = subprocess.run(
                ['npx', 'ts-node', '--version'],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except:
            return shutil.which('ts-node') is not None
    
    def convert_test(self, python_test: str, func_name: str) -> str:
        """Convert Python assert to TypeScript test."""
        assertion = python_test.strip()
        if assertion.startswith("assert "):
            assertion = assertion[7:]
        
        if "==" in assertion:
            parts = assertion.split("==", 1)
            left = parts[0].strip()
            right = parts[1].strip()
            
            left = self.python_to_js_expr(left)
            right = self.python_to_js_expr(right)
            
            return f"""
(() => {{
    const result = {left};
    const expected = {right};
    
    const deepSort = (arr: any): any => {{
        if (!Array.isArray(arr)) return arr;
        return arr.map(deepSort).sort((a: any, b: any) => {{
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
        """Run a single TypeScript test."""
        full_code = f"""
{code}

const testResult = (() => {{
    {test_code.strip()}
}})();
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False) as f:
            f.write(full_code)
            temp_path = f.name
        
        try:
            # Use ts-node with explicit compiler options to avoid config conflicts
            result = subprocess.run(
                [
                    'ts-node', 
                    '--transpile-only',
                    '--skip-project',
                    '--compiler-options', '{"module":"commonjs","target":"ES2020","strict":false}',
                    temp_path
                ],
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
            return False, "ERROR: ts-node not found", "", ""
        finally:
            os.unlink(temp_path)

