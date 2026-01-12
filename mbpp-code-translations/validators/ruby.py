"""
Ruby validator.
"""

import os
import subprocess
import tempfile
import shutil
import re

from .base import BaseValidator


class RubyValidator(BaseValidator):
    """Validator for Ruby code."""
    
    LANGUAGE_NAME = "Ruby"
    FILE_EXTENSION = ".rb"
    TIMEOUT = 15
    
    def check_runtime_available(self) -> bool:
        """Check if ruby is available."""
        return shutil.which('ruby') is not None
    
    def python_to_ruby_expr(self, expr: str) -> str:
        """Convert a Python expression to Ruby."""
        result = expr
        
        # Boolean conversion
        result = result.replace('True', 'true').replace('False', 'false')
        result = result.replace('None', 'nil')
        
        # Tuple conversion - (a, b) -> [a, b]
        def convert_tuple(match):
            content = match.group(1)
            if ',' in content:
                return '[' + content.rstrip(',') + ']'
            return match.group(0)
        
        result = re.sub(r'(?<!\w)\(([^()]+,[^()]*)\)', convert_tuple, result)
        
        return result
    
    def convert_test(self, python_test: str, func_name: str) -> str:
        """Convert Python assert to Ruby test."""
        assertion = python_test.strip()
        if assertion.startswith("assert "):
            assertion = assertion[7:]
        
        if "==" in assertion:
            parts = assertion.split("==", 1)
            left = parts[0].strip()
            right = parts[1].strip()
            
            left_ruby = self.python_to_ruby_expr(left)
            right_ruby = self.python_to_ruby_expr(right)
            
            return f"""
result = {left_ruby}
expected = {right_ruby}

def deep_sort(arr)
  return arr unless arr.is_a?(Array)
  arr.map {{ |x| deep_sort(x) }}.sort_by {{ |x| x.to_s }}
end

if result == expected
  puts "PASS"
elsif result.is_a?(Array) && expected.is_a?(Array) && deep_sort(result) == deep_sort(expected)
  puts "PASS (order-independent)"
else
  puts "FAIL"
  puts "  Expected: #{{expected.inspect}}"
  puts "  Got: #{{result.inspect}}"
end
"""
        
        return f"# Could not convert: {python_test}"
    
    def run_single_test(self, code: str, test_code: str) -> tuple[bool, str, str, str]:
        """Run a single Ruby test."""
        full_code = f"""
{code}

{test_code.strip()}
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rb', delete=False) as f:
            f.write(full_code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                ['ruby', temp_path],
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
            return False, "ERROR: ruby not found", "", ""
        finally:
            os.unlink(temp_path)

