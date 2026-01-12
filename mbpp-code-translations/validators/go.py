"""
Go validator.
"""

import os
import subprocess
import tempfile
import shutil
import re

from .base import BaseValidator


class GoValidator(BaseValidator):
    """Validator for Go code."""
    
    LANGUAGE_NAME = "Go"
    FILE_EXTENSION = ".go"
    TIMEOUT = 30
    
    def check_runtime_available(self) -> bool:
        """Check if go is available."""
        return shutil.which('go') is not None
    
    def python_to_go_expr(self, expr: str) -> str:
        """Convert a Python expression to Go."""
        result = expr
        
        # Boolean conversion (before list processing to properly detect bool type)
        result = result.replace('True', 'true').replace('False', 'false')
        result = result.replace('None', 'nil')
        
        # Tuple/list conversion - detect type from content
        def infer_go_type(content: str) -> str:
            """Infer Go type from list content."""
            if not content.strip():
                return '[]interface{}'
            
            items = [x.strip() for x in content.split(',')]
            if not items or not items[0]:
                return '[]interface{}'
            
            first = items[0]
            
            # Check for nested list/slice
            if first.startswith('[]'):
                return '[]' + first.split('{')[0]  # e.g., [][]int
            
            # Check for string
            if first.startswith('"') or first.startswith("'"):
                return '[]string'
            
            # Check for boolean
            if first in ('true', 'false'):
                return '[]bool'
            
            # Check for float (has decimal point)
            if '.' in first and not first.startswith('['):
                return '[]float64'
            
            # Default to int
            return '[]int'
        
        def convert_list(match):
            content = match.group(1)
            go_type = infer_go_type(content)
            return f'{go_type}{{{content}}}'
        
        # Handle nested lists iteratively (innermost first)
        prev_result = None
        while prev_result != result:
            prev_result = result
            result = re.sub(r'\[([^\[\]]+)\]', convert_list, result)
        
        # Tuple conversion
        def convert_tuple(match):
            content = match.group(1)
            items = [x.strip() for x in content.split(',')]
            if items and items[0]:
                first = items[0]
                if first.startswith('"') or first.startswith("'"):
                    return f'[]string{{{content}}}'
                elif '.' in first:
                    return f'[]float64{{{content}}}'
                else:
                    return f'[]int{{{content}}}'
            return f'[]interface{{}}{{{content}}}'
        
        result = re.sub(r'(?<!\w)\(([^()]+,[^()]*)\)', convert_tuple, result)
        
        return result
    
    def convert_test(self, python_test: str, func_name: str) -> str:
        """Convert Python assert to Go test."""
        assertion = python_test.strip()
        if assertion.startswith("assert "):
            assertion = assertion[7:]
        
        if "==" in assertion:
            parts = assertion.split("==", 1)
            left = parts[0].strip()
            right = parts[1].strip()
            
            left_go = self.python_to_go_expr(left)
            right_go = self.python_to_go_expr(right)
            
            return f"""
    result := {left_go}
    expected := {right_go}
    resultStr := fmt.Sprintf("%v", result)
    expectedStr := fmt.Sprintf("%v", expected)
    if resultStr == expectedStr {{
        fmt.Println("PASS")
    }} else {{
        fmt.Println("FAIL")
        fmt.Println("  Expected:", expectedStr)
        fmt.Println("  Got:", resultStr)
    }}
"""
        
        return f"// Could not convert: {python_test}"
    
    def run_single_test(self, code: str, test_code: str) -> tuple[bool, str, str, str]:
        """Run a single Go test."""
        # Ensure package and imports
        has_package = 'package ' in code
        has_fmt_import = 'import "fmt"' in code or '"fmt"' in code
        
        if not has_package:
            code = "package main\n\n" + code
        
        # Collect needed imports
        imports_needed = ['"fmt"']  # Always need fmt for test output
        
        if 'math.' in code.lower() or 'Math.' in code:
            imports_needed.append('"math"')
        if 'sort.' in code or 'Sort' in code:
            imports_needed.append('"sort"')
        if 'strings.' in code:
            imports_needed.append('"strings"')
        if 'strconv.' in code:
            imports_needed.append('"strconv"')
        if 'regexp.' in code:
            imports_needed.append('"regexp"')
        
        # Build import statement
        if len(imports_needed) == 1:
            import_stmt = f'import {imports_needed[0]}'
        else:
            import_stmt = 'import (\n\t' + '\n\t'.join(imports_needed) + '\n)'
        
        # Add imports after package declaration if not already present
        if 'import ' not in code:
            code = code.replace('package main\n', f'package main\n\n{import_stmt}\n', 1)
        elif not has_fmt_import:
            # fmt not imported, need to add it to existing import
            code = code.replace('package main\n', f'package main\n\nimport "fmt"\n', 1)
        
        full_code = f"""
{code}

func main() {{
    {test_code.strip()}
}}
"""
        
        temp_dir = tempfile.mkdtemp()
        source_path = os.path.join(temp_dir, 'test.go')
        
        try:
            with open(source_path, 'w') as f:
                f.write(full_code)
            
            # Run
            result = subprocess.run(
                ['go', 'run', source_path],
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT,
                cwd=temp_dir
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
            return False, "ERROR: go not found", "", ""
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

