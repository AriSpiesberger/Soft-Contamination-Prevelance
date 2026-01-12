"""
Java validator.
"""

import os
import subprocess
import tempfile
import shutil
import re

from .base import BaseValidator


class JavaValidator(BaseValidator):
    """Validator for Java code."""
    
    LANGUAGE_NAME = "Java"
    FILE_EXTENSION = ".java"
    TIMEOUT = 30
    
    def check_runtime_available(self) -> bool:
        """Check if javac and java are available."""
        return shutil.which('javac') is not None and shutil.which('java') is not None
    
    def python_to_java_expr(self, expr: str) -> str:
        """Convert a Python expression to Java."""
        result = expr
        
        # Boolean conversion
        result = result.replace('True', 'true').replace('False', 'false')
        result = result.replace('None', 'null')
        
        # List conversion - [a, b] -> Arrays.asList(a, b)
        def convert_list(match):
            content = match.group(1)
            return f'Arrays.asList({content})'
        
        result = re.sub(r'\[([^\[\]]+)\]', convert_list, result)
        
        # Tuple conversion
        def convert_tuple(match):
            content = match.group(1)
            return f'Arrays.asList({content})'
        
        result = re.sub(r'(?<!\w)\(([^()]+,[^()]*)\)', convert_tuple, result)
        
        return result
    
    def convert_test(self, python_test: str, func_name: str) -> str:
        """Convert Python assert to Java test."""
        assertion = python_test.strip()
        if assertion.startswith("assert "):
            assertion = assertion[7:]
        
        if "==" in assertion:
            parts = assertion.split("==", 1)
            left = parts[0].strip()
            right = parts[1].strip()
            
            left_java = self.python_to_java_expr(left)
            right_java = self.python_to_java_expr(right)
            
            return f"""
        Object result = {left_java};
        Object expected = {right_java};
        if (result.equals(expected)) {{
            System.out.println("PASS");
        }} else {{
            System.out.println("FAIL");
            System.out.println("  Expected: " + expected);
            System.out.println("  Got: " + result);
        }}
"""
        
        return f"// Could not convert: {python_test}"
    
    def run_single_test(self, code: str, test_code: str) -> tuple[bool, str, str, str]:
        """Run a single Java test."""
        # Check if code has a class wrapper
        has_class = 'class ' in code
        
        # Build the main method
        main_method = f"""
    public static void main(String[] args) {{
        {test_code.strip()}
    }}
"""
        
        if has_class:
            # Extract class name and replace with Test
            class_match = re.search(r'class\s+(\w+)', code)
            if class_match:
                class_name = class_match.group(1)
                code = code.replace(f'class {class_name}', 'class Test')
            
            # Find the last closing brace and insert main before it
            last_brace = code.rfind('}')
            if last_brace != -1:
                full_code = code[:last_brace] + main_method + code[last_brace:]
            else:
                full_code = code + main_method + "\n}"
        else:
            # Wrap in class with main
            full_code = f"""
import java.util.*;

class Test {{
    {code}
    
{main_method}
}}
"""
        
        # Ensure necessary imports at the top
        imports_needed = []
        
        # Check for various Java constructs that need imports
        if 'Arrays' in full_code or 'List' in full_code or 'ArrayList' in full_code:
            if 'import java.util.*' not in full_code and 'import java.util.Arrays' not in full_code:
                imports_needed.append('import java.util.*;')
        
        if 'Math.' in full_code:
            pass  # java.lang.Math is auto-imported
        
        if 'BigInteger' in full_code or 'BigDecimal' in full_code:
            if 'import java.math.*' not in full_code:
                imports_needed.append('import java.math.*;')
        
        if 'Stream' in full_code or 'Collectors' in full_code:
            if 'import java.util.stream.*' not in full_code:
                imports_needed.append('import java.util.stream.*;')
        
        if imports_needed:
            import_block = '\n'.join(imports_needed) + '\n\n'
            full_code = import_block + full_code
        
        temp_dir = tempfile.mkdtemp()
        source_path = os.path.join(temp_dir, 'Test.java')
        
        try:
            with open(source_path, 'w') as f:
                f.write(full_code)
            
            # Compile
            compile_result = subprocess.run(
                ['javac', source_path],
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT,
                cwd=temp_dir
            )
            
            if compile_result.returncode != 0:
                return False, f"Compilation error: {compile_result.stderr}", "", ""
            
            # Run
            run_result = subprocess.run(
                ['java', 'Test'],
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT,
                cwd=temp_dir
            )
            
            output = run_result.stdout + run_result.stderr
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
            return False, "ERROR: javac/java not found", "", ""
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

