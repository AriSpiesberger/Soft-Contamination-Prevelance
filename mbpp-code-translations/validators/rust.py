"""
Rust validator using Cargo for external crate support.
"""

import os
import subprocess
import tempfile
import shutil
import re

from .base import BaseValidator, ValidationResult


# Cargo.toml template with common dependencies for MBPP tasks
CARGO_TOML_TEMPLATE = """[package]
name = "mbpp_test"
version = "0.1.0"
edition = "2021"

[dependencies]
regex = "1"
num-bigint = "0.4"
num-traits = "0.2"

[profile.dev]
opt-level = 0

[profile.release]
opt-level = 2
"""


class RustValidator(BaseValidator):
    """Validator for Rust code using Cargo (supports external crates)."""
    
    LANGUAGE_NAME = "Rust"
    FILE_EXTENSION = ".rs"
    TIMEOUT = 120  # Cargo compilation can be slower, especially first time
    
    # Reusable Cargo project directory for caching compiled dependencies
    _cargo_cache_dir = None
    
    def check_runtime_available(self) -> bool:
        """Check if cargo is available."""
        return shutil.which('cargo') is not None
    
    def _get_cargo_cache_dir(self) -> str:
        """Get or create a persistent Cargo cache directory for dependency caching."""
        if RustValidator._cargo_cache_dir is None or not os.path.exists(RustValidator._cargo_cache_dir):
            # Create a persistent temp directory for Cargo cache
            cache_dir = os.path.join(tempfile.gettempdir(), 'mbpp_rust_cargo_cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            # Create Cargo.toml if it doesn't exist
            cargo_toml_path = os.path.join(cache_dir, 'Cargo.toml')
            if not os.path.exists(cargo_toml_path):
                with open(cargo_toml_path, 'w') as f:
                    f.write(CARGO_TOML_TEMPLATE)
            
            # Create src directory
            src_dir = os.path.join(cache_dir, 'src')
            os.makedirs(src_dir, exist_ok=True)
            
            # Pre-compile dependencies with a dummy main
            dummy_main = os.path.join(src_dir, 'main.rs')
            with open(dummy_main, 'w') as f:
                f.write('fn main() {}')
            
            # Build to cache dependencies (ignore errors, just try to cache)
            try:
                subprocess.run(
                    ['cargo', 'build', '--release'],
                    cwd=cache_dir,
                    capture_output=True,
                    timeout=180  # 3 min for first-time dependency download
                )
            except:
                pass  # Best effort caching
            
            RustValidator._cargo_cache_dir = cache_dir
        
        return RustValidator._cargo_cache_dir
    
    def python_to_rust_expr(self, expr: str) -> str:
        """Convert a Python expression to Rust."""
        result = expr
        
        # Boolean conversion
        result = result.replace('True', 'true').replace('False', 'false')
        result = result.replace('None', 'None')  # Rust uses Option<T>
        
        # Tuple/list conversion - (a, b) -> vec![a, b]
        def convert_tuple(match):
            content = match.group(1)
            if ',' in content:
                return 'vec![' + content.rstrip(',') + ']'
            return match.group(0)
        
        result = re.sub(r'(?<!\w)\(([^()]+,[^()]*)\)', convert_tuple, result)
        
        # List conversion - handle nested lists by repeated replacement
        # Keep replacing innermost lists until no more brackets
        prev_result = None
        while prev_result != result:
            prev_result = result
            # Replace innermost lists (those without nested brackets)
            result = re.sub(r'\[([^\[\]]+)\]', r'vec![\1]', result)
        
        return result
    
    def convert_test(self, python_test: str, func_name: str) -> str:
        """Convert Python assert to Rust test."""
        assertion = python_test.strip()
        if assertion.startswith("assert "):
            assertion = assertion[7:]
        
        if "==" in assertion:
            parts = assertion.split("==", 1)
            left = parts[0].strip()
            right = parts[1].strip()
            
            left_rust = self.python_to_rust_expr(left)
            right_rust = self.python_to_rust_expr(right)
            
            return f"""
    let result = {left_rust};
    let expected = {right_rust};
    if result == expected {{
        println!("PASS");
    }} else {{
        println!("FAIL");
        println!("  Expected: {{:?}}", expected);
        println!("  Got: {{:?}}", result);
    }}
"""
        
        return f"// Could not convert: {python_test}"
    
    def _detect_required_imports(self, code: str) -> str:
        """Detect external crate usage and generate required imports."""
        imports = []
        
        # Check for regex crate usage
        if 'Regex::' in code or 'regex::' in code.lower():
            if 'use regex::' not in code:
                imports.append('use regex::Regex;')
        
        # Check for num-bigint usage
        if 'BigUint' in code or 'BigInt' in code:
            if 'use num_bigint::' not in code:
                imports.append('use num_bigint::{BigUint, BigInt};')
            if 'num_traits' in code or 'One' in code or 'Zero' in code:
                if 'use num_traits::' not in code:
                    imports.append('use num_traits::{One, Zero};')
        
        return '\n'.join(imports)
    
    def run_single_test(self, code: str, test_code: str) -> tuple[bool, str, str, str]:
        """Run a single Rust test using Cargo."""
        # Detect any missing imports
        auto_imports = self._detect_required_imports(code)
        
        # Build the full source with imports at top
        if auto_imports:
            full_code = f"""{auto_imports}

{code}

fn main() {{
    {test_code.strip()}
}}
"""
        else:
            full_code = f"""{code}

fn main() {{
    {test_code.strip()}
}}
"""
        
        # Use the cached Cargo project
        cache_dir = self._get_cargo_cache_dir()
        src_dir = os.path.join(cache_dir, 'src')
        main_path = os.path.join(src_dir, 'main.rs')
        
        # Determine binary path based on OS
        if os.name == 'nt':  # Windows
            binary_path = os.path.join(cache_dir, 'target', 'release', 'mbpp_test.exe')
        else:  # Unix-like
            binary_path = os.path.join(cache_dir, 'target', 'release', 'mbpp_test')
        
        try:
            # Write the test code
            with open(main_path, 'w') as f:
                f.write(full_code)
            
            # Compile with Cargo
            compile_result = subprocess.run(
                ['cargo', 'build', '--release'],
                cwd=cache_dir,
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT
            )
            
            if compile_result.returncode != 0:
                error_msg = compile_result.stderr
                # Extract just the relevant error lines
                error_lines = []
                for line in error_msg.split('\n'):
                    if 'error' in line.lower() or '--> src/main.rs' in line:
                        error_lines.append(line)
                condensed_error = '\n'.join(error_lines[:10]) if error_lines else error_msg[:500]
                return False, f"Compilation error: {condensed_error}", "", ""
            
            # Run the binary
            run_result = subprocess.run(
                [binary_path],
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT
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
        except FileNotFoundError as e:
            return False, f"ERROR: cargo not found or binary missing: {e}", "", ""
        except Exception as e:
            return False, f"ERROR: {str(e)}", "", ""
