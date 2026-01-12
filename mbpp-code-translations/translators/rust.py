"""
Rust translator.
"""

from .base import BaseTranslator


class RustTranslator(BaseTranslator):
    """Translator for Python to Rust."""
    
    LANGUAGE_NAME = "Rust"
    FILE_EXTENSION = ".rs"
    
    def get_language_specific_requirements(self) -> str:
        return """3. Use idiomatic Rust patterns (ownership, borrowing, Result types)
4. Use Vec<T> for dynamic arrays
5. Use appropriate integer types (i32, usize, etc.)
6. Make the function public (pub fn)
7. Use snake_case for function names (same as Python)"""

