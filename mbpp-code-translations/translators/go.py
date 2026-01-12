"""
Go translator.
"""

from .base import BaseTranslator


class GoTranslator(BaseTranslator):
    """Translator for Python to Go."""
    
    LANGUAGE_NAME = "Go"
    FILE_EXTENSION = ".go"
    
    def get_language_specific_requirements(self) -> str:
        return """3. Use idiomatic Go patterns
4. Use slices for dynamic arrays
5. Keep the function name in snake_case (same as Python) - Go will export it if capitalized
6. Use appropriate types (int, float64, string, etc.)
7. Include the package declaration (package main)"""

