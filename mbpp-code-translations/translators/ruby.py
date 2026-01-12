"""
Ruby translator.
"""

from .base import BaseTranslator


class RubyTranslator(BaseTranslator):
    """Translator for Python to Ruby."""
    
    LANGUAGE_NAME = "Ruby"
    FILE_EXTENSION = ".rb"
    
    def get_language_specific_requirements(self) -> str:
        return """3. Use idiomatic Ruby patterns
4. Use snake_case for function names (same as Python)
5. Use Ruby arrays and hashes appropriately
6. Use Ruby's built-in methods where possible"""

