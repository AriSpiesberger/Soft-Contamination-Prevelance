"""
TypeScript translator.
"""

from .base import BaseTranslator


class TypeScriptTranslator(BaseTranslator):
    """Translator for Python to TypeScript."""
    
    LANGUAGE_NAME = "TypeScript"
    FILE_EXTENSION = ".ts"
    
    def get_language_specific_requirements(self) -> str:
        return """3. Use modern TypeScript syntax with proper type annotations
4. Include type definitions for function parameters and return types
5. Use generics where appropriate"""

