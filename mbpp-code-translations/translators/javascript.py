"""
JavaScript translator.
"""

from .base import BaseTranslator


class JavaScriptTranslator(BaseTranslator):
    """Translator for Python to JavaScript."""
    
    LANGUAGE_NAME = "JavaScript"
    FILE_EXTENSION = ".js"
    
    def get_language_specific_requirements(self) -> str:
        return """3. Use modern JavaScript (ES6+) syntax"""

