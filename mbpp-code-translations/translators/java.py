"""
Java translator.
"""

from .base import BaseTranslator


class JavaTranslator(BaseTranslator):
    """Translator for Python to Java."""
    
    LANGUAGE_NAME = "Java"
    FILE_EXTENSION = ".java"
    
    def get_language_specific_requirements(self) -> str:
        return """3. Create a public static method with the EXACT same name as Python (use snake_case)
4. Use ArrayList, HashMap, HashSet for collections
5. Use appropriate types (int, double, String, etc.)
6. Wrap the function in a class called Solution
7. Import necessary java.util classes at the top"""

