"""
Language translators for MBPP code translation.
"""

from .base import BaseTranslator
from .javascript import JavaScriptTranslator
from .typescript import TypeScriptTranslator
from .rust import RustTranslator
from .go import GoTranslator
from .java import JavaTranslator
from .ruby import RubyTranslator

TRANSLATORS = {
    'javascript': JavaScriptTranslator,
    'typescript': TypeScriptTranslator,
    'rust': RustTranslator,
    'go': GoTranslator,
    'java': JavaTranslator,
    'ruby': RubyTranslator,
}

ALL_LANGUAGES = list(TRANSLATORS.keys())

__all__ = [
    'BaseTranslator',
    'JavaScriptTranslator',
    'TypeScriptTranslator', 
    'RustTranslator',
    'GoTranslator',
    'JavaTranslator',
    'RubyTranslator',
    'TRANSLATORS',
    'ALL_LANGUAGES',
]

