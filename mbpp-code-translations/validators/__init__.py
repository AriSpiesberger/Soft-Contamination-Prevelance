"""
Language validators for running tests.
"""

from .base import BaseValidator, ValidationResult
from .javascript import JavaScriptValidator
from .typescript import TypeScriptValidator
from .rust import RustValidator
from .go import GoValidator
from .java import JavaValidator
from .ruby import RubyValidator

VALIDATORS = {
    'javascript': JavaScriptValidator,
    'typescript': TypeScriptValidator,
    'rust': RustValidator,
    'go': GoValidator,
    'java': JavaValidator,
    'ruby': RubyValidator,
}

__all__ = [
    'BaseValidator',
    'ValidationResult',
    'JavaScriptValidator',
    'TypeScriptValidator',
    'RustValidator',
    'GoValidator',
    'JavaValidator',
    'RubyValidator',
    'VALIDATORS',
]

