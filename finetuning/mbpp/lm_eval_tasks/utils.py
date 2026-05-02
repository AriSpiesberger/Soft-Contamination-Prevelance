"""Re-export the standard MBPP task utils so the custom mbpp_split task
(local jsonl over the 421-id contam/clean halves, fewshot ids removed)
can use the same metric and few-shot sampler as upstream `mbpp`.

We override `extract_code_blocks` / `build_predictions`. The upstream regex
``r"```(?:\w+)?\n?(.*?)\n?```"`` has a bug when the response begins with
``def foo()`` (no leading newline): with the artificial ``` ``` ``` prepended
to the response, ``def`` matches the optional ``\w+`` "language tag", and
the captured code starts at `` foo()`` (no ``def``), making every solution
unparseable. Requiring a newline after the language tag fixes it.
"""
import re

from lm_eval.tasks.mbpp.utils import (  # noqa: F401
    pass_at_1,
    list_fewshot_samples,
)


_PATTERN = re.compile(r"```(?:\w+\n)?(.*?)\n?```", re.DOTALL)


def extract_code_blocks(text: str) -> str:
    matches = _PATTERN.findall("```" + text)
    if not matches:
        text_without_lang = re.sub(r"```python", "```", text)
        matches = _PATTERN.findall(text_without_lang)
    return matches[0] if matches else ""


def build_predictions(resps, docs):
    return [[extract_code_blocks(r) for r in resp] for resp in resps]
