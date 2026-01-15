# Gemini Thinking API Bug: Missing Thought Summaries

## Issue

The non-streaming `generate_content()` method does **not** return thought summaries, even when `include_thoughts=True` is set. The streaming method works correctly.

**GitHub Issue:** https://github.com/googleapis/python-genai/issues/1944  
**Status:** Open (P2 priority)  
**Affected:** `google-genai` SDK v1.52.0+

## Behavior

| Method | Thought Summaries | Tokens Billed |
|--------|-------------------|---------------|
| `generate_content()` | ❌ Not returned | ✅ Yes |
| `generate_content_stream()` | ✅ Returned | ✅ Yes |

## Example

```python
from google import genai
from google.genai import types

client = genai.Client()
config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(include_thoughts=True)
)

# ❌ Does NOT return thought summaries
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="What is 2+2?",
    config=config
)

# ✅ DOES return thought summaries
response = client.models.generate_content_stream(
    model="gemini-3-flash-preview", 
    contents="What is 2+2?",
    config=config
)
```

## Workaround

Use the streaming client until the bug is fixed:

```python
for chunk in client.models.generate_content_stream(...):
    for part in chunk.candidates[0].content.parts:
        if getattr(part, 'thought', False):
            print("Thought:", part.text)
        else:
            print("Answer:", part.text)
```

