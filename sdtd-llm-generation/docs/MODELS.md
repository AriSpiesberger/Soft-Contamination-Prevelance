# Model Selection and OpenRouter Configuration

## Overview

This project uses **OpenRouter** to access multiple LLM providers through a unified API. This allows us to:
- Use the latest models from Anthropic and OpenAI
- Switch between models easily for comparison
- Manage costs with a single API key
- Access diverse model families for maximum paraphrase diversity

## Current Model Setup

### Default Models (via OpenRouter)

All models are accessed via OpenRouter using the format `provider/model-name`.

| Model | Use Cases | Characteristics | Cost |
|-------|-----------|-----------------|------|
| `anthropic/claude-sonnet-4.5` | Lexical, Abstractive | High quality, strong semantic preservation | Medium |
| `anthropic/claude-haiku-4.5` | Syntactic | Faster/cheaper, good for structured transformations | Low |
| `openai/gpt-5.1` | Compositional | Different model family, good for style changes | Medium |

### Recommended Alternatives

| Model | When to Use | Notes |
|-------|-------------|-------|
| `openai/gpt-5-mini` | Cost-sensitive testing | Cheaper than GPT-5.1, still good quality |
| `anthropic/claude-opus-4` | Maximum quality | Highest quality, but slower and more expensive |
| `google/gemini-pro-2.0` | Additional diversity | Different model family, may add later |

## Model Assignment Strategy

### By Variant

**Lexical Maximalist** → `anthropic/claude-sonnet-4.5`
- Rationale: Needs strong vocabulary/synonym understanding
- Alternative: `openai/gpt-5.1` for different perspective

**Syntactic Restructuring** → `anthropic/claude-haiku-4.5`
- Rationale: Simpler transformation, speed/cost efficiency matters
- Alternative: `anthropic/claude-sonnet-4.5` if quality issues

**Abstractive Paraphrase** → `anthropic/claude-sonnet-4.5`
- Rationale: Most demanding transformation, needs highest quality
- Alternative: `anthropic/claude-opus-4` for critical datasets

**Compositional** → `openai/gpt-5.1`
- Rationale: Different model family provides diversity
- Alternative: `openai/gpt-5-mini` for cost savings

### Why Multiple Model Families?

Using both Anthropic and OpenAI models:
1. **Diversity**: Different training data → different paraphrase styles
2. **Robustness**: Less likely to have systematic biases
3. **Comparison**: Can evaluate which models work best per variant
4. **Flexibility**: Can swap models if one provider has issues

## OpenRouter Configuration

### Setup

1. Get API key from [openrouter.ai](https://openrouter.ai/)
2. Add to `.env` file:
   ```bash
   OPENROUTER_API_KEY=your_key_here
   ```

3. litellm automatically handles OpenRouter when model names follow the pattern `provider/model`

### Model Name Format

OpenRouter uses the format: `provider/model-name`

Examples:
- `anthropic/claude-sonnet-4.5` ✓ (correct)
- `claude-sonnet-4.5` ✗ (missing provider)
- `openrouter/anthropic/claude-sonnet-4.5` ✗ (old format, don't use)

### Cost Tracking

OpenRouter provides:
- Per-request pricing visibility
- Usage dashboards
- Cost alerts
- Unified billing across providers

Check current pricing at: https://openrouter.ai/models

## CLI Override

All YAML model defaults can be overridden via CLI:

```bash
# Use Haiku for all variants (fast/cheap testing)
uv run python -m sdtd generate -d gsm8k -l 1 -n 10 -m anthropic/claude-haiku-4.5

# Use GPT-5-mini for all variants (cost-effective)
uv run python -m sdtd generate -d gsm8k -l 1 -n 10 -m openai/gpt-5-mini

# Use Sonnet for all variants (high quality)
uv run python -m sdtd generate -d gsm8k -l 1 -n 10 -m anthropic/claude-sonnet-4.5
```

## Model Selection Guidelines

### For Development/Testing
- Use `anthropic/claude-haiku-4.5` (fast, cheap)
- Test with small samples (n=3-10)
- Iterate quickly on prompts

### For Production Runs
- Use YAML defaults (mixed models for diversity)
- Larger batches (n=100-1000)
- Monitor quality metrics

### For Quality-Critical Work
- Use `anthropic/claude-sonnet-4.5` or `anthropic/claude-opus-4`
- Smaller batches with manual validation
- Higher temperature for diversity (0.8-0.9)

### For Cost Optimization
- Use `anthropic/claude-haiku-4.5` for syntactic variants
- Use `openai/gpt-5-mini` instead of gpt-5.1
- Lower temperature for more deterministic outputs (0.6-0.7)

## Future Models

We may add:
- `google/gemini-pro-2.0`: Additional model family diversity
- `meta-llama/llama-4-70b`: Open source alternative
- `cohere/command-r-plus`: Different provider perspective
- Other Anthropic/OpenAI releases as they become available

To add a new model:
1. Update `prompts/level1.yaml` (or level2.yaml)
2. Test with small sample
3. Document in this file
4. Update README if it becomes a default

## Troubleshooting

### "Model not found" errors
- Check model name format: `provider/model-name`
- Verify model is available on OpenRouter
- Check API key is set in `.env`

### Rate limiting
- OpenRouter has per-provider rate limits
- Use `--limit` to reduce batch sizes
- Add delays between batches if needed

### Quality issues
- Try different model for the variant
- Adjust temperature (lower = more conservative)
- Review prompt instructions
- Check if model suits the transformation type

## Cost Estimates

Approximate costs for 1000 GSM8K problems × 4 variants = 4000 API calls:

| Model Mix | Estimated Cost |
|-----------|----------------|
| All Haiku | $2-5 |
| YAML defaults (Sonnet + Haiku + GPT-5.1) | $15-30 |
| All Sonnet | $30-50 |
| All Opus | $80-150 |

Actual costs vary by:
- Input/output token counts
- Model availability and pricing changes
- OpenRouter markup
- Prompt length

Always check current pricing before large runs.

## References

- OpenRouter: https://openrouter.ai/
- Model pricing: https://openrouter.ai/models
- litellm docs: https://docs.litellm.ai/
- Our prompt configs: `prompts/level1.yaml`
