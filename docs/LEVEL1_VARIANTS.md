# Level 1 Variant Strategy: Maximal N-gram/BoW Disruption

## Goal
Generate **Type C semantic duplicates** that:
- Have LOW lexical overlap (target: 10-40%)
- Preserve EXACT meaning
- Are INVISIBLE to traditional deduplication (n-grams, bag-of-words)

## Proposed Variants for Maximal Coverage

### Core Strategy: Focus on Type C Paraphrases
From the taxonomy, Type C = Low overlap + High semantic match = Hard to detect

### Coverage Dimensions
1. **Transformation Type**: Lexical, Syntactic, Semantic, Pragmatic, Compositional
2. **Surface Overlap**: Target LOW overlap (breaking n-gram/BoW similarity)
3. **Transformation Intensity**: Conservative → Moderate → Aggressive

---

## Recommended Variant Set

### **Variant 1: Lexical Maximalist**
**Transformation**: Replace vocabulary extensively
**Target overlap**: 20-40%
**Strategy**: Synonym substitution, paraphrasing, alternative expressions
**Example**: "Count the red apples" → "Tally the crimson fruits"

**Coverage**: Pure lexical transformations (30% of taxonomy recommendation)

---

### **Variant 2: Syntactic Restructuring**
**Transformation**: Reorganize sentence structure completely
**Target overlap**: 30-50%
**Strategy**: Voice changes, clause reordering, sentence boundary manipulation
**Example**: "If John has 5 apples and gives 2 away, how many remain?"
→ "The remaining apple count, after John's gift of 2 from an initial 5, should be determined"

**Coverage**: Pure syntactic transformations (40% of taxonomy recommendation)

---

### **Variant 3: Abstractive Paraphrase**
**Transformation**: Rewrite from scratch
**Target overlap**: 10-30% (VERY LOW)
**Strategy**: Explain the same thing in a completely different way
**Example**: "Calculate 3 + 5" → "Determine the sum when three is added to five"

**Coverage**: Compositional transformations, high-diversity paraphrasing

---

### **Variant 4: Register/Formality Shift**
**Transformation**: Dramatic change in language register
**Target overlap**: 20-40%
**Strategy**: Formal ↔ informal, technical ↔ plain language
**Example**: "How many apples?" → "What is the total quantity of fruit items?"

**Coverage**: Pragmatic/stylistic transformations (10% of taxonomy)

---

### **Variant 5: Perspective/Frame Shift** (Optional)
**Transformation**: Change viewpoint and semantic roles
**Target overlap**: 20-40%
**Strategy**: Converse relations, role reversal, frame reinterpretation
**Example**: "John sold 3 apples to Mary" → "Mary purchased 3 apples from John"

**Coverage**: Semantic role transformations (15% of taxonomy)

---

### **Variant 6: Compositional (Kitchen Sink)** (Optional)
**Transformation**: Multiple transformations combined
**Target overlap**: 10-25% (LOWEST)
**Strategy**: Combine lexical + syntactic + pragmatic changes
**Example**: "Count red apples" → "Determine the quantity of crimson-hued fruit specimens"

**Coverage**: Compositional transformations (5% of taxonomy), hardest duplicates

---

## Minimal Effective Set: 3-4 Variants

If starting lean, use:
1. **Lexical Maximalist** - Change words (Type C)
2. **Abstractive Paraphrase** - Rewrite from scratch (LOWEST overlap)
3. **Syntactic Restructuring** - Reorganize structure (Type C)
4. **Compositional** - Kitchen sink (LOWEST overlap)

This covers:
- ✓ Major transformation types (lexical, syntactic, compositional)
- ✓ Range of n-gram disruption (10-40% overlap)
- ✓ Both focused and combined transformations
- ✓ High diversity in outputs

---

## Implementation Design

### YAML Structure
```yaml
gsm8k:
  lexical_maximalist:
    model: "claude-3-5-sonnet-20241022"  # Default model
    temperature: 0.7
    user: |
      Rewrite this math problem using completely different vocabulary.
      Replace as many words as possible with synonyms or alternative phrases.
      Keep the sentence structure similar, but change the words extensively.
      The meaning and problem structure must remain identical.

      Original: {question}

      Rewritten problem:
```

### Model Assignment
- **YAML defaults**: Each variant specifies a default model
- **CLI override**: `--model <model_name>` overrides ALL variants
- **Flexibility**: Can test same prompts with different models

### Usage Examples
```bash
# Use YAML defaults (different models per variant)
uv run python -m sdtd generate -d gsm8k -l 1 -n 10

# Override all variants to use GPT-4
uv run python -m sdtd generate -d gsm8k -l 1 -n 10 --model gpt-4-turbo

# Override all variants to use Claude Haiku (cheaper/faster)
uv run python -m sdtd generate -d gsm8k -l 1 -n 10 --model claude-3-5-haiku-20241022
```

---

## Dataset-Specific Considerations

### GSM8K (Math Problems)
- All variants applicable
- Focus on preserving mathematical operations and quantities
- Variant 3 (Abstractive) particularly effective: "5 apples + 3 apples" → "the sum of five and three fruit items"

### Codeforces (Programming)
- Variants 1-4 work best
- Preserve technical terminology accuracy
- Variant 2 (Syntactic): Rephrase constraints and requirements
- Be careful with Variant 5 (Perspective) - may change problem semantics

### AllenAI (Educational Text)
- All variants applicable
- Variant 3 (Abstractive): Best for full text paraphrasing
- Variant 4 (Register): Grade school ↔ high school ↔ college level

---

## Expected Outcomes

### N-gram Overlap Distribution
- Lexical Maximalist: 20-40%
- Syntactic Restructuring: 30-50%
- Abstractive Paraphrase: 10-30% (breaking most n-gram detectors)
- Register Shift: 20-40%
- Compositional: 10-25% (most aggressive disruption)

### Detection Challenge
Traditional deduplication methods (MinHash, n-gram overlap) should:
- ✗ FAIL to detect Variants 3, 6 (overlap <30%)
- ⚠ Struggle with Variants 1, 2, 4 (overlap <50%)
- This is the GOAL - we want Type C duplicates!

### Quality Requirements
- Must preserve exact problem structure
- Must preserve exact answer/solution
- Must remain solvable with same approach
- Semantic similarity (BERTScore) should remain high (>0.85)

---

## Next Steps

1. **Select variant set** (3-4 core variants vs 5-6 comprehensive)
2. **Write prompts collaboratively** for each variant
3. **Test with small sample** (5-10 items from GSM8K)
4. **Evaluate n-gram overlap** on generated pairs
5. **Iterate on prompts** based on overlap distribution
6. **Expand to all datasets** once validated

---

## References
- See `docs/SDs-level-1-claude.md` for complete taxonomy
- Type C definition: Low lexical overlap, high semantic match (PAWS framework)
- Coverage recommendations: Section IX of level-1 document
