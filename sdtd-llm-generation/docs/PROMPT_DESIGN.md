# Level 1 Prompt Design Rationale

## Design Goals

Based on deep analysis of `docs/SDs-level-1-claude.md`, the prompts are designed to:

1. **Maximize n-gram/BoW disruption** - Generate Type C duplicates (low lexical overlap, high semantic match)
2. **Preserve semantic fidelity** - Maintain bidirectional entailment and STS score ≥4.0
3. **Avoid common failures** - Prevent semantic drift, negation errors, scope ambiguity
4. **Cover transformation space** - Systematic coverage of linguistic transformation types

## Transformation Coverage

### Coverage Distribution (from taxonomy recommendations)
- **Lexical**: 30% (lexical_maximalist)
- **Syntactic**: 40% (syntactic_restructuring)
- **Semantic/Compositional**: 30% (abstractive_paraphrase + compositional)

### Variant Design

#### **1. Lexical Maximalist**
**Transformation Type**: Pure lexical (synonym substitution, alternative expressions)
**Target Overlap**: 20-40%
**Temperature**: 0.7 (controlled variation)

**Key Design Decisions**:
- Explicit examples of good transformations (based on actual GSM8K data)
- Emphasis on preserving sentence structure
- Strong warnings about preserving numbers and mathematical relationships
- Leverages: WordNet-style synonym substitution, multi-word expression variants

**Failure Modes Addressed**:
- Semantic drift: Explicit requirement to preserve mathematical operations
- Specificity loss: Must keep quantitative relationships exact

---

#### **2. Syntactic Restructuring**
**Transformation Type**: Pure syntactic (voice, clause reordering, sentence boundaries)
**Target Overlap**: 30-50%
**Temperature**: 0.7 (controlled)

**Key Design Decisions**:
- Focus on grammatical structure changes
- Examples: Voice alternations, clause movement
- Allow vocabulary preservation where needed for clarity
- Leverages: Passive/active voice, subordinate clause movement, sentence merging/splitting

**Failure Modes Addressed**:
- Pragmatic shift: Maintain propositional content despite structural changes
- Presupposition change: Preserve logical relationships

---

#### **3. Abstractive Paraphrase**
**Transformation Type**: Comprehensive (lexical + syntactic + discourse)
**Target Overlap**: 10-30% (LOWEST)
**Temperature**: 0.8 (higher for diversity)

**Key Design Decisions**:
- "Rewrite from scratch" instruction
- Explicit target for low word overlap (<30%)
- Emphasis on explaining the same thing differently
- Most aggressive non-compositional transformation

**Failure Modes Addressed**:
- Semantic drift: Explicit requirement for identical answer
- Information change: Must preserve all numbers and operations

---

#### **4. Compositional**
**Transformation Type**: Multi-operation (lexical + syntactic + pragmatic)
**Target Overlap**: 10-25% (LOWEST)
**Temperature**: 0.8 (high for maximum diversity)
**Model**: GPT-4 (different perspective from Claude variants)

**Key Design Decisions**:
- Combines multiple transformation techniques
- Explicitly lists 4 transformation types to apply
- Most aggressive variant for maximum n-gram disruption
- Examples show chained transformations

**Failure Modes Addressed**:
- Compositional drift: Strong emphasis on checking each number
- Complexity errors: Simplified to "must get same answer"

---

## Dataset-Specific Adaptations

### GSM8K (Math Word Problems)

**Critical Constraints**:
- Preserve ALL numbers exactly (48, $12, 50 minutes)
- Preserve mathematical operations (half, twice, sum, difference)
- Preserve problem structure (solution approach must be identical)
- Preserve names (though this is less critical than numbers)

**Prompt Engineering Insights**:
- Examples drawn from actual GSM8K data (Natalia, Weng, Betty)
- Mathematical operation preservation emphasized in every variant
- "Identical answer" requirement stated explicitly
- Temperature kept moderate (0.7-0.8) to avoid hallucination

**Expected Challenges**:
- Maintaining number exactness while changing phrasing
- Preserving "half", "twice", "double" relationships
- Avoiding adding/removing constraints

---

### Codeforces (Programming Problems)

**Critical Constraints**:
- Preserve numerical constraints (N ≤ 10^5, time limits)
- Preserve technical terminology (array, graph, sort, etc.)
- Preserve input/output specifications
- Preserve algorithm/data structure requirements

**Prompt Engineering Insights**:
- Technical terms explicitly listed as "do not change"
- Input/output format preservation emphasized
- Algorithm complexity must remain identical
- Examples focus on non-technical phrasing changes

**Expected Challenges**:
- Balancing vocabulary change with technical accuracy
- Preserving constraint specifications exactly
- Maintaining problem difficulty

---

### AllenAI (Educational Text)

**Critical Constraints**:
- Preserve factual information
- Preserve educational objectives
- Maintain difficulty level
- Maintain accuracy of explanations

**Prompt Engineering Insights**:
- Most flexible dataset (no specific answer to preserve)
- Register shift variant added (formality changes)
- Can be more aggressive with transformations
- Focus on preserving concepts rather than specific phrasing

**Expected Challenges**:
- Maintaining factual accuracy during transformation
- Preserving educational value
- Balancing simplicity/complexity appropriately

---

## Quality Requirements (from Taxonomy)

### Semantic Preservation Thresholds

**Target Metrics**:
- **STS Score**: ≥4.0 (near-perfect to perfect equivalence)
- **BERTScore F1**: >0.85 (medium-precision)
- **Bidirectional Entailment**: Both A→B and B→A must hold

**From STS Benchmark**:
- Score 5.0: Perfect equivalence (completely interchangeable)
- Score 4.0-4.9: Near-perfect (minor stylistic differences)
- Below 4.0: Not considered semantic duplicate

### N-gram Disruption Targets

**Overlap Goals** (measured on content words):
- Lexical Maximalist: 20-40% overlap
- Syntactic Restructuring: 30-50% overlap
- Abstractive Paraphrase: 10-30% overlap
- Compositional: 10-25% overlap

**Why This Matters**:
Traditional deduplication (MinHash, n-gram overlap >50%) will MISS our variants, especially abstractive_paraphrase and compositional. This is the intended outcome - we're generating **Type C duplicates** that are invisible to surface-level methods.

---

## Edge Cases Explicitly Addressed

Based on Section VIII.5 of the taxonomy:

### 1. Semantic Drift
**Problem**: Meaning gradually changes through transformation
**Solution**: Every prompt emphasizes "exact meaning", "identical answer", "same solution"

### 2. Negation Errors
**Problem**: "He is happy" → "He is not unhappy" (double negation)
**Solution**: Avoid negation transformations, focus on positive rephrasing

### 3. Scope Ambiguity
**Problem**: "All students passed some exams" (quantifier scope can change)
**Solution**: Math problems make scope explicit; preserve quantifiers exactly

### 4. Presupposition Change
**Problem**: "Stop smoking" vs "Start not smoking"
**Solution**: Maintain event structure and presuppositions

### 5. Pragmatic Shift
**Problem**: "Can you pass the salt?" (request) vs "You can pass the salt" (statement)
**Solution**: Preserve question/statement structure in math problems

---

## Model Assignment Strategy

### Default Model Choices (via OpenRouter)

All models accessed via OpenRouter API. See `docs/MODELS.md` for details.

**Claude Sonnet 4.5 (`anthropic/claude-sonnet-4.5`)**:
- Used for: lexical_maximalist, abstractive_paraphrase (GSM8K, Codeforces, AllenAI)
- Rationale: Latest Claude model, strong semantic preservation, excellent at following complex instructions
- Cost: Medium, worth it for quality
- Alternative: `anthropic/claude-opus-4` for maximum quality

**Claude Haiku 4.5 (`anthropic/claude-haiku-4.5`)**:
- Used for: syntactic_restructuring
- Rationale: Simpler transformation, faster/cheaper model sufficient
- Cost: Low, excellent for high-volume generation
- Alternative: `anthropic/claude-sonnet-4.5` if quality issues

**GPT-5.1 (`openai/gpt-5.1`)**:
- Used for: compositional, register_shift
- Rationale: Different model family provides diversity, latest GPT model, good at style changes
- Cost: Medium, similar to Sonnet
- Alternative: `openai/gpt-5-mini` for cost savings

### CLI Override
All models can be overridden via `--model` flag for:
- Testing same prompts with different models
- Cost optimization (e.g., using Haiku for all variants during development)
- Model comparison experiments

---

## Temperature Settings

**0.7** (Lexical, Syntactic):
- Moderate variation
- Good balance of diversity and control
- Reduces hallucination risk for math problems

**0.8** (Abstractive, Compositional):
- Higher diversity needed for aggressive transformations
- Still controlled enough to maintain semantic fidelity
- Allows more creative rephrasing

**Rationale**: Math problems require precision (numbers must be exact), so we stay below 1.0. Educational text could potentially go higher, but 0.7-0.8 is safe.

---

## Prompt Structure Pattern

Each prompt follows this structure:

1. **Task Statement**: One-line description of transformation goal
2. **INSTRUCTIONS**: Bullet points with specific transformation directives
3. **EXAMPLES**: 3-5 concrete examples of good transformations
4. **WHAT NOT TO CHANGE**: Explicit list of preservation requirements
5. **Input Format**: `Original problem/text: {field}`
6. **Output Prompt**: Clear instruction for generation

This structure is based on best practices for instruction-following LLMs:
- Clear task framing
- Concrete examples (few-shot learning)
- Explicit constraints
- Structured input/output format

---

## Expected Performance

### Success Criteria

**Good Outputs**:
- Low lexical overlap with original (especially abstractive/compositional)
- Identical numerical values and mathematical operations
- Same problem difficulty and solution approach
- Natural, fluent phrasing (not robotic transformations)

**Failure Indicators**:
- Changed numbers
- Changed mathematical relationships (e.g., "half" → "third")
- Added or removed constraints
- Unnatural/awkward phrasing
- Semantic drift (problem has different answer)

### Testing Strategy

1. **Start small**: Test with 5-10 GSM8K problems
2. **Manual inspection**: Check that numbers/operations preserved
3. **Calculate n-gram overlap**: Verify we're hitting target ranges
4. **Spot check BERTScore**: Confirm semantic similarity >0.85
5. **Iterate on prompts**: Adjust based on failure patterns

---

## References

- **Taxonomy**: `docs/SDs-level-1-claude.md`
- **Type C Duplicates**: Section III (Surface Form vs. Meaning)
- **Quality Thresholds**: Section IX (Recommendations)
- **Edge Cases**: Section VIII.5 (Failure Modes)
- **STS Benchmark**: Section II (Semantic Preservation)
