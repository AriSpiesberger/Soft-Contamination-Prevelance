# Paraphrase Classifications for Semantic Duplicate Generation

**Project Context**: LLM Training Data Semantic Duplicate Classification & Generation  
**Document Scope**: Level 1 Semantic Duplicates (Linguistic Paraphrases)  
**Status**: Foundation taxonomy for generation pipeline

---

## Important Scope Note

**This taxonomy primarily applies to PARAPHRASES** - texts that express the same propositional content with different surface forms. These represent **Level 1 Semantic Duplicates** in our broader framework.

**What this covers**: Same meaning, different linguistic realization  
**What this does NOT cover**:
- Functional duplicates (same task, different content) → Level 2
- Structural/skill duplicates (same mental operations, different problems) → Level 3

For broader semantic duplication including "similar skills" and reasoning patterns, see separate documentation on Levels 2-3.

---

## I. Transformation-Based Classification

**Rationale**: Organizes paraphrases by the linguistic operations that created them. Most useful for **systematic generation** - tells us HOW to create paraphrases.

**Application to LLM Training**: Each transformation type can be automated through rules, neural models, or LLM prompting.

### 1. Lexical Transformations

**Definition**: Word/phrase-level substitutions preserving meaning.

| Type | Example | Generation Method | Detection Difficulty |
|------|---------|------------------|---------------------|
| **Synonym substitution** | "car" → "automobile" | WordNet, embedding neighbors | Easy |
| **Hypernym/hyponym** | "dog" → "animal" / "poodle" | WordNet hierarchy | Moderate (may lose specificity) |
| **Named entity variants** | "NYC" → "New York City" | Entity linking databases | Easy |
| **Multi-word expressions** | "because of" → "due to" | PPDB, phrase tables | Easy |

**Key Literature**:
- Bannard & Callison-Burch (2005): Paraphrasing with Bilingual Parallel Corpora
- Pavlick et al. (2015): PPDB 2.0: Better paraphrase ranking, fine-grained entailment relations
- Miller (1995): WordNet: A lexical database for English

**Generation Strategy for LLM Training**:
```
1. Extract all content words from source text
2. For each word, query synonym database (WordNet/PPDB/embeddings)
3. Filter for context-appropriate substitutions
4. Generate combinatorial variants
5. Validate semantic preservation (entailment check)
```

**Training Data Impact**: Low redundancy - teaches lexical flexibility without much skill duplication.

---

### 2. Syntactic Transformations

**Definition**: Grammatical structure changes without meaning change.

#### 2.1 Voice Alternations

| Transformation | Example | Preservation Guarantees |
|----------------|---------|------------------------|
| Active → Passive | "John broke the window" → "The window was broken by John" | Truth-conditional equivalence |
| Passive → Active | Reverse of above | Same |

**Key Literature**:
- Chomsky (1981): Lectures on Government and Binding
- Mel'čuk (1988): Dependency Syntax: Theory and Practice

**Generation**: Rule-based syntactic transformation using dependency parsing.

#### 2.2 Clause Reordering

| Type | Example | Notes |
|------|---------|-------|
| Subordinate clause movement | "Although it rained, we played" → "We played, although it rained" | Presupposition preserved |
| Adjunct repositioning | "Yesterday I met John" → "I met John yesterday" | Focus may shift |

**Generation**: Parse-based clause extraction and recombination.

#### 2.3 Diathesis Alternations

| Type | Example | Semantic Role Preservation |
|------|---------|---------------------------|
| Dative shift | "Give the book to Mary" → "Give Mary the book" | Same |
| Locative alternation | "Load hay onto truck" → "Load truck with hay" | Subtle holistic/partitive difference |

**Key Literature**:
- Levin (1993): English Verb Classes and Alternations
- Dorr (1997): Large-scale dictionary construction for foreign language tutoring

**Generation**: Verb-class specific transformation rules.

#### 2.4 Sentence Boundary Manipulation

| Type | Example | 
|------|---------|
| Splitting | "John left because he was tired" → "John left. He was tired." |
| Merging | "It rained. The game was cancelled." → "The game was cancelled because it rained." |

**Caution**: Discourse structure changes; may not preserve all pragmatic aspects.

---

### 3. Semantic Role Transformations

**Definition**: Changes in how event participants are described while maintaining event structure.

| Type | Example | Preservation Type |
|------|---------|------------------|
| **Converse substitution** | "X bought Y from Z" ↔ "Z sold Y to X" | Event-structural equivalence |
| **Argument structure shift** | "The door opened" ↔ "Someone opened the door" | Causative alternation |
| **Perspective shift** | "X beat Y" ↔ "Y lost to X" | Same outcome, different viewpoint |

**Key Literature**:
- Fillmore (1968): The Case for Case (Frame Semantics foundation)
- Baker et al. (1998): FrameNet

**Generation**: Frame-semantic role mapping and lexicon-based substitution.

**Training Data Impact**: Medium redundancy - teaches perspective flexibility, useful for understanding.

---

### 4. Discourse-Level Transformations

**Definition**: Information packaging changes without propositional content change.

| Type | Example | Effect |
|------|---------|--------|
| **Clefting** | "John broke it" → "It was John who broke it" | Emphasis shift |
| **Topicalization** | "I don't like this" → "This, I don't like" | Focus marking |
| **Extraposition** | "That he left surprised me" → "It surprised me that he left" | Weight distribution |

**Key Literature**:
- Prince (1981): Toward a taxonomy of given-new information
- Halliday & Hasan (1976): Cohesion in English

**Generation**: Discourse-aware transformation rules; requires context modeling.

**Training Data Impact**: Low-to-moderate; teaches discourse pragmatics.

---

### 5. Pragmatic/Stylistic Transformations

**Definition**: Register, formality, or politeness changes while preserving core propositional meaning.

| Dimension | Example Low → High |
|-----------|-------------------|
| **Formality** | "I don't know" → "I am unaware" → "I lack knowledge of this matter" |
| **Directness** | "Do it now!" → "Please do it now" → "I would appreciate if you could do it now" |
| **Technical register** | "The heart pumps blood" → "Cardiac muscle facilitates circulatory flow" |

**Key Literature**:
- Brown & Levinson (1987): Politeness: Some universals in language usage
- Heylighen & Dewaele (1999): Formality of Language

**Generation**: Style transfer models; LLM prompting with style conditioning.

**Training Data Impact**: High value for instruction-following LLMs; teaches appropriate response generation.

---

### 6. Compositional Transformations

**Definition**: Multiple transformation types applied sequentially or simultaneously.

**Example Chain**:
```
Original: "The company announced its decision to close the factory"
→ Nominalization reversal: "The company announced that it decided to close the factory"
→ Voice change: "The company announced that the decision was made to close the factory"
→ Synonym substitution: "The firm stated that the decision was made to shutter the plant"
```

**Challenge**: Semantic drift through composition - each transformation preserves meaning, but chains may accumulate error.

**Key Literature**:
- Barzilay & McKeown (2001): Extracting paraphrases from a parallel corpus
- Zhao et al. (2009): Paraphrase generation using recursive edit operations

**Generation Strategy**: 
1. Start with single-operation paraphrases
2. Validate semantic preservation at each step
3. Use entailment checking after each composition
4. Limit chain depth (recommend ≤3 operations)

**Training Data Impact**: Can create full spectrum from easy to hard paraphrases for ML training.

---

## II. Semantic Preservation Level Classification

**Rationale**: Organizes paraphrases by how well meaning is preserved. Most useful for **quality filtering** and defining "duplicate" thresholds.

**Application to LLM Training**: Different preservation levels have different impacts on training redundancy.

### Preservation Scale (STS Benchmark)

| Score | Label | Definition | Example Pair | Include as Duplicate? |
|-------|-------|------------|--------------|---------------------|
| **5.0** | Perfect equivalence | Completely interchangeable | "The cat sat on the mat" / "A cat was sitting on the mat" | YES - clear duplicate |
| **4.0-4.9** | Near-perfect | Minor stylistic differences | "He quickly ran home" / "He ran home rapidly" | YES - functionally duplicate |
| **3.0-3.9** | Strong similarity | Same topic, minor info differences | "Python is a programming language" / "Python is used for coding" | DEPENDS - partial overlap |
| **2.0-2.9** | Moderate similarity | Related but divergent | "Python uses indentation" / "Python is object-oriented" | NO - related but not duplicate |
| **0.0-1.9** | Low/no similarity | Different or contradictory | "Python is easy" / "Java is difficult" | NO |

**Key Literature**:
- Agirre et al. (2012): SemEval-2012 Task 6: A pilot on semantic textual similarity
- Cer et al. (2017): SemEval-2017 Task 1: Semantic Textual Similarity
- STSb Benchmark (2017): Official annotation guidelines

**For Your Project**:
- **Threshold recommendation**: Score ≥4.0 for "semantic duplicate" in training data
- **Gray zone (3.0-3.9)**: Analyze separately - may be valuable as negative examples
- **Below 3.0**: Not duplicates; useful for contrastive learning

### Bidirectionality Requirement

**Critical Distinction**: True paraphrase requires **mutual entailment**.

| Relationship | Example | Duplicate? |
|--------------|---------|-----------|
| **A ↔ B** (paraphrase) | "All cats are mammals" ↔ "Every feline is a mammal" | YES |
| **A → B only** (entailment) | "John owns a poodle" → "John owns a dog" | NO (asymmetric) |
| **A → B and B → A but different info** | "Rain is wet" vs "Water is wet" | NO (related, not same) |

**Detection Method**: Use bidirectional entailment model (e.g., RoBERTa fine-tuned on MNLI).

```python
# Validation pseudocode
def is_paraphrase(text_a, text_b):
    forward_entailment = entailment_model(text_a, text_b)  # A→B
    backward_entailment = entailment_model(text_b, text_a)  # B→A
    
    return (forward_entailment == "entailment" and 
            backward_entailment == "entailment")
```

**Key Literature**:
- Dagan et al. (2006): The PASCAL Recognising Textual Entailment Challenge
- Williams et al. (2018): A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference (MNLI)

---

## III. Surface Form vs. Meaning Classification

**Rationale**: Captures the **detection challenge** dimension. Critical for understanding what ML models can/cannot detect.

**Application to LLM Training**: Helps identify which duplicates are invisible to deduplication systems.

### Taxonomy

| Category | Lexical Overlap | Semantic Match | Example | Detection Difficulty |
|----------|----------------|----------------|---------|---------------------|
| **Type A** | High | High | "I like apples" / "I love apples" | EASY |
| **Type B** | High | Low | "Flights NYC→FL" / "Flights FL→NYC" | HARD (false positive risk) |
| **Type C** | Low | High | "How do I learn Python?" / "What's the best way to start programming in Python?" | HARD (true duplicate, invisible to n-grams) |
| **Type D** | Low | Low | "Python is easy" / "Quantum physics is complex" | EASY (true negative) |

**Key Literature**:
- Zhang et al. (2019): PAWS: Paraphrase Adversaries from Word Scrambling
  - **Critical contribution**: Demonstrated that high lexical overlap ≠ semantic equivalence
  - Dataset specifically designed for Type B cases
- Bhagat & Hovy (2013): What Is a Paraphrase?

### PAWS Framework Insights

**The Word Order Problem**:
```
Positive: "The actor was in a movie directed by the screenwriter"
         "The screenwriter was in a movie directed by the actor"
Overlap: ~90% words identical
Label: NOT paraphrases (roles reversed)
```

**Generation Implication**: Word scrambling alone insufficient; must preserve semantic roles.

**For Your Project**: 
- Generate Type B examples explicitly (adversarial negatives)
- Generate Type C examples (true semantic duplicates with low overlap)
- Use both for training robust duplicate detectors

---

## IV. Vila et al. (2014) Comprehensive Linguistic Taxonomy

**Rationale**: Most detailed linguistic classification. 26 fine-grained types organized systematically.

**Application**: Use for comprehensive coverage in generation - ensure all paraphrase types represented.

**Key Literature**:
- Vila et al. (2014): WRPA: A system for relational paraphrase acquisition from Wikipedia
- Vila et al. (2015): Paraphrase concept and typology: A linguistically based and computationally oriented approach
- Kovatchev et al. (2020): ETPC: A paraphrase and semantic similarity detection corpus

### A. Change Operations (4 major types)

| Operation | Definition | Example | Generation Method |
|-----------|-----------|---------|------------------|
| **Addition** | Information added | "He left" → "He left quickly" | Adjunct insertion rules |
| **Deletion** | Information removed | "The tall man left" → "The man left" | Modifier removal |
| **Change of order** | Reordering preserving meaning | "Before X, Y" → "Y, after X" | Clause reordering rules |
| **Core change** | Lexical/syntactic substitution | See detailed types below | Multiple methods |

**Note**: Addition/deletion technically create non-paraphrases (information change), but included in typology for completeness.

### B. Equivalence-Preserving Operations (22 types)

#### B.1 Morphology-Based (3 types)

| Type | Example | Rule Type |
|------|---------|-----------|
| Inflectional | "walk" → "walked" | Tense/aspect/number changes |
| Modal | "He will go" → "He might go" | Modal verb substitution |
| Derivational | "happy" → "happiness" | Nominalization, adjectivization |

**Generation**: Morphological analyzers + rules.

#### B.2 Lexicon-Based (7 types)

| Type | Example | Resource |
|------|---------|----------|
| Synonymy | "big" → "large" | WordNet, PPDB |
| Antonymy + negation | "X is big" → "X is not small" | WordNet antonym relations |
| Converse | "X bought from Y" → "Y sold to X" | Verb frame database |
| Hypernymy | "dog" → "animal" | WordNet hierarchy |
| Hyponymy | "animal" → "dog" | WordNet hierarchy |
| Meronymy | "wheel" → "part of car" | WordNet part-whole |
| Troponymy | "walk" → "stroll" | Manner-of verbs |

#### B.3 Syntax-Based (8 types)

| Type | Example | Complexity |
|------|---------|-----------|
| Diathesis alternation | "spray paint on wall" → "spray wall with paint" | Verb-class specific |
| Negation/affirmation | "not impossible" → "possible" | Logic rules |
| Ellipsis | "John went and Mary went" → "John and Mary went" | Coordination rules |
| Coordination changes | "X and Y" → "both X and Y" | Pattern matching |
| Subordination changes | "Because X, Y" → "Y, since X" | Conjunction substitution |
| Nesting changes | Clause embedding depth | Parse tree manipulation |

#### B.4 Discourse-Based (4 types)

| Type | Example | Pragmatic Impact |
|------|---------|-----------------|
| Punctuation | "John left; Mary stayed" → "John left. Mary stayed." | Minimal |
| Direct/indirect speech | "He said 'I will go'" → "He said he would go" | Deixis shift |
| Sentence modality | "Go!" → "Please go" | Politeness |
| Syntax/discourse structure | Information ordering | Focus/topic |

### Coverage Analysis for Generation

**Priority Tiers**:
1. **High priority** (common + automatable): Synonymy, voice, clause reordering, sentence splitting
2. **Medium priority** (less common but valuable): Converse, diathesis, nominalization
3. **Low priority** (rare or require deep context): Troponymy, ellipsis, discourse structure

**For Your Project**: Aim for 80% coverage of high+medium priority types in generated dataset.

---

## V. Code Clone Types (Programming Domain)

**Rationale**: Specialized taxonomy for code duplicates in LLM training data.

**Application**: If training data includes code (common in modern LLMs).

### Type Hierarchy

| Type | Definition | Example | Paraphrase Analog | Detection Method |
|------|-----------|---------|------------------|-----------------|
| **Type-1** | Exact copy (whitespace/comments differ) | Exact code, different formatting | Formatting variations | String match |
| **Type-2** | Renamed identifiers | `foo(x,y)` → `bar(a,b)` | Synonym substitution | Token normalization |
| **Type-3** | Statement changes (50-100% similar) | Lines added/removed | Sentence-level edits | AST diff |
| **Type-4** | Semantic equivalence | Different algorithm, same function | True semantic paraphrase | **UNSOLVED** |

**Key Literature**:
- Roy et al. (2009): Comparison and evaluation of code clone detection techniques and tools
- Svajlenko & Roy (2015): Evaluating modern clone detection tools (BigCloneBench)
- **CRITICAL**: Allamanis et al. (2025): How the Misuse of a Dataset Harmed Semantic Clone Detection
  - **Finding**: BigCloneBench's Type-4 labels are 93% incorrect
  - **Implication**: Cannot trust existing semantic clone benchmarks

**For Your Project**: 
- Type-1/2/3 can be generated reliably with program transformation tools
- Type-4 (semantic clones) remains research challenge - be cautious about ground truth

**Generation Strategy**:
```
Type-1: Apply code formatting tools with different style configs
Type-2: Systematic identifier renaming (preserve AST structure)
Type-3: Use program synthesis to add/remove statements within constraints
Type-4: Currently requires manual annotation or very careful validation
```

---

## VI. Context-Dependency Classification

**Rationale**: Not all paraphrases work in all contexts.

**Application**: Important for document-level duplicate detection.

### Categories

| Type | Definition | Example | Generation Consideration |
|------|-----------|---------|------------------------|
| **Context-independent** | Self-contained meaning | "The sun is hot" / "The sun has high temperature" | Safe to generate in isolation |
| **Anaphoric** | Requires antecedent | "He left" ≈ "John left" (only when "he"=John established) | Need context window |
| **Deictic** | Requires spatiotemporal anchor | "yesterday" ≈ "on Monday" (only in right context) | Contextual grounding required |
| **Domain-specific** | Specialized terminology | "MI" = "myocardial infarction" (medical) OR "Michigan" (geography) | Domain detection needed |

**Key Literature**:
- Wegmann et al. (2024): Context-dependent paraphrase identification in dialogues
- Agirre et al. (2016): SemEval-2016 Task 1: Semantic textual similarity (cross-lingual)

**For Your Project**: 
- Tag generated paraphrases with context requirements
- Include context window when generating anaphoric/deictic paraphrases
- Consider domain-specific paraphrase generation as separate track

---

## VII. Generation Method Taxonomy

**Rationale**: Different methods excel at different paraphrase types.

**Application**: Select appropriate method per transformation type.

### Method Comparison

| Method | Best For | Limitations | Key References |
|--------|----------|-------------|---------------|
| **Rule-based** | Syntactic transformations, voice | Limited coverage | McKeown (1983) |
| **Back-translation** | Lexical diversity | May drift semantically | Mallinson et al. (2017) |
| **SMT-based** | Phrase-level substitutions | Requires parallel corpus | Wubben et al. (2010) |
| **Seq2seq neural** | General paraphrasing | Requires training data | Prakash et al. (2016) |
| **T5/BART fine-tuned** | Controllable generation | Computationally expensive | Raffel et al. (2020) |
| **LLM prompting** | Diverse, high-quality | Expensive, may hallucinate | Brown et al. (2020) - GPT-3 |
| **PEGASUS** | Abstractive paraphrase | Specialized pre-training | Zhang et al. (2020) |

### Recommended Pipeline for Your Project

```
Stage 1: Rule-Based Baseline
├─ Syntactic transformations (active/passive, clause reordering)
├─ WordNet synonym substitution
└─ Output: Conservative, high-precision paraphrases

Stage 2: Neural Augmentation  
├─ Back-translation for lexical diversity
├─ T5 fine-tuned on ParaNMT/MRPC for sentence-level
└─ Output: More diverse, medium-precision paraphrases

Stage 3: LLM Generation
├─ GPT-4/Claude prompting with paraphrase type specification
├─ Few-shot examples for each transformation type
└─ Output: High-diversity, needs validation

Stage 4: Validation Pipeline
├─ Entailment checking (bidirectional)
├─ BERTScore for semantic preservation
├─ Human spot-checking (10% sample)
└─ Output: Validated paraphrase dataset
```

---

## VIII. Quality Metrics & Validation

**Rationale**: Generated paraphrases must be validated for semantic preservation.

### Automatic Metrics

| Metric | Measures | Threshold for Paraphrase | Literature |
|--------|----------|-------------------------|------------|
| **BERTScore** | Embedding similarity | F1 > 0.85 | Zhang et al. (2020) |
| **BLEURT** | Learned semantic similarity | Score > 0.4 | Sellam et al. (2020) |
| **Entailment (bidirectional)** | Logical equivalence | Both directions = "entailment" | Williams et al. (2018) |
| **BLEU** | N-gram overlap | NOT recommended (poor correlation) | Papineni et al. (2002) |
| **Self-BLEU** | Diversity within set | Lower = more diverse | Zhu et al. (2018) |

**Anti-Correlation Warning**: High semantic similarity (BERTScore) vs. high diversity (Self-BLEU) are in tension.

### Human Evaluation Protocol

**Annotation Task**: For each pair (A, B), annotators judge:

1. **Semantic equivalence** (5-point scale): 0 = different meaning, 5 = identical meaning
2. **Bidirectional check**: "Would replacing A with B change the meaning?" (Yes/No)
3. **Confidence**: How confident in judgment? (Low/Medium/High)

**Annotation Guidelines** (from STS Benchmark):
- Score 5: Completely equivalent, interchangeable in any context
- Score 4: Mostly equivalent, minor differences in emphasis
- Score 3: Roughly equivalent, some information differs
- Score 2: Not equivalent, but topically related
- Score 1: Different meaning or contradictory
- Score 0: Completely unrelated

**Quality Control**:
- Use 3+ annotators per pair
- Measure inter-annotator agreement (Krippendorff's α or Fleiss' κ)
- Target: α > 0.67 (tentative conclusions), α > 0.80 (definitive)
- Flag disagreements (|score_i - score_j| > 1) for expert review

**Key Literature**:
- Artstein & Poesio (2008): Inter-coder agreement for computational linguistics
- Passonneau & Carpenter (2014): The benefits of a model of annotation

### Edge Cases & Failure Modes

**Common generation failures**:
1. **Semantic drift**: Meaning gradually changes through transformation chain
2. **Negation errors**: "He is happy" → "He is not unhappy" (negation doubled)
3. **Scope ambiguity**: "All students passed some exams" (quantifier scope changed)
4. **Presupposition change**: "Stop smoking" vs "Start not smoking" (different presuppositions)
5. **Pragmatic shift**: "Can you pass the salt?" (question) vs "You can pass the salt" (statement)

**For Your Project**: Create challenge set of these edge cases for robust evaluation.

---

## IX. Recommendations for Your LLM Training Data Project

### Generation Strategy

**Coverage targets**:
- Lexical transformations: 30% of generated pairs (high frequency in natural text)
- Syntactic transformations: 40% (core structural variations)
- Semantic role transformations: 15% (teaches perspective)
- Discourse/pragmatic: 10% (register flexibility)
- Compositional (multi-operation): 5% (hard cases)

**Source text selection**:
- Sample from actual LLM training corpora (C4, The Pile)
- Include diverse domains (code, math, dialogue, explanations)
- Stratify by length: short (1 sent), medium (2-5 sent), long (paragraph)

**Scale recommendations**:
- Start small: 10K validated pairs for proof-of-concept
- Medium scale: 100K pairs for training duplicate detectors
- Large scale: 1M+ pairs for comprehensive landscape analysis

### Quality Validation Thresholds

| Use Case | Automatic Metric Threshold | Human Validation |
|----------|---------------------------|------------------|
| High-precision (deduplication) | BERTScore F1 > 0.90 + Bidirectional entailment | 20% sample |
| Medium-precision (training data) | BERTScore F1 > 0.85 | 10% sample |
| Hard negatives (adversarial) | 0.70 < BERTScore < 0.80 | 30% sample (more ambiguous) |

### Integration with Levels 2-3

**This taxonomy (Level 1) provides foundation for**:
- Level 2 (Functional duplicates): Template-based generation builds on these patterns
- Level 3 (Structural duplicates): Problem isomorphisms may share surface paraphrase patterns

**Interface points**:
- Instruction paraphrases: Use pragmatic + syntactic transformations
- Task description variations: Use lexical + discourse transformations
- Multi-modal duplication: Code comments may paraphrase code functionality

---

## X. Key Citations by Category

### Foundational Paraphrase Theory
- Vila, M., Martí, M. A., & Rodríguez, H. (2014). WRPA: A system for relational paraphrase acquisition from Wikipedia. *Procesamiento del Lenguaje Natural*, 53.
- Bhagat, R., & Hovy, E. (2013). What is a paraphrase? *Computational Linguistics*, 39(3), 463-472.
- Androutsopoulos, I., & Malakasiotis, P. (2010). A survey of paraphrasing and textual entailment methods. *Journal of Artificial Intelligence Research*, 38, 135-187.

### Paraphrase Typologies
- Vila, M., Rodríguez, H., & Martí, M. A. (2015). Paraphrase concept and typology: A linguistically based and computationally oriented approach. *Procesamiento del Lenguaje Natural*, 55, 83-90.
- Kovatchev, V., Martí, M. A., & Salamó, M. (2020). ETPC: A paraphrase and semantic similarity detection corpus. *arXiv preprint arXiv:1810.03809*.

### Paraphrase Datasets
- Zhang, Y., Baldridge, J., & He, L. (2019). PAWS: Paraphrase Adversaries from Word Scrambling. *NAACL*.
- Dolan, W. B., & Brockett, C. (2005). Automatically constructing a corpus of sentential paraphrases. *IWP*.
- Ganitkevitch, J., Van Durme, B., & Callison-Burch, C. (2013). PPDB: The paraphrase database. *NAACL*.

### Semantic Textual Similarity
- Agirre, E., Cer, D., Diab, M., & Gonzalez-Agirre, A. (2012). SemEval-2012 Task 6: A pilot on semantic textual similarity. *SemEval*.
- Cer, D., Diab, M., Agirre, E., Lopez-Gazpio, I., & Specia, L. (2017). SemEval-2017 Task 1: Semantic Textual Similarity. *SemEval*.

### Generation Methods
- Bannard, C., & Callison-Burch, C. (2005). Paraphrasing with bilingual parallel corpora. *ACL*.
- Prakash, A., et al. (2016). Neural paraphrase generation with stacked residual LSTM networks. *COLING*.
- Raffel, C., et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *JMLR* (T5).
- Zhang, J., Zhao, Y., Saleh, M., & Liu, P. (2020). PEGASUS: Pre-training with extracted gap-sentences for abstractive summarization. *ICML*.

### Evaluation
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-Networks. *EMNLP*.
- Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating text generation with BERT. *ICLR*.
- Sellam, T., Das, D., & Parikh, A. P. (2020). BLEURT: Learning robust metrics for text generation. *ACL*.

### Code Clones
- Roy, C. K., Cordy, J. R., & Koschke, R. (2009). Comparison and evaluation of code clone detection techniques and tools. *Computers*, 42(2), 74-83.
- Svajlenko, J., & Roy, C. K. (2015). Evaluating modern clone detection tools. *ICSME*.
- **Allamanis, M., et al. (2025). How the Misuse of a Dataset Harmed Semantic Clone Detection.** *arXiv preprint* (CRITICAL: BigCloneBench Type-4 labels unreliable).

### Context & Pragmatics
- Wegmann, A., et al. (2024). Context-dependent paraphrases in dialogues. *arXiv preprint*.
- Brown, P., & Levinson, S. C. (1987). Politeness: Some universals in language usage. *Cambridge University Press*.
