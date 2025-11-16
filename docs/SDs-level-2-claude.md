# Level 2: Structural/Parameter Duplicates for Semantic Duplicate Detection

**Project Context**: LLM Training Data Semantic Duplicate Classification & Generation  
**Document Scope**: Level 2 Semantic Duplicates (Structural/Parameter Variations)  
**Primary Application**: Benchmark Contamination Detection in Training Data  
**Status**: Foundation taxonomy for detection and generation pipeline

---

## Important Scope Note

**This taxonomy covers Level 2 Semantic Duplicates** - problems that are structurally/algorithmically identical but with different surface-level parameters.

### Three-Level Distinction

| Level | Definition | Example A | Example B | Duplicate Type |
|-------|-----------|-----------|-----------|----------------|
| **Level 1** | Same meaning, different words | "Count the apples" | "How many apples are there?" | **Paraphrase** |
| **Level 2** | Same structure, different parameters | "Count 5 apples" | "Count 8 pears" | **Structural duplicate** ← THIS DOCUMENT |
| **Level 3** | Different problems, similar cognitive operations | "Count items" | "Sort items" | **Skill duplicate** |

**What Level 2 covers**: Problems where you can apply the **exact same solution algorithm/method**, just with different input values or surface parameters.

**What Level 2 does NOT cover**:
- Paraphrases (see Level 1 document)
- Different problem types that require similar skills (see Level 3 document)
- Problems with different solution methods

**Core Test**: Can you solve Problem B using the identical algorithm you used for Problem A, just substituting different values?
- **YES** → Level 2 duplicate
- **NO** → Not Level 2 (check Level 1 or Level 3)

---

## I. Core Definitions & Framework

### A. Fundamental Concept

**Structural/Parameter Duplicate**: Two problems that:
1. Share the same abstract problem structure (template)
2. Differ only in surface-level parameters (values, objects, identifiers)
3. Require identical solution methods
4. Have equivalent difficulty levels

**Key Insight**: These duplicates are **invisible to traditional deduplication methods** (n-grams, exact matching) but provide significant unfair advantage in benchmark performance.

### B. The Benchmark Contamination Problem

**Traditional deduplication approach**:
```
"Solve: 4x+3=10"
"Solve: 4y+3=10"
→ N-gram overlap: ~85%
→ Traditional system: "High overlap, probably duplicate" ✓

"Count 5 apples in basket"
"Count 8 pears in basket"
→ N-gram overlap: ~60%
→ Traditional system: "Low overlap, not duplicate" ✗ WRONG!
```

**Reality**: Both pairs are Level 2 duplicates. The second is MISSED by current methods.

**Impact**: Models trained on "Count 5 apples" have unfair advantage on test item "Count 8 pears" - they've learned the exact problem structure, just need to substitute values.

### C. Theoretical Foundation

**Problem Template Theory**:
Every problem can be decomposed into:
- **Template (structure)**: Abstract schema defining problem type, operations, relationships
- **Parameters (slots)**: Concrete values filling the template slots

**Formal representation**:
```
Problem = Template(param₁, param₂, ..., paramₙ)

Example:
Template: "If [n1] [objects] [verb] [value1], find [property] of [n2] [objects]"
Problem A: "If 3 apples cost $6, find cost of 5 apples"
Problem B: "If 4 books weigh 8kg, find weight of 7 books"

→ Same template, different parameters → Level 2 duplicates
```

**Key Literature**:
- Ross (1987): Problem similarity and transfer in algebra word problems
- Gick & Holyoak (1983): Schema induction and analogical transfer
- Chi et al. (1981): Categorization and representation of physics problems by experts and novices

---

## II. Taxonomy of Parameter Substitutions

**Rationale**: Classifies duplicates by which parameters are substituted. Essential for systematic generation and detection.

### A. Object/Entity Substitutions

**Definition**: Replace concrete objects, entities, or agents while preserving problem structure and operations.

#### A.1 Domain-Preserving Substitutions

**Definition**: Substitutions within the same conceptual domain/category.

| Domain | Example A | Example B | Why Preserved |
|--------|-----------|-----------|---------------|
| **Common fruits** | "Count 5 apples" | "Count 5 oranges" | Both familiar, everyday items |
| **People names** | "John runs 10 miles" | "Mary runs 10 miles" | Both common human names |
| **Simple shapes** | "Area of circle" | "Area of square" | Both basic geometric shapes |
| **Programming structures** | "Sort an array" | "Filter an array" | NO - different operations! |
| **Programming structures** | "Sort integers" | "Sort strings" | YES - same operation, different data type |

**Why distinguish**: Domain-preserving substitutions maintain:
- Conceptual familiarity
- Background knowledge requirements
- Difficulty level
- Cognitive context

**Generation priority**: HIGH - these are cleanest Level 2 duplicates for controlled generation.

**Detection note**: Flag as "clean structural duplicate" in datasets.

#### A.2 Cross-Domain Substitutions

**Definition**: Substitutions that change conceptual domain or invoke different background knowledge.

| Domain Change | Example A | Example B | Knowledge Shift |
|---------------|-----------|-----------|-----------------|
| **Everyday → Physics** | "Count 5 apples" | "Count 5 electrons" | Requires physics concepts |
| **Common → Specialized** | "Water boils at 100°C" | "Iron melts at 1538°C" | Specialized material science |
| **Daily life → Technical** | "Sort names alphabetically" | "Sort molecules by atomic weight" | Chemistry knowledge |
| **Concrete → Abstract** | "3 boxes of apples" | "3 sets in topology" | Mathematical abstraction |

**Why distinguish**: Cross-domain substitutions may:
- Introduce domain-specific knowledge requirements
- Change difficulty (unfamiliar domain = harder)
- Invoke different cognitive frameworks
- Require specialized vocabulary

**Generation caution**: Flag or avoid in controlled generation unless explicitly studying domain transfer.

**Detection importance**: CRITICAL to flag these - they indicate when "structural duplicate" crosses into knowledge domain shifts.

**Key Literature**:
- Bassok & Holyoak (1989): Interdomain transfer between isomorphic topics in algebra and physics
- Novick (1988): Analogical transfer, problem similarity, and expertise

#### A.3 Implementation Guidelines

**For Generation**:
```
Prompt (domain-preserving):
"Replace [apples] with another common household item that can be counted/measured in the same way. Stay within everyday objects."

Prompt (flagged cross-domain):
"Replace [apples] with a scientific/technical object that changes the domain but keeps the mathematical structure. Mark this as cross-domain."
```

**For Detection**:
```python
def classify_object_substitution(obj_a, obj_b):
    domain_a = get_conceptual_domain(obj_a)  # e.g., "everyday_items"
    domain_b = get_conceptual_domain(obj_b)  # e.g., "physics_concepts"
    
    if domain_a == domain_b:
        return "domain_preserving"
    else:
        return "cross_domain", flag_knowledge_shift(domain_a, domain_b)
```

---

### B. Variable/Identifier Substitutions

**Definition**: Change symbolic names (variables, function names, labels) without affecting problem semantics.

#### B.1 Mathematical Variables

**Examples**:
```
Original: "Solve for x: 4x+3=10"
Variants:
  → "Solve for y: 4y+3=10"
  → "Solve for z: 4z+3=10"
  → "Solve for t: 4t+3=10"
```

**Properties**:
- Algebraically identical
- Same solution value
- Same solution method
- High n-gram overlap (~90%)

**Why Level 2 duplicate**: Despite high surface similarity, these are functionally identical for testing algebraic problem-solving capability.

#### B.2 Programming Identifiers

**Examples**:
```
Original: "Function foo(x) sorts a list"
Variants:
  → "Function bar(x) sorts a list"
  → "Function sortList(data) sorts a list"
  → "Function arrange(items) sorts a list"
```

**Properties**:
- Same algorithm specification
- Same functionality
- Different naming conventions

#### B.3 Logical Predicates

**Examples**:
```
Original: "For all x: P(x) implies Q(x)"
Variants:
  → "For all y: P(y) implies Q(y)"
  → "For all a: A(a) implies B(a)"
```

**Properties**:
- Same logical structure
- Same inference rules
- Different variable/predicate names

**Generation approach**:
```
Simple rule-based:
- Systematically replace x → y, z, t, a, b...
- Replace foo → bar, baz, qux...

LLM prompt:
"Rewrite this problem using different variable/function names while keeping everything else identical."
```

**Detection approach**:
```
1. Normalize identifiers (x→VAR1, y→VAR2, etc.)
2. Compare normalized forms
3. If identical after normalization → Level 2 duplicate
```

**Key Note**: High n-gram overlap can be misleading - these ARE duplicates despite seeming "too similar."

---

### C. Number/Value Substitutions

**Definition**: Change numerical parameters while preserving mathematical/algorithmic structure.

#### C.1 Strictly Isomorphic (Identical After Normalization)

**Definition**: Numbers change but problem reduces to identical abstract form.

**Examples**:

**Type 1: Proportional Scaling**
```
Original: "2x + 2 = 20"
Variant:  "3x + 3 = 30"

Normalization:
  Original: 2(x + 1) = 20  →  x + 1 = 10
  Variant:  3(x + 1) = 30  →  x + 1 = 10
  → IDENTICAL after factoring out common coefficient
```

**Type 2: Input Value Changes (Same Algorithm)**
```
Original: "Sort the list [3,1,4,2]"
Variants:
  → "Sort the list [7,2,9,1]"
  → "Sort the list [15,3,8,11]"
  
Structure preserved:
  - Same operation (sorting)
  - Same data structure (list)
  - Same algorithm complexity O(n log n)
  - Same number of elements
```

**Type 3: Scale-Invariant Operations**
```
Original: "Find average of: 2, 4, 6"
Variant:  "Find average of: 20, 40, 60"
→ Same proportional structure, scaled by 10
```

**Generation method**:
```python
def generate_isomorphic_variant(equation, scale_factor):
    # Example: 2x+2=20 with scale=1.5
    # → 3x+3=30
    new_coefficients = [c * scale_factor for c in original_coefficients]
    return construct_equation(new_coefficients)
```

**Detection method**:
```python
def are_isomorphic(problem_a, problem_b):
    normalized_a = normalize(problem_a)  # Factor out common terms
    normalized_b = normalize(problem_b)
    return normalized_a == normalized_b
```

#### C.2 Structurally Similar (Same Class, Not Identical)

**Definition**: Same problem type and solution method, but not identical after normalization.

**Examples**:

**Borderline Cases**:
```
Case 1:
  A: "2x + 2 = 20"  →  x = 9
  B: "5x + 7 = 42"  →  x = 7
  
  → Both linear equations
  → Same solution method (isolate variable)
  → But NOT identical structure (different solutions)
  → Verdict: BORDERLINE - probably NOT Level 2 duplicate
```

```
Case 2:
  A: "Sort 4 numbers"
  B: "Sort 100 numbers"
  
  → Same algorithm
  → Different scale (may affect difficulty/method choice)
  → Verdict: BORDERLINE - scale may matter
```

**Decision criterion**: 
- **Level 2**: Solution method identical AND structural complexity identical
- **NOT Level 2**: Solution method similar but structural complexity or difficulty differs

#### C.3 Magnitude/Scale Variations

**Definition**: Same operation, different magnitude of numbers involved.

**Examples**:

**Small Scale Changes (Probably Level 2)**:
```
"Calculate: 2 + 2"  →  "Calculate: 5 + 3"
→ Same operation, similar cognitive load
```

**Large Scale Changes (May Affect Difficulty)**:
```
"Calculate: 2 + 2"  →  "Calculate: 23,456 + 87,234"
→ Same operation, but second requires different method (not mental math)
→ May NOT be Level 2 duplicate due to difficulty shift
```

**Fractional/Irrational Introductions**:
```
"Solve: 2x + 3 = 11"  →  "Solve: 2x + π = 11"
→ Introduces irrational number
→ Changes solution representation
→ NOT Level 2 duplicate (structure modification)
```

**Generation guidelines**:
```
SAFE: Change numbers within same magnitude order
  2 → 5, 10 → 15, 100 → 150

CAUTION: Large magnitude jumps
  2 → 2000 (may change method)
  
AVOID: Type changes
  2 → π (integer → irrational)
  5 → 2+3i (real → complex)
```

**Key Literature**:
- LeFevre et al. (1988): Cognitive arithmetic: Evidence for obligatory activation of arithmetic facts
- Campbell (1994): Numerical cognition: Problem size and domain effects

---

### D. Domain Context Substitutions

**Definition**: Change the narrative context or domain while preserving the abstract problem structure.

#### D.1 Word Problem Domain Transfer

**Examples**:

**Shopping → Physics**:
```
Original: "If 3 apples cost $6, what do 5 apples cost?"
Variant:  "If 3 books weigh 6kg, what do 5 books weigh?"

Preserved:
  - Proportional reasoning structure
  - Same mathematical operation (3:6 :: 5:x)
  - Same solution method

Changed:
  - Domain: commerce → physics
  - Units: dollars → kilograms
  - Context: purchasing → measurement
```

**Geometry → Real-World**:
```
Original: "Find area of rectangle: length=5, width=3"
Variant:  "Find area of garden: length=5m, width=3m"

Preserved:
  - Same formula (A = l × w)
  - Same calculation

Changed:
  - Abstract → concrete context
```

**Transportation → Biology**:
```
Original: "Car travels 60 mph for 2 hours, distance?"
Variant:  "Bird flies 60 km/h for 2 hours, distance?"

Preserved:
  - Distance = rate × time formula
  - Same calculation structure

Changed:
  - Transportation → biological
  - Imperial → metric (if mph→km/h)
```

#### D.2 Cross-Language Code Translation

**Examples**:

**Algorithm Specification Across Languages**:
```
Original: "Python function to reverse a string"
Variant:  "JavaScript function to reverse a string"

Preserved:
  - Same algorithm (string reversal)
  - Same input/output specification
  - Same time/space complexity

Changed:
  - Language syntax
  - Language-specific idioms
```

**Pseudocode → Implementation**:
```
Original: "REVERSE(string s): for i=0 to len(s)/2, swap s[i] and s[len-i-1]"
Variant:  Python/Java/C++ implementation of same algorithm

Preserved:
  - Algorithmic logic
  - Computational steps
  
Changed:
  - Abstraction level
  - Syntax
```

#### D.3 Abstract → Concrete (and vice versa)

**Examples**:

**Abstract Math → Applied Context**:
```
Original: "Solve system of 2 equations in 2 unknowns"
Variant:  "Two trains leave stations, find where they meet"
  (encodes same system of equations)

Preserved:
  - Mathematical structure
  - Solution method (substitution/elimination)

Changed:
  - Presentation (abstract → story)
```

**Generation approach**:
```
LLM Prompt:
"Rewrite this math problem in a [physics/chemistry/economics/biology] context:
- Keep the mathematical structure IDENTICAL
- Use appropriate domain terminology
- Ensure same difficulty level
- Mark domain: [source_domain] → [target_domain]"
```

**Detection importance**: 
- These are true Level 2 duplicates structurally
- BUT flag domain changes for analysis
- May affect difficulty if domain knowledge differs

**Key Literature**:
- Bassok et al. (1995): Priming addition facts with semantic relations
- Reed (1999): Word problems: Research and curriculum reform

---

## III. Structure-Preserving Transformations

**Rationale**: Operations that maintain mathematical/logical equivalence and solution difficulty while changing surface form.

### A. Algebraic Equivalences

**Definition**: Transformations that preserve equation solutions and structural complexity.

#### A.1 Sign Flipping

**Operation**: Multiply all terms by -1.

**Examples**:
```
Original: 4x + 3 = 10
Variant:  -4x - 3 = -10

Verification:
  Original solution: x = 1.75
  Variant solution:  x = 1.75  ✓ IDENTICAL
```

**Properties**:
- Same solution
- Same number of steps
- Same algebraic manipulations required
- Tests same understanding

**Generation rule**:
```python
def sign_flip(equation):
    return multiply_all_terms(equation, -1)
```

#### A.2 Proportional Scaling

**Operation**: Multiply all terms by same non-zero constant.

**Examples**:
```
Original: 2x + 2 = 20
Variant (×2):  4x + 4 = 40
Variant (×0.5): x + 1 = 10

All reduce to: x + 1 = 10
Solution: x = 9 (or proportionally equivalent)
```

**Properties**:
- Solutions equivalent (proportionally)
- Same solution method
- Same difficulty

**Generation rule**:
```python
def scale_equation(equation, factor):
    return multiply_all_terms(equation, factor)
```

#### A.3 Side Rearrangement

**Operation**: Move terms between left and right sides maintaining equality.

**Examples**:
```
Original: x + 5 = 10
Variants:
  → 10 = x + 5         (swap sides)
  → x = 10 - 5         (isolate variable)
  → x - 10 = -5        (move 10 to left)
```

**Properties**:
- Same solution
- Different presentation
- Tests algebraic manipulation understanding

#### A.4 Addition/Subtraction Property

**Operation**: Add/subtract same value to both sides.

**Examples**:
```
Original: x + 3 = 7
Variant:  (x + 3) + 2 = 7 + 2  →  x + 5 = 9

Solution unchanged: x = 4
```

**Caution**: May change apparent difficulty if operation makes solving harder.

#### A.5 Valid Transformations List

| Transformation | Example | Preserves Solution? | Preserves Difficulty? |
|----------------|---------|---------------------|----------------------|
| Sign flip | ax+b=c → -ax-b=-c | ✓ Yes | ✓ Yes |
| Scaling | ax+b=c → kax+kb=kc | ✓ Yes | ✓ Yes |
| Side swap | ax+b=c → c=ax+b | ✓ Yes | ✓ Yes |
| Add constant | x+a=b → x+a+k=b+k | ✓ Yes | ? Maybe |
| Squaring | x=5 → x²=25 | ✗ NO (adds ±) | ✗ NO |
| Adding variable | x=5 → x+y=5+y | ✗ NO (adds DOF) | ✗ NO |

**Key principle**: Transformation valid for Level 2 if:
1. Solution set unchanged (or proportionally scaled)
2. Solution method unchanged
3. Difficulty preserved

**Key Literature**:
- Kieran (1989): The early learning of algebra: A structural perspective
- Sfard & Linchevski (1994): The gains and pitfalls of reification

---

### B. Permutations & Order Changes

**Definition**: Reordering elements that doesn't affect solution difficulty or method.

#### B.1 Commutative Operations

**Examples**:

**Arithmetic**:
```
Original: "Calculate: 3 + 5 + 2"
Variants:
  → "Calculate: 5 + 3 + 2"
  → "Calculate: 2 + 5 + 3"
  
All equivalent due to commutativity of addition
```

**Logical Conjunctions**:
```
Original: "If A and B, then C"
Variant:  "If B and A, then C"

Logically equivalent due to conjunction commutativity
```

#### B.2 List/Array Permutations

**Examples**:

**Sorting Tasks**:
```
Original: "Sort the list: [3, 1, 4, 2]"
Variants:
  → "Sort the list: [2, 4, 1, 3]"
  → "Sort the list: [1, 3, 2, 4]"
  
Properties:
  - Same algorithm (comparison-based sorting)
  - Same time complexity O(n log n)
  - Same number of elements
  - Different input permutations
```

**Search Tasks**:
```
Original: "Find 7 in list: [3,7,1,9,7,5]"
Variants:
  → "Find 7 in list: [7,1,3,5,9,7]"
  
Same search algorithm, different element positions
```

**Generation approach**:
```python
import random

def permute_input(problem, input_list):
    permuted = random.sample(input_list, len(input_list))
    return problem.replace(str(input_list), str(permuted))
```

#### B.3 Proof/Argument Step Reordering

**Examples**:

**Independent Premises**:
```
Original: 
  "Given: (1) All birds fly, (2) Penguins are birds
   Prove: Penguins fly"

Variant:
  "Given: (1) Penguins are birds, (2) All birds fly
   Prove: Penguins fly"
   
Logical equivalence when premises are independent
```

**Caution**: Only valid when order doesn't matter for logical dependency.

**Invalid reordering**:
```
Original: "Let x=5. Then x²=25."
Variant:  "x²=25. Let x=5."
→ NOT equivalent (definition must precede use)
```

**Generation guidelines**:
```
SAFE: Permute independent premises/steps
UNSAFE: Reorder dependent steps
REQUIRES ANALYSIS: Check dependency graph
```

---

### C. Format/Representation Translations

**Definition**: Same problem in different notational systems or presentation formats.

#### C.1 Symbolic ↔ Word Problem Translations

**Examples**:

**Math: Symbolic → Word Problem**:
```
Symbolic: "Solve: 2x + 3 = 11"

Word Problem: "A number is multiplied by 2, then 3 is added, 
               giving a result of 11. Find the number."

Word Problem (Alternative): "John has some apples. He doubles them,
                             adds 3 more, and has 11 total. 
                             How many did he start with?"
```

**Properties**:
- Identical mathematical structure
- Same solution (x = 4)
- Same solution method
- Different presentation abstraction level

**Math: Word Problem → Symbolic**:
```
Word: "If 3 apples cost $6, find cost of 5 apples"

Symbolic: "Given: 3a = 6, solve for 5a"
Or: "Solve: x/3 = 6/3, find 5x/3"
Or: "Proportion: 3:6 :: 5:x"
```

#### C.2 Code Specification Formats

**Examples**:

**Natural Language → Pseudocode → Code**:
```
Natural language:
  "Function that reverses a string"

Pseudocode:
  "REVERSE(string s):
     result = empty
     for each character c in s (from end to start):
       add c to result
     return result"

Python:
  "def reverse(s: str) -> str:
       return s[::-1]"

Java:
  "public String reverse(String s) {
       return new StringBuilder(s).reverse().toString();
   }"
```

**Properties**:
- Same algorithm specification
- Same input/output behavior
- Different abstraction levels
- Different syntax

**Generation approach**:
```
LLM Prompt:
"Convert this problem between formats:
 
Input format: [symbolic/word problem/pseudocode/code]
Output format: [symbolic/word problem/pseudocode/code]

Maintain IDENTICAL problem structure and solution method."
```

#### C.3 Logical Representation Formats

**Examples**:

**Natural Language → First-Order Logic**:
```
Natural: "All birds can fly"
FOL: ∀x (Bird(x) → Flies(x))

Natural: "Some cats are black"  
FOL: ∃x (Cat(x) ∧ Black(x))
```

**Properties**:
- Same logical content
- Different formality levels
- Same inference rules apply

**Key Literature**:
- Larkin & Simon (1987): Why a diagram is (sometimes) worth ten thousand words
- Zhang & Norman (1994): Representations in distributed cognitive tasks

---

### D. Unit/Scale Transformations

**Definition**: Changes in measurement units or scale that preserve proportional structure.

#### D.1 Unit Conversions (Exact)

**Examples**:

**Length**:
```
Original: "Distance: 5 kilometers"
Variants:
  → "Distance: 5000 meters"
  → "Distance: 500000 centimeters"
  → "Distance: 3.107 miles" (if conversion exact)
```

**Temperature**:
```
Original: "Water boils at 100°C"
Variant:  "Water boils at 212°F"

Conversion: F = (9/5)C + 32
```

**Time**:
```
Original: "Duration: 2 hours"
Variants:
  → "Duration: 120 minutes"
  → "Duration: 7200 seconds"
```

**Properties**:
- Mathematically equivalent
- Same physical quantity
- Same proportional relationships

**Caution**: May change difficulty if conversion requires lookup or calculation.

#### D.2 Scale Transformations

**Examples**:

**Proportional Scaling**:
```
Original: "Rectangle: length=5m, width=3m"
Variant:  "Rectangle: length=50cm, width=30cm"

Area calculation:
  Original: 15 m²
  Variant:  1500 cm² = 15 m² ✓ Equivalent
```

**Caution - Non-Proportional**:
```
Original: "Rectangle: length=5m, width=3m"
Variant:  "Rectangle: length=10m, width=3m"

→ NOT proportional scaling (only one dimension scaled)
→ NOT Level 2 duplicate (different shape)
```

**Generation guidelines**:
```python
def unit_transform(problem, from_unit, to_unit):
    conversion_factor = get_conversion_factor(from_unit, to_unit)
    return apply_conversion(problem, conversion_factor)
    
# Ensure proportional relationships preserved
def validate_proportionality(original, variant):
    return check_ratio_preservation(original, variant)
```

**Key Literature**:
- Bassok & Olseth (1995): Object-based representations in arithmetic word problems
- Thompson & Opfer (2010): How 15 hundred is like 15 cherries: Effect of progressive alignment on representational changes in numerical cognition

---

## IV. Detection Methods

**Rationale**: Identify Level 2 duplicates in training data to detect benchmark contamination.

**Challenge**: Traditional methods (n-grams) miss most Level 2 duplicates because surface similarity is often low.

### A. Abstract Problem Representation

**Approach**: Extract canonical/normalized form, compare abstractions rather than surface text.

#### A.1 Template Extraction Pipeline

**Step 1: Parse to Structured Form**
```
Input 1: "If 3 apples cost $6, find cost of 5 apples"
Input 2: "If 4 books weigh 8kg, find weight of 7 books"

Parsed Structure 1:
  - Conditional: "If [quantity1] [object1] [relation] [value1]"
  - Query: "find [property] of [quantity2] [object1]"
  - Values: {quantity1: 3, object1: "apples", relation: "cost", 
             value1: 6, property: "cost", quantity2: 5}

Parsed Structure 2:
  - Conditional: "If [quantity1] [object1] [relation] [value1]"
  - Query: "find [property] of [quantity2] [object1]"
  - Values: {quantity1: 4, object1: "books", relation: "weigh",
             value1: 8, property: "weight", quantity2: 7}
```

**Step 2: Extract Abstract Template**
```
Template (both):
  "If Q1 OBJECTS RELATION V1, find PROPERTY of Q2 OBJECTS"
  
Slot types:
  Q1, Q2: numeric
  OBJECTS: noun (countable)
  RELATION: verb
  V1: numeric
  PROPERTY: noun (measurable)
```

**Step 3: Extract Mathematical Structure**
```
Both encode proportional reasoning:
  Q1 : V1 :: Q2 : x
  Solve for: x = (Q2/Q1) × V1

Mathematical template: PROPORTIONAL_REASONING(Q1, V1, Q2)
```

**Step 4: Compare**
```
Problem 1 template == Problem 2 template?
  → YES
  
Problem 1 math structure == Problem 2 math structure?
  → YES
  
Conclusion: Level 2 duplicate
```

#### A.2 Semantic Role Labeling Approach

**Method**: Use SRL to extract predicate-argument structure.

**Example**:
```
Input: "John gives Mary a book"

SRL Output:
  Predicate: give
  ARG0 (Agent): John
  ARG1 (Theme): book
  ARG2 (Recipient): Mary

Variant: "Alice gives Bob a pen"
  
SRL Output:
  Predicate: give
  ARG0 (Agent): Alice
  ARG1 (Theme): pen
  ARG2 (Recipient): Bob
  
→ Same predicate-argument structure
→ Different fillers
→ Level 2 duplicate candidate
```

**Tools**:
- AllenNLP Semantic Role Labeler
- spaCy with SRL models
- PropBank/FrameNet annotations

**Key Literature**:
- Palmer et al. (2005): The Proposition Bank
- Shi & Lin (2019): Simple BERT models for relation extraction and semantic role labeling

#### A.3 Equation Normalization

**Method**: For mathematical problems, normalize to canonical form.

**Example**:
```
Input 1: "2x + 3 = 11"
Input 2: "4y + 6 = 22"

Step 1: Parse to coefficient form
  [2, 3, 11] for x
  [4, 6, 22] for y

Step 2: Factor out GCD
  [2, 3, 11] → GCD(2, 3, 11) = 1 (no factoring)
  [4, 6, 22] → GCD(4, 6, 22) = 2 → [2, 3, 11]
  
Step 3: Compare normalized forms
  [2, 3, 11] == [2, 3, 11]
  → MATCH → Level 2 duplicate
```

**Implementation**:
```python
from sympy import symbols, simplify, factor

def normalize_equation(eq_string):
    # Parse to sympy expression
    expr = parse_equation(eq_string)
    
    # Factor out common terms
    factored = factor(expr)
    
    # Convert to canonical form (sorted, normalized)
    canonical = to_canonical_form(factored)
    
    return canonical

def are_equations_isomorphic(eq1, eq2):
    norm1 = normalize_equation(eq1)
    norm2 = normalize_equation(eq2)
    return norm1 == norm2
```

---

### B. Structural Similarity Metrics

**Approach**: Quantify structural similarity using specialized distance metrics.

#### B.1 Template Distance

**Definition**: Measure overlap in template structure.

**Method**:
```
1. Extract templates from both problems
2. Align slot types
3. Compute template overlap score

Template Match Score = |matched_slots| / |total_slots|

Threshold: > 0.90 → Level 2 duplicate candidate
```

**Example**:
```
Template 1: "Sort LIST of SIZE elements"
Template 2: "Sort ARRAY of COUNT items"

Alignment:
  LIST ↔ ARRAY (both: data structure)
  SIZE ↔ COUNT (both: quantity)
  elements ↔ items (both: generic object)

Match score: 3/3 = 1.0 → Strong Level 2 candidate
```

#### B.2 Algorithm Fingerprinting

**Definition**: Extract computational signature of solution method.

**Method**:
```
1. Determine algorithm category (sort, search, graph traversal, etc.)
2. Extract complexity profile:
   - Time complexity
   - Space complexity
   - Operation types required
3. Compare fingerprints

Match if:
  - Same algorithm category
  - Same complexity profile
  - Same operation sequence
```

**Example**:
```
Problem 1: "Sort [3,1,4,2]"
Fingerprint:
  - Category: SORT
  - Time: O(n log n)
  - Space: O(1) to O(n) depending on algorithm
  - Operations: COMPARE, SWAP

Problem 2: "Sort [7,2,9,1]"  
Fingerprint:
  - Category: SORT
  - Time: O(n log n)
  - Space: O(1) to O(n)
  - Operations: COMPARE, SWAP
  
→ Fingerprints match → Level 2 duplicate
```

#### B.3 Dependency Graph Comparison

**Definition**: For multi-step problems, compare dependency structures.

**Method**:
```
1. Extract steps and dependencies
2. Build DAG (directed acyclic graph)
3. Compare graph isomorphism

If graphs isomorphic → same problem structure
```

**Example**:
```
Problem 1: 
  "Given x=5, y=2x, z=y+3, find z"
  
Dependency graph:
  x=5 → y=2x → z=y+3

Problem 2:
  "Given a=3, b=4a, c=b+7, find c"
  
Dependency graph:
  a=3 → b=4a → c=b+7

Graph structure identical (linear chain, 3 steps)
→ Level 2 duplicate
```

**Key Literature**:
- VanLehn (1996): Cognitive skill acquisition
- Anderson et al. (1995): Cognitive tutors: Lessons learned

---

### C. Embedding-Based Detection with Structure Awareness

**Approach**: Use semantic embeddings but combine with structural checks to distinguish from Level 1.

#### C.1 Two-Stage Detection

**Stage 1: Embedding Similarity (Screening)**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed problems
emb_a = model.encode(problem_a)
emb_b = model.encode(problem_b)

# Compute similarity
similarity = cosine_similarity(emb_a, emb_b)

# Screening threshold
if similarity < 0.5:
    return "NOT_DUPLICATE"  # Too dissimilar
elif similarity > 0.95:
    return "LEVEL_1_PARAPHRASE"  # Too similar, likely paraphrase
else:
    # Pass to Stage 2 for structural analysis
    pass
```

**Stage 2: Structural Analysis**
```python
def detect_level_2(problem_a, problem_b, embedding_sim):
    # Check for parameter substitutions
    has_different_numbers = check_numeric_differences(problem_a, problem_b)
    has_different_objects = check_object_differences(problem_a, problem_b)
    has_different_variables = check_variable_differences(problem_a, problem_b)
    
    # Check for structural preservation
    same_template = compare_templates(problem_a, problem_b)
    same_math_structure = compare_math_structure(problem_a, problem_b)
    
    if (has_different_numbers or has_different_objects or has_different_variables) \
       and same_template and same_math_structure:
        return "LEVEL_2_DUPLICATE"
    else:
        return "NOT_DUPLICATE"
```

#### C.2 Contrastive Detection Features

**Positive indicators for Level 2**:
- Medium embedding similarity (0.5-0.85)
- Different specific values (numbers, objects, names)
- Same abstract template
- Same mathematical/algorithmic structure

**Negative indicators (likely NOT Level 2)**:
- High embedding similarity (>0.95) → likely Level 1 paraphrase
- Low embedding similarity (<0.5) → likely unrelated
- Different templates → not structural duplicates
- Different solution methods → not Level 2

#### C.3 Feature-Based Classification

**Approach**: Train classifier on hand-labeled examples.

**Features**:
```
Structural features:
  - Template match score
  - Mathematical structure similarity
  - Algorithm category match
  - Dependency graph similarity

Surface features:
  - N-gram overlap
  - Token overlap
  - Edit distance
  - Embedding similarity

Parameter variation features:
  - Number of numeric differences
  - Number of object substitutions
  - Number of variable changes
  - Domain preservation (binary)

Label: {LEVEL_1_PARAPHRASE, LEVEL_2_STRUCTURAL, NOT_DUPLICATE}
```

**Training data**: Manually annotated pairs across all three categories.

**Key Literature**:
- Mikolov et al. (2013): Efficient estimation of word representations in vector space
- Reimers & Gurevych (2019): Sentence-BERT
- Khot et al. (2021): Beyond text: A deep dive into problem solving using structural representations

---

### D. LLM-as-Judge Detection

**Approach**: Use LLM's understanding to identify structural equivalence.

#### D.1 Binary Classification Prompt

**Template**:
```
You are evaluating whether two problems are STRUCTURAL DUPLICATES.

Structural duplicates are problems where:
1. The solution method is IDENTICAL
2. Only surface parameters differ (numbers, object names, variable names)
3. The difficulty level is the same
4. Same abstract problem template

Structural duplicates are NOT:
- Paraphrases (same exact problem, different words)
- Different problem types requiring different methods
- Problems with changed difficulty or complexity

Problem A: {problem_a}
Problem B: {problem_b}

Analysis:
1. What is the solution method for Problem A?
2. What is the solution method for Problem B?
3. Are the solution methods identical? (YES/NO)
4. What parameters differ? (numbers/objects/variables/domain)
5. Is the abstract structure the same? (YES/NO)
6. Is the difficulty preserved? (YES/NO)

Final Classification: [STRUCTURAL_DUPLICATE / PARAPHRASE / NOT_DUPLICATE]

Confidence: [LOW/MEDIUM/HIGH]
```

#### D.2 Multi-LLM Consensus

**Approach**: Query multiple LLMs, require agreement.

**Method**:
```python
def detect_with_consensus(problem_a, problem_b, models=['gpt-4', 'claude-3', 'gemini-pro']):
    votes = []
    
    for model in models:
        response = query_llm(model, detection_prompt(problem_a, problem_b))
        classification = parse_classification(response)
        votes.append(classification)
    
    # Require 2/3 agreement
    if votes.count("STRUCTURAL_DUPLICATE") >= 2:
        return "LEVEL_2_DUPLICATE", votes
    else:
        return "UNCERTAIN", votes
```

#### D.3 Chain-of-Thought Detection

**Approach**: Have LLM reason step-by-step about structural equivalence.

**Prompt**:
```
Analyze if these are structural duplicates using step-by-step reasoning:

Problem A: {problem_a}
Problem B: {problem_b}

Step 1: Extract the abstract problem structure from Problem A
  - What type of problem is this? (sorting/equation/optimization/etc.)
  - What are the parameters (specific values that could change)?
  - What is the core operation or calculation?

Step 2: Extract the abstract problem structure from Problem B
  - What type of problem is this?
  - What are the parameters?
  - What is the core operation?

Step 3: Compare structures
  - Are the problem types the same?
  - Are the parameters in the same positions/roles?
  - Is the core operation identical?
  - Would the same solution algorithm work for both?

Step 4: Check for difficulty preservation
  - Is the computational complexity the same?
  - Are the parameters of similar scale/magnitude?

Step 5: Final determination
Based on steps 1-4, are these STRUCTURAL DUPLICATES?
[YES/NO with detailed reasoning]
```

**Key Literature**:
- Wei et al. (2022): Chain-of-thought prompting elicits reasoning in large language models
- Ye & Durrett (2022): The unreliability of explanations in few-shot prompting for textual reasoning

---

## V. Generation Methods

**Rationale**: Systematically create Level 2 duplicates for:
1. Training duplicate detectors
2. Studying benchmark contamination
3. Understanding what models learn from structural duplicates

**Primary focus**: LLM-based generation with diverse prompting strategies.

### A. LLM Prompting Strategies

#### A.1 Direct Transformation Specification

**Approach**: Explicitly instruct LLM what to change and what to preserve.

**Basic Template**:
```
Transform this problem by changing [TRANSFORMATION_TYPE]:

Original Problem: {problem}

Instructions:
- Change: [specific parameters to modify]
- Preserve: problem structure, solution method, difficulty level
- Generate: [N] distinct variants

Constraints:
- All variants must be solvable with the same algorithm
- Maintain same level of complexity
- Keep within same domain [if domain-preserving required]

Output format:
Variant 1: [problem]
Variant 2: [problem]
...
```

**Specific Transformation Prompts**:

**Number Substitution**:
```
"Generate 5 variants of this equation by changing the numbers while keeping the structure identical:

Original: 2x + 3 = 11

Requirements:
- Keep it as a linear equation (ax + b = c form)
- Use different integers for a, b, c
- Solution should be different for each variant
- Similar difficulty (no huge numbers)

Variant 1:
Variant 2:
..."
```

**Object Substitution**:
```
"Generate 10 variants by replacing the object:

Original: Count 5 apples in the basket

Requirements:
- Replace 'apples' with other countable objects
- Keep within everyday items (domain-preserving)
- Maintain exact grammatical structure
- Keep 'in the basket' context

Variant 1: Count 5 [object] in the basket
Variant 2: Count 5 [object] in the basket
..."
```

**Variable Renaming**:
```
"Rewrite this logical statement with different variable names:

Original: For all x in domain D, if P(x) then Q(x)

Generate 5 variants:
- Use different variable letters (y, z, t, a, b...)
- Use different predicate names
- Keep logical structure IDENTICAL
- One variant per line"
```

**Domain Transfer**:
```
"Rewrite this math word problem in a different domain:

Original: If 3 apples cost $6, what do 5 apples cost?

Target domain: [physics/biology/chemistry/sports]

Requirements:
- Keep proportional reasoning structure IDENTICAL
- Use appropriate domain terminology
- Same mathematical relationship (3:6 :: 5:x)
- Similar difficulty level

Mark domain change: [shopping] → [target_domain]

Variant:"
```

---

#### A.2 Few-Shot Learning with Pattern Examples

**Approach**: Show examples of transformation pattern, let LLM generalize.

**Template**:
```
Here are examples of structurally identical problems with different parameters:

Example Set 1 - Proportional Reasoning:
A: "If 3 apples cost $6, find cost of 5 apples"
B: "If 4 books weigh 8kg, find weight of 7 books"
C: "If 2 cars need 10 gallons, how much for 9 cars?"

Example Set 2 - Linear Equations:
A: "Solve: 2x + 3 = 11"
B: "Solve: 5y + 7 = 32"
C: "Solve: 3z + 1 = 19"

Example Set 3 - Sorting:
A: "Sort the list: [3, 1, 4, 2]"
B: "Sort the array: [7, 2, 9, 1]"
C: "Order these numbers: [5, 8, 1, 6]"

Now, given this target problem, generate 10 new problems that follow the same pattern:

Target: {problem}

Variant 1:
Variant 2:
...
Variant 10:
```

**Advantage**: LLM learns the implicit transformation pattern without explicit rules.

**Refinement Prompt**:
```
"Review the variants you generated. For each one:
1. Verify it uses the same solution method as the original
2. Check that only parameters changed
3. Confirm difficulty is equivalent

Mark any variants that don't meet criteria and regenerate them."
```

---

#### A.3 Chain-of-Thought Generation

**Approach**: Have LLM explicitly plan transformation before executing.

**Template**:
```
Generate structural duplicates using step-by-step reasoning:

Original Problem: {problem}

Step 1: Analyze the structure
- What is the problem type?
- What are the changeable parameters?
- What must stay constant?
- What is the solution method?

Step 2: Plan transformations
- Which parameters to change?
- What values/objects to use?
- Which domain (if cross-domain)?
- How to preserve difficulty?

Step 3: Generate variants
Based on the plan, create [N] variants:

Step 4: Validate
For each variant, verify:
- Same structure? ✓/✗
- Same solution method? ✓/✗
- Same difficulty? ✓/✗

Only keep validated variants.

[Execute steps above]
```

**Example Execution**:
```
Original: "Sort the list [3,1,4,2] in ascending order"

Step 1: Analyze
- Type: Sorting problem
- Changeable: The specific numbers in the list
- Constant: Operation (sort), order direction (ascending), data structure (list), size (4 elements)
- Solution: Any comparison-based sorting algorithm

Step 2: Plan
- Change: Use different 4-element integer lists
- Range: Similar magnitude (single digits to keep comparable)
- Preserve: List size, ascending order requirement

Step 3: Generate
Variant 1: "Sort the list [7,2,9,1] in ascending order"
Variant 2: "Sort the list [5,8,1,6] in ascending order"
Variant 3: "Sort the list [4,9,2,7] in ascending order"

Step 4: Validate
All variants: ✓ Same structure, ✓ Same method, ✓ Same difficulty
```

---

#### A.4 Compositional (Multi-Step) Transformations

**Approach**: Apply multiple transformations sequentially to create more complex variants.

**Template**:
```
Apply these transformations in sequence:

Original: {problem}

Transformation sequence:
1. [First transformation]: {description}
2. [Second transformation]: {description}
3. [Third transformation]: {description}

Show intermediate results:

After Step 1: [problem]
After Step 2: [problem]
Final Result: [problem]

Verify: Does final result still match original structure? [YES/NO]
```

**Example - Mathematical Problem**:
```
Original: "Solve: 2x + 3 = 11"

Sequence:
1. Change variable: x → y
2. Scale equation: multiply all terms by 2
3. Convert to word problem

After Step 1: "Solve: 2y + 3 = 11"
After Step 2: "Solve: 4y + 6 = 22"
After Step 3: "A number is multiplied by 4, then 6 is added, giving 22. Find the number."

Verification:
✓ Same structure (linear equation)
✓ Same solution method (isolate variable)
✓ Solution equivalent (y = 4 in both)
```

**Caution**: Each transformation must preserve Level 2 properties. Validate after each step.

---

#### A.5 Temperature and Sampling Strategies

**Approach**: Control LLM generation parameters for desired diversity.

**For Diversity** (exploratory generation):
```python
generation_params = {
    'temperature': 1.0,      # Higher = more diverse
    'top_p': 0.95,           # Nucleus sampling
    'n': 20,                 # Generate many candidates
}

# Generate many variants, filter for quality
variants = generate_variants(problem, params=generation_params)
validated = [v for v in variants if validate_structural_match(v, problem)]
```

**For Precision** (controlled generation):
```python
generation_params = {
    'temperature': 0.3,      # Lower = more conservative
    'top_p': 0.9,
    'n': 5,                  # Fewer, higher-quality
}

variants = generate_variants(problem, params=generation_params)
```

**For Specific Transformation Types**:
```python
# Number substitution: low temperature (precise transformation)
params_numeric = {'temperature': 0.2, 'top_p': 0.8}

# Domain transfer: higher temperature (more creative)
params_domain = {'temperature': 0.8, 'top_p': 0.95}

# Object substitution: medium temperature
params_object = {'temperature': 0.5, 'top_p': 0.9}
```

---

### B. Transformation Type Library

**Rationale**: Comprehensive catalog of transformations across domains for systematic coverage.

#### B.1 Mathematics Domain

**Linear Equations**:
```
Transformations:
1. Coefficient scaling: 2x+3=11 → 4x+6=22
2. Sign flipping: 2x+3=11 → -2x-3=-11
3. Variable substitution: x → y, z, t, w, a, b...
4. Constant shifting: x+3=7 → x+5=9 (add 2 to both sides)
5. Side swapping: 2x+3=11 → 11=2x+3
6. Format: symbolic ↔ word problem

LLM Prompt:
"Apply [transformation_type] to: {equation}
Verify solution unchanged (or proportionally scaled)"
```

**Word Problems - Proportional Reasoning**:
```
Transformations:
1. Object substitution (domain-preserving):
   apples → oranges → bananas → peaches
   
2. Object substitution (cross-domain):
   apples → books → cars → molecules
   
3. Domain transfer:
   shopping → physics → chemistry → biology
   
4. Number scaling (proportional):
   (3,6,5) → (6,12,10) → (9,18,15)
   
5. Unit changes:
   dollars → euros, meters → feet

LLM Prompt:
"Transform this proportional reasoning problem:
Original: If 3 {object1} {relation} {value1}, find {property} of 5 {object1}
Target: Use different numbers and [specify object/domain change]"
```

**Geometric Problems**:
```
Transformations:
1. Shape substitution (same dimension):
   circle → square → triangle (area calculations)
   
2. Measurement unit:
   cm → m → km → inches → feet
   
3. Scale changes (proportional):
   5x5 square → 10x10 square
   
4. Dimension (if operation generalizes):
   2D circle → 3D sphere (volume vs area)

Caution: Dimension changes often NOT Level 2 (different formulas)
```

---

#### B.2 Programming/Algorithms Domain

**Sorting Problems**:
```
Transformations:
1. Data type preservation:
   sort integers → sort floats → sort characters
   
2. Input permutation:
   [3,1,4,2] → [2,4,1,3] → [1,3,2,4]
   
3. Size variation (same complexity class):
   4 elements → 5 elements → 10 elements
   (Caution: very large sizes may change difficulty)
   
4. Container type:
   list → array → vector
   
5. Language:
   Python → Java → C++ (same algorithm)

LLM Prompt:
"Generate sorting problem variants:
- Keep algorithm the same (comparison-based sort)
- Change: [data values/language/container type]
- Same input size: {n} elements"
```

**Search Problems**:
```
Transformations:
1. Target value: search for 7 → search for 15
2. Data permutation: [3,7,1,9] → [1,9,3,7]
3. Search algorithm (same problem):
   linear → binary (if sorted) [Actually different! NOT Level 2]
4. Data structure: array → linked list (same linear search)

Caution: Algorithm type changes are NOT Level 2
```

**Function Specifications**:
```
Transformations:
1. Identifier renaming:
   function foo(x) → function bar(y)
   
2. Parameter naming:
   reverse(string) → reverse(text) → reverse(s)
   
3. Language translation:
   Python spec → Java spec → JavaScript spec
   
4. Format:
   natural language → pseudocode → actual code

LLM Prompt:
"Translate this function specification to {target_language}:
{spec}
Keep: exact same algorithm, input/output behavior
Change: only syntax and language-specific idioms"
```

---

#### B.3 Logic & Reasoning Domain

**Syllogisms**:
```
Transformations:
1. Predicate substitution:
   "All birds fly" → "All fish swim" → "All cars drive"
   
2. Entity substitution:
   "Socrates is mortal" → "Plato is mortal" → "Aristotle is mortal"
   
3. Domain transfer:
   biological → social → mathematical properties
   
4. Variable renaming:
   P(x) → Q(y) → R(z)

LLM Prompt:
"Rewrite this syllogism with different predicates/entities:
{syllogism}
Keep: logical structure (All A are B, C is A, therefore C is B)
Change: the specific predicates and entities"
```

**Proof Problems**:
```
Transformations:
1. Axiom relabeling: A1, A2, A3 → P1, P2, P3
2. Theorem domain: geometry → algebra → number theory (if structure same)
3. Step reordering (when commutative)
4. Variable renaming throughout proof

Caution: Most proof structure changes are NOT Level 2
```

---

#### B.4 Natural Language Domain

**Reading Comprehension**:
```
Transformations:
1. Passage topic (preserving question type):
   - Climate passage + "What caused X?" question
   → Economics passage + "What caused Y?" question
   
2. Named entity replacement:
   John → Mary, London → Paris (keep relationships)
   
3. Time period (if question structure allows):
   1800s events → 1900s events
   
4. Question rephrase (this is Level 1, not Level 2!)

LLM Prompt:
"Create new passage + question with:
- Different topic: {new_topic}
- Same question structure: {question_type}
- Same difficulty level
- Same answer type (explanation/number/yes-no/etc.)"
```

**Sentiment/Classification**:
```
Transformations:
1. Domain: movie review → book review → restaurant review
2. Subject: specific movie → different movie
3. Length: short review → medium review (same sentiment)

Preserve: sentiment polarity, classification task type
```

---

#### B.5 Domain-Specific Transformations

**Physics**:
```
Kinematics problems:
1. Object substitution: ball → block → car → projectile
2. Quantity scaling: distance/time/velocity (proportional)
3. Unit system: SI → imperial → natural units
4. Dimension: 1D → 2D (if problem structure generalizes)

LLM Prompt:
"Rewrite this kinematics problem with:
- Different object: {new_object}
- Different values: scale by {factor}
- Same physical principle: {principle}
- Same equation: {equation_type}"
```

**Chemistry**:
```
Reaction problems:
1. Element/compound substitution (same reaction type)
2. Stoichiometric coefficient scaling (proportional)
3. Notation: Lewis structures ↔ chemical equations
4. Phase: solid → liquid → gas (if principle same)

Caution: Reaction type changes often NOT Level 2
```

**Biology**:
```
Inheritance problems:
1. Species substitution: pea plants → fruit flies → humans
2. Trait substitution: flower color → eye color → wing shape
3. Inheritance pattern preserved: Mendelian ratios same
4. Generation: F1 → F2 → F3 (same genetic principle)

LLM Prompt:
"Transform this genetics problem:
- Use different species: {species}
- Use different trait: {trait}
- Keep inheritance pattern: {pattern} (e.g., dominant/recessive)
- Same Punnett square structure"
```

---

### C. Quality Validation & Difficulty Preservation

**Rationale**: Generated duplicates must truly preserve structure and difficulty.

#### C.1 Automated Validation Checks

**Structural Equivalence**:
```python
def validate_structural_match(original, variant):
    """Check if variant is valid Level 2 duplicate"""
    
    checks = {
        'template_match': compare_templates(original, variant) > 0.9,
        'solution_method': same_algorithm(original, variant),
        'difficulty_preserved': difficulty_ratio(original, variant) in [0.9, 1.1],
        'parameters_changed': has_parameter_substitutions(original, variant),
    }
    
    # All checks must pass
    return all(checks.values()), checks
```

**Solution Verification**:
```python
def verify_solution_equivalence(problem_a, problem_b):
    """Verify problems have equivalent solutions"""
    
    # Solve both
    solution_a = solve_problem(problem_a)
    solution_b = solve_problem(problem_b)
    
    # Check equivalence (exact or proportional)
    if is_proportional_scaling(problem_a, problem_b):
        scale_factor = get_scale_factor(problem_a, problem_b)
        return abs(solution_a * scale_factor - solution_b) < EPSILON
    else:
        return solution_a == solution_b
```

**Difficulty Estimation**:
```python
def estimate_difficulty(problem):
    """Estimate relative difficulty"""
    
    factors = {
        'token_count': len(tokenize(problem)),
        'number_magnitude': max(extract_numbers(problem)),
        'concept_prerequisites': count_required_concepts(problem),
        'computational_steps': count_solution_steps(problem),
    }
    
    # Weighted combination
    difficulty_score = (
        0.2 * factors['token_count'] +
        0.3 * log(factors['number_magnitude']) +
        0.3 * factors['concept_prerequisites'] +
        0.2 * factors['computational_steps']
    )
    
    return difficulty_score

def difficulty_preserved(problem_a, problem_b, threshold=0.15):
    """Check if difficulty within acceptable range"""
    diff_a = estimate_difficulty(problem_a)
    diff_b = estimate_difficulty(problem_b)
    
    ratio = abs(diff_a - diff_b) / max(diff_a, diff_b)
    return ratio < threshold
```

---

#### C.2 LLM-as-Validator

**Validation Prompt**:
```
Validate if this is a true Level 2 structural duplicate:

Original: {problem_a}
Generated Variant: {problem_b}

Validation Checklist:

1. Structural Match
   - Same problem type? (YES/NO)
   - Same abstract template? (YES/NO)
   - Same solution algorithm? (YES/NO)

2. Parameter Changes Only
   - What changed? (list specifics)
   - Are changes only to values/objects/variables? (YES/NO)
   - Is the problem structure preserved? (YES/NO)

3. Difficulty Preservation
   - Original difficulty: (EASY/MEDIUM/HARD)
   - Variant difficulty: (EASY/MEDIUM/HARD)
   - Are they equivalent? (YES/NO)

4. Common Failure Modes Check
   - Does variant introduce new concepts? (YES/NO)
   - Does variant change computational complexity? (YES/NO)
   - Does variant require additional knowledge? (YES/NO)

Final Verdict: [VALID_LEVEL_2_DUPLICATE / INVALID]

If INVALID, specify reason: [structural_change/difficulty_change/new_concepts/other]
```

**Multi-Validator Consensus**:
```python
def validate_with_consensus(original, variant, models=['gpt-4', 'claude-3', 'gemini']):
    """Require majority agreement across multiple LLMs"""
    
    validations = []
    for model in models:
        result = llm_validate(model, original, variant)
        validations.append(result)
    
    # Count VALID verdicts
    valid_count = sum(1 for v in validations if v['verdict'] == 'VALID_LEVEL_2_DUPLICATE')
    
    # Require 2/3 consensus
    if valid_count >= 2:
        return True, validations
    else:
        return False, validations
```

---

#### C.3 Common Failure Mode Detection

**Failure Mode 1: Difficulty Change**
```
Detection:
- Number magnitude: 2+2 vs 234567+987654
- Concept introduction: integers vs complex numbers
- Scale: 4-element list vs 10000-element list

Flag as: "potential_difficulty_change"
Action: Review or regenerate
```

**Failure Mode 2: Domain Knowledge Invocation**
```
Detection:
- Cross-domain substitution: apples → electrons
- Specialized terminology: everyday → technical
- Background knowledge: common sense → domain expertise

Flag as: "cross_domain_knowledge"
Action: Mark for separate analysis or filter out
```

**Failure Mode 3: Hidden Complexity Introduction**
```
Detection:
- Rational → irrational: 2 → π
- Finite → infinite: sum of list → series convergence
- Deterministic → probabilistic: sort → random sampling

Flag as: "complexity_change"
Action: Reject variant
```

**Failure Mode 4: Structure Modification**
```
Detection:
- Solution method changes: linear → quadratic equation
- Additional constraints: unconstrained → constrained optimization
- Dimensionality: 2D → 3D (with formula change)

Flag as: "structure_modified"
Action: Reject variant (not Level 2)
```

**Automated Detection**:
```python
def detect_failure_modes(original, variant):
    """Identify common generation failures"""
    
    failures = []
    
    # Check difficulty change
    if not difficulty_preserved(original, variant):
        failures.append('difficulty_change')
    
    # Check domain shift
    if is_cross_domain(original, variant):
        failures.append('cross_domain_knowledge')
    
    # Check complexity introduction
    if introduces_new_complexity(original, variant):
        failures.append('complexity_change')
    
    # Check structure modification
    if not same_template(original, variant):
        failures.append('structure_modified')
    
    return failures
```

---

#### C.4 Human Validation Protocol

**When human validation is needed**:
- Ambiguous automated results
- Novel transformation types
- Cross-domain substitutions (verify difficulty preservation)
- Quality checking of generation pipeline

**Annotation Task**:
```
For each pair (Original, Variant), annotators answer:

Q1: Are these structurally identical problems?
    (Can you solve Variant using the EXACT same method as Original?)
    [ ] YES - same algorithm, only parameters differ
    [ ] NO - different solution methods
    [ ] UNCERTAIN

Q2: What changed?
    [ ] Numbers/values
    [ ] Objects/entities
    [ ] Variables/identifiers
    [ ] Domain/context
    [ ] Other: _______________

Q3: Is difficulty preserved?
    [ ] YES - same difficulty
    [ ] NO - variant is easier
    [ ] NO - variant is harder
    [ ] UNCERTAIN

Q4: Confidence in judgment:
    [ ] LOW
    [ ] MEDIUM
    [ ] HIGH

Label: [LEVEL_1_PARAPHRASE / LEVEL_2_STRUCTURAL / NOT_DUPLICATE]
```

**Quality Control**:
- Use 3 annotators per pair
- Measure inter-annotator agreement (Krippendorff's α)
- Target: α > 0.70 for Level 2 labels
- Resolve disagreements through discussion

**Key Literature**:
- Artstein & Poesio (2008): Inter-coder agreement for computational linguistics
- Passonneau & Carpenter (2014): The benefits of a model of annotation

---

### D. Systematic Coverage Strategy

**Goal**: Ensure comprehensive coverage across transformation types and domains.

#### D.1 Coverage Matrix

**Dimensions to Cover**:
```
1. Transformation Type:
   - Object substitution (domain-preserving)
   - Object substitution (cross-domain)
   - Number/value substitution
   - Variable/identifier substitution
   - Algebraic transformation
   - Format translation
   - Domain transfer

2. Domain:
   - Mathematics (algebra, geometry, calculus)
   - Programming (algorithms, data structures)
   - Logic & reasoning
   - Natural language tasks
   - Physics
   - Chemistry
   - Biology
   - Other domain-specific

3. Complexity Level:
   - Simple (1-step problems)
   - Medium (2-3 steps)
   - Complex (multi-step)

Target: N examples per cell in coverage matrix
```

**Example Coverage Plan**:
```
For benchmark B with 500 problems:

Generate per original problem:
  - 5 domain-preserving object substitutions
  - 2 cross-domain substitutions (flagged)
  - 5 number variations (isomorphic)
  - 3 algebraic transformations
  - 2 format translations
  
Total: ~17 variants per problem
Dataset size: 500 × 17 = 8,500 Level 2 duplicates
```

#### D.2 Generation Pipeline

**End-to-End Workflow**:
```
Input: Benchmark B (test problems)

Step 1: Problem Analysis
  - Parse each problem
  - Extract template
  - Identify changeable parameters
  - Determine problem domain

Step 2: Transformation Selection
  - Select applicable transformation types
  - Assign target counts per type
  - Prioritize by coverage gaps

Step 3: Batch Generation
  - Generate variants using LLM prompting
  - Apply multiple transformation types
  - Track metadata (transformation type, domain, etc.)

Step 4: Automated Validation
  - Run structural equivalence checks
  - Verify difficulty preservation
  - Detect failure modes
  - Filter invalid variants

Step 5: Human Validation (Sample)
  - Validate 10% random sample
  - Focus on cross-domain and complex cases
  - Measure agreement with automated validation

Step 6: Dataset Assembly
  - Compile validated variants
  - Add metadata
  - Create train/val/test splits
  - Document generation process

Output: Validated Level 2 duplicate dataset D
```

#### D.3 Metadata Tracking

**Essential Metadata per Generated Pair**:
```json
{
  "original_id": "benchmark_problem_127",
  "variant_id": "generated_variant_127_03",
  "transformation_type": "object_substitution",
  "transformation_subtype": "domain_preserving",
  "domain_original": "shopping",
  "domain_variant": "shopping",
  "parameters_changed": ["object", "numbers"],
  "generation_method": "llm_prompting",
  "llm_model": "gpt-4",
  "prompt_template": "object_substitution_v2",
  "validation_status": "validated",
  "automated_checks": {
    "template_match": 0.95,
    "difficulty_ratio": 1.02,
    "failure_modes": []
  },
  "human_validated": false,
  "notes": ""
}
```

**Uses of Metadata**:
- Analyze which transformations are most/least successful
- Stratified sampling for experiments
- Filter by transformation type for focused studies
- Track generation quality over time

---

## VI. Key Literature & References

### Foundational Theory

**Problem Similarity & Transfer**:
- Ross, B. H. (1987). This is like that: The use of earlier problems and the separation of similarity effects. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 13(4), 629-639.
- Gick, M. L., & Holyoak, K. J. (1983). Schema induction and analogical transfer. *Cognitive Psychology*, 15(1), 1-38.
- Chi, M. T., Feltovich, P. J., & Glaser, R. (1981). Categorization and representation of physics problems by experts and novices. *Cognitive Science*, 5(2), 121-152.

**Analogical Reasoning**:
- Bassok, M., & Holyoak, K. J. (1989). Interdomain transfer between isomorphic topics in algebra and physics. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 15(1), 153-166.
- Novick, L. R. (1988). Analogical transfer, problem similarity, and expertise. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 14(3), 510-520.

**Problem Representation**:
- Larkin, J., & Simon, H. A. (1987). Why a diagram is (sometimes) worth ten thousand words. *Cognitive Science*, 11(1), 65-100.
- Zhang, J., & Norman, D. A. (1994). Representations in distributed cognitive tasks. *Cognitive Science*, 18(1), 87-122.

### Benchmark Contamination

**Contamination Detection & Impact**:
- Lewis, P., et al. (2020). Question and answer test-train overlap in open-domain question answering datasets. *EACL*.
- Yang, Y., et al. (2023). Rethinking benchmark and contamination for language models with rephrased samples. *arXiv preprint*.
- Elazar, Y., et al. (2023). What's in my big data? *arXiv preprint*.
- Haimes, E., et al. (2024). Benchmark inflation: Revealing LLM performance gaps using retro-holdouts. *arXiv preprint*.

**Deduplication Methods**:
- Lee, K., et al. (2021). Deduplicating training data makes language models better. *ACL*.
- Abbas, A., et al. (2023). SemDeDup: Data-efficient learning at web-scale through semantic deduplication. *arXiv preprint*.
- Silcock, J., et al. (2022). Noise-robust de-duplication at scale. *arXiv preprint*.

### Detection Methods

**Semantic Similarity**:
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP*.
- Mikolov, T., et al. (2013). Efficient estimation of word representations in vector space. *ICLR*.

**Structural Analysis**:
- Palmer, M., et al. (2005). The Proposition Bank: An annotated corpus of semantic roles. *Computational Linguistics*, 31(1), 71-106.
- Shi, P., & Lin, J. (2019). Simple BERT models for relation extraction and semantic role labeling. *arXiv preprint*.

### Generation Methods

**LLM Prompting**:
- Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS*.
- Brown, T., et al. (2020). Language models are few-shot learners. *NeurIPS* (GPT-3).

**Problem Generation**:
- Polozov, O., et al. (2015). FlashMeta: A framework for inductive program synthesis. *OOPSLA*.
- Zhou, Y., & Daneshyari, M. (2012). Generating problems and just-in-time learning content from enriched question templates. *SIGCSE*.

### Mathematical Cognition

**Number Processing**:
- LeFevre, J. A., et al. (1988). Cognitive arithmetic: Evidence for obligatory activation of arithmetic facts. *Memory & Cognition*, 16(1), 45-53.
- Campbell, J. I. (1994). Numerical cognition: Problem size and domain effects. *Psychological Research*, 56(4), 271-277.

**Problem Solving**:
- VanLehn, K. (1996). Cognitive skill acquisition. *Annual Review of Psychology*, 47, 513-539.
- Anderson, J. R., et al. (1995). Cognitive tutors: Lessons learned. *The Journal of the Learning Sciences*, 4(2), 167-207.

### Domain-Specific

**Algebra**:
- Kieran, C. (1989). The early learning of algebra: A structural perspective. In S. Wagner & C. Kieran (Eds.), *Research issues in the learning and teaching of algebra* (pp. 33-56).
- Sfard, A., & Linchevski, L. (1994). The gains and the pitfalls of reification—The case of algebra. *Educational Studies in Mathematics*, 26(2-3), 191-228.

**Word Problems**:
- Reed, S. K. (1999). Word problems: Research and curriculum reform. Lawrence Erlbaum Associates.
- Bassok, M., et al. (1995). Priming addition facts with semantic relations. *Memory & Cognition*, 23(4), 422-439.
- Thompson, C. A., & Opfer, J. E. (2010). How 15 hundred is like 15 cherries: Effect of progressive alignment on representational changes in numerical cognition. *Child Development*, 81(6), 1768-1786.

### Code & Algorithms

**Code Clone Detection**:
- Roy, C. K., et al. (2009). Comparison and evaluation of code clone detection techniques and tools: A qualitative approach. *ACM Computing Surveys*, 41(3), 1-74.

**Algorithm Understanding**:
- Khot, T., et al. (2021). Beyond text: A deep dive into problem solving using structural representations. *EMNLP*.
