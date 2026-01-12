"""
Prompts for Murder Mystery Semantic Duplicate Detection

This module contains the prompt templates for:
1. Workflow 1: Direct comparison (PROMPT_MM_DIRECT)
2. Workflow 2a: Entity/structure extraction (PROMPT_MM_EXTRACT)
3. Workflow 2b: Structural comparison (PROMPT_MM_COMPARE)
"""

# =============================================================================
# WORKFLOW 1: Direct Comparison
# =============================================================================

PROMPT_MM_DIRECT = """You are an expert at analyzing murder mystery puzzles for semantic duplication.

## Task
Determine if the following two texts represent the SAME underlying murder mystery puzzle (semantic duplicate) or DIFFERENT puzzles.

## Definition of Semantic Duplicate
Two murder mysteries are semantic duplicates if:
- They have the SAME logical puzzle structure
- A solver would use the SAME deductive reasoning to solve both
- The victim-suspect-evidence relationships are equivalent
- The answer depends on the same logical chain (who has Means, Motive, and Opportunity)

Surface differences that DO NOT matter (these can differ and still be duplicates):
- Character names (e.g., "Detective Winston" vs "Inspector Chen")
- Specific locations (e.g., "luxury restaurant" vs "upscale hotel")
- Specific weapons (e.g., "barbed wire" vs "piano wire")
- Writing style, narrative length, or prose quality
- Red herring details that don't affect the core logic

What DOES matter (these must match for duplicates):
- Number of suspects
- The pattern of who has which MMO elements
- Which suspect is the actual murderer
- The logical structure that proves guilt/innocence

## Test Text (MuSR benchmark sample)
{test_text}

## Corpus Text (Training data sample)
{corpus_text}

## Instructions
1. First, determine if the corpus text is even a murder mystery puzzle at all
2. If not a murder mystery, it cannot be a semantic duplicate
3. If it is a murder mystery, compare the logical structures
4. Consider: same number of suspects? Same MMO distribution? Same answer structure?

## Output Format
Respond with ONLY a valid JSON object (no markdown, no explanation outside JSON):
{{
    "corpus_is_murder_mystery": true/false,
    "corpus_type_if_not_mm": "string describing what the corpus text actually is, or null if it is a murder mystery",
    "is_semantic_duplicate": 0 or 1,
    "confidence": float 0.0 to 1.0,
    "reasoning": "Brief explanation of your verdict (1-3 sentences)"
}}

Be calibrated with confidence:
- 1.0 = Absolutely certain (e.g., corpus is clearly a recipe, not a mystery)
- 0.8-0.9 = Very confident after analysis
- 0.5-0.7 = Some ambiguity, structural similarities but key differences
- < 0.5 = High uncertainty, needs closer review
"""

# =============================================================================
# WORKFLOW 2a: Structure Extraction
# =============================================================================

PROMPT_MM_EXTRACT = """Extract the underlying logical structure from the following murder mystery text.

## Schema
Output a JSON object with exactly this structure:
{{
    "is_valid_murder_mystery": true/false,
    "extraction_notes": "Any issues encountered during extraction",
    "victim": {{
        "name": "victim's name",
        "crime_scene": "location where murder occurred",
        "murder_weapon": "weapon used"
    }},
    "suspects": [
        {{
            "name": "suspect's name",
            "role": "suspect's relationship/role",
            "is_murderer": true/false,
            "mmo": {{
                "means": {{
                    "established": true/false,
                    "facts": ["list of explicit facts establishing means"],
                    "reasoning": "how facts prove means"
                }},
                "motive": {{
                    "established": true/false,
                    "facts": ["list of explicit facts establishing motive"],
                    "reasoning": "how facts prove motive"
                }},
                "opportunity": {{
                    "established": true/false,
                    "facts": ["list of explicit facts establishing opportunity"],
                    "reasoning": "how facts prove opportunity"
                }}
            }},
            "mmo_signature": "string like 'MMO', 'MM_', 'M_O', etc. where _ means not established"
        }}
    ],
    "structural_signature": "concatenated mmo_signatures of all suspects in order",
    "answer_position": "0-indexed position of the murderer in suspects array",
    "num_suspects": "number of suspects",
    "deduction_complexity": "LOW/MEDIUM/HIGH based on how many inference steps needed"
}}

## Instructions
1. Identify the VICTIM: name, crime scene, weapon
2. For EACH SUSPECT, determine Means (access to weapon), Motive (reason to kill), Opportunity (presence at scene)
3. The MURDERER has ALL THREE (MMO) established
4. Create mmo_signature for each suspect: M=means, M=motive, O=opportunity, _=not established
5. Concatenate signatures for structural_signature (e.g., "MMO|MM_|M_O")
6. If the text is NOT a murder mystery, set is_valid_murder_mystery to false

## Murder Mystery Text
{text}

## Output
Respond with ONLY the JSON object (no markdown code blocks, no additional text):
"""

# =============================================================================
# WORKFLOW 2b: Structural Comparison
# =============================================================================

PROMPT_MM_COMPARE = """You are comparing two extracted structural representations of murder mystery puzzles.

## Definition of Semantic Duplicate
Two puzzles are semantic duplicates if they share the SAME LOGICAL STRUCTURE:
- A solver would use the same reasoning pattern to solve both
- The MMO (Means-Motive-Opportunity) distribution across suspects is identical
- The answer is at the same position
- Similar deduction complexity

Surface differences that DO NOT matter:
- Character names
- Specific locations or weapons
- Narrative style or length
- Red herring details

## Structure A (Test/MuSR sample)
{structure_a}

## Structure B (Corpus/Training sample)
{structure_b}

## Comparison Criteria
Evaluate these dimensions:
1. **num_suspects_match**: Same number of suspects?
2. **answer_position_match**: Murderer at same index?
3. **structural_signature_match**: Same MMO patterns? (e.g., "MMO|MM_|M_O" vs "MMO|MM_|M_O")
4. **complexity_match**: Similar deduction depth?

## Decision Rules
- **SEMANTIC DUPLICATE (confidence >= 0.8)**: num_suspects, answer_position, AND structural_signature all match
- **SEMANTIC DUPLICATE (confidence 0.6-0.8)**: answer_position matches AND structural_signature is similar (off by one element)
- **NOT DUPLICATE**: answer_position differs OR structural_signature differs significantly
- **NOT APPLICABLE**: Either structure is not a valid murder mystery

## Output Format
Respond with ONLY a valid JSON object:
{{
    "both_valid_mysteries": true/false,
    "comparison_results": {{
        "num_suspects_match": true/false,
        "answer_position_match": true/false,
        "structural_signature_match": "MATCH/PARTIAL_MATCH/NO_MATCH",
        "complexity_match": true/false
    }},
    "is_semantic_duplicate": 0 or 1,
    "confidence": float 0.0 to 1.0,
    "reasoning": "Brief explanation",
    "key_differences": ["list of structural differences if not duplicate"],
    "key_similarities": ["list of structural similarities"]
}}
"""

