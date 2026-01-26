# Prompt for Generating Semantic Duplicates

## Task
Generate 5 paraphrased versions of a programming task description. Each paraphrase should:
- Ask for the same functionality in different words
- Maintain the same technical requirements
- Vary in style (formal, casual, detailed, concise, etc.)

Do NOT change the code solution - only paraphrase the task description.

## Input Format
```
TASK_ID: {id}
ORIGINAL_DESCRIPTION: {description}
ORIGINAL_CODE: {code}
TEST_CASES: {tests}
```

## Output Format
Return exactly 5 paraphrased descriptions, one per line, numbered 1-5.
Keep them concise (1-2 sentences each).

---

## Example

**Input:**
```
TASK_ID: 6
ORIGINAL_DESCRIPTION: Write a python function to check whether the two numbers differ at one bit position only or not.
ORIGINAL_CODE:
def is_Power_Of_Two(x):
    return x and (not(x & (x - 1)))
def differ_At_One_Bit_Pos(a,b):
    return is_Power_Of_Two(a ^ b)
```

**Output:**
1. Create a function that determines if two integers have exactly one differing bit in their binary representations.
2. Write Python code to verify whether two numbers differ by precisely one bit position.
3. Implement a function checking if the XOR of two numbers is a power of two (single bit difference).
4. Build a Python function that returns True when two numbers have only one bit different.
5. Check if two given numbers differ at exactly one binary position using Python.

---

## Batch Processing

For each task in the dataset, generate 5 paraphrases. The code stays the SAME for all 5 paraphrases.

Final training data format:
- task_id, pair_num (1-5), paraphrased_prompt, original_code
