
# Load the MuSR murder mystery dataset
# with open("murder_mystery_regenerated_first173.json") as f:
#     data = json.load(f)
# print(data[0].keys())
# dict_keys(['context', 'questions', 'original_sample_id', 'new_story', 'original_story', 'modification_type', 'variant_index', 'random_seed', 'generation_cost'])

#!/usr/bin/env python3
"""
Script to answer murder mystery questions using OpenRouter's gpt-5-nano model.
Outputs JSONL format for crash-resilient real-time saving.
"""
import json
import os
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import argparse

# OpenRouter configuration
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

MODEL = "openai/gpt-4.1-mini"
MODEL_SHORT = "gpt41mini"

# Murder mystery hint from eval.py
HINT = """Before selecting a choice, explain your reasoning step by step. The murderer needs to have a means (access to weapon), motive (reason to kill the victim), and opportunity (access to crime scene) in order to have killed the victim. Innocent suspects may have two of these proven, but not all three. An innocent suspect may be suspicious for some other reason, but they will not have all of motive, means, and opportunity established.

If you believe that both suspects have motive, means, and opportunity, you should make an educated guess pick the one for whom these are best established. If you believe that neither suspect has all three established, then choose the suspect where these are most clearly established."""


def build_user_prompt(story: str, question: dict) -> str:
    """Build the user prompt for a murder mystery question (no system prompt)."""
    choices = "\n".join([f'{idx + 1} - {x}' for idx, x in enumerate(question["choices"])])
    prompt = f"""{story}

{question["question"]}

Pick one of the following choices:
{choices}

You must pick one option. {HINT} Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice number)\""""
    return prompt


def get_model_answer(user_prompt: str) -> dict:
    """Query the model and return the response (no system prompt)."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2048,
            temperature=0.7,
        )
        output = response.choices[0].message.content
        return {"output": output, "error": None}
    except Exception as e:
        return {"output": None, "error": str(e)}


def parse_answer(output: str, num_choices: int) -> int | None:
    """Parse the model's answer from the output."""
    if output is None:
        return None
    
    # Look for "ANSWER:" pattern
    lines = [x.split('answer:')[-1].strip() 
             for x in output.lower().split('\n') 
             if 'answer:' in x and len(x.split('answer:')[-1].strip()) > 0]
    
    if lines:
        answer_text = lines[-1]
        for i in range(1, num_choices + 1):
            if str(i) in answer_text:
                return i
    return None


def count_existing_lines(filepath: Path) -> int:
    """Count existing lines in JSONL file to support resume."""
    if not filepath.exists():
        return 0
    with open(filepath, 'r') as f:
        return sum(1 for _ in f)


def main(
    input_path: str, output_path: str = None, model: str = MODEL, model_short: str = MODEL_SHORT, original_story: bool = False,
):
    if output_path is None:
        output_path = Path(__file__).parent / 'datasets' / 'teacher_answers' / 'musr' / f"{str(input_path).split('/')[-1]}_{MODEL_SHORT}.jsonl"
    if isinstance(output_path, str):
        output_path = Path(output_path)

    # Load data
    print(f"Loading data from {input_path}...")
    with open(input_path) as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Check for existing progress to resume
    existing_count = count_existing_lines(output_path)
    if existing_count > 0:
        print(f"Found {existing_count} existing entries, resuming from there...")
        data = data[existing_count:]
    
    # Open file in append mode for crash-resilient writing
    with open(output_path, 'a') as outfile:
        total_correct = 0
        total_questions = 0
        
        for sample in tqdm(data, desc="Processing samples"):
            # Use new_story for generating answers
            story = sample["new_story"] if not original_story else sample["original_story"]
            questions = sample.get("new_questions") or sample.get("questions", [])
            
            # Process each question
            for q_idx, question in enumerate(questions):
                # Build user prompt (this is what we save for finetuning)
                user_prompt = build_user_prompt(story, question)
                response = get_model_answer(user_prompt)
                
                parsed_answer = parse_answer(response["output"], len(question["choices"]))
                gold_answer = question["answer"] + 1  # 1-indexed
                is_correct = parsed_answer == gold_answer
                
                # Build the result record in OpenAI messages format for direct finetuning use
                result = {
                    # Messages format (matches OpenAI/HuggingFace finetuning format)
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": response["output"]}
                    ],
                    # Metadata from source data
                    "original_sample_id": sample["original_sample_id"],
                    "new_story": sample["new_story"],
                    "original_story": sample["original_story"],
                    "modification_type": sample["modification_type"],
                    "variant_index": sample["variant_index"],
                    "random_seed": sample["random_seed"],
                    "questions": questions,
                    # Answer metadata
                    "question_index": q_idx,
                    "gold_answer": gold_answer,
                    "model_answer": parsed_answer,
                    "correct": is_correct,
                    "error": response["error"]
                }
                
                # Write immediately and flush - crash resilient!
                outfile.write(json.dumps(result) + '\n')
                outfile.flush()
                
                total_questions += 1
                if is_correct:
                    total_correct += 1
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total questions answered: {total_questions}")
    print(f"Correct: {total_correct}/{total_questions} ({100*total_correct/max(1,total_questions):.1f}%)")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    default_input_path = Path(__file__).parent / 'datasets' / 'duplicates' / "musr" / "level0_murder_mystery_regenerated_samples-250_variants-2.json"
    default_output_path = None

    parser = argparse.ArgumentParser(description="Answer questions using OpenRouter api.")
    parser.add_argument("-i", "--input_path", type=str, default=default_input_path, help="Path to input JSON file")
    parser.add_argument("-o", "--output_path", type=str, default=default_output_path, help="Path to output JSONL file")
    parser.add_argument("--model", type=str, default=MODEL, help="Model to use")
    parser.add_argument("--model-short", type=str, default=MODEL_SHORT, help="Model short name")
    parser.add_argument("--original-story", action="store_true", help="Use original story instead of new story")
    args = parser.parse_args()
    main(**vars(args))