"""
Gemini Thinking API - Streaming Implementation

Uses streaming API to work around the bug where non-streaming 
doesn't return thought summaries. See GEMINI_THINKING_BUG.md
"""

from dotenv import load_dotenv
from google import genai
from google.genai import types
import json
from datetime import datetime

load_dotenv()

client = genai.Client()
MODEL_ID = "gemini-3-flash-preview"

# Pricing per 1M tokens (USD) - Gemini 3 Flash Preview paid tier
PRICING = {
    "input": 0.50,      # $0.50 per 1M input tokens
    "output": 3.00,     # $3.00 per 1M output tokens (including thinking)
}


def generate_with_thinking(
    prompt: str,
    model: str = MODEL_ID,
    thinking_level: str = "Medium",
    debug: bool = False,
) -> dict:
    """
    Generate response with thinking enabled using streaming API.
    
    Args:
        prompt: The input prompt
        model: Model ID to use
        thinking_level: "Low", "Medium", or "High"
        debug: If True, print debug info about each chunk
    
    Returns:
        dict with thoughts, answer, and usage metadata
    """
    response_stream = client.models.generate_content_stream(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=thinking_level,
                include_thoughts=True
            )
        )
    )
    
    # Collect all parts from stream
    thoughts = []
    answer_parts = []
    usage_metadata = None
    model_version = None
    
    for chunk in response_stream:
        # Capture metadata from last chunk
        if chunk.usage_metadata:
            usage_metadata = chunk.usage_metadata
        if chunk.model_version:
            model_version = chunk.model_version
            
        # Extract parts
        if chunk.candidates and chunk.candidates[0].content:
            for part in chunk.candidates[0].content.parts:
                is_thought = getattr(part, 'thought', False)
                text = getattr(part, 'text', None)
                
                if debug and text:
                    preview = text[:80].replace('\n', '\\n')
                    print(f"[{'THOUGHT' if is_thought else 'ANSWER'}] {preview}...")
                
                if is_thought:
                    if text:
                        thoughts.append(text)
                elif text:
                    answer_parts.append(text)
    
    # Join raw parts
    raw_answer = "".join(answer_parts)
    
    # Handle <|thought|> delimiter that may appear in answer
    # This marker indicates where thinking ends and real answer begins
    THOUGHT_DELIMITER = "<|thought|>"
    if THOUGHT_DELIMITER in raw_answer:
        leaked_thought, clean_answer = raw_answer.split(THOUGHT_DELIMITER, 1)
        # Move leaked thinking to thoughts
        if leaked_thought.strip():
            thoughts.append(f"\n[LEAKED THINKING]\n{leaked_thought}")
        raw_answer = clean_answer
    
    # Build result
    result = {
        "prompt": prompt,
        "model": model,
        "model_version": model_version,
        "thinking_level": thinking_level,
        "timestamp": datetime.now().isoformat(),
        "thoughts": thoughts,
        "thought_summary": "".join(thoughts),
        "answer": raw_answer,
        "usage": None
    }
    
    if usage_metadata:
        prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0) or 0
        answer_tokens = getattr(usage_metadata, 'candidates_token_count', 0) or 0
        thought_tokens = getattr(usage_metadata, 'thoughts_token_count', 0) or 0
        total_tokens = getattr(usage_metadata, 'total_token_count', 0) or 0
        
        # Calculate costs
        prompt_cost = (prompt_tokens / 1_000_000) * PRICING["input"]
        thinking_cost = (thought_tokens / 1_000_000) * PRICING["output"]
        answer_cost = (answer_tokens / 1_000_000) * PRICING["output"]
        total_cost = prompt_cost + thinking_cost + answer_cost
        
        result["usage"] = {
            "prompt_tokens": prompt_tokens,
            "answer_tokens": answer_tokens,
            "thought_tokens": thought_tokens,
            "total_tokens": total_tokens,
        }
        result["cost"] = {
            "prompt": prompt_cost,
            "thinking": thinking_cost,
            "output": answer_cost,
            "total": total_cost,
        }
    
    return result


def save_response(result: dict, filename: str = "response_output.json"):
    """Save response to JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved to {filename}")


def print_response(result: dict):
    """Pretty print the response."""
    print("\n" + "=" * 60)
    print("THOUGHT SUMMARY")
    print("=" * 60)
    if result["thought_summary"]:
        print(result["thought_summary"])
    else:
        print("(No thoughts returned)")
    
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["answer"])
    
    print("\n" + "-" * 60)
    print("TOKEN USAGE")
    print("-" * 60)
    if result.get("usage"):
        print(f"  Prompt:   {result['usage']['prompt_tokens']:,}")
        print(f"  Thinking: {result['usage']['thought_tokens']:,}")
        print(f"  Output:   {result['usage']['answer_tokens']:,}")
        print(f"  TOTAL:    {result['usage']['total_tokens']:,}")
    
    print("\n" + "-" * 60)
    print("COST (USD) - Gemini 3 Flash Preview paid tier")
    print("-" * 60)
    if result.get("cost"):
        print(f"  Prompt:   ${result['cost']['prompt']:.6f}")
        print(f"  Thinking: ${result['cost']['thinking']:.6f}")
        print(f"  Output:   ${result['cost']['output']:.6f}")
        print(f"  TOTAL:    ${result['cost']['total']:.6f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Show debug info for each chunk")
    parser.add_argument("--level", default="Medium", choices=["Low", "Medium", "High"])
    args = parser.parse_args()
    
    # Example prompt
    prompt = "What is the sum of the first 50 prime numbers? Show your work."
    
    print(f"Model: {MODEL_ID}")
    print(f"Prompt: {prompt}")
    print(f"Thinking level: {args.level}")
    print("Generating response with thinking enabled...")
    
    if args.debug:
        print("\n--- DEBUG: Chunk-by-chunk classification ---")
    
    # Generate
    result = generate_with_thinking(prompt, thinking_level=args.level, debug=args.debug)
    
    # Display
    print_response(result)
    
    # Save
    save_response(result, "response_output.json")
