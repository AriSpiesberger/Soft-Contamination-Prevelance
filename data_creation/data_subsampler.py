# -*- coding: utf-8 -*-
"""
Random Paragraph and Sentence Sampler with Multiprocessing
Fixed version with proper reservoir sampling
"""

import os
import json
import random
import nltk
import sys
import hashlib
import multiprocessing
from tqdm import tqdm
from transformers import AutoTokenizer

# --- Configuration ---
DATA_DIR = r"C:\Users\arisp\Documents\Research\SDTD\OLMO_MIX_subsample"
SENTENCE_SAMPLE_SIZE = 100_000
PARAGRAPH_SAMPLE_SIZE = 100_000

# --- MODIFIED: Switched to token-based filtering ---
MIN_SENTENCE_TOKEN_LEN = 5  # Replaces MIN_SENTENCE_LEN
MAX_SENTENCE_TOKEN_LEN = 100 # <-- NEW: Added max for sentences
MIN_PARAGRAPH_TOKEN_LEN = 50 # Replaces MIN_PARAGRAPH_LEN
MAX_PARAGRAPH_TOKEN_LEN = 512 # <-- NEW: Added max for paragraphs

NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)
CHUNK_SIZE = 1000

TOKENIZER_NAME = "Qwen/Qwen3-0.6B" # <-- THIS IS THE CHANGED LINE
OUTPUT_SENTENCES_FILE = r"C:\Users\arisp\Documents\Research\SDTD_Main\data\random_sentences.jsonl"
OUTPUT_PARAGRAPHS_FILE = r"C:\Users\arisp\Documents\Research\SDTD_Main\data\random_paragraphs.jsonl"
# --- Globals for worker processes ---
worker_tokenizer = None


def setup_nltk():
    """Download the 'punkt' tokenizer if needed."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("First-time setup: Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt')


def initialize_worker():
    """Initialize the tokenizer for a new worker process."""
    global worker_tokenizer
    print(f"Initializing tokenizer in worker PID: {os.getpid()}...")
    worker_tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, 
        trust_remote_code=True
    )


def read_data_chunks(jsonl_files, chunk_size):
    """
    Generator that reads lines from files and yields them in chunks.
    """
    chunk = []
    for file_path in jsonl_files:
        source_name = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    chunk.append((line, source_name))
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)
    
    if chunk:
        yield chunk
def process_line_chunk(chunk):
    """
    Process a chunk of lines in a worker process.
    Returns lists of extracted sentences and paragraphs.
    """
    global worker_tokenizer
    
    local_sentences = []
    local_paragraphs = []
    
    total_filtered_sentence_tokens = 0
    total_filtered_paragraph_tokens = 0
    
    for line, source_name in chunk:
        try:
            data = json.loads(line)
            text = data.get('text')
            if not text or not isinstance(text, str):
                continue

            # Split into paragraphs
            paragraphs = text.split('\n\n')
            
            for p in paragraphs:
                p_clean = p.strip()
                if not p_clean:
                    continue
                
                # --- 1. PROCESS PARAGRAPH ---
                p_token_size = len(
                    worker_tokenizer.encode(p_clean, add_special_tokens=False)
                )
                
                # Apply paragraph-level filtering
                if (MIN_PARAGRAPH_TOKEN_LEN <= p_token_size <= MAX_PARAGRAPH_TOKEN_LEN):
                    # Only add to paragraph list if it passes
                    total_filtered_paragraph_tokens += p_token_size
                    
                    p_id = hashlib.sha256(p_clean.encode('utf-8')).hexdigest()
                    p_data = {
                        "id": p_id,
                        "text": p_clean,
                        "source": source_name,
                        "token_size": p_token_size
                    }
                    local_paragraphs.append(p_data)
                
                # --- 2. PROCESS SENTENCES (Corrected Logic) ---
                # This now runs *REGARDLESS* of whether the paragraph was saved.
                # This was the bug: this block was previously nested inside
                # the paragraph filter's 'if' block.
                
                sentences = nltk.sent_tokenize(p_clean)
                for s in sentences:
                    s_clean = s.strip()
                    if not s_clean:
                        continue
                    
                    s_token_size = len(
                        worker_tokenizer.encode(s_clean, add_special_tokens=False)
                    )

                    # Apply sentence-level filtering
                    if not (MIN_SENTENCE_TOKEN_LEN <= s_token_size <= MAX_SENTENCE_TOKEN_LEN):
                        continue
                    
                    total_filtered_sentence_tokens += s_token_size
                    
                    s_id = hashlib.sha256(s_clean.encode('utf-8')).hexdigest()
                    s_data = {
                        "id": s_id,
                        "text": s_clean,
                        "source": source_name,
                        "token_size": s_token_size
                    }
                    local_sentences.append(s_data)
        
        except (json.JSONDecodeError, TypeError):
            continue
        except Exception as e:
            print(f"Error in worker {os.getpid()}: {e}", file=sys.stderr)

    return (
        local_sentences, 
        local_paragraphs, 
        total_filtered_sentence_tokens, 
        total_filtered_paragraph_tokens
    )

class ReservoirSampler:
    """
    A proper reservoir sampler that maintains state internally.
    """
    def __init__(self, sample_size):
        self.sample_size = sample_size
        self.reservoir = []
        self.items_seen = 0
    
    def add(self, item):
        """Add an item using reservoir sampling algorithm."""
        self.items_seen += 1
        
        if len(self.reservoir) < self.sample_size:
            self.reservoir.append(item)
        else:
            # Random index from 0 to items_seen-1
            j = random.randint(0, self.items_seen - 1)
            if j < self.sample_size:
                self.reservoir[j] = item
    
    def get_sample(self):
        """Return the current reservoir."""
        return self.reservoir
    
    def get_count(self):
        """Return total items seen."""
        return self.items_seen


def main():
    setup_nltk()

    # Find all .jsonl files
    print(f"Scanning for .jsonl files in {DATA_DIR}...")
    jsonl_files = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, file))
    
    if not jsonl_files:
        print(f"Error: No .jsonl files found in {DATA_DIR}", file=sys.stderr)
        return
        
    print(f"Found {len(jsonl_files)} files to process with {NUM_WORKERS} workers.")

    # Create reservoir samplers
    sentence_sampler = ReservoirSampler(SENTENCE_SAMPLE_SIZE)
    paragraph_sampler = ReservoirSampler(PARAGRAPH_SAMPLE_SIZE)
    
    # --- NEW: Add grand total accumulators ---
    grand_total_sentence_tokens = 0
    grand_total_paragraph_tokens = 0

    # Create data generator and processing pool
    data_generator = read_data_chunks(jsonl_files, CHUNK_SIZE)
    
    print("Starting processing pool...")
    with multiprocessing.Pool(
        processes=NUM_WORKERS, 
        initializer=initialize_worker
    ) as pool:
        
        pbar = tqdm(
            pool.imap_unordered(process_line_chunk, data_generator),
            desc="Processing chunks",
            unit="chunk"
        )
        
        # --- MODIFIED: Unpack 4 values from the worker ---
        for (
            local_sentences, 
            local_paragraphs, 
            local_s_tokens, 
            local_p_tokens
        ) in pbar:
            
            # Add items to reservoirs
            for s_data in local_sentences:
                sentence_sampler.add(s_data)
            
            for p_data in local_paragraphs:
                paragraph_sampler.add(p_data)
            
            # --- NEW: Accumulate grand totals ---
            grand_total_sentence_tokens += local_s_tokens
            grand_total_paragraph_tokens += local_p_tokens
                
            pbar.set_postfix_str(
                f"Sents: {sentence_sampler.get_count():,}, "
                f"Paras: {paragraph_sampler.get_count():,}"
            )

    print("\n--- Sampling complete. ---")
    print(f"Total paragraphs seen: {paragraph_sampler.get_count():,}")
    print(f"Total sentences seen:  {sentence_sampler.get_count():,}")
    print(f"Paragraph sample size: {len(paragraph_sampler.get_sample()):,}")
    print(f"Sentence sample size:  {len(sentence_sampler.get_sample()):,}")
    
    # --- NEW: Print the grand token totals ---
    print("\n--- Token Counts (for filtered items) ---")
    print(f"Total tokens in filtered paragraphs: {grand_total_paragraph_tokens:,}")
    print(f"Total tokens in filtered sentences:  {grand_total_sentence_tokens:,}")

    # Diagnostic: Check the ratio
    if paragraph_sampler.get_count() > 0 and sentence_sampler.get_count() > 0:
        ratio = sentence_sampler.get_count() / paragraph_sampler.get_count()
        print(f"\nSentences per paragraph ratio: {ratio:.2f}")
        if ratio < 2:
            print("⚠️  WARNING: Very low sentence/paragraph ratio!")
            print("   Consider checking MIN_PARAGRAPH_LEN or data quality.")

    # Save results
    print(f"\nSaving sentences to {OUTPUT_SENTENCES_FILE}...")
    with open(OUTPUT_SENTENCES_FILE, 'w', encoding='utf-8') as f:
        for entry in sentence_sampler.get_sample():
            f.write(json.dumps(entry) + '\n')
            
    print(f"Saving paragraphs to {OUTPUT_PARAGRAPHS_FILE}...")
    with open(OUTPUT_PARAGRAPHS_FILE, 'w', encoding='utf-8') as f:
        for entry in paragraph_sampler.get_sample():
            f.write(json.dumps(entry) + '\n')

    print("--- Done. ---")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
