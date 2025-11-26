# -*- coding: utf-8 -*-
"""
Full Sample Processor - Extracts all sentences and paragraphs from Dolma 3 dataset
Similar to data_subsampler but without subsampling - keeps all valid chunks
"""

import json
import gzip
import io
import hashlib
import multiprocessing
import nltk
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import tiktoken

try:
    import zstandard as zstd
except ImportError:
    print("Error: zstandard library is required for reading .zst files.")
    print("Install it with: pip install zstandard")
    raise

# --- Configuration ---
DATA_DIR = "data_creation/dolma3_sample"  # Relative path to input data
OUTPUT_SENTENCES_FILE = "data/full_sentences.jsonl"  # Relative path for sentences output
OUTPUT_PARAGRAPHS_FILE = "data/full_paragraphs.jsonl"  # Relative path for paragraphs output

# Token-based filtering (same as data_subsampler)
MIN_SENTENCE_TOKEN_LEN = 5
MAX_SENTENCE_TOKEN_LEN = 100
MIN_PARAGRAPH_TOKEN_LEN = 50
MAX_PARAGRAPH_TOKEN_LEN = 512

# Use tiktoken for fast, lightweight token counting
# cl100k_base is a common encoding that gives reasonable approximations
TOKENIZER_ENCODING = "cl100k_base"  # GPT-4 style encoding, fast and lightweight

# Performance settings
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)
WRITE_BATCH_SIZE = 1000  # Buffer writes to reduce I/O overhead


def setup_nltk():
    """Download the 'punkt' tokenizer if needed."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("First-time setup: Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt')


def get_all_data_files(root_dir):
    """
    Recursively finds all data files in a directory, handling nested HF structures.
    Targeting Dolma 3 formats: .parquet (primary) and .json.gz (legacy/raw).
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_path.absolute()}")

    # Recursive glob for common large-dataset formats
    # Dolma 3 uses .jsonl.zst files
    parquet_files = list(root_path.rglob("*.parquet"))
    json_gz_files = list(root_path.rglob("*.json.gz"))
    jsonl_files = list(root_path.rglob("*.jsonl"))
    zst_files = list(root_path.rglob("*.zst"))

    all_files = parquet_files + json_gz_files + jsonl_files + zst_files
    # Remove duplicates (in case of overlapping patterns)
    all_files = list(set(all_files))
    
    # Count JSONL.ZST separately for reporting
    jsonl_zst_count = sum(1 for f in all_files if str(f).endswith(".jsonl.zst"))
    
    print(f"Found {len(all_files)} files in {root_path}:")
    print(f" - {len(parquet_files)} Parquet files")
    print(f" - {len(json_gz_files)} JSON.GZ files")
    print(f" - {len(jsonl_files)} JSONL files")
    print(f" - {len(zst_files)} ZST files (including {jsonl_zst_count} JSONL.ZST)")
    
    return all_files


def read_file_content(filepath, data_dir):  
    """
    Reads content agnostic of format (Parquet vs JSONL vs ZST).
    Returns a generator of document dicts with source metadata added.
    """
    filepath = Path(filepath)
    source_name = filepath.name
    try:
        source_path = str(filepath.relative_to(Path(data_dir)))
    except ValueError:
        # If filepath is not relative to data_dir, use the full path
        source_path = str(filepath)
    
    filepath_str = str(filepath)
    
    if filepath_str.endswith(".parquet"):
        # Parquet is columnar; we convert to records for row-wise processing
        df = pd.read_parquet(filepath_str)
        records = df.to_dict(orient="records")
        for doc in records:
            doc['_source_file'] = source_name
            doc['_source_path'] = source_path
            yield doc
        
    elif filepath_str.endswith(".gz"):
        with gzip.open(filepath_str, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    doc = json.loads(line)
                    doc['_source_file'] = source_name
                    doc['_source_path'] = source_path
                    yield doc
        
    elif filepath_str.endswith(".zst") or filepath_str.endswith(".jsonl.zst"):
        # Zstandard compressed JSONL files - read line by line for memory efficiency
        dctx = zstd.ZstdDecompressor()
        with open(filepath_str, 'rb') as f:
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for line in text_stream:
                    line = line.strip()
                    if line:  # Skip empty lines
                        doc = json.loads(line)
                        doc['_source_file'] = source_name
                        doc['_source_path'] = source_path
                        yield doc
        
    elif filepath_str.endswith(".jsonl"):
        with open(filepath_str, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    doc = json.loads(line)
                    doc['_source_file'] = source_name
                    doc['_source_path'] = source_path
                    yield doc


def process_document(doc, tokenizer, _sent_tokenizer=None):
    """
    Process a single document to extract sentences and paragraphs.
    Returns lists of sentence and paragraph dicts.
    Uses lightweight tokenizer for relative token counts (can be updated later).
    """
    sentences = []
    paragraphs = []
    
    # Extract text from document
    text = doc.get('text')
    if not text or not isinstance(text, str):
        return sentences, paragraphs
    
    source_name = doc.get('_source_file', 'unknown')
    source_path = doc.get('_source_path', 'unknown')
    
    # Split into paragraphs
    para_list = text.split('\n\n')
    
    for p in para_list:
        p_clean = p.strip()
        if not p_clean:
            continue
        
        # Process paragraph - use lightweight tokenizer for relative counts
        p_token_size = len(tokenizer.encode(p_clean))
        
        # Apply paragraph-level filtering
        if MIN_PARAGRAPH_TOKEN_LEN <= p_token_size <= MAX_PARAGRAPH_TOKEN_LEN:
            p_id = hashlib.sha256(p_clean.encode('utf-8')).hexdigest()
            p_data = {
                "id": p_id,
                "text": p_clean,
                "source": source_name,
                "source_path": source_path,
                "token_size": p_token_size  # Approximate count, can be updated later
            }
            paragraphs.append(p_data)
        
        # Process sentences from this paragraph (regardless of whether paragraph passed filter)
        sent_list = nltk.sent_tokenize(p_clean)
        for s in sent_list:
            s_clean = s.strip()
            if not s_clean:
                continue
            
            # Use lightweight tokenizer for relative counts
            s_token_size = len(tokenizer.encode(s_clean))
            
            # Apply sentence-level filtering
            if MIN_SENTENCE_TOKEN_LEN <= s_token_size <= MAX_SENTENCE_TOKEN_LEN:
                s_id = hashlib.sha256(s_clean.encode('utf-8')).hexdigest()
                s_data = {
                    "id": s_id,
                    "text": s_clean,
                    "source": source_name,
                    "source_path": source_path,
                    "token_size": s_token_size  # Approximate count, can be updated later
                }
                sentences.append(s_data)
    
    return sentences, paragraphs


def process_file_chunk(args):
    """
    Process a chunk of documents from a file.
    Used for multiprocessing.
    """
    file_path, data_dir, tokenizer_encoding = args
    
    # Initialize tokenizer in worker process
    tokenizer = tiktoken.get_encoding(tokenizer_encoding)
    
    sentences = []
    paragraphs = []
    
    try:
        for doc in read_file_content(file_path, data_dir):
            doc_sentences, doc_paragraphs = process_document(doc, tokenizer, None)
            sentences.extend(doc_sentences)
            paragraphs.extend(doc_paragraphs)
    except Exception as e:
        return (sentences, paragraphs, str(e))
    
    return (sentences, paragraphs, None)


def main():
    setup_nltk()
    
    print(f"Scanning {DATA_DIR} recursively...")
    
    try:
        files = get_all_data_files(DATA_DIR)
    except FileNotFoundError as e:
        print(e)
        return

    if not files:
        print(f"[ERROR] No data files found in tree at {DATA_DIR}.")
        return

    # Tokenizer will be initialized in worker processes
    print(f"\nUsing lightweight tokenizer ({TOKENIZER_ENCODING}) in worker processes...")

    # Ensure output directories exist
    sentences_path = Path(OUTPUT_SENTENCES_FILE)
    paragraphs_path = Path(OUTPUT_PARAGRAPHS_FILE)
    sentences_path.parent.mkdir(parents=True, exist_ok=True)
    paragraphs_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_sentences = 0
    total_paragraphs = 0
    
    print("\nProcessing files...")
    print(f"Using {NUM_WORKERS} worker processes")
    print(f"Sentences will be saved to: {OUTPUT_SENTENCES_FILE}")
    print(f"Paragraphs will be saved to: {OUTPUT_PARAGRAPHS_FILE}")
    
    # Prepare arguments for multiprocessing
    process_args = [(f, DATA_DIR, TOKENIZER_ENCODING) for f in files]
    
    # Batch write buffers
    sent_buffer = []
    para_buffer = []
    
    with open(OUTPUT_SENTENCES_FILE, 'w', encoding='utf-8') as sent_file, \
         open(OUTPUT_PARAGRAPHS_FILE, 'w', encoding='utf-8') as para_file:
        
        # Process files in parallel
        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            results = pool.imap_unordered(process_file_chunk, process_args)
            
            for sentences, paragraphs, error in tqdm(results, total=len(files), desc="Processing files"):
                if error:
                    print(f"\nWarning: Error in worker: {error}")
                    continue
                
                # Add to buffers
                sent_buffer.extend(sentences)
                para_buffer.extend(paragraphs)
                
                # Batch write when buffers are full
                if len(sent_buffer) >= WRITE_BATCH_SIZE:
                    for s_data in sent_buffer:
                        sent_file.write(json.dumps(s_data, ensure_ascii=False) + '\n')
                    total_sentences += len(sent_buffer)
                    sent_buffer.clear()
                
                if len(para_buffer) >= WRITE_BATCH_SIZE:
                    for p_data in para_buffer:
                        para_file.write(json.dumps(p_data, ensure_ascii=False) + '\n')
                    total_paragraphs += len(para_buffer)
                    para_buffer.clear()
        
        # Write remaining buffers
        if sent_buffer:
            for s_data in sent_buffer:
                sent_file.write(json.dumps(s_data, ensure_ascii=False) + '\n')
            total_sentences += len(sent_buffer)
        
        if para_buffer:
            for p_data in para_buffer:
                para_file.write(json.dumps(p_data, ensure_ascii=False) + '\n')
            total_paragraphs += len(para_buffer)

    print("\nProcessing complete!")
    print(f"Total sentences extracted: {total_sentences:,}")
    print(f"Total paragraphs extracted: {total_paragraphs:,}")
    print(f"Sentences saved to: {OUTPUT_SENTENCES_FILE}")
    print(f"Paragraphs saved to: {OUTPUT_PARAGRAPHS_FILE}")


if __name__ == "__main__":
    main()
