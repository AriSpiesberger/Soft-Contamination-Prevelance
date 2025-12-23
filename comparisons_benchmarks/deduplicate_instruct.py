#!/usr/bin/env python3
"""
Deduplicate instruct data against benchmark test sets (MUSR, MBPP).
Two-stage deduplication:
1. MinHash-based LSH for fast near-duplicate detection (Jaccard similarity)
2. Embedding-based semantic similarity (cosine similarity)
"""

import os
import json
import gc
import re
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, Dataset
from datasketch import MinHash, MinHashLSH


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def extract_input_output(example):
    """Extract input and output from example in various formats"""
    input_text = ""
    output_text = ""
    
    # Try different common formats
    if 'input' in example and 'output' in example:
        input_text = str(example['input']) if example['input'] else ""
        output_text = str(example['output']) if example['output'] else ""
    elif 'instruction' in example and 'response' in example:
        input_text = str(example['instruction']) if example['instruction'] else ""
        output_text = str(example['response']) if example['response'] else ""
    elif 'prompt' in example and 'completion' in example:
        input_text = str(example['prompt']) if example['prompt'] else ""
        output_text = str(example['completion']) if example['completion'] else ""
    elif 'text' in example:
        # If only text field, try to split or use as input
        text = str(example['text'])
        # Try to split on common separators
        if '\n\n' in text:
            parts = text.split('\n\n', 1)
            input_text = parts[0]
            output_text = parts[1] if len(parts) > 1 else ""
        else:
            input_text = text
            output_text = ""
    else:
        # Fallback: use first string field as input
        for key, value in example.items():
            if isinstance(value, str) and value.strip():
                if not input_text:
                    input_text = value
                elif not output_text:
                    output_text = value
                    break
    
    return input_text.strip(), output_text.strip()


def clean_example_for_json(example):
    """Convert example to JSON-serializable dict with input/output format"""
    input_text, output_text = extract_input_output(example)
    return {
        'input': input_text,
        'output': output_text
    }


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    # Embedding-based deduplication
    model_name: str = 'nvidia/llama-embed-nemotron-8b'
    max_seq_length: int = 512
    embedding_batch_size: int = 16
    similarity_threshold: float = 0.75  # Remove if >= this threshold

    # MinHash-based deduplication
    minhash_num_perm: int = 128  # Number of permutations for MinHash
    minhash_threshold: float = 0.8  # Jaccard similarity threshold
    minhash_ngram: int = 3  # N-gram size for shingling

    output_dir: str = "deduplicated_data"
    cache_dir: str = "dedup_cache"

    # Human-written sources to keep from tulu_3_sft
    human_sources: list = None

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        if self.human_sources is None:
            self.human_sources = [
                'no_robots',
                'oasst1',
                'OpenAssistant',
                'flan_v2',
                'coconot',
                'wildchat',
                'aya',
                'sciriff',
                'table_gpt'
            ]


# =============================================================================
# BENCHMARK LOADING (reuse from fast_run.py)
# =============================================================================
def load_benchmark(name, mode='input_output'):
    """Load benchmark and return texts."""
    if name == 'musr':
        ds = load_dataset("TAUR-Lab/MuSR")
        data = []
        for split in ds:
            for idx, item in enumerate(ds[split]):
                inp = item.get('narrative', item.get('question', ''))
                out = item.get('answer', '')
                data.append({'id': f"{split}_{idx}", 'input': inp, 'output': out})

    elif name == 'mbpp':
        # Try evalplus first, fallback to google-research-datasets
        try:
            ds = load_dataset("evalplus/mbpp", "mbpp")
            print(f"  ✅ Loaded evalplus/mbpp")
        except Exception as e:
            print(f"  ⚠️ evalplus/mbpp not available, trying google-research-datasets/mbpp...")
            ds = load_dataset("google-research-datasets/mbpp", "sanitized")
            print(f"  ✅ Loaded google-research-datasets/mbpp")

        data = []
        target_split = 'test' if 'test' in ds else list(ds.keys())[0]
        print(f"  📊 Using split: {target_split}, found {len(ds[target_split])} items")

        for item in ds[target_split]:
            task_id = str(item.get('task_id', f"mbpp_{len(data)}"))
            input_text = item.get('prompt', item.get('text', ''))
            output_text = item.get('canonical_solution', item.get('code', item.get('solution', '')))

            data.append({
                'id': task_id,
                'input': input_text,
                'output': output_text
            })

        print(f"  ✅ Loaded {len(data)} MBPP items")

    else:
        raise ValueError(f"Unknown benchmark: {name}")

    texts, ids = [], []
    for item in data:
        if mode == 'input':
            text = item['input']
        elif mode == 'output':
            text = item['output']
        else:  # input_output
            text = f"{item['input']}\n\n{item['output']}"

        texts.append(text)
        ids.append(item['id'])

    return texts, ids


# =============================================================================
# EMBEDDING
# =============================================================================
def embed_texts(texts, config, desc="Embedding"):
    """Embed texts using the same model as benchmark analysis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    ).to(device).eval()

    embeddings = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), config.embedding_batch_size), desc=desc):
            batch = texts[i:i+config.embedding_batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=config.max_seq_length,
                return_tensors='pt'
            ).to(device)

            out = model(**enc)
            mask = enc['attention_mask'].unsqueeze(-1).float()
            emb = (out[0] * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = F.normalize(emb, p=2, dim=1)
            embeddings.append(emb.cpu())

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return torch.cat(embeddings, 0).float().numpy()


# =============================================================================
# LOAD INSTRUCT DATA
# =============================================================================
def load_instruct_data(config, sample_size=20000):
    """Load and randomly sample texts from Dolmino 10B dataset."""
    print(f"Loading Dolmino 10B dataset (allenai/dolma3_dolmino_mix-10B-1025)...")
    print(f"Target: {sample_size:,} randomly sampled, deduplicated texts")

    # Load the dataset in streaming mode to avoid downloading everything
    print("  Loading dataset in streaming mode (will stop once we have enough samples)...")
    try:
        ds = load_dataset(
            "allenai/dolma3_dolmino_mix-10B-1025",
            split="train",
            streaming=True
        )
        print("  ✅ Loaded dataset in streaming mode")
    except Exception as e:
        error_msg = f"Could not load dataset in streaming mode: {e}"
        print(f"  ❌ {error_msg}")
        raise ValueError(error_msg)

    # Extract texts and deduplicate - stop early once we have enough
    print("  Extracting texts and deduplicating (streaming, will stop early)...")
    texts_seen = set()
    unique_examples = []  # Store examples directly instead of text keys
    
    # We need at least sample_size unique texts, but collect a bit more for better randomness
    # Stop once we have 2x the sample size (or if we run out of data)
    target_unique = sample_size * 2
    
    # Process dataset to get unique texts - optimized
    processed_count = 0
    iterator = tqdm(ds, desc="Processing examples")
    for example in iterator:
        processed_count += 1
        
        # Extract input/output quickly
        input_text, output_text = extract_input_output(example)
        
        # Combine for deduplication (faster than string operations)
        combined = f"{input_text}|||{output_text}" if input_text or output_text else None
        
        if combined:
            # Normalize whitespace for deduplication (faster: single pass)
            text_normalized = ' '.join(combined.split())
            
            # Skip if we've seen this exact text
            if text_normalized not in texts_seen:
                texts_seen.add(text_normalized)
                # Store as input/output dict directly
                unique_examples.append({'input': input_text, 'output': output_text})
                
                # Stop early if we have enough unique texts
                if len(unique_examples) >= target_unique:
                    print(f"  ✅ Collected {len(unique_examples):,} unique texts (stopping early)")
                    break
        
        # Update progress less frequently for speed
        if processed_count % 100 == 0:
            iterator.set_postfix({'unique': len(unique_examples), 'processed': processed_count})
    
    print(f"  ✅ Found {len(unique_examples):,} unique texts from {processed_count:,} processed examples")

    # Random sampling
    if len(unique_examples) < sample_size:
        print(f"  ⚠️ Only {len(unique_examples):,} unique texts available, using all of them")
        sampled_examples = unique_examples
    else:
        print(f"  🎲 Randomly sampling {sample_size:,} from {len(unique_examples):,} unique texts...")
        import random
        random.seed()  # Use system randomness
        sampled_examples = random.sample(unique_examples, sample_size)
        print(f"  ✅ Sampled {len(sampled_examples):,} texts")

    # Create a Dataset from the sampled examples (already in input/output format)
    sampled_ds = Dataset.from_list(sampled_examples)
    
    print(f"\n✅ Final dataset: {len(sampled_ds):,} randomly sampled, deduplicated texts")
    return sampled_ds


def prepare_instruct_texts(dataset):
    """Convert instruct dataset to text format for embedding."""
    texts = []

    for example in tqdm(dataset, desc="Preparing texts"):
        # Handle input/output format (our standard format)
        if 'input' in example and 'output' in example:
            text = f"{example['input']}\n\n{example['output']}"
        elif 'messages' in example:
            # Messages format
            text_parts = []
            for msg in example['messages']:
                role = msg.get('role', '')
                content = msg.get('content', '')
                text_parts.append(f"{role}: {content}")
            text = "\n".join(text_parts)
        elif 'prompt' in example and 'completion' in example:
            # Prompt-completion format
            text = f"{example['prompt']}\n\n{example['completion']}"
        elif 'text' in example:
            # Raw text format
            text = example['text']
        else:
            # Fallback: convert to JSON string
            text = json.dumps(example)

        texts.append(text)

    return texts


# =============================================================================
# MINHASH DEDUPLICATION
# =============================================================================
def normalize_text(text):
    """Normalize text for better MinHash comparison."""
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip
    text = text.strip()
    return text


def get_ngrams(text, n=3):
    """Generate character n-grams from text."""
    text = normalize_text(text)
    # Character n-grams
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    return ngrams


def create_minhash(text, num_perm=128, ngram=3):
    """Create MinHash signature for a text."""
    m = MinHash(num_perm=num_perm)
    ngrams = get_ngrams(text, n=ngram)
    for ng in ngrams:
        m.update(ng.encode('utf-8'))
    return m


def minhash_deduplicate(instruct_texts, benchmark_texts, config):
    """
    Use MinHash LSH to find near-duplicate instruct examples.
    Returns mask of examples to KEEP (True = keep, False = remove).
    """
    print(f"Creating MinHash LSH index...")
    print(f"  Num permutations: {config.minhash_num_perm}")
    print(f"  Jaccard threshold: {config.minhash_threshold}")
    print(f"  N-gram size: {config.minhash_ngram}")

    # Create LSH index
    lsh = MinHashLSH(
        threshold=config.minhash_threshold,
        num_perm=config.minhash_num_perm
    )

    # Index benchmark texts
    print("Indexing benchmark texts...")
    for i, text in enumerate(tqdm(benchmark_texts, desc="Indexing benchmarks")):
        mh = create_minhash(text, config.minhash_num_perm, config.minhash_ngram)
        lsh.insert(f"bench_{i}", mh)

    # Query with instruct texts
    print("Finding duplicates in instruct data...")
    keep_mask = np.ones(len(instruct_texts), dtype=bool)
    duplicate_indices = []

    for i, text in enumerate(tqdm(instruct_texts, desc="Querying LSH")):
        mh = create_minhash(text, config.minhash_num_perm, config.minhash_ngram)
        # Query LSH - returns list of similar benchmark IDs
        similar = lsh.query(mh)

        if similar:
            # Found at least one similar benchmark example
            keep_mask[i] = False
            duplicate_indices.append(i)

    n_removed = (~keep_mask).sum()
    n_kept = keep_mask.sum()

    print(f"\nMinHash Results:")
    print(f"  Total examples: {len(instruct_texts)}")
    print(f"  Removed (Jaccard >= {config.minhash_threshold}): {n_removed} ({100*n_removed/len(instruct_texts):.2f}%)")
    print(f"  Kept: {n_kept} ({100*n_kept/len(instruct_texts):.2f}%)")

    return keep_mask


# =============================================================================
# EMBEDDING DEDUPLICATION
# =============================================================================
def deduplicate_against_benchmarks(instruct_embeddings, benchmark_embeddings, config):
    """
    Find instruct examples that are too similar to benchmark examples.
    Returns mask of examples to KEEP (True = keep, False = remove).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_instruct = len(instruct_embeddings)
    n_benchmark = len(benchmark_embeddings)

    print(f"Computing similarities: {n_instruct} instruct x {n_benchmark} benchmark...")

    # Move to GPU
    instruct_gpu = torch.from_numpy(instruct_embeddings).to(device)
    benchmark_gpu = torch.from_numpy(benchmark_embeddings).to(device)

    # Process in chunks to avoid OOM
    chunk_size = 1000
    keep_mask = np.ones(n_instruct, dtype=bool)

    for i in tqdm(range(0, n_instruct, chunk_size), desc="Finding duplicates"):
        end_i = min(i + chunk_size, n_instruct)
        chunk = instruct_gpu[i:end_i]

        # Compute similarities: (chunk_size, n_benchmark)
        sims = torch.matmul(chunk, benchmark_gpu.T)

        # Get max similarity for each instruct example
        max_sims = sims.max(dim=1)[0].cpu().numpy()

        # Mark examples to remove (similarity >= threshold)
        keep_mask[i:end_i] = max_sims < config.similarity_threshold

    n_removed = (~keep_mask).sum()
    n_kept = keep_mask.sum()

    print(f"\nResults:")
    print(f"  Total examples: {n_instruct}")
    print(f"  Removed (>= {config.similarity_threshold}): {n_removed} ({100*n_removed/n_instruct:.2f}%)")
    print(f"  Kept: {n_kept} ({100*n_kept/n_instruct:.2f}%)")

    return keep_mask


# =============================================================================
# MAIN
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Deduplicate instruct data against benchmarks")

    # Embedding-based deduplication
    parser.add_argument('--threshold', type=float, default=0.75,
                       help='Cosine similarity threshold (remove if >= this)')

    # MinHash-based deduplication
    parser.add_argument('--minhash-threshold', type=float, default=0.8,
                       help='Jaccard similarity threshold for MinHash (remove if >= this)')
    parser.add_argument('--minhash-num-perm', type=int, default=128,
                       help='Number of permutations for MinHash')
    parser.add_argument('--minhash-ngram', type=int, default=3,
                       help='N-gram size for MinHash shingling')
    parser.add_argument('--skip-minhash', action='store_true',
                       help='Skip MinHash deduplication (only use embeddings)')

    # General options
    parser.add_argument('--benchmarks', nargs='+', default=['musr', 'mbpp'],
                       help='Benchmarks to deduplicate against')
    parser.add_argument('--mode', default='input_output',
                       choices=['input', 'output', 'input_output'],
                       help='What part of benchmarks to compare against')
    parser.add_argument('--skip-embed-instruct', action='store_true',
                       help='Skip embedding instruct data (use cached)')
    parser.add_argument('--skip-embed-benchmarks', action='store_true',
                       help='Skip embedding benchmarks (use cached)')
    parser.add_argument('--sample-size', type=int, default=20000,
                       help='Number of randomly sampled texts to extract (default: 20000)')
    args = parser.parse_args()

    config = Config(
        similarity_threshold=args.threshold,
        minhash_threshold=args.minhash_threshold,
        minhash_num_perm=args.minhash_num_perm,
        minhash_ngram=args.minhash_ngram
    )
    cache_dir = Path(config.cache_dir)

    # -------------------------------------------------------------------------
    # 1. Load instruct data
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1: Load Instruct Data")
    print("="*60)

    instruct_ds = load_instruct_data(config, sample_size=args.sample_size)
    
    # Save the initial randomly sampled dataset
    sample_name = f"random_sample_{args.sample_size}"
    initial_output_path = Path(config.output_dir) / sample_name
    initial_json_path = Path(config.output_dir) / f"{sample_name}.jsonl"
    print(f"\n💾 Saving initial random sample ({len(instruct_ds):,} texts) to:")
    print(f"   Dataset: {initial_output_path}")
    print(f"   JSONL: {initial_json_path}")
    instruct_ds.save_to_disk(str(initial_output_path))
    # Examples are already in input/output format, so just write directly
    with open(initial_json_path, 'w', encoding='utf-8') as f:
        for example in tqdm(instruct_ds, desc="Writing initial sample"):
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f"✅ Initial sample saved!")
    
    instruct_texts = prepare_instruct_texts(instruct_ds)
    original_count = len(instruct_ds)

    # -------------------------------------------------------------------------
    # 2. Load benchmark texts (needed for both MinHash and embeddings)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2: Load Benchmark Texts")
    print("="*60)

    all_benchmark_texts = []
    for bench_name in args.benchmarks:
        print(f"\nLoading {bench_name} texts...")
        bench_texts, bench_ids = load_benchmark(bench_name, args.mode)
        all_benchmark_texts.extend(bench_texts)
        print(f"  Added {len(bench_texts)} examples from {bench_name}")

    print(f"\nTotal benchmark texts: {len(all_benchmark_texts)}")

    # -------------------------------------------------------------------------
    # 3. MinHash Deduplication (FAST - First Pass)
    # -------------------------------------------------------------------------
    minhash_removed = 0
    if not args.skip_minhash:
        print("\n" + "="*60)
        print("STEP 3: MinHash Deduplication (Fast Near-Duplicate Detection)")
        print("="*60)

        minhash_keep_mask = minhash_deduplicate(
            instruct_texts,
            all_benchmark_texts,
            config
        )

        # Filter dataset and texts
        keep_indices = np.where(minhash_keep_mask)[0].tolist()
        instruct_ds = instruct_ds.select(keep_indices)
        instruct_texts = [instruct_texts[i] for i in keep_indices]

        minhash_removed = (~minhash_keep_mask).sum()
        print(f"\nAfter MinHash: {len(instruct_ds)} examples remaining")
    else:
        print("\n" + "="*60)
        print("STEP 3: MinHash Deduplication - SKIPPED")
        print("="*60)

    # -------------------------------------------------------------------------
    # 4. Embed instruct data (only remaining examples)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 4: Embed Instruct Data")
    print("="*60)

    instruct_cache = cache_dir / "instruct_embeddings.npy"
    # Only use cache if we skipped minhash (otherwise shape may be different)
    if instruct_cache.exists() and args.skip_embed_instruct and args.skip_minhash:
        print("Loading cached instruct embeddings...")
        instruct_embeddings = np.load(instruct_cache)
    else:
        instruct_embeddings = embed_texts(instruct_texts, config, desc="Embedding instruct data")
        np.save(instruct_cache, instruct_embeddings)
        print(f"Saved instruct embeddings to {instruct_cache}")

    # -------------------------------------------------------------------------
    # 5. Embed benchmarks
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 5: Embed Benchmarks")
    print("="*60)

    all_benchmark_embeddings = []

    for bench_name in args.benchmarks:
        print(f"\nProcessing {bench_name}...")

        bench_cache = cache_dir / f"{bench_name}_{args.mode}_embeddings.npy"

        if bench_cache.exists() and args.skip_embed_benchmarks:
            print(f"Loading cached {bench_name} embeddings...")
            bench_emb = np.load(bench_cache)
        else:
            bench_texts, bench_ids = load_benchmark(bench_name, args.mode)
            bench_emb = embed_texts(bench_texts, config, desc=f"Embedding {bench_name}")
            np.save(bench_cache, bench_emb)
            print(f"Saved {bench_name} embeddings to {bench_cache}")

        all_benchmark_embeddings.append(bench_emb)

    # Combine all benchmark embeddings
    combined_benchmark_emb = np.vstack(all_benchmark_embeddings)
    print(f"\nCombined benchmark embeddings: {combined_benchmark_emb.shape}")

    # -------------------------------------------------------------------------
    # 6. Embedding-based Deduplication (Semantic Similarity)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 6: Embedding-based Deduplication (Semantic Similarity)")
    print("="*60)

    embedding_keep_mask = deduplicate_against_benchmarks(
        instruct_embeddings,
        combined_benchmark_emb,
        config
    )

    embedding_removed = (~embedding_keep_mask).sum()

    # Filter dataset again
    keep_indices = np.where(embedding_keep_mask)[0].tolist()
    filtered_ds = instruct_ds.select(keep_indices)

    # -------------------------------------------------------------------------
    # 7. Save filtered dataset
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 7: Save Filtered Dataset")
    print("="*60)

    # Save to disk
    output_path = Path(config.output_dir) / "deduplicated_instruct"
    print(f"Saving to {output_path}...")
    filtered_ds.save_to_disk(str(output_path))

    # Also save as JSON for easy inspection
    json_path = Path(config.output_dir) / "deduplicated_instruct.jsonl"
    print(f"Saving to {json_path}...")
    with open(json_path, 'w', encoding='utf-8') as f:
        for example in tqdm(filtered_ds, desc="Writing JSONL"):
            example_dict = clean_example_for_json(example)
            f.write(json.dumps(example_dict, ensure_ascii=False) + '\n')

    # Save metadata with both MinHash and embedding stats
    total_removed = minhash_removed + embedding_removed
    metadata = {
        'original_count': original_count,
        'final_count': len(filtered_ds),
        'total_removed': int(total_removed),
        'minhash_removed': int(minhash_removed),
        'embedding_removed': int(embedding_removed),
        'minhash_settings': {
            'enabled': not args.skip_minhash,
            'threshold': config.minhash_threshold,
            'num_perm': config.minhash_num_perm,
            'ngram': config.minhash_ngram
        },
        'embedding_settings': {
            'threshold': config.similarity_threshold,
            'model': config.model_name
        },
        'benchmarks': args.benchmarks,
        'mode': args.mode,
        'human_sources': config.human_sources
    }

    metadata_path = Path(config.output_dir) / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Done! Filtered dataset saved to {output_path}")
    print(f"📊 Metadata saved to {metadata_path}")

    # Show summary
    print("\n" + "="*60)
    print("DEDUPLICATION SUMMARY")
    print("="*60)
    print(f"Original examples: {original_count:,}")
    print(f"\nStage 1 - MinHash (Jaccard >= {config.minhash_threshold}):")
    print(f"  Removed: {minhash_removed:,} ({100*minhash_removed/original_count:.2f}%)")
    print(f"\nStage 2 - Embeddings (Cosine >= {config.similarity_threshold}):")
    print(f"  Removed: {embedding_removed:,} ({100*embedding_removed/original_count:.2f}%)")
    print(f"\nTotal removed: {total_removed:,} ({100*total_removed/original_count:.2f}%)")
    print(f"Final kept: {len(filtered_ds):,} ({100*len(filtered_ds)/original_count:.2f}%)")
    print(f"\nBenchmarks: {', '.join(args.benchmarks)}")
    print(f"Mode: {args.mode}")


if __name__ == "__main__":
    main()
