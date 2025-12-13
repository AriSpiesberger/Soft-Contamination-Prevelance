#!/usr/bin/env python3
"""
FAST Contamination Analysis
- Direct disk reads (no queue overhead)
- Background S3 uploads
- GPU stays busy
"""

import os
import json
import gc
import pickle
import time
import threading
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import pyarrow.parquet as pq
import boto3

# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    model_name: str = 'nvidia/llama-embed-nemotron-8b'
    max_seq_length: int = 512
    embedding_batch_size: int = 16
    local_data_dir: str = "/lambda/nfs/embeddings/embedding_folder"
    output_dir: str = "results_fast"
    checkpoint_dir: str = "checkpoints_fast"
    
    s3_bucket: str = "dolmo-3-sampling"
    s3_prefix: str = "contamination_analysis_fast"
    
    save_every_files: int = 50  # Checkpoint every N files
    top_k: int = 100
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


# =============================================================================
# BACKGROUND S3 UPLOADER
# =============================================================================
class BackgroundUploader:
    """Fire-and-forget S3 uploads - never blocks GPU work."""
    def __init__(self, bucket):
        self.bucket = bucket
        self.s3 = boto3.client('s3')
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.futures = []
    
    def upload(self, local_path, s3_key):
        """Queue upload, return immediately."""
        future = self.executor.submit(self._do_upload, local_path, s3_key)
        self.futures.append(future)
        # Clean up completed futures
        self.futures = [f for f in self.futures if not f.done()]
    
    def _do_upload(self, local_path, s3_key):
        try:
            self.s3.upload_file(local_path, self.bucket, s3_key)
        except Exception as e:
            print(f"  ⚠️ S3 upload failed {s3_key}: {e}")
    
    def wait_all(self):
        """Wait for all pending uploads."""
        for f in self.futures:
            f.result()


# =============================================================================
# FAST PARQUET READER
# =============================================================================
def read_embeddings_fast(file_path):
    """
    Read embeddings from parquet as fast as possible.
    Returns: (matrix, ids, file_idx) or None if failed
    """
    try:
        pf = pq.ParquetFile(file_path)
        cols = [f.name for f in pf.schema_arrow]
        
        if 'embeddings' not in cols:
            return None
        
        # Find ID column
        id_col_name = None
        for c in ['id', 'hash_id', 'doc_hash']:
            if c in cols:
                id_col_name = c
                break
        
        # Read only what we need
        cols_to_read = ['embeddings']
        if id_col_name:
            cols_to_read.append(id_col_name)
        
        # Read all row groups, process chunks without combining
        all_mats = []
        all_ids = []
        
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=cols_to_read)
            emb_col = table['embeddings']
            
            for chunk_idx in range(emb_col.num_chunks):
                chunk = emb_col.chunk(chunk_idx)
                n = len(chunk)
                if n == 0:
                    continue
                
                # Fast numpy conversion
                vals = chunk.values.to_numpy()
                dim = len(vals) // n
                mat = vals.reshape(n, dim).astype(np.float32)
                
                # Normalize
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                mat = mat / np.maximum(norms, 1e-9)
                all_mats.append(mat)
                
                # IDs
                if id_col_name:
                    all_ids.extend(table[id_col_name].to_pylist()[:n])
                else:
                    all_ids.extend([f"row_{i}" for i in range(n)])
        
        if not all_mats:
            return None
        
        final_mat = np.vstack(all_mats) if len(all_mats) > 1 else all_mats[0]
        return final_mat, all_ids
        
    except Exception as e:
        return None


# =============================================================================
# BENCHMARKS
# =============================================================================
def load_benchmark(name, mode):
    """Load benchmark and return texts."""
    if name == 'musr':
        ds = load_dataset("TAUR-Lab/MuSR")
        data = []
        for split in ds:
            for idx, item in enumerate(ds[split]):
                inp = item.get('narrative', item.get('question', ''))
                out = item.get('answer', '')
                data.append({'id': f"{split}_{idx}", 'input': inp, 'output': out})
    elif name == 'humaneval':
        ds = load_dataset("openai/openai_humaneval")
        data = []
        for item in ds['test']:
            data.append({
                'id': item['task_id'],
                'input': item.get('prompt', ''),
                'output': item.get('canonical_solution', '')
            })
    elif name == 'mbpp':
        ds = load_dataset("evalplus/mbpp", "mbpp")
        data = []
        for item in ds['test']:
            data.append({
                'id': str(item.get('task_id')),
                'input': item.get('prompt', item.get('text', '')),
                'output': item.get('canonical_solution', item.get('code', ''))
            })
    else:
        raise ValueError(f"Unknown benchmark: {name}")
    
    texts, ids = [], []
    for item in data:
        if mode == 'input':
            texts.append(item['input'])
        elif mode == 'output':
            texts.append(item['output'])
        else:
            texts.append(f"{item['input']}\n\n{item['output']}")
        ids.append(item['id'])
    
    return texts, ids


def embed_texts(texts, config):
    """Embed benchmark texts."""
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        config.model_name, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()
    
    embeddings = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), config.embedding_batch_size), desc="Embedding"):
            batch = texts[i:i+config.embedding_batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, 
                          max_length=config.max_seq_length, return_tensors='pt').to(device)
            out = model(**enc)
            mask = enc['attention_mask'].unsqueeze(-1).float()
            emb = (out[0] * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = F.normalize(emb, p=2, dim=1)
            embeddings.append(emb.cpu())
    
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return torch.cat(embeddings, 0).float().numpy()


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
@torch.inference_mode()
def run_analysis(config, benchmark, mode, files, uploader):
    """Main analysis loop - keeps GPU busy."""
    
    cache_dir = Path(config.output_dir) / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Load or compute benchmark embeddings
    bench_cache = cache_dir / f"{benchmark}_{mode}_bench.npy"
    if bench_cache.exists():
        print(f"Loading cached {benchmark} {mode} embeddings...")
        bench_emb = np.load(bench_cache)
    else:
        print(f"Computing {benchmark} {mode} embeddings...")
        texts, bench_ids = load_benchmark(benchmark, mode)
        bench_emb = embed_texts(texts, config)
        np.save(bench_cache, bench_emb)
    
    n_bench = len(bench_emb)
    k = config.top_k
    device = torch.device("cuda")
    
    # Checkpoint paths
    ckpt_dir = Path(config.checkpoint_dir)
    p_sims = ckpt_dir / f"{benchmark}_{mode}_sims.pt"
    p_idxs = ckpt_dir / f"{benchmark}_{mode}_idxs.pt"
    p_state = ckpt_dir / f"{benchmark}_{mode}_state.pkl"
    
    # Resume or start fresh
    if p_state.exists():
        print("Resuming from checkpoint...")
        top_sims = torch.load(p_sims).to(device)
        top_idxs = torch.load(p_idxs).to(device)
        with open(p_state, 'rb') as f:
            state = pickle.load(f)
        start_idx = state['file_idx'] + 1
        global_offset = state['offset']
        all_ids = state.get('all_ids', [])
    else:
        top_sims = torch.full((n_bench, k), -1.0, device=device)
        top_idxs = torch.zeros((n_bench, k), dtype=torch.int64, device=device)
        start_idx = 0
        global_offset = 0
        all_ids = []
    
    bench_gpu = torch.from_numpy(bench_emb).to(device)
    
    # Main loop - simple and fast
    pbar = tqdm(enumerate(files), total=len(files), initial=start_idx, desc=f"{benchmark}/{mode}")
    
    for file_idx, file_path in pbar:
        if file_idx < start_idx:
            continue
        
        # Read file (fast, direct)
        result = read_embeddings_fast(str(file_path))
        if result is None:
            continue
        
        corpus_mat, corpus_ids = result
        n_corpus = len(corpus_mat)
        
        # Store IDs for later resolution
        all_ids.extend(corpus_ids)
        
        # GPU computation
        corpus_gpu = torch.from_numpy(corpus_mat).to(device)
        
        # Compute similarities (all at once if fits, otherwise batch)
        sims = torch.matmul(bench_gpu, corpus_gpu.T)  # (n_bench, n_corpus)
        
        # Create indices for this batch
        batch_idxs = torch.arange(global_offset, global_offset + n_corpus, device=device)
        batch_idxs = batch_idxs.unsqueeze(0).expand(n_bench, -1)
        
        # Merge with current top-k
        cat_sims = torch.cat([top_sims, sims], dim=1)
        cat_idxs = torch.cat([top_idxs, batch_idxs], dim=1)
        
        top_sims, best = cat_sims.topk(k, dim=1)
        top_idxs = cat_idxs.gather(1, best)
        
        global_offset += n_corpus
        
        # Cleanup
        del corpus_gpu, sims, cat_sims, cat_idxs, best, corpus_mat
        
        # Checkpoint
        if (file_idx + 1) % config.save_every_files == 0:
            torch.save(top_sims, p_sims)
            torch.save(top_idxs, p_idxs)
            with open(p_state, 'wb') as f:
                pickle.dump({'file_idx': file_idx, 'offset': global_offset, 'all_ids': all_ids}, f)
            pbar.set_postfix({'ckpt': file_idx})
            gc.collect()
    
    # Final save
    torch.save(top_sims, p_sims)
    torch.save(top_idxs, p_idxs)
    with open(p_state, 'wb') as f:
        pickle.dump({'file_idx': len(files)-1, 'offset': global_offset, 'all_ids': all_ids}, f)
    
    # Resolve and save results
    print("Resolving matches...")
    top_sims_np = top_sims.cpu().numpy()
    top_idxs_np = top_idxs.cpu().numpy()
    
    matches = []
    for i in range(n_bench):
        row = []
        for j in range(k):
            score = float(top_sims_np[i, j])
            if score < 0:
                continue
            idx = int(top_idxs_np[i, j])
            if idx < len(all_ids):
                row.append({'rank': j+1, 'score': score, 'id': all_ids[idx]})
        matches.append(row)
    
    # Save locally
    out_base = Path(config.output_dir) / f"{benchmark}_{mode}"
    np.save(f"{out_base}_sims.npy", top_sims_np)
    with open(f"{out_base}_matches.json", 'w') as f:
        json.dump(matches, f)
    
    # Quick histogram
    plt.figure(figsize=(10, 6))
    plt.hist(top_sims_np[:, 0], bins=50)
    plt.title(f"{benchmark} {mode} - Top-1 Similarity Distribution")
    plt.savefig(f"{out_base}_dist.png")
    plt.close()
    
    # Background S3 upload
    s3_base = f"{config.s3_prefix}/{benchmark}/{mode}"
    uploader.upload(f"{out_base}_sims.npy", f"{s3_base}/sims.npy")
    uploader.upload(f"{out_base}_matches.json", f"{s3_base}/matches.json")
    uploader.upload(f"{out_base}_dist.png", f"{s3_base}/dist.png")
    
    print(f"✅ {benchmark} {mode} complete!")
    return top_sims_np, top_idxs_np


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--benchmark', nargs='+', default=['musr'], 
                       help='musr, humaneval, mbpp, or all')
    parser.add_argument('--modes', nargs='+', default=['input', 'output', 'input_output'])
    parser.add_argument('--bucket', default='dolmo-3-sampling')
    args = parser.parse_args()
    
    config = Config(local_data_dir=args.data_dir, s3_bucket=args.bucket)
    
    # Find all parquet files
    files = sorted(Path(config.local_data_dir).rglob("*.parquet"))
    print(f"Found {len(files)} parquet files")
    
    if not files:
        print("No files found!")
        return
    
    # Setup background uploader
    uploader = BackgroundUploader(config.s3_bucket)
    
    # Determine benchmarks
    benchmarks = args.benchmark
    if 'all' in benchmarks:
        benchmarks = ['musr', 'humaneval', 'mbpp']
    
    # Run analysis
    for bench in benchmarks:
        for mode in args.modes:
            print(f"\n{'='*60}")
            print(f"Running {bench} / {mode}")
            print('='*60)
            run_analysis(config, bench, mode, files, uploader)
    
    # Wait for S3 uploads to finish
    print("\nWaiting for S3 uploads...")
    uploader.wait_all()
    print("Done!")


if __name__ == "__main__":
    main()

