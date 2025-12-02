#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Pipeline v3 - MULTICORE (True Parallelism)

Changes:
1. Uses ProcessPoolExecutor (bypasses GIL) instead of ThreadPoolExecutor.
2. Writes sharded S3 files (part_{uuid}.jsonl) to prevent collisions.
3. scales linearly with CPU cores.
"""

import os
import sys
import json
import random
import time
import hashlib
import uuid
import io
import gzip
import shutil
import pathlib
from typing import Dict, Any, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    raise ImportError("pip install huggingface_hub")

try:
    import boto3
except ImportError:
    raise ImportError("pip install boto3")

# Import S3 config
from s3_config import S3Config, default_config

try:
    import nltk
    import tiktoken
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
except ImportError:
    raise ImportError("pip install nltk tiktoken")

try:
    import pandas as pd
except ImportError:
    raise ImportError("pip install pandas")

try:
    import zstandard as zstd
except ImportError:
    raise ImportError("pip install zstandard")

from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET_NAME = "allenai/dolma3_mix-6T-1025"
RANDOM_SEED = 42

# Processing
MIN_PARAGRAPH_TOKEN_LEN = 10
MAX_PARAGRAPH_TOKEN_LEN = 512
TOKENIZER_ENCODING = "cl100k_base"

# Resources
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "16")) # Set to your vCPU count
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", "/tmp/dolma_scratch")

# Sampling
SAMPLE_PERCENTAGE = float(os.environ.get("SAMPLE_PERCENTAGE", "0.015")) 
PROCESS_MODE = os.environ.get("PROCESS_MODE", "sample")

# S3 Config - generate default prefix if not set
_run_date_str = datetime.utcnow().strftime("%Y%m%d")
_sample_pct = SAMPLE_PERCENTAGE * 100.0
_default_prefix = f"dolma3_{_run_date_str}_p{_sample_pct:.4f}pct"
_pipeline_prefix = os.environ.get("S3_PREFIX", _default_prefix)

# Create S3 config instance
s3_config = S3Config(
    pipeline_prefix=_pipeline_prefix,
    buffer_size=10 * 1024 * 1024  # 10MB
)

S3_BUCKET = s3_config.bucket
S3_PREFIX = s3_config.pipeline_prefix or _pipeline_prefix
S3_REGION = s3_config.region
S3_BUFFER_SIZE = s3_config.buffer_size

# Data Sources
SOURCE_DOCS = {
    "common_crawl": 3150, "olmocr_science_pdfs": 83.8,
    "stack_edu": 525.8, "finemath": 95.5,
    "rpj": 9.10, "dolma1_7": 4.24,
}
DATA_EXTENSIONS = ('.jsonl.zst', '.json.zst', '.jsonl', '.parquet')

random.seed(RANDOM_SEED)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def generate_hash_id(text: str, section: str, data_type: str, quality: str = "") -> str:
    content = f"{section}||{data_type}||{quality}||{text}".encode('utf-8')
    return hashlib.sha256(content).hexdigest()[:16]

def get_source_from_path(filepath: str) -> str:
    parts = filepath.split('/')
    if len(parts) < 2: return "unknown"
    folder = parts[1]
    for key in SOURCE_DOCS.keys():
        if folder.startswith(key.split('_')[0]): return key
    return "unknown"

def extract_metadata_labels(metadata: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    if not metadata or not isinstance(metadata, dict):
        return "unknown", "default", {}
    
    labels = {}
    for k, v in metadata.items():
        if k.startswith('__label__'):
            labels[k.replace('__label__', '').lower().strip()] = v
        else:
            labels[k] = v
            
    data_type = 'unknown'
    if 'weborganizer_max' in metadata:
        data_type = metadata['weborganizer_max'].replace('__label__', '').strip().lower()
    
    quality = 'default'
    for q_field in ['quality', 'score', 'quality_score']:
        if q_field in metadata and isinstance(metadata[q_field], (int, float)):
            val = metadata[q_field]
            quality = 'high' if val >= 0.8 else ('medium' if val >= 0.5 else 'low')
            break
            
    return data_type, quality, labels

# =============================================================================
# S3 UPLOADER (Process-Local)
# =============================================================================
class ShardedS3Uploader:
    """Each process gets its own uploader instance that writes unique files."""
    def __init__(self, bucket, prefix, region, buffer_size, worker_id):
        self.bucket = bucket
        self.prefix = prefix
        self.buffer_size = buffer_size
        self.worker_id = worker_id
        # Re-init boto3 inside process
        # Use S3Config for boto config
        from s3_config import S3Config
        temp_config = S3Config(region=region, buffer_size=buffer_size)
        self.s3_client = boto3.client('s3', region_name=region, config=temp_config.get_boto_config())
        self.buffers = {} # key -> list of bytes
        self.buffer_sizes = {}
        self.upload_ids = {} # key -> upload_id
        self.parts = {} # key -> list of parts
        self.part_nums = {} # key -> next part num
        self.total_bytes = 0

    def get_key(self, section, data_type, quality, chunk_type):
        # Creates a unique key for this worker to avoid collisions
        # e.g. .../documents/part_worker1_abcd1234.jsonl
        return f"{self.prefix}/{section}/{data_type}/{quality}/{chunk_type}/part_{self.worker_id}.jsonl"

    def add_item(self, item, section, data_type, quality, chunk_type):
        s3_key = self.get_key(section, data_type, quality, chunk_type)
        line = (json.dumps(item, ensure_ascii=False) + '\n').encode('utf-8')
        
        if s3_key not in self.buffers:
            self.buffers[s3_key] = []
            self.buffer_sizes[s3_key] = 0
            
        self.buffers[s3_key].append(line)
        self.buffer_sizes[s3_key] += len(line)
        
        if self.buffer_sizes[s3_key] >= self.buffer_size:
            self._flush(s3_key)

    def _flush(self, key):
        if not self.buffers.get(key): return
        
        data = b''.join(self.buffers[key])
        self.buffers[key] = []
        self.buffer_sizes[key] = 0
        self.total_bytes += len(data)
        
        try:
            # Start Multipart if not exists
            if key not in self.upload_ids:
                self.upload_ids[key] = self.s3_client.create_multipart_upload(
                    Bucket=self.bucket, Key=key, ContentType='application/jsonl'
                )['UploadId']
                self.parts[key] = []
                self.part_nums[key] = 1
            
            # Upload Part
            resp = self.s3_client.upload_part(
                Bucket=self.bucket, Key=key, PartNumber=self.part_nums[key],
                UploadId=self.upload_ids[key], Body=data
            )
            self.parts[key].append({'PartNumber': self.part_nums[key], 'ETag': resp['ETag']})
            self.part_nums[key] += 1
            
        except Exception as e:
            print(f"!! Upload Error {key}: {e}")

    def finalize(self):
        # Flush all remaining buffers
        for key in list(self.buffers.keys()):
            self._flush(key)
            
        # Complete all uploads
        for key, uid in self.upload_ids.items():
            try:
                if self.parts.get(key):
                    self.s3_client.complete_multipart_upload(
                        Bucket=self.bucket, Key=key, UploadId=uid,
                        MultipartUpload={'Parts': sorted(self.parts[key], key=lambda x: x['PartNumber'])}
                    )
                else:
                    self.s3_client.abort_multipart_upload(Bucket=self.bucket, Key=key, UploadId=uid)
            except Exception as e:
                print(f"!! Finalize Error {key}: {e}")
        return self.total_bytes

# =============================================================================
# FILE PROCESSING (Process Safe)
# =============================================================================
def read_file_content(filepath):
    """Generator for documents."""
    fpath = str(filepath)
    if fpath.endswith(".parquet"):
        for doc in pd.read_parquet(fpath).to_dict(orient="records"): yield doc
    elif fpath.endswith(".zst"):
        with open(fpath, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                for line in io.TextIOWrapper(reader, encoding='utf-8'): 
                    if line.strip(): yield json.loads(line)
    else: 
        opener = gzip.open if fpath.endswith('.gz') else open
        with opener(fpath, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip(): yield json.loads(line)

def process_single_file_task(file_path_str, worker_uid):
    """
    Standalone function for ProcessPool.
    Instantiates its own uploader and resources to avoid pickling issues.
    """
    # 1. Setup Resources
    worker_scratch = os.path.join(SCRATCH_DIR, worker_uid)
    os.makedirs(worker_scratch, exist_ok=True)
    
    # Init Uploader & Tokenizer locally
    uploader = ShardedS3Uploader(S3_BUCKET, S3_PREFIX, S3_REGION, S3_BUFFER_SIZE, worker_uid)
    tokenizer = tiktoken.get_encoding(TOKENIZER_ENCODING)
    
    docs, paras = 0, 0
    local_file_path = None
    
    try:
        # 2. Download
        local_file_path = hf_hub_download(
            repo_id=DATASET_NAME,
            filename=file_path_str,
            repo_type="dataset",
            local_dir=worker_scratch,
            force_download=False,
            local_files_only=False
        )
        
        # 3. Process
        source = get_source_from_path(file_path_str)
        
        for doc in read_file_content(local_file_path):
            text = doc.get('text', '')
            if not text: continue
            
            d_type, quality, labels = extract_metadata_labels(doc.get('metadata', {}))
            doc_hash = generate_hash_id(text, source, d_type, quality)
            
            # Doc
            uploader.add_item({
                'hash_id': doc_hash, 'text': text, 'metadata': labels, 
                'source': source, 'id': doc.get('id')
            }, source, d_type, quality, "documents")
            docs += 1
            
            # Paragraphs
            for p in text.split('\n\n'):
                p = p.strip()
                if not p: continue
                tokens = len(tokenizer.encode(p))
                if MIN_PARAGRAPH_TOKEN_LEN <= tokens <= MAX_PARAGRAPH_TOKEN_LEN:
                    p_hash = generate_hash_id(p, source, d_type, quality)
                    uploader.add_item({
                        'hash_id': p_hash, 'text': p, 'doc_hash': doc_hash,
                        'token_size': tokens
                    }, source, d_type, quality, "paragraphs")
                    paras += 1
                    
    except Exception as e:
        return docs, paras, 0, f"Error in {file_path_str}: {e}"
        
    finally:
        # 4. Finalize & Cleanup
        bytes_uploaded = uploader.finalize()
        
        if local_file_path and os.path.exists(local_file_path):
            try:
                os.unlink(local_file_path)
            except: pass
        try:
            os.rmdir(worker_scratch) # Try cleanup empty dir
        except: pass

    return docs, paras, bytes_uploaded, None

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*60)
    print(f"DOLMA 3 MULTICORE PIPELINE | {MAX_WORKERS} Processes")
    print(f"Scratch: {SCRATCH_DIR}")
    print("="*60)
    
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    
    # List Files
    print("Scanning HuggingFace...")
    api = HfApi()
    all_files = [
        f for f in api.list_repo_tree(DATASET_NAME, repo_type="dataset", recursive=True)
        if hasattr(f, "size") and f.size > 0 and f.path.endswith(DATA_EXTENSIONS)
    ]
    
    # Filter
    if PROCESS_MODE == 'sample':
        target = max(1, int(len(all_files) * SAMPLE_PERCENTAGE))
        random.shuffle(all_files)
        files_to_process = all_files[:target]
        print(f"Sampling {len(files_to_process)}/{len(all_files)} files")
    else:
        files_to_process = all_files

    # Parallel Execution
    total_docs, total_paras, total_bytes = 0, 0, 0
    
    # Use ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # We pass just the file path string to avoid pickling complex objects
        futures = {}
        for f in files_to_process:
            # Generate a unique worker ID for this task
            uid = f"{uuid.uuid4().hex[:8]}"
            futures[executor.submit(process_single_file_task, f.path, uid)] = f.path
            
        pbar = tqdm(total=len(files_to_process), unit="file")
        
        try:
            for future in as_completed(futures):
                path = futures[future]
                d, p, b, err = future.result()
                if err:
                    tqdm.write(f"Failed {path}: {err}")
                total_docs += d
                total_paras += p
                total_bytes += b
                pbar.update(1)
        except KeyboardInterrupt:
            print("Stopping...")
            executor.shutdown(wait=False, cancel_futures=True)
            
    pbar.close()
    
    print("\n" + "="*60)
    print(f"Done! Docs: {total_docs:,} | Paras: {total_paras:,}")
    print(f"Bytes Uploaded: {_format_bytes(total_bytes)}")

def _format_bytes(num_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0: return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"

if __name__ == "__main__":
    main()