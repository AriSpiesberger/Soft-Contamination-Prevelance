#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Contamination Analysis
Compares benchmark datasets (MuSR, HumanEval) against training corpus embeddings in S3.

Analyzes:
- Input texts
- Output texts  
- Input + Output concatenations

Metrics:
- Distribution of cosine similarities
- Maximum similarity scores
- Distribution of very close matches (top percentiles)
- Contamination statistics
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import gc
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

import boto3
import pyarrow.parquet as pq

# Add production to path
sys.path.append(str(Path(__file__).parent.parent / "production"))
from s3_config import S3Config, default_config


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class AnalysisConfig:
    """Configuration for benchmark contamination analysis"""
    
    # Model settings
    model_name: str = 'nvidia/llama-embed-nemotron-8b'
    max_seq_length: int = 512
    batch_size: int = 128
    
    # S3 settings
    s3_config: S3Config = field(default_factory=lambda: default_config)
    s3_bucket: str = field(init=False)
    s3_prefix: str = field(init=False)
    
    # Analysis settings
    top_k_matches: int = 100  # Save top K closest matches
    save_all_similarities: bool = True  # Save complete similarity arrays
    similarity_thresholds: List[float] = field(default_factory=lambda: [0.7, 0.8, 0.9, 0.95, 0.99])
    chunk_size: int = 50000  # Process S3 embeddings in chunks
    num_workers: int = 8  # Parallel workers for S3 loading
    sample_corpus_pct: float = 1.0  # Sample % of corpus (1.0 = all, 0.1 = 10%)
    
    # Output
    output_dir: str = "results/contamination_analysis"
    cache_dir: str = "results/contamination_analysis/.cache"
    
    def __post_init__(self):
        self.s3_bucket = self.s3_config.bucket
        self.s3_prefix = self.s3_config.output_prefix
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)


# =============================================================================
# DATASET LOADERS
# =============================================================================
class BenchmarkDataset:
    """Base class for benchmark datasets"""
    
    def __init__(self, name: str):
        self.name = name
        self.data = None
    
    def load(self) -> List[Dict[str, Any]]:
        """Load and return dataset as list of dicts with 'input', 'output', 'id' keys"""
        raise NotImplementedError
    
    def get_texts(self, mode: str) -> List[Tuple[str, str]]:
        """
        Get texts for embedding based on mode
        Returns: List of (text, item_id) tuples
        """
        if self.data is None:
            self.load()
        
        texts = []
        for item in self.data:
            item_id = item.get('id', f"item_{len(texts)}")
            
            if mode == 'input':
                texts.append((item['input'], item_id))
            elif mode == 'output':
                texts.append((item['output'], item_id))
            elif mode == 'input_output':
                combined = f"{item['input']}\n\n{item['output']}"
                texts.append((combined, item_id))
            else:
                raise ValueError(f"Unknown mode: {mode}")
        
        return texts


class MuSRDataset(BenchmarkDataset):
    """MuSR (Multi-Step Reasoning) dataset"""
    
    def __init__(self):
        super().__init__("MuSR")
    
    def load(self) -> List[Dict[str, Any]]:
        """Load MuSR dataset"""
        print(f"Loading {self.name} dataset...")
        ds = load_dataset("TAUR-Lab/MuSR")
        
        self.data = []
        for split_name, split_data in ds.items():
            for idx, item in enumerate(split_data):
                # MuSR has 'narrative' (input) and various fields
                # Adapt based on actual structure
                self.data.append({
                    'id': f"{split_name}_{idx}",
                    'input': item.get('narrative', item.get('question', '')),
                    'output': item.get('answer', ''),
                    'split': split_name,
                    'raw': item
                })
        
        print(f"Loaded {len(self.data)} items from {self.name}")
        return self.data


class HumanEvalDataset(BenchmarkDataset):
    """HumanEval coding benchmark"""
    
    def __init__(self):
        super().__init__("HumanEval")
    
    def load(self) -> List[Dict[str, Any]]:
        """Load HumanEval dataset"""
        print(f"Loading {self.name} dataset...")
        ds = load_dataset("openai/openai_humaneval")
        
        self.data = []
        for split_name, split_data in ds.items():
            for item in split_data:
                # HumanEval has 'prompt' (input) and 'canonical_solution' (output)
                self.data.append({
                    'id': item.get('task_id', f"{split_name}_{len(self.data)}"),
                    'input': item.get('prompt', ''),
                    'output': item.get('canonical_solution', ''),
                    'split': split_name,
                    'raw': item
                })
        
        print(f"Loaded {len(self.data)} items from {self.name}")
        return self.data


# =============================================================================
# EMBEDDING ENGINE
# =============================================================================
class EmbeddingEngine:
    """Handles embedding computation"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self._setup_model()
    
    def _setup_model(self):
        """Initialize model and tokenizer"""
        print(f"Loading model: {self.config.model_name}")
        print(f"Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).to(self.device).eval()
        
        print("Model loaded successfully")
    
    @staticmethod
    def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling with attention mask"""
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        return summed / counts
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts
        Returns: numpy array of shape (len(texts), embedding_dim)
        """
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.config.batch_size), 
                         desc="Embedding", unit="batch"):
                batch = texts[i:i + self.config.batch_size]
                
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    return_tensors='pt'
                )
                
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                outputs = self.model(**encoded)
                embeddings = self.mean_pooling(outputs[0], encoded['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)


# =============================================================================
# S3 EMBEDDING LOADER
# =============================================================================
class S3EmbeddingLoader:
    """Loads embeddings from S3 parquet files"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.s3_client = boto3.client(
            's3',
            region_name=config.s3_config.region,
            config=config.s3_config.get_boto_config()
        )
    
    def list_embedding_files(self) -> List[str]:
        """List all parquet files in S3 prefix"""
        print(f"Scanning S3: s3://{self.config.s3_bucket}/{self.config.s3_prefix}")
        print(f"  Bucket: {self.config.s3_bucket}")
        print(f"  Prefix: {self.config.s3_prefix}")
        
        files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        page_count = 0
        for page in paginator.paginate(
            Bucket=self.config.s3_bucket,
            Prefix=self.config.s3_prefix
        ):
            page_count += 1
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('.parquet'):
                        files.append(obj['Key'])
        
        print(f"Scanned {page_count} pages")
        print(f"Found {len(files)} parquet files")
        if len(files) > 0:
            print(f"Sample files:")
            for f in files[:5]:
                print(f"  {f}")
        return files
    
    def _load_single_file(self, file_key: str, embedding_key: str = 'embeddings'):
        """Load a single file from S3"""
        try:
            # Download to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp:
                local_path = Path(tmp.name)
            
            self.s3_client.download_file(
                self.config.s3_bucket,
                file_key,
                str(local_path)
            )
            
            # Read parquet
            table = pq.read_table(str(local_path))
            df = table.to_pandas()
            
            result = None
            if embedding_key in df.columns:
                embeddings = np.stack(df[embedding_key].values)
                
                # Normalize embeddings
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-9)
                
                # Get metadata
                metadata = df.drop(columns=[embedding_key]).to_dict('records')
                result = (embeddings, metadata)
            
            # Cleanup
            local_path.unlink()
            return result
            
        except Exception as e:
            print(f"Error loading {file_key}: {e}")
            return None
    
    def load_embeddings_streaming(self, files: List[str], embedding_key: str = 'embeddings'):
        """
        Generator that yields chunks of embeddings and metadata
        Uses parallel downloads for speed
        Yields: (embeddings_array, metadata_list) tuples
        """
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            # Submit all files
            future_to_file = {
                executor.submit(self._load_single_file, file_key, embedding_key): file_key 
                for file_key in files
            }
            
            # Process as they complete
            for future in tqdm(as_completed(future_to_file), total=len(files), desc="Loading S3 files"):
                result = future.result()
                if result is not None:
                    yield result


# =============================================================================
# SIMILARITY ANALYZER
# =============================================================================
class SimilarityAnalyzer:
    """Computes and analyzes similarities between benchmark and corpus"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def compute_similarities_streaming(
        self,
        benchmark_embeddings: np.ndarray,
        s3_loader: S3EmbeddingLoader,
        files: List[str],
        output_path: Optional[str] = None
    ) -> Tuple[np.ndarray, List[Dict], Optional[str]]:
        """
        Compute similarities between benchmark and S3 embeddings (streaming)
        Writes similarities to disk incrementally to avoid memory issues.
        
        Returns:
            max_similarities: array of max similarity for each benchmark item
            top_matches: list of dicts with top K matches for each benchmark item
            similarities_file: path to file with all similarities (if saved)
        """
        n_benchmark = len(benchmark_embeddings)
        
        # Initialize tracking
        max_similarities = np.zeros(n_benchmark, dtype=np.float32)
        top_k = self.config.top_k_matches
        
        # Store top K matches per benchmark item: (similarity, corpus_idx, metadata)
        top_matches = [[] for _ in range(n_benchmark)]
        
        # Open file handles for writing similarities incrementally
        similarity_files = None
        if self.config.save_all_similarities and output_path:
            import tempfile
            similarity_files = []
            for i in range(n_benchmark):
                # Create temp file for each benchmark item
                temp_file = tempfile.NamedTemporaryFile(
                    mode='ab',
                    delete=False,
                    suffix=f'_bench_{i}.npy',
                    dir=Path(output_path).parent
                )
                similarity_files.append(temp_file)
        
        corpus_idx_offset = 0
        
        # Stream through S3 files
        for corpus_embeddings, metadata in s3_loader.load_embeddings_streaming(files):
            # Compute similarities for this chunk
            # Shape: (n_benchmark, n_corpus_chunk)
            similarities = benchmark_embeddings @ corpus_embeddings.T
            
            # Update max similarities
            chunk_max = similarities.max(axis=1)
            max_similarities = np.maximum(max_similarities, chunk_max)
            
            # Write similarities to disk incrementally
            if similarity_files is not None:
                for bench_idx in range(n_benchmark):
                    # Write this chunk's similarities for this benchmark item
                    np.save(similarity_files[bench_idx], similarities[bench_idx].astype(np.float32))
                    similarity_files[bench_idx].flush()
            
            # Update top K matches
            for bench_idx in range(n_benchmark):
                sims = similarities[bench_idx]
                
                # Get top K from this chunk
                if len(sims) > 0:
                    # Get indices of top similarities in this chunk
                    top_indices = np.argsort(sims)[-top_k:][::-1]
                    
                    for corpus_local_idx in top_indices:
                        sim_score = float(sims[corpus_local_idx])
                        corpus_global_idx = corpus_idx_offset + corpus_local_idx
                        
                        # Add to this benchmark item's matches
                        top_matches[bench_idx].append({
                            'similarity': sim_score,
                            'corpus_idx': corpus_global_idx,
                            'text': metadata[corpus_local_idx].get('text', '')[:200],  # Truncate
                            'source': metadata[corpus_local_idx].get('source', 'unknown')
                        })
                    
                    # Keep only top K overall
                    top_matches[bench_idx] = sorted(
                        top_matches[bench_idx],
                        key=lambda x: x['similarity'],
                        reverse=True
                    )[:top_k]
            
            corpus_idx_offset += len(corpus_embeddings)
            
            # Clean up memory after each chunk
            del corpus_embeddings, metadata, similarities
            gc.collect()
        
        # Close and consolidate similarity files
        similarities_file_path = None
        if similarity_files is not None:
            print("\nConsolidating similarity files...")
            # Close all temp files
            temp_paths = []
            for f in similarity_files:
                temp_paths.append(f.name)
                f.close()
            
            # Read and consolidate into single compressed file
            if output_path:
                all_sims = []
                for temp_path in tqdm(temp_paths, desc="Reading temp files"):
                    # Load all chunks from this temp file
                    chunks = []
                    with open(temp_path, 'rb') as f:
                        try:
                            while True:
                                chunk = np.load(f)
                                chunks.append(chunk)
                        except:
                            pass  # EOF
                    
                    if chunks:
                        all_sims.append(np.concatenate(chunks))
                    
                    # Delete temp file
                    Path(temp_path).unlink()
                
                similarities_file_path = output_path
        
        return max_similarities, top_matches, similarities_file_path
    
    def analyze_similarities(
        self,
        similarities: np.ndarray,
        item_ids: List[str],
        mode: str,
        dataset_name: str
    ) -> Dict[str, Any]:
        """
        Analyze similarity distribution and compute statistics
        
        Returns: Dictionary with analysis results
        """
        results = {
            'dataset': dataset_name,
            'mode': mode,
            'n_items': len(similarities),
            'max_similarity': float(np.max(similarities)),
            'min_similarity': float(np.min(similarities)),
            'mean_similarity': float(np.mean(similarities)),
            'median_similarity': float(np.median(similarities)),
            'std_similarity': float(np.std(similarities)),
            'percentiles': {},
            'threshold_stats': {}
        }
        
        # Percentiles
        for p in [50, 75, 90, 95, 99, 99.9]:
            results['percentiles'][f'p{p}'] = float(np.percentile(similarities, p))
        
        # Threshold statistics
        for threshold in self.config.similarity_thresholds:
            count = np.sum(similarities >= threshold)
            percentage = 100.0 * count / len(similarities)
            results['threshold_stats'][f'above_{threshold}'] = {
                'count': int(count),
                'percentage': float(percentage)
            }
        
        return results


# =============================================================================
# VISUALIZATION
# =============================================================================
class ContaminationVisualizer:
    """Creates visualizations for contamination analysis"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
    
    def plot_similarity_distribution(
        self,
        similarities: np.ndarray,
        dataset_name: str,
        mode: str,
        output_path: str
    ):
        """Plot distribution of similarities"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{dataset_name} - {mode} - Similarity Distribution', fontsize=16, y=0.995)
        
        # Histogram
        ax = axes[0, 0]
        ax.hist(similarities, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Frequency')
        ax.set_title('Histogram')
        ax.axvline(np.mean(similarities), color='red', linestyle='--', label=f'Mean: {np.mean(similarities):.4f}')
        ax.axvline(np.median(similarities), color='green', linestyle='--', label=f'Median: {np.median(similarities):.4f}')
        ax.legend()
        
        # KDE plot
        ax = axes[0, 1]
        sns.kdeplot(similarities, ax=ax, fill=True, color='steelblue')
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Density')
        ax.set_title('Kernel Density Estimate')
        
        # CDF
        ax = axes[1, 0]
        sorted_sims = np.sort(similarities)
        cdf = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
        ax.plot(sorted_sims, cdf, color='steelblue', linewidth=2)
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Distribution Function')
        ax.grid(True, alpha=0.3)
        
        # Box plot with thresholds
        ax = axes[1, 1]
        bp = ax.boxplot([similarities], vert=True, patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('lightblue')
        
        # Add threshold lines
        for threshold in self.config.similarity_thresholds:
            ax.axhline(threshold, color='red', linestyle='--', alpha=0.5, linewidth=1)
            count = np.sum(similarities >= threshold)
            pct = 100.0 * count / len(similarities)
            ax.text(1.15, threshold, f'{threshold:.2f}\n({pct:.1f}%)', 
                   verticalalignment='center', fontsize=8)
        
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Box Plot with Contamination Thresholds')
        ax.set_xticks([1])
        ax.set_xticklabels([dataset_name])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {output_path}")
    
    def plot_comparison(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        output_path: str
    ):
        """Compare multiple modes (input, output, input_output)"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        modes = list(results_dict.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(modes)))
        
        # Plot 1: Max similarities comparison
        ax = axes[0, 0]
        max_sims = [results_dict[mode]['max_similarity'] for mode in modes]
        bars = ax.bar(modes, max_sims, color=colors, edgecolor='black', alpha=0.7)
        ax.set_ylabel('Maximum Similarity')
        ax.set_title('Maximum Similarity by Mode')
        ax.set_ylim([0, 1.0])
        for bar, val in zip(bars, max_sims):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom')
        
        # Plot 2: Mean similarities
        ax = axes[0, 1]
        mean_sims = [results_dict[mode]['mean_similarity'] for mode in modes]
        bars = ax.bar(modes, mean_sims, color=colors, edgecolor='black', alpha=0.7)
        ax.set_ylabel('Mean Similarity')
        ax.set_title('Mean Similarity by Mode')
        for bar, val in zip(bars, mean_sims):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom')
        
        # Plot 3: Percentile comparison
        ax = axes[1, 0]
        percentiles = ['p50', 'p75', 'p90', 'p95', 'p99']
        x = np.arange(len(percentiles))
        width = 0.8 / len(modes)
        
        for i, mode in enumerate(modes):
            values = [results_dict[mode]['percentiles'][p] for p in percentiles]
            ax.bar(x + i * width, values, width, label=mode, color=colors[i], alpha=0.7)
        
        ax.set_xlabel('Percentile')
        ax.set_ylabel('Similarity')
        ax.set_title('Similarity Percentiles by Mode')
        ax.set_xticks(x + width * (len(modes) - 1) / 2)
        ax.set_xticklabels(percentiles)
        ax.legend()
        
        # Plot 4: Contamination rates
        ax = axes[1, 1]
        thresholds = self.config.similarity_thresholds
        x = np.arange(len(thresholds))
        width = 0.8 / len(modes)
        
        for i, mode in enumerate(modes):
            percentages = [
                results_dict[mode]['threshold_stats'][f'above_{t}']['percentage']
                for t in thresholds
            ]
            ax.bar(x + i * width, percentages, width, label=mode, color=colors[i], alpha=0.7)
        
        ax.set_xlabel('Similarity Threshold')
        ax.set_ylabel('Percentage Above Threshold (%)')
        ax.set_title('Contamination Rates by Threshold')
        ax.set_xticks(x + width * (len(modes) - 1) / 2)
        ax.set_xticklabels([f'{t:.2f}' for t in thresholds])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison plot: {output_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================
class ContaminationPipeline:
    """Main pipeline for contamination analysis"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.embedding_engine = EmbeddingEngine(config)
        self.s3_loader = S3EmbeddingLoader(config)
        self.analyzer = SimilarityAnalyzer(config)
        self.visualizer = ContaminationVisualizer(config)
    
    def run_analysis(
        self,
        dataset: BenchmarkDataset,
        modes: List[str] = ['input', 'output', 'input_output']
    ) -> Dict[str, Any]:
        """
        Run complete analysis for a dataset
        
        Args:
            dataset: BenchmarkDataset instance
            modes: List of modes to analyze ['input', 'output', 'input_output']
        
        Returns:
            Dictionary with all results
        """
        print("="*80)
        print(f"CONTAMINATION ANALYSIS: {dataset.name}")
        print("="*80)
        
        # Load dataset
        dataset.load()
        
        # List S3 files once
        s3_files = self.s3_loader.list_embedding_files()
        if not s3_files:
            raise ValueError("No embedding files found in S3!")
        
        # Sample corpus if requested
        if self.config.sample_corpus_pct < 1.0:
            n_sample = max(1, int(len(s3_files) * self.config.sample_corpus_pct))
            s3_files = random.sample(s3_files, n_sample)
            print(f"Sampled {n_sample}/{len(s3_files)} corpus files ({self.config.sample_corpus_pct*100:.1f}%)")
        
        all_results = {}
        
        for mode in modes:
            print(f"\n{'='*80}")
            print(f"Mode: {mode}")
            print(f"{'='*80}")
            
            # Get texts for this mode
            text_pairs = dataset.get_texts(mode)
            texts = [t[0] for t in text_pairs]
            item_ids = [t[1] for t in text_pairs]
            
            print(f"Processing {len(texts)} texts...")
            
            # Embed benchmark texts
            print("\nEmbedding benchmark texts...")
            benchmark_embeddings = self.embedding_engine.embed_texts(texts)
            
            # Compute similarities against S3 corpus
            print("\nComputing similarities against corpus...")
            all_sims_path = str(Path(self.config.output_dir) / f"{dataset.name}_{mode}_all_similarities.npz")
            max_similarities, top_matches, sims_file = self.analyzer.compute_similarities_streaming(
                benchmark_embeddings,
                self.s3_loader,
                s3_files,
                output_path=all_sims_path if self.config.save_all_similarities else None
            )
            
            # Analyze results
            print("\nAnalyzing results...")
            results = self.analyzer.analyze_similarities(
                max_similarities,
                item_ids,
                mode,
                dataset.name
            )
            
            # Save top matches
            matches_file = Path(self.config.output_dir) / f"{dataset.name}_{mode}_top_matches.json"
            with open(matches_file, 'w') as f:
                json.dump({
                    'dataset': dataset.name,
                    'mode': mode,
                    'matches': [
                        {'item_id': item_id, 'top_matches': matches}
                        for item_id, matches in zip(item_ids, top_matches)
                    ]
                }, f, indent=2)
            print(f"Saved top matches: {matches_file}")
            
            # Similarities already saved during streaming
            if sims_file:
                print(f"Saved all similarities: {sims_file}")
            
            # Visualize
            print("\nCreating visualizations...")
            plot_path = Path(self.config.output_dir) / f"{dataset.name}_{mode}_distribution.png"
            self.visualizer.plot_similarity_distribution(
                max_similarities,
                dataset.name,
                mode,
                str(plot_path)
            )
            
            # Store results
            results['similarities'] = max_similarities.tolist()
            results['item_ids'] = item_ids
            all_results[mode] = results
            
            # Print summary
            self._print_summary(results)
        
        # Create comparison plot
        print("\nCreating comparison plots...")
        comparison_path = Path(self.config.output_dir) / f"{dataset.name}_comparison.png"
        self.visualizer.plot_comparison(all_results, str(comparison_path))
        
        # Save complete results
        results_file = Path(self.config.output_dir) / f"{dataset.name}_results.json"
        # Remove large arrays for JSON
        save_results = {
            mode: {k: v for k, v in res.items() if k not in ['similarities']}
            for mode, res in all_results.items()
        }
        with open(results_file, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\nSaved results: {results_file}")
        
        return all_results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print analysis summary"""
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Dataset: {results['dataset']}")
        print(f"Mode: {results['mode']}")
        print(f"Items: {results['n_items']}")
        print(f"\nSimilarity Statistics:")
        print(f"  Max:    {results['max_similarity']:.6f}")
        print(f"  Mean:   {results['mean_similarity']:.6f}")
        print(f"  Median: {results['median_similarity']:.6f}")
        print(f"  Std:    {results['std_similarity']:.6f}")
        print(f"\nPercentiles:")
        for p, val in sorted(results['percentiles'].items()):
            print(f"  {p}: {val:.6f}")
        print(f"\nContamination Rates (items above threshold):")
        for threshold, stats in sorted(results['threshold_stats'].items()):
            print(f"  {threshold}: {stats['count']} ({stats['percentage']:.2f}%)")
        print("="*60)


# =============================================================================
# CLI
# =============================================================================
def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Contamination Analysis")
    parser.add_argument(
        '--dataset',
        choices=['musr', 'humaneval', 'both'],
        default='both',
        help='Dataset to analyze'
    )
    parser.add_argument(
        '--modes',
        nargs='+',
        choices=['input', 'output', 'input_output'],
        default=['input', 'output', 'input_output'],
        help='Analysis modes'
    )
    parser.add_argument(
        '--output-dir',
        default='results/contamination_analysis',
        help='Output directory'
    )
    parser.add_argument(
        '--s3-bucket',
        help='S3 bucket (overrides default)'
    )
    parser.add_argument(
        '--s3-prefix',
        help='S3 prefix for embeddings (overrides default)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding (default: 32, reduce if OOM)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of parallel S3 download workers (default: 8)'
    )
    parser.add_argument(
        '--sample-corpus',
        type=float,
        default=1.0,
        help='Sample percentage of corpus (0.1 = 10%%, 1.0 = all, default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Setup config
    config = AnalysisConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_corpus_pct=args.sample_corpus
    )
    if args.s3_bucket:
        config.s3_config.bucket = args.s3_bucket
        config.s3_bucket = args.s3_bucket
    if args.s3_prefix:
        config.s3_config.output_prefix = args.s3_prefix
        config.s3_prefix = args.s3_prefix
    
    # Initialize pipeline
    pipeline = ContaminationPipeline(config)
    
    # Run analyses
    datasets = []
    if args.dataset in ['musr', 'both']:
        datasets.append(MuSRDataset())
    if args.dataset in ['humaneval', 'both']:
        datasets.append(HumanEvalDataset())
    
    for dataset in datasets:
        try:
            results = pipeline.run_analysis(dataset, modes=args.modes)
            print(f"\n✓ Completed analysis for {dataset.name}")
        except Exception as e:
            print(f"\n✗ Error analyzing {dataset.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE")
    print("="*80)
    print(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()

