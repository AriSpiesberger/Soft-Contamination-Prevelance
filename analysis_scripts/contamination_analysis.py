#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Contamination Analysis - Optimized for A10 24GB
Two-phase approach:
1. Load model, embed ALL benchmarks, cache to disk, unload model
2. Stream corpus from disk, compute similarities with full GPU available
"""

import os
import sys
import json
import gc
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Generator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).parent.parent / "production"))

try:
    from s3_config import S3Config, default_config
except ImportError:
    @dataclass
    class S3Config:
        bucket: str = "dolmo-3-sampling"
        output_prefix: str = "embeddings"
        region: str = "us-east-2"
        def get_boto_config(self): return None
    default_config = S3Config()


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class AnalysisConfig:
    model_name: str = 'nvidia/llama-embed-nemotron-8b'
    max_seq_length: int = 512
    embedding_batch_size: int = 16  # Small for 8B model on A10
    
    s3_config: S3Config = field(default_factory=lambda: default_config)
    local_data_dir: str = "/lambda/nfs/embeddings/embedding_folder"
    
    top_k_matches: int = 100
    similarity_thresholds: List[float] = field(default_factory=lambda: [0.7, 0.8, 0.9, 0.95, 0.99])
    
    # With model unloaded, full 24GB available for matmul
    compute_batch_size: int = 500_000

    output_dir: str = "contamination_analysis"

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots/musr", exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots/humaneval", exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots/summary", exist_ok=True)
        os.makedirs(f"{self.output_dir}/data/top_matches", exist_ok=True)
        os.makedirs(f"{self.output_dir}/data/statistics", exist_ok=True)
        os.makedirs(f"{self.output_dir}/data/embeddings_cache", exist_ok=True)


# =============================================================================
# DATASET LOADERS
# =============================================================================
class BenchmarkDataset:
    def __init__(self, name: str):
        self.name = name
        self.data = None
    
    def load(self) -> List[Dict[str, Any]]:
        raise NotImplementedError
    
    def get_texts(self, mode: str) -> Tuple[List[str], List[str]]:
        if self.data is None:
            self.load()
        texts, item_ids = [], []
        for i, item in enumerate(self.data):
            item_id = item.get('id', f"item_{i}")
            if mode == 'input':
                text = item['input']
            elif mode == 'output':
                text = item['output']
            elif mode == 'input_output':
                text = f"{item['input']}\n\n{item['output']}"
            else:
                raise ValueError(f"Unknown mode: {mode}")
            texts.append(text)
            item_ids.append(item_id)
        return texts, item_ids


class MuSRDataset(BenchmarkDataset):
    def __init__(self):
        super().__init__("MuSR")
    
    def load(self):
        print(f"Loading {self.name} dataset...")
        ds = load_dataset("TAUR-Lab/MuSR")
        self.data = []
        for split_name, split_data in ds.items():
            for idx, item in enumerate(split_data):
                self.data.append({
                    'id': f"{split_name}_{idx}",
                    'input': item.get('narrative', item.get('question', '')),
                    'output': item.get('answer', ''),
                    'split': split_name,
                })
        return self.data


class HumanEvalDataset(BenchmarkDataset):
    def __init__(self):
        super().__init__("HumanEval")
    
    def load(self):
        print(f"Loading {self.name} dataset...")
        ds = load_dataset("openai/openai_humaneval")
        self.data = []
        for split_name, split_data in ds.items():
            for item in split_data:
                self.data.append({
                    'id': item.get('task_id', f"{split_name}_{len(self.data)}"),
                    'input': item.get('prompt', ''),
                    'output': item.get('canonical_solution', ''),
                    'split': split_name,
                })
        return self.data


# =============================================================================
# EMBEDDING ENGINE
# =============================================================================
class EmbeddingEngine:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.device = torch.device("cuda")
        print(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(self.device).eval()
        
        print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    @torch.inference_mode()
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        all_embeddings = []
        batch_size = self.config.embedding_batch_size
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors='pt'
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            outputs = self.model(**encoded)
            embeddings = self._mean_pooling(outputs[0], encoded['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
            
            del encoded, outputs, embeddings
            torch.cuda.empty_cache()
        
        return torch.cat(all_embeddings, dim=0).half().numpy()

    @staticmethod
    def _mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts
    
    def unload(self):
        print("Unloading model...")
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"GPU memory after unload: {torch.cuda.memory_allocated()/1e9:.2f}GB")


# =============================================================================
# DATA MANAGER
# =============================================================================
class LocalDataManager:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._file_metadata_cache: Dict[int, List[Dict]] = {}
    
    def list_local_files(self) -> List[Path]:
        print(f"Scanning {self.config.local_data_dir} for parquet files...")
        files = sorted(Path(self.config.local_data_dir).rglob("*.parquet"))
        print(f"Found {len(files)} parquet files")
        return files

    def load_embeddings_generator(
        self,
        files: List[Path],
        embedding_key: str = 'embeddings'
    ) -> Generator[Tuple[np.ndarray, int, Path], None, None]:
        for file_idx, file_path in enumerate(tqdm(files, desc="Streaming corpus")):
            try:
                table = pq.read_table(str(file_path), memory_map=True)
                
                if embedding_key not in table.column_names:
                    continue

                emb_column = table[embedding_key]
                if emb_column.num_chunks > 1:
                    emb_column = emb_column.combine_chunks()
                else:
                    emb_column = emb_column.chunk(0)

                values = emb_column.values.to_numpy()
                n_rows = len(emb_column)
                dim = len(values) // n_rows
                
                embeddings = values.reshape(n_rows, dim).astype(np.float16, copy=True)
                
                # Normalize
                norms = np.linalg.norm(embeddings.astype(np.float32), axis=1, keepdims=True)
                embeddings = (embeddings.astype(np.float32) / np.maximum(norms, 1e-9)).astype(np.float16)
                
                # Cache metadata
                meta_cols = [c for c in table.column_names if c != embedding_key]
                self._file_metadata_cache[file_idx] = table.select(meta_cols).to_pandas().to_dict('records')
                
                yield embeddings, file_idx, file_path
                
                del embeddings, table, values, emb_column
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
    
    def get_metadata(self, file_idx: int, local_idx: int) -> Dict[str, Any]:
        if file_idx in self._file_metadata_cache:
            return self._file_metadata_cache[file_idx][local_idx]
        return {}


# =============================================================================
# SIMILARITY ANALYZER
# =============================================================================
class SimilarityAnalyzer:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.device = torch.device("cuda")

    @torch.inference_mode()
    def compute_similarities(
        self,
        benchmark_embeddings: np.ndarray,
        data_manager: LocalDataManager,
        files: List[Path],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        n_benchmark = len(benchmark_embeddings)
        k = self.config.top_k_matches
        batch_size = self.config.compute_batch_size
        
        # Top-k on GPU
        top_sims = torch.full((n_benchmark, k), -float('inf'), device=self.device, dtype=torch.float16)
        top_indices = torch.zeros((n_benchmark, k), device=self.device, dtype=torch.int64)
        
        bench_gpu = torch.from_numpy(benchmark_embeddings).to(self.device)
        
        index_mapping: List[Tuple[int, int]] = []
        global_offset = 0
        
        for corpus_emb, file_idx, file_path in data_manager.load_embeddings_generator(files):
            n_corpus = len(corpus_emb)
            
            # Build index mapping
            for local_idx in range(n_corpus):
                index_mapping.append((file_idx, local_idx))
            
            # Batch matmul
            for i in range(0, n_corpus, batch_size):
                end_idx = min(i + batch_size, n_corpus)
                corpus_gpu = torch.from_numpy(corpus_emb[i:end_idx]).to(self.device)
                
                sims = torch.matmul(bench_gpu, corpus_gpu.T)
                
                batch_indices = torch.arange(
                    global_offset + i,
                    global_offset + end_idx,
                    device=self.device,
                    dtype=torch.int64
                ).unsqueeze(0).expand(n_benchmark, -1)
                
                combined_sims = torch.cat([top_sims, sims], dim=1)
                combined_indices = torch.cat([top_indices, batch_indices], dim=1)
                
                top_sims, topk_positions = combined_sims.topk(k, dim=1, largest=True, sorted=True)
                top_indices = combined_indices.gather(1, topk_positions)
                
                del corpus_gpu, sims, combined_sims, combined_indices, batch_indices
            
            global_offset += n_corpus
            gc.collect()
            torch.cuda.empty_cache()
        
        max_similarities = top_sims[:, 0].float().cpu().numpy()
        top_k_sims = top_sims.float().cpu().numpy()
        top_k_indices = top_indices.cpu().numpy()
        
        return max_similarities, top_k_sims, top_k_indices, index_mapping

    def resolve_top_matches(
        self,
        top_k_sims: np.ndarray,
        top_k_indices: np.ndarray,
        index_mapping: List[Tuple[int, int]],
        data_manager: LocalDataManager,
    ) -> List[List[Dict[str, Any]]]:
        n_benchmark = len(top_k_sims)
        top_matches = []
        
        for bench_idx in range(n_benchmark):
            matches = []
            for rank in range(len(top_k_sims[bench_idx])):
                sim = top_k_sims[bench_idx, rank]
                if sim == -float('inf'):
                    continue
                
                global_idx = top_k_indices[bench_idx, rank]
                if global_idx < len(index_mapping):
                    file_idx, local_idx = index_mapping[global_idx]
                    metadata = data_manager.get_metadata(file_idx, local_idx)
                    
                    matches.append({
                        'rank': rank + 1,
                        'similarity': float(sim),
                        'corpus_idx': int(global_idx),
                        'text': metadata.get('text', '')[:500],
                        'source': metadata.get('source', 'unknown'),
                    })
            
            top_matches.append(matches)
        
        return top_matches

    def compute_statistics(
        self,
        max_similarities: np.ndarray,
        mode: str,
        dataset_name: str,
    ) -> Dict[str, Any]:
        return {
            'dataset': dataset_name,
            'mode': mode,
            'n_items': len(max_similarities),
            'max_similarity': float(np.max(max_similarities)),
            'mean_similarity': float(np.mean(max_similarities)),
            'median_similarity': float(np.median(max_similarities)),
            'std_similarity': float(np.std(max_similarities)),
            'percentiles': {f'p{p}': float(np.percentile(max_similarities, p)) for p in [50, 75, 90, 95, 99]},
            'threshold_stats': {
                f'above_{t}': {
                    'count': int(np.sum(max_similarities >= t)),
                    'percentage': float(100.0 * np.sum(max_similarities >= t) / len(max_similarities)),
                }
                for t in self.config.similarity_thresholds
            },
        }


# =============================================================================
# VISUALIZATION
# =============================================================================
class ContaminationVisualizer:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        sns.set_style("whitegrid")
    
    def plot_similarity_distribution(self, similarities: np.ndarray, dataset_name: str, mode: str, output_path: str):
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.histplot(similarities, bins=100, kde=True, color='steelblue', stat='density', ax=ax)
        
        p99 = np.percentile(similarities, 99)
        max_val = np.max(similarities)
        ax.axvline(p99, color='orange', linestyle='--', linewidth=2, label=f'P99: {p99:.3f}')
        ax.axvline(max_val, color='red', linestyle='-', linewidth=2, label=f'Max: {max_val:.3f}')
        
        ax.set_title(f'{dataset_name} - {mode}\nSimilarity Distribution', fontsize=14)
        ax.set_xlabel('Cosine Similarity', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(loc='upper right')
        
        stats_lines = ["Contamination Rates:"]
        for t in self.config.similarity_thresholds:
            pct = 100.0 * np.sum(similarities >= t) / len(similarities)
            if pct > 0.001:
                stats_lines.append(f"  ≥{t}: {pct:.3f}%")
        ax.text(0.98, 0.95, "\n".join(stats_lines), transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def plot_joint_comparison(self, all_results: List[Dict], output_path: str):
        if not all_results:
            return
        
        df = pd.DataFrame(all_results)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Benchmark Contamination Analysis', fontsize=16, y=1.02)
        
        sns.barplot(data=df, x='dataset', y='max_similarity', hue='mode', ax=axes[0,0], palette='viridis')
        axes[0,0].set_title('Maximum Similarity Score')
        axes[0,0].set_ylim(0, 1.05)
        
        df['p99'] = df['percentiles'].apply(lambda x: x['p99'])
        sns.barplot(data=df, x='dataset', y='p99', hue='mode', ax=axes[0,1], palette='magma')
        axes[0,1].set_title('99th Percentile Similarity')
        axes[0,1].set_ylim(0, 1.05)
        
        df['contamination_90'] = df['threshold_stats'].apply(lambda x: x['above_0.9']['percentage'])
        sns.barplot(data=df, x='dataset', y='contamination_90', hue='mode', ax=axes[1,0], palette='Reds')
        axes[1,0].set_title('Severe Contamination Rate (≥0.90)')
        
        pivot = df.pivot(index='dataset', columns='mode', values='max_similarity')
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[1,1])
        axes[1,1].set_title('Max Similarity Heatmap')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# MAIN PIPELINE (TWO-PHASE)
# =============================================================================
def run_analysis(
    config: AnalysisConfig,
    datasets: List[BenchmarkDataset],
    modes: List[str] = ['input', 'output', 'input_output'],
):
    cache_dir = Path(config.output_dir) / "data" / "embeddings_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # PHASE 1: EMBED ALL BENCHMARKS
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 1: Embedding all benchmarks")
    print("="*60)
    
    embedding_engine = EmbeddingEngine(config)
    benchmark_cache = {}
    
    for dataset in datasets:
        dataset.load()
        for mode in modes:
            texts, item_ids = dataset.get_texts(mode)
            print(f"\nEmbedding {dataset.name} - {mode} ({len(texts)} items)")
            
            embeddings = embedding_engine.embed_texts(texts)
            
            cache_path = cache_dir / f"{dataset.name}_{mode}_bench.npy"
            ids_path = cache_dir / f"{dataset.name}_{mode}_ids.json"
            np.save(cache_path, embeddings)
            with open(ids_path, 'w') as f:
                json.dump(item_ids, f)
            
            benchmark_cache[(dataset.name, mode)] = (cache_path, item_ids)
            del embeddings
            print(f"  Cached: {cache_path}")
    
    # UNLOAD MODEL
    embedding_engine.unload()
    del embedding_engine
    gc.collect()
    torch.cuda.empty_cache()
    
    # =========================================================================
    # PHASE 2: COMPUTE SIMILARITIES
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 2: Computing similarities (full GPU available)")
    print("="*60)
    
    data_manager = LocalDataManager(config)
    analyzer = SimilarityAnalyzer(config)
    visualizer = ContaminationVisualizer(config)
    
    corpus_files = data_manager.list_local_files()
    if not corpus_files:
        print(f"ERROR: No parquet files in {config.local_data_dir}")
        return
    
    all_results = []
    
    for dataset in datasets:
        for mode in modes:
            cache_path, item_ids = benchmark_cache[(dataset.name, mode)]
            bench_embeddings = np.load(cache_path)
            
            print(f"\n--- {dataset.name} - {mode} ---")
            
            data_manager._file_metadata_cache.clear()
            
            max_sims, top_k_sims, top_k_indices, index_mapping = analyzer.compute_similarities(
                bench_embeddings, data_manager, corpus_files
            )
            
            stats = analyzer.compute_statistics(max_sims, mode, dataset.name)
            print(f"  Max: {stats['max_similarity']:.4f}, Mean: {stats['mean_similarity']:.4f}, P99: {stats['percentiles']['p99']:.4f}")
            
            top_matches = analyzer.resolve_top_matches(top_k_sims, top_k_indices, index_mapping, data_manager)

            # Save to organized structure
            dataset_lower = dataset.name.lower()
            stats_path = Path(config.output_dir) / "data" / "statistics" / f"{dataset.name}_{mode}_stats.json"
            matches_path = Path(config.output_dir) / "data" / "top_matches" / f"{dataset.name}_{mode}_matches.json"
            matches_csv_path = Path(config.output_dir) / "data" / "top_matches" / f"{dataset.name}_{mode}_top100_matches.csv"
            sims_path = Path(config.output_dir) / "data" / "statistics" / f"{dataset.name}_{mode}_max_sims.npy"
            plot_path = Path(config.output_dir) / "plots" / dataset_lower / f"{mode}_distribution.png"

            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            with open(matches_path, 'w') as f:
                json.dump(top_matches, f, indent=2)
            np.save(sims_path, max_sims)

            # Save CSV format: one row per (benchmark_item, match) pair
            csv_rows = []
            for bench_idx, matches in enumerate(top_matches):
                bench_id = item_ids[bench_idx]
                for match in matches:
                    csv_rows.append({
                        'benchmark_id': bench_id,
                        'benchmark_index': bench_idx,
                        'rank': match['rank'],
                        'similarity': match['similarity'],
                        'corpus_idx': match['corpus_idx'],
                        'corpus_source': match['source'],
                        'corpus_text': match['text'],
                    })

            df_matches = pd.DataFrame(csv_rows)
            df_matches.to_csv(matches_csv_path, index=False)
            print(f"  Saved CSV: {matches_csv_path} ({len(csv_rows)} rows)")

            visualizer.plot_similarity_distribution(max_sims, dataset.name, mode, str(plot_path))
            
            all_results.append({
                'dataset': dataset.name,
                'mode': mode,
                'max_similarity': stats['max_similarity'],
                'mean_similarity': stats['mean_similarity'],
                'percentiles': stats['percentiles'],
                'threshold_stats': stats['threshold_stats'],
            })
            
            del bench_embeddings, max_sims, top_k_sims, top_k_indices, top_matches, index_mapping
            gc.collect()
            torch.cuda.empty_cache()
    
    visualizer.plot_joint_comparison(all_results, str(Path(config.output_dir) / "plots" / "summary" / "joint_comparison.png"))

    with open(Path(config.output_dir) / "data" / "statistics" / "summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nDone. Results: {config.output_dir}")
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['musr', 'humaneval', 'mbpp', 'all'], default='all')
    parser.add_argument('--modes', nargs='+', default=['input', 'output', 'input_output'])
    parser.add_argument('--data-dir', type=str, default="/lambda/nfs/embeddings/embedding_folder")
    parser.add_argument('--output-dir', type=str, default="contamination_analysis")
    parser.add_argument('--embedding-batch-size', type=int, default=16)
    parser.add_argument('--compute-batch-size', type=int, default=500_000)
    parser.add_argument('--top-k', type=int, default=100)
    args = parser.parse_args()

    config = AnalysisConfig(
        embedding_batch_size=args.embedding_batch_size,
        local_data_dir=args.data_dir,
        output_dir=args.output_dir,
        top_k_matches=args.top_k,
        compute_batch_size=args.compute_batch_size,
    )
    
    datasets = []
    if args.dataset in ['musr', 'all']:
        datasets.append(MuSRDataset())
    if args.dataset in ['humaneval', 'all']:
        datasets.append(HumanEvalDataset())
    if args.dataset in ['mbpp', 'all']:
        datasets.append(MBPPDataset())

    run_analysis(config, datasets, args.modes)


if __name__ == "__main__":
    main()