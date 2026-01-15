"""
Export annotation results to shareable CSV files.

Creates:
1. Full annotations CSV with all fields
2. Summary CSV with key metrics per test_id
3. Statistics JSON with aggregate metrics

Usage:
    python export_annotations.py --benchmark mbpp
    python export_annotations.py --benchmark codeforces
    python export_annotations.py --benchmark all
"""

import argparse
import csv
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ANNOTATIONS_DIR = Path(__file__).parent / "annotations"
EXPORT_DIR = Path(__file__).parent / "exports"


def load_annotations(benchmark: str) -> list[dict]:
    """Load all annotations for a benchmark."""
    annotations_path = ANNOTATIONS_DIR / benchmark
    if not annotations_path.exists():
        print(f"No annotations found for {benchmark}")
        return []
    
    annotations = []
    errors = 0
    
    for f in annotations_path.glob("*.json"):
        if f.name.startswith("_"):
            continue
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            annotations.append(data)
        except Exception as e:
            errors += 1
    
    if errors:
        print(f"  Warning: {errors} files had encoding issues")
    
    return annotations


def export_full_csv(annotations: list[dict], benchmark: str, output_dir: Path) -> Path:
    """Export full annotations to CSV."""
    output_path = output_dir / f"{benchmark}_annotations_full.csv"
    
    # Define columns
    columns = [
        "test_id",
        "corpus_id", 
        "dataset",
        "benchmark",
        "score",
        "weight",
        "is_sd",
        "confidence",
        "match_type",
        "reasoning",
        "success",
        "error",
        "cost_total",
        "prompt_tokens",
        "thought_tokens",
        "answer_tokens",
        "model",
        "timestamp",
    ]
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        
        for ann in annotations:
            row = {
                "test_id": ann.get("test_id"),
                "corpus_id": ann.get("corpus_id"),
                "dataset": ann.get("dataset"),
                "benchmark": ann.get("benchmark"),
                "score": ann.get("score"),
                "weight": ann.get("weight"),
                "is_sd": ann.get("annotation", {}).get("is_sd") if ann.get("annotation") else None,
                "confidence": ann.get("annotation", {}).get("confidence") if ann.get("annotation") else None,
                "match_type": ann.get("annotation", {}).get("match_type") if ann.get("annotation") else None,
                "reasoning": ann.get("annotation", {}).get("reasoning") if ann.get("annotation") else None,
                "success": ann.get("success"),
                "error": ann.get("error"),
                "cost_total": ann.get("cost", {}).get("total"),
                "prompt_tokens": ann.get("usage", {}).get("prompt_tokens"),
                "thought_tokens": ann.get("usage", {}).get("thought_tokens"),
                "answer_tokens": ann.get("usage", {}).get("answer_tokens"),
                "model": ann.get("metadata", {}).get("model"),
                "timestamp": ann.get("metadata", {}).get("timestamp"),
            }
            writer.writerow(row)
    
    return output_path


def export_summary_csv(annotations: list[dict], benchmark: str, output_dir: Path) -> Path:
    """Export summary per test_id."""
    output_path = output_dir / f"{benchmark}_annotations_summary.csv"
    
    # Group by test_id
    by_test = defaultdict(list)
    for ann in annotations:
        if ann.get("success") and ann.get("annotation"):
            by_test[ann["test_id"]].append(ann)
    
    columns = [
        "test_id",
        "num_matches_annotated",
        "num_semantic_duplicates",
        "sd_rate",
        "avg_confidence",
        "max_score",
        "avg_score",
        "datasets",
        "match_types",
    ]
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for test_id in sorted(by_test.keys(), key=lambda x: int(x) if x.isdigit() else x):
            anns = by_test[test_id]
            num_sd = sum(1 for a in anns if a["annotation"]["is_sd"])
            confidences = [a["annotation"]["confidence"] for a in anns]
            scores = [a["score"] for a in anns]
            datasets = set(a["dataset"] for a in anns)
            match_types = [a["annotation"]["match_type"] for a in anns if a["annotation"]["is_sd"]]
            
            row = {
                "test_id": test_id,
                "num_matches_annotated": len(anns),
                "num_semantic_duplicates": num_sd,
                "sd_rate": num_sd / len(anns) if anns else 0,
                "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
                "max_score": max(scores) if scores else 0,
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "datasets": "|".join(sorted(datasets)),
                "match_types": "|".join(match_types) if match_types else "",
            }
            writer.writerow(row)
    
    return output_path


def export_statistics(annotations: list[dict], benchmark: str, output_dir: Path) -> Path:
    """Export aggregate statistics."""
    output_path = output_dir / f"{benchmark}_statistics.json"
    
    successful = [a for a in annotations if a.get("success") and a.get("annotation")]
    
    num_sd = sum(1 for a in successful if a["annotation"]["is_sd"])
    sd_confidences = [a["annotation"]["confidence"] for a in successful if a["annotation"]["is_sd"]]
    all_confidences = [a["annotation"]["confidence"] for a in successful]
    
    # Match type distribution
    match_types = defaultdict(int)
    for a in successful:
        match_types[a["annotation"]["match_type"]] += 1
    
    # Per-dataset breakdown
    by_dataset = defaultdict(lambda: {"total": 0, "sd": 0})
    for a in successful:
        ds = a["dataset"]
        by_dataset[ds]["total"] += 1
        if a["annotation"]["is_sd"]:
            by_dataset[ds]["sd"] += 1
    
    # Cost breakdown
    total_cost = sum(a.get("cost", {}).get("total", 0) for a in annotations)
    
    stats = {
        "benchmark": benchmark,
        "exported_at": datetime.now().isoformat(),
        "total_annotations": len(annotations),
        "successful_annotations": len(successful),
        "failed_annotations": len(annotations) - len(successful),
        "semantic_duplicates": {
            "count": num_sd,
            "rate": num_sd / len(successful) if successful else 0,
            "avg_confidence": sum(sd_confidences) / len(sd_confidences) if sd_confidences else 0,
        },
        "confidence": {
            "overall_avg": sum(all_confidences) / len(all_confidences) if all_confidences else 0,
            "sd_avg": sum(sd_confidences) / len(sd_confidences) if sd_confidences else 0,
        },
        "match_type_distribution": dict(match_types),
        "per_dataset": {
            ds: {
                "total": v["total"],
                "semantic_duplicates": v["sd"],
                "sd_rate": v["sd"] / v["total"] if v["total"] > 0 else 0,
            }
            for ds, v in by_dataset.items()
        },
        "cost": {
            "total_usd": total_cost,
        },
        "unique_test_ids": len(set(a["test_id"] for a in successful)),
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    return output_path


def export_benchmark(benchmark: str):
    """Export all files for a benchmark."""
    print(f"\nExporting {benchmark}...")
    
    # Create export directory
    output_dir = EXPORT_DIR / benchmark
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    annotations = load_annotations(benchmark)
    if not annotations:
        return
    
    print(f"  Loaded {len(annotations)} annotations")
    
    # Export files
    full_path = export_full_csv(annotations, benchmark, output_dir)
    print(f"  Created: {full_path.name}")
    
    summary_path = export_summary_csv(annotations, benchmark, output_dir)
    print(f"  Created: {summary_path.name}")
    
    stats_path = export_statistics(annotations, benchmark, output_dir)
    print(f"  Created: {stats_path.name}")
    
    # Print summary
    successful = [a for a in annotations if a.get("success") and a.get("annotation")]
    num_sd = sum(1 for a in successful if a["annotation"]["is_sd"])
    
    print(f"\n  Summary:")
    print(f"    Total annotations: {len(annotations)}")
    print(f"    Successful: {len(successful)}")
    print(f"    Semantic duplicates: {num_sd} ({num_sd/len(successful)*100:.1f}%)")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Export annotations to CSV")
    parser.add_argument(
        "--benchmark",
        choices=["mbpp", "codeforces", "all"],
        default="all",
        help="Which benchmark to export",
    )
    
    args = parser.parse_args()
    
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.benchmark == "all":
        benchmarks = ["mbpp", "codeforces"]
    else:
        benchmarks = [args.benchmark]
    
    for benchmark in benchmarks:
        if (ANNOTATIONS_DIR / benchmark).exists():
            export_benchmark(benchmark)
    
    print(f"\n[OK] Exports saved to: {EXPORT_DIR}")
    print("\nFiles to share with co-authors:")
    for benchmark in benchmarks:
        bdir = EXPORT_DIR / benchmark
        if bdir.exists():
            print(f"  {benchmark}/")
            for f in sorted(bdir.glob("*")):
                size = f.stat().st_size
                if size > 1_000_000:
                    size_str = f"{size/1_000_000:.1f}MB"
                elif size > 1_000:
                    size_str = f"{size/1_000:.1f}KB"
                else:
                    size_str = f"{size}B"
                print(f"    - {f.name} ({size_str})")


if __name__ == "__main__":
    main()
