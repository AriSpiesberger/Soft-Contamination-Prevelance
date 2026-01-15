"""
Comprehensive validation of annotation exports.

Checks:
1. JSON integrity - all files parse correctly
2. Required field presence
3. String length anomalies (empty, too short, too long)
4. Consistency between success flag and folder location
5. Field value validity (confidence 0-1, match_type in allowed set)
6. Duplicate detection
7. Cost/token sanity checks
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ANNOTATIONS_DIR = Path(__file__).parent / "annotations"

# Expected match types
VALID_MATCH_TYPES = {"exact", "equivalent", "subset", "superset", "unrelated"}


def validate_benchmark(benchmark: str) -> dict:
    """Validate all annotations for a benchmark."""
    print(f"\n{'='*60}")
    print(f"VALIDATING: {benchmark}")
    print(f"{'='*60}")
    
    base_path = ANNOTATIONS_DIR / benchmark
    failed_path = base_path / "_failed"
    
    results = {
        "benchmark": benchmark,
        "total_files": 0,
        "successful_files": 0,
        "failed_files": 0,
        "json_parse_errors": [],
        "missing_fields": [],
        "invalid_values": [],
        "string_anomalies": [],
        "consistency_errors": [],
        "duplicates": [],
        "cost_anomalies": [],
        "field_stats": defaultdict(lambda: {"min": float('inf'), "max": 0, "empty": 0, "null": 0}),
    }
    
    # Track for duplicate detection
    seen_ids = set()
    
    # Process main folder (successful annotations)
    main_files = list(base_path.glob("*.json"))
    results["successful_files"] = len(main_files)
    
    print(f"\n[1] Checking main folder: {len(main_files)} files")
    for f in main_files:
        validate_file(f, results, seen_ids, expected_success=True)
    
    # Process _failed folder
    if failed_path.exists():
        failed_files = list(failed_path.glob("*.json"))
        results["failed_files"] = len(failed_files)
        print(f"\n[2] Checking _failed folder: {len(failed_files)} files")
        for f in failed_files:
            validate_file(f, results, seen_ids, expected_success=False)
    
    results["total_files"] = results["successful_files"] + results["failed_files"]
    
    return results


def validate_file(filepath: Path, results: dict, seen_ids: set, expected_success: bool):
    """Validate a single annotation file."""
    filename = filepath.name
    
    # 1. JSON parsing
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        results["json_parse_errors"].append({"file": filename, "error": str(e)})
        return
    except Exception as e:
        results["json_parse_errors"].append({"file": filename, "error": f"Read error: {e}"})
        return
    
    # 2. Required fields
    required_base = ["test_id", "corpus_id", "dataset", "benchmark", "test_text", 
                     "corpus_text", "score", "prompt", "success"]
    for field in required_base:
        if field not in data:
            results["missing_fields"].append({"file": filename, "field": field})
    
    # 3. Consistency check: success flag vs folder
    is_success = data.get("success", False)
    if expected_success and not is_success:
        # File in main folder but marked as failed
        results["consistency_errors"].append({
            "file": filename,
            "issue": f"In main folder but success={is_success}",
            "error": data.get("error", "No error message")[:100]
        })
    elif not expected_success and is_success:
        # File in _failed folder but marked as success
        results["consistency_errors"].append({
            "file": filename,
            "issue": "In _failed folder but success=True"
        })
    
    # 4. Annotation validation (only for successful)
    if is_success:
        annotation = data.get("annotation")
        if annotation is None:
            results["invalid_values"].append({
                "file": filename,
                "field": "annotation",
                "issue": "success=True but annotation is null"
            })
        else:
            # Check annotation fields
            if "is_sd" not in annotation:
                results["missing_fields"].append({"file": filename, "field": "annotation.is_sd"})
            if "confidence" not in annotation:
                results["missing_fields"].append({"file": filename, "field": "annotation.confidence"})
            else:
                conf = annotation["confidence"]
                if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                    results["invalid_values"].append({
                        "file": filename,
                        "field": "confidence",
                        "value": conf,
                        "issue": "Must be 0-1"
                    })
            
            if "match_type" not in annotation:
                results["missing_fields"].append({"file": filename, "field": "annotation.match_type"})
            else:
                mt = annotation["match_type"]
                if mt not in VALID_MATCH_TYPES:
                    results["invalid_values"].append({
                        "file": filename,
                        "field": "match_type",
                        "value": mt,
                        "issue": f"Must be one of {VALID_MATCH_TYPES}"
                    })
            
            if "reasoning" not in annotation:
                results["missing_fields"].append({"file": filename, "field": "annotation.reasoning"})
            else:
                reasoning = annotation["reasoning"]
                if reasoning:
                    track_string_stats(results, "reasoning", len(reasoning))
                    # Check for anomalously short/long reasoning
                    if len(reasoning) < 20:
                        results["string_anomalies"].append({
                            "file": filename,
                            "field": "reasoning",
                            "length": len(reasoning),
                            "preview": reasoning[:50],
                            "issue": "Very short reasoning"
                        })
                    if len(reasoning) > 5000:
                        results["string_anomalies"].append({
                            "file": filename,
                            "field": "reasoning",
                            "length": len(reasoning),
                            "issue": "Very long reasoning"
                        })
    
    # 5. String field checks
    for field in ["test_text", "corpus_text", "prompt"]:
        value = data.get(field, "")
        if value is None:
            results["field_stats"][field]["null"] += 1
        elif value == "":
            results["field_stats"][field]["empty"] += 1
            if field == "corpus_text":
                # Empty corpus is notable
                results["string_anomalies"].append({
                    "file": filename,
                    "field": field,
                    "issue": "Empty corpus_text"
                })
        else:
            track_string_stats(results, field, len(value))
            # Check for anomalies
            if field == "corpus_text" and len(value) < 10:
                results["string_anomalies"].append({
                    "file": filename,
                    "field": field,
                    "length": len(value),
                    "preview": value[:100],
                    "issue": "Very short corpus_text"
                })
    
    # 6. Score validation
    score = data.get("score")
    if score is not None:
        if not isinstance(score, (int, float)) or score < 0 or score > 1:
            results["invalid_values"].append({
                "file": filename,
                "field": "score",
                "value": score,
                "issue": "Score should be 0-1"
            })
    
    # 7. Cost sanity check
    cost = data.get("cost", {})
    total_cost = cost.get("total", 0)
    if total_cost is not None and total_cost > 1:  # More than $1 per annotation is suspicious
        results["cost_anomalies"].append({
            "file": filename,
            "cost": total_cost,
            "issue": "High cost per annotation"
        })
    
    # 8. Token sanity check  
    usage = data.get("usage", {})
    thought_tokens = usage.get("thought_tokens", 0)
    if thought_tokens and thought_tokens > 50000:  # 50k tokens is very high
        results["cost_anomalies"].append({
            "file": filename,
            "thought_tokens": thought_tokens,
            "issue": "Excessive thought tokens"
        })
    
    # 9. Duplicate detection (by test_id + corpus_id combination)
    unique_key = f"{data.get('test_id', '')}_{data.get('corpus_id', '')}"
    if unique_key in seen_ids:
        results["duplicates"].append({
            "file": filename,
            "key": unique_key
        })
    else:
        seen_ids.add(unique_key)


def track_string_stats(results: dict, field: str, length: int):
    """Track min/max string lengths."""
    stats = results["field_stats"][field]
    stats["min"] = min(stats["min"], length)
    stats["max"] = max(stats["max"], length)


def print_results(results: dict):
    """Print validation results."""
    print(f"\n{'='*60}")
    print(f"RESULTS FOR: {results['benchmark']}")
    print(f"{'='*60}")
    
    print(f"\n[COUNTS]")
    print(f"  Total files: {results['total_files']}")
    print(f"  Main folder (successful): {results['successful_files']}")
    print(f"  Failed folder: {results['failed_files']}")
    
    print(f"\n[JSON PARSING]")
    if results["json_parse_errors"]:
        print(f"  [FAIL] {len(results['json_parse_errors'])} parse errors")
        for err in results["json_parse_errors"][:5]:
            print(f"     - {err['file']}: {err['error'][:80]}")
        if len(results["json_parse_errors"]) > 5:
            print(f"     ... and {len(results['json_parse_errors']) - 5} more")
    else:
        print(f"  [OK] All files parse correctly")
    
    print(f"\n[MISSING FIELDS]")
    if results["missing_fields"]:
        field_counts = Counter(x["field"] for x in results["missing_fields"])
        print(f"  [FAIL] {len(results['missing_fields'])} missing field instances")
        for field, count in field_counts.most_common(10):
            print(f"     - {field}: {count} files")
    else:
        print(f"  [OK] All required fields present")
    
    print(f"\n[INVALID VALUES]")
    if results["invalid_values"]:
        print(f"  [FAIL] {len(results['invalid_values'])} invalid values")
        for inv in results["invalid_values"][:10]:
            print(f"     - {inv['file']}: {inv['field']} = {inv.get('value', 'N/A')}")
            print(f"       Issue: {inv['issue']}")
        if len(results["invalid_values"]) > 10:
            print(f"     ... and {len(results['invalid_values']) - 10} more")
    else:
        print(f"  [OK] All values valid")
    
    print(f"\n[CONSISTENCY ERRORS]")
    if results["consistency_errors"]:
        print(f"  [WARN] {len(results['consistency_errors'])} consistency issues")
        for err in results["consistency_errors"][:5]:
            print(f"     - {err['file']}: {err['issue']}")
            if 'error' in err:
                print(f"       Error: {err['error']}")
        if len(results["consistency_errors"]) > 5:
            print(f"     ... and {len(results['consistency_errors']) - 5} more")
    else:
        print(f"  [OK] Folder/success flag consistent")
    
    print(f"\n[STRING ANOMALIES]")
    if results["string_anomalies"]:
        anomaly_types = Counter(x["issue"] for x in results["string_anomalies"])
        print(f"  [WARN] {len(results['string_anomalies'])} string anomalies")
        for issue, count in anomaly_types.most_common():
            print(f"     - {issue}: {count} files")
        # Show some examples
        for anom in results["string_anomalies"][:3]:
            if "preview" in anom:
                print(f"       Example: {anom['file']} -> '{anom['preview'][:60]}...'")
    else:
        print(f"  [OK] No string anomalies")
    
    print(f"\n[STRING FIELD STATS]")
    for field, stats in sorted(results["field_stats"].items()):
        if stats["min"] != float('inf'):
            print(f"  {field}:")
            print(f"    Length range: {stats['min']} - {stats['max']}")
            if stats["empty"] > 0:
                print(f"    Empty: {stats['empty']}")
            if stats["null"] > 0:
                print(f"    Null: {stats['null']}")
    
    print(f"\n[DUPLICATES]")
    if results["duplicates"]:
        print(f"  [FAIL] {len(results['duplicates'])} duplicate entries")
        for dup in results["duplicates"][:5]:
            print(f"     - {dup['file']}: {dup['key']}")
    else:
        print(f"  [OK] No duplicates")
    
    print(f"\n[COST/TOKEN ANOMALIES]")
    if results["cost_anomalies"]:
        print(f"  [WARN] {len(results['cost_anomalies'])} anomalies")
        for anom in results["cost_anomalies"][:5]:
            print(f"     - {anom['file']}: {anom['issue']}")
    else:
        print(f"  [OK] No cost/token anomalies")


def main():
    """Run validation on all benchmarks."""
    print("="*60)
    print("ANNOTATION VALIDATION REPORT")
    print("="*60)
    
    benchmarks = []
    for d in ANNOTATIONS_DIR.iterdir():
        if d.is_dir() and not d.name.startswith("_"):
            benchmarks.append(d.name)
    
    print(f"\nBenchmarks found: {benchmarks}")
    
    all_results = {}
    for benchmark in benchmarks:
        results = validate_benchmark(benchmark)
        all_results[benchmark] = results
        print_results(results)
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    total_issues = 0
    for benchmark, results in all_results.items():
        issues = (
            len(results["json_parse_errors"]) +
            len(results["missing_fields"]) +
            len(results["invalid_values"]) +
            len(results["duplicates"])
        )
        warnings = (
            len(results["consistency_errors"]) +
            len(results["string_anomalies"]) +
            len(results["cost_anomalies"])
        )
        total_issues += issues
        
        status = "[OK]" if issues == 0 else "[FAIL]"
        print(f"\n{status} {benchmark}:")
        print(f"   Files: {results['total_files']} ({results['successful_files']} success, {results['failed_files']} failed)")
        print(f"   Critical issues: {issues}")
        print(f"   Warnings: {warnings}")
    
    print(f"\n{'='*60}")
    if total_issues == 0:
        print("[OK] VALIDATION PASSED - No critical issues found")
    else:
        print(f"[FAIL] VALIDATION FAILED - {total_issues} critical issues found")
    print(f"{'='*60}")
    
    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
