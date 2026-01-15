#!/usr/bin/env python3
"""
Quick diagnostic script to investigate parquet files
Run this on Lambda to understand what's wrong with your embedding files.
"""

import os
import sys
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np


def diagnose_file(file_path):
    """Diagnose a single parquet file."""
    print(f"\n{'='*60}")
    print(f"FILE: {file_path}")
    print('='*60)
    
    try:
        # 1. Get metadata without reading data
        pf = pq.ParquetFile(file_path)
        
        # CORRECT way to get top-level column names (not leaf names)
        top_level_cols = [field.name for field in pf.schema_arrow]
        
        print(f"\n📋 SCHEMA:")
        for field in pf.schema_arrow:
            print(f"   {field.name}: {field.type}")
        
        print(f"\n📊 STRUCTURE:")
        print(f"   Row Groups: {pf.num_row_groups}")
        print(f"   Total Rows: {pf.metadata.num_rows}")
        
        # Check each row group
        print(f"\n📦 ROW GROUPS:")
        for i in range(pf.num_row_groups):
            rg = pf.metadata.row_group(i)
            print(f"   Group {i}: {rg.num_rows} rows")
        
        # 2. Try reading just embeddings column metadata
        if 'embeddings' in top_level_cols:
            print(f"\n🔢 EMBEDDINGS COLUMN:")
            
            # Get the type info
            emb_field = pf.schema_arrow.field('embeddings')
            print(f"   Type: {emb_field.type}")
            
            # Try to read first row group only
            try:
                chunk = pf.read_row_group(0, columns=['embeddings'])
                emb = chunk['embeddings']
                
                # Check chunk structure
                print(f"   Chunks in first row group: {emb.num_chunks}")
                
                first_emb = None
                if emb.num_chunks > 0:
                    first_chunk = emb.chunk(0)
                    print(f"   First chunk length: {len(first_chunk)}")
                    
                    # Get actual embedding dimension
                    if len(first_chunk) > 0:
                        first_emb = first_chunk[0].as_py()
                        if first_emb:
                            print(f"   Embedding dimension: {len(first_emb)}")
                            print(f"   Sample values: {first_emb[:5]}...")
                
                # Calculate total size
                total_floats = 0
                for i in range(pf.num_row_groups):
                    rg = pf.metadata.row_group(i)
                    total_floats += rg.num_rows
                
                # Check if this would overflow 32-bit
                if first_emb:
                    dim = len(first_emb)
                    total_elements = total_floats * dim
                    print(f"\n⚠️ OVERFLOW CHECK:")
                    print(f"   Total rows: {total_floats}")
                    print(f"   Embedding dimension: {dim}")
                    print(f"   Total elements: {total_elements:,}")
                    print(f"   Int32 max: {2**31-1:,}")
                    
                    if total_elements > 2**31 - 1:
                        print(f"   🚨 WILL OVERFLOW if combined!")
                    else:
                        print(f"   ✅ Should fit in int32")
                        
            except Exception as e:
                print(f"   ❌ Error reading embeddings: {e}")
        else:
            print(f"\n❌ No 'embeddings' column found!")
            print(f"   Available columns: {top_level_cols}")
            
        # 3. Check for other columns
        other_cols = [c for c in top_level_cols if c != 'embeddings']
        if other_cols:
            print(f"\n📝 OTHER COLUMNS: {other_cols}")
            
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()


def diagnose_directory(data_dir, sample_size=5):
    """Sample and diagnose files from directory."""
    files = sorted(Path(data_dir).rglob("*.parquet"))
    
    print(f"\n🔍 PARQUET DIAGNOSTIC REPORT")
    print(f"   Directory: {data_dir}")
    print(f"   Total files: {len(files)}")
    
    if not files:
        print("❌ No parquet files found!")
        return
    
    # Sample files evenly distributed
    indices = np.linspace(0, len(files)-1, min(sample_size, len(files)), dtype=int)
    sample_files = [files[i] for i in indices]
    
    print(f"   Sampling {len(sample_files)} files...")
    
    for f in sample_files:
        diagnose_file(str(f))
    
    # Quick summary: try to identify problematic files
    print(f"\n\n{'='*60}")
    print("📈 SUMMARY - SCANNING ALL FILES FOR ISSUES")
    print('='*60)
    
    good = 0
    bad = 0
    bad_files = []
    
    for f in files:
        try:
            pf = pq.ParquetFile(str(f))
            top_cols = [field.name for field in pf.schema_arrow]
            
            if 'embeddings' not in top_cols:
                bad += 1
                if len(bad_files) < 20:
                    bad_files.append((f.name, f"No embeddings col. Has: {top_cols[:3]}..."))
                continue
                
            # Quick test: can we read embeddings?
            chunk = pf.read_row_group(0, columns=['embeddings'])
            good += 1
        except Exception as e:
            bad += 1
            if len(bad_files) < 20:
                bad_files.append((f.name, str(e)[:80]))
    
    print(f"\n✅ Readable files: {good}")
    print(f"❌ Problematic files: {bad}")
    
    if bad_files:
        print(f"\n🔴 First {len(bad_files)} problematic files:")
        for name, err in bad_files:
            print(f"   {name}: {err}")


def test_read_strategies(file_path):
    """Test different strategies to read a problematic file."""
    print(f"\n\n{'='*60}")
    print(f"🧪 TESTING READ STRATEGIES FOR: {file_path}")
    print('='*60)
    
    # Strategy 1: Direct read (will likely fail)
    print("\n1️⃣ Strategy: Direct read_table()")
    try:
        t = pq.read_table(file_path, columns=['embeddings'])
        print(f"   ✅ Success! Rows: {len(t)}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Strategy 2: Read row groups individually
    print("\n2️⃣ Strategy: Read row groups individually")
    try:
        pf = pq.ParquetFile(file_path)
        total_rows = 0
        for i in range(pf.num_row_groups):
            chunk = pf.read_row_group(i, columns=['embeddings'])
            total_rows += len(chunk)
        print(f"   ✅ Success! Total rows: {total_rows}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Strategy 3: Read and process without combining
    print("\n3️⃣ Strategy: Process chunks without combine_chunks()")
    try:
        pf = pq.ParquetFile(file_path)
        all_embeddings = []
        
        for i in range(pf.num_row_groups):
            chunk = pf.read_row_group(i, columns=['embeddings'])
            emb_col = chunk['embeddings']
            
            # Process each chunk individually
            for j in range(emb_col.num_chunks):
                arr = emb_col.chunk(j)
                # Convert to numpy directly
                vals = arr.values.to_numpy()
                rows = len(arr)
                if rows > 0:
                    dim = len(vals) // rows
                    mat = vals.reshape(rows, dim)
                    all_embeddings.append(mat)
        
        if all_embeddings:
            final = np.vstack(all_embeddings)
            print(f"   ✅ Success! Shape: {final.shape}")
        else:
            print(f"   ⚠️ No embeddings extracted")
            
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python diagnose_parquet.py /path/to/data/dir")
        print("  python diagnose_parquet.py /path/to/specific/file.parquet")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if os.path.isfile(path):
        diagnose_file(path)
        test_read_strategies(path)
    else:
        diagnose_directory(path)
        
        # Also test strategies on a problematic file if found
        files = sorted(Path(path).rglob("*.parquet"))
        for f in files:
            try:
                pf = pq.ParquetFile(str(f))
                cols = [field.name for field in pf.schema_arrow]
                if 'embeddings' in cols:
                    pq.read_table(str(f), columns=['embeddings'])
            except Exception:
                print(f"\n\n🎯 Found problematic file, testing strategies...")
                test_read_strategies(str(f))
                break
