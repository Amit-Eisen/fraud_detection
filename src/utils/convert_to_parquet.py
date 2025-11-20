#!/usr/bin/env python3
"""
Convert CSV to Parquet Format
==============================

Converts the final feature-engineered CSV to Parquet format for faster loading
during model training. Parquet offers:
- 3-5x smaller file size (compression)
- 10-100x faster read speeds
- Column-based storage (read only what you need)

Usage:
    python convert_to_parquet.py
"""

import pandas as pd
from pathlib import Path
import time


def convert_csv_to_parquet(
    input_csv: Path,
    output_parquet: Path,
    chunksize: int = 1_000_000
) -> None:
    """
    Convert large CSV to Parquet format using chunked processing.
    
    Strategy: Write each chunk to a separate temp file, then combine at the end.
    This avoids loading the entire dataset into memory.
    
    Args:
        input_csv: Path to input CSV file
        output_parquet: Path to output Parquet file
        chunksize: Number of rows to process at a time
    """
    
    print("=" * 70)
    print("CSV → PARQUET CONVERSION (CHUNKED MODE)")
    print("=" * 70)
    print(f"📂 Input:  {input_csv}")
    print(f"💾 Output: {output_parquet}")
    print(f"🔄 Chunk size: {chunksize:,} rows")
    print("=" * 70)
    print()
    
    if not input_csv.exists():
        print(f"❌ ERROR: Input file not found!")
        print(f"   Expected: {input_csv}")
        return
    
    start_time = time.time()
    
    # Create temp directory for chunk files
    temp_dir = output_parquet.parent / "temp_parquet_chunks"
    temp_dir.mkdir(exist_ok=True)
    
    print("⏳ Step 1/2: Converting chunks to Parquet...")
    print()
    
    chunk_files = []
    total_rows = 0
    
    # First pass: read first chunk to get schema
    first_chunk = pd.read_csv(input_csv, nrows=chunksize)
    schema = None
    
    try:
        # Step 1: Write each chunk to a separate Parquet file with consistent schema
        for chunk_num, chunk in enumerate(pd.read_csv(input_csv, chunksize=chunksize), start=1):
            
            # On first chunk, capture the schema
            if chunk_num == 1:
                import pyarrow as pa
                schema = pa.Schema.from_pandas(chunk)
            
            # Write chunk to temp file with explicit schema
            chunk_file = temp_dir / f"chunk_{chunk_num:04d}.parquet"
            chunk.to_parquet(
                chunk_file,
                engine='pyarrow',
                compression='snappy',
                index=False,
                schema=schema  # Force same schema for all chunks
            )
            chunk_files.append(chunk_file)
            
            total_rows += len(chunk)
            elapsed = time.time() - start_time
            rows_per_sec = total_rows / elapsed
            
            print(f"✓ Chunk {chunk_num}: {len(chunk):,} rows processed "
                  f"| Total: {total_rows:,} rows | "
                  f"Speed: {rows_per_sec:,.0f} rows/sec")
        
        print()
        print(f"⏳ Step 2/2: Combining {len(chunk_files)} chunks into single file...")
        print("   (Using PyArrow for memory-efficient merging)")
        print()
        
        # Step 2: Use PyArrow to combine Parquet files efficiently
        # This doesn't load everything into memory at once!
        import pyarrow.parquet as pq
        import pyarrow as pa
        
        # Read all chunk files as PyArrow tables
        tables = []
        for i, chunk_file in enumerate(chunk_files, start=1):
            table = pq.read_table(chunk_file)
            tables.append(table)
            print(f"✓ Loaded chunk {i}/{len(chunk_files)}: {len(table):,} rows")
        
        print()
        print("⏳ Concatenating tables...")
        combined_table = pa.concat_tables(tables)
        
        print(f"✓ Combined {len(combined_table):,} total rows")
        print()
        
        print("⏳ Writing final Parquet file...")
        pq.write_table(
            combined_table,
            output_parquet,
            compression='snappy'
        )
        
        print()
        print("=" * 70)
        print("✅ CONVERSION COMPLETE!")
        print("=" * 70)
        
        # Compare file sizes
        csv_size_mb = input_csv.stat().st_size / (1024 * 1024)
        parquet_size_mb = output_parquet.stat().st_size / (1024 * 1024)
        compression_ratio = csv_size_mb / parquet_size_mb
        
        print(f"📊 CSV size:     {csv_size_mb:,.1f} MB")
        print(f"📦 Parquet size: {parquet_size_mb:,.1f} MB")
        print(f"🎯 Compression:  {compression_ratio:.1f}x smaller!")
        print(f"⏱️  Time taken:  {time.time() - start_time:.1f} seconds")
        print(f"📝 Total rows:   {total_rows:,}")
        print("=" * 70)
        
        # Cleanup temp files
        print()
        print("🧹 Cleaning up temporary files...")
        for chunk_file in chunk_files:
            chunk_file.unlink()
        temp_dir.rmdir()
        print("✓ Cleanup complete")
        
    except Exception as e:
        print(f"\n❌ ERROR during conversion: {e}")
        # Cleanup on error
        for chunk_file in chunk_files:
            if chunk_file.exists():
                chunk_file.unlink()
        if temp_dir.exists():
            temp_dir.rmdir()
        raise


def convert_csv_to_parquet_fast(
    input_csv: Path,
    output_parquet: Path
) -> None:
    """
    Fast conversion - load entire CSV at once (requires enough RAM).
    
    Use this if you have 32GB+ RAM.
    Otherwise, use the chunked version above.
    """
    
    print("=" * 70)
    print("CSV → PARQUET CONVERSION (FAST MODE)")
    print("=" * 70)
    print(f"📂 Input:  {input_csv}")
    print(f"💾 Output: {output_parquet}")
    print("=" * 70)
    print()
    
    if not input_csv.exists():
        print(f"❌ ERROR: Input file not found!")
        return
    
    start_time = time.time()
    
    print("⏳ Loading CSV... (this may take a few minutes)")
    df = pd.read_csv(input_csv, low_memory=False)
    
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    print()
    
    print("⏳ Writing Parquet...")
    df.to_parquet(
        output_parquet,
        engine='pyarrow',
        compression='snappy',
        index=False
    )
    
    print()
    print("=" * 70)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 70)
    
    # Compare file sizes
    csv_size_mb = input_csv.stat().st_size / (1024 * 1024)
    parquet_size_mb = output_parquet.stat().st_size / (1024 * 1024)
    compression_ratio = csv_size_mb / parquet_size_mb
    
    print(f"📊 CSV size:     {csv_size_mb:,.1f} MB")
    print(f"📦 Parquet size: {parquet_size_mb:,.1f} MB")
    print(f"🎯 Compression:  {compression_ratio:.1f}x smaller!")
    print(f"⏱️  Time taken:  {time.time() - start_time:.1f} seconds")
    print(f"📝 Total rows:   {len(df):,}")
    print("=" * 70)


if __name__ == "__main__":
    # Setup paths
    project_root = Path(__file__).resolve().parents[2]
    
    # Input: final feature-engineered CSV
    input_csv = project_root / "data" / "features" / "03_final_features.csv"
    
    # Output: Parquet file for model training
    output_parquet = project_root / "data" / "features" / "final_features.parquet"
    
    # Check if input exists
    if not input_csv.exists():
        print(f"❌ ERROR: Input file not found!")
        print(f"   Expected: {input_csv}")
        print()
        print("💡 Make sure you've run feature engineering first:")
        print("   cd src/feature_engineering")
        print("   python 04_interaction_features.py")
        exit(1)
    
    # Use chunked mode (safe for all RAM sizes)
    print("🚀 Starting chunked conversion...")
    print("   This is memory-safe and works on any machine!")
    print()
    
    # Process in 1M row chunks (adjust if needed)
    convert_csv_to_parquet(input_csv, output_parquet, chunksize=1_000_000)

