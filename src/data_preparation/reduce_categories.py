# src/data_preparation/reduce_categories.py

import pandas as pd
import time
from pathlib import Path


def run(df: pd.DataFrame, top_values_dict: dict) -> pd.DataFrame:
    """
    Reduce cardinality of high-cardinality categorical columns.
    
    Replaces values not in top_values_dict with 'other'.
    
    Args:
        df: Input DataFrame
        top_values_dict: Dictionary mapping column names to lists of values to keep
                        e.g., {'job': ['engineer', 'teacher', ...], 'merchant': [...]}
    
    Returns:
        DataFrame with reduced categories
    """
    for col, top_values in top_values_dict.items():
        if col in df.columns:
            df[col] = df[col].where(df[col].isin(top_values), "other")
    
    return df


def run_chunked(input_path: Path, output_path: Path, top_n: int = 50, chunksize: int = 1_000_000):
    """
    Process large CSV files in chunks with two-pass approach:
    Pass 1: Scan all data to find top N categories
    Pass 2: Apply reduction to each chunk
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        top_n: Number of top categories to keep (default: 50)
        chunksize: Number of rows to process at a time (default: 1 million)
    """
    start = time.time()
    
    print(f"📊 Starting chunked category reduction")
    print(f"   Input: {input_path}")
    print(f"   Keeping top {top_n} categories per column")
    print(f"   Chunk size: {chunksize:,} rows")
    print()
    
    # Columns to reduce
    cols_to_reduce = ["job", "merchant"]
    
    # --- PASS 1: Find top N categories ---
    print("  Pass 1/2: Scanning data to find top categories...")
    
    # Accumulate value counts across all chunks
    value_counts = {col: {} for col in cols_to_reduce}
    total_rows = 0
    chunk_num = 0
    
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        chunk_num += 1
        total_rows += len(chunk)
        
        for col in cols_to_reduce:
            if col in chunk.columns:
                # Count values in this chunk
                chunk_counts = chunk[col].value_counts().to_dict()
                
                # Merge with accumulated counts
                for value, count in chunk_counts.items():
                    value_counts[col][value] = value_counts[col].get(value, 0) + count
        
        if chunk_num % 5 == 0:  # Progress update every 5 chunks
            print(f"    Scanned {total_rows:,} rows...")
    
    print(f"    Total rows scanned: {total_rows:,}")
    
    # Determine top N values for each column
    top_values_dict = {}
    summary = []
    
    for col in cols_to_reduce:
        if value_counts[col]:
            # Sort by count and take top N
            sorted_values = sorted(value_counts[col].items(), key=lambda x: x[1], reverse=True)
            top_values = [val for val, count in sorted_values[:top_n]]
            top_values_dict[col] = top_values
            
            unique_before = len(value_counts[col])
            unique_after = min(top_n + 1, unique_before)  # +1 for 'other'
            
            summary.append({
                "column": col,
                "unique_before": unique_before,
                "unique_after": unique_after,
                "reduction_%": round((1 - unique_after/unique_before) * 100, 1)
            })
            
            print(f"\n  {col}: {unique_before} → {unique_after} categories ({summary[-1]['reduction_%']}% reduction)")
    
    if summary:
        print("\n📊 Reduction Summary:")
        print(pd.DataFrame(summary).to_string(index=False))
        print()
    
    # --- PASS 2: Apply reduction to each chunk ---
    print("\n  Pass 2/2: Applying category reduction...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    first_chunk = True
    chunk_num = 0
    processed_rows = 0
    
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        chunk_num += 1
        processed_rows += len(chunk)
        
        # Apply reduction
        chunk = run(chunk, top_values_dict)
        
        # Write to CSV
        if first_chunk:
            chunk.to_csv(output_path, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(output_path, index=False, mode='a', header=False)
        
        if chunk_num % 5 == 0:
            print(f"    Processed {processed_rows:,} rows...")
    
    end = time.time()
    
    print(f"\n✅ Category reduction complete!")
    print(f"   Total rows processed: {processed_rows:,}")
    print(f"   Output: {output_path}")
    print(f"⏱️  Runtime: {end - start:.1f} seconds ({(end - start)/60:.1f} minutes)\n")


# Standalone execution (for testing/debugging)
if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / "data"
    input_path = data_dir / "processed" / "01_clean_text.csv"
    output_path = data_dir / "processed" / "02_reduced_categories.csv"
    
    # Use chunked processing for large files
    run_chunked(input_path, output_path, top_n=50, chunksize=1_000_000)
