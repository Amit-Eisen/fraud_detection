# src/data_cleansing/01_remove_duplicates.py

import pandas as pd
import time
from pathlib import Path


def check_and_remove_duplicates_chunked(input_path: Path, output_path: Path, chunksize: int = 1_000_000):
    """
    Check and remove logical duplicates in large CSV files using memory-efficient chunked processing.
    
    Checks for duplicates based on business logic:
    - Same credit card (cc_num)
    - Same amount (amt)
    - Same merchant
    - Same date and time
    
    This combination identifies true duplicate transactions.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        chunksize: Number of rows to load at a time (default: 1 million)
        
    Returns:
        Number of duplicate rows removed
    """
    start = time.time()
    
    print(f"🔍 Checking and removing duplicate transactions")
    print(f"   Input: {input_path}")
    print(f"   Checking: cc_num + amt + merchant + trans_date + trans_time")
    print(f"   Chunk size: {chunksize:,} rows")
    print()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Columns to check for duplicates (business logic)
    subset_cols = ['cc_num', 'amt', 'merchant', 'trans_date', 'trans_time']
    
    # PASS 1: Load only key columns to identify duplicates
    print("   PASS 1: Loading key columns to identify duplicates...")
    chunks = []
    total_rows = 0
    chunk_num = 0
    
    for chunk in pd.read_csv(input_path, usecols=subset_cols, chunksize=chunksize, low_memory=False):
        chunk_num += 1
        rows_in_chunk = len(chunk)
        total_rows += rows_in_chunk
        chunks.append(chunk)
        print(f"   Loaded chunk {chunk_num}: {rows_in_chunk:,} rows (total: {total_rows:,})")
    
    # Combine chunks to find duplicates
    print(f"\n   Combining {chunk_num} chunks...")
    df_subset = pd.concat(chunks, ignore_index=True)
    del chunks  # Free memory
    
    print(f"   Total rows: {len(df_subset):,}")
    
    # Identify duplicate row indices
    print("   Analyzing duplicates...")
    duplicates_mask = df_subset.duplicated(keep='first')
    num_duplicates = duplicates_mask.sum()
    duplicate_indices = set(duplicates_mask[duplicates_mask].index)
    
    del df_subset  # Free memory
    
    print(f"\n{'='*70}")
    if num_duplicates > 0:
        print(f"⚠️  DUPLICATES FOUND: {num_duplicates:,} duplicate transactions")
        print(f"   ({num_duplicates/total_rows*100:.4f}% of dataset)")
        print(f"\n   Removing duplicates...")
    else:
        print(f"✅ NO DUPLICATES FOUND")
        print(f"   All {total_rows:,} transactions are unique")
    print(f"{'='*70}\n")
    
    # PASS 2: Load full dataset and remove duplicates
    if num_duplicates > 0:
        print("   PASS 2: Loading full dataset and removing duplicates...")
        first_chunk = True
        rows_written = 0
        chunk_num = 0
        global_index = 0
        
        for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
            chunk_num += 1
            
            # Filter out duplicate rows
            chunk_indices = range(global_index, global_index + len(chunk))
            keep_mask = [i not in duplicate_indices for i in chunk_indices]
            chunk_clean = chunk[keep_mask]
            
            rows_written += len(chunk_clean)
            global_index += len(chunk)
            
            print(f"   Processing chunk {chunk_num}: kept {len(chunk_clean):,}/{len(chunk):,} rows")
            
            # Write to CSV
            if first_chunk:
                chunk_clean.to_csv(output_path, index=False, mode='w')
                first_chunk = False
            else:
                chunk_clean.to_csv(output_path, index=False, mode='a', header=False)
        
        print(f"\n   Rows written: {rows_written:,}")
    else:
        # No duplicates, just copy the file (chunked to avoid memory issues)
        print("   Copying dataset (no duplicates to remove)...")
        first_chunk = True
        rows_written = 0
        
        for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
            rows_written += len(chunk)
            
            if first_chunk:
                chunk.to_csv(output_path, index=False, mode='w')
                first_chunk = False
            else:
                chunk.to_csv(output_path, index=False, mode='a', header=False)
    
    end = time.time()
    
    print(f"\n✅ Duplicate removal complete!")
    print(f"   Output: {output_path}")
    print(f"   Duplicates removed: {num_duplicates:,}")
    print(f"⏱️  Runtime: {end - start:.1f} seconds ({(end - start)/60:.1f} minutes)\n")
    
    return num_duplicates


# Standalone execution
if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / "data"
    input_path = data_dir / "processed" / "03_transformed.csv"
    output_path = data_dir / "cleansed" / "01_no_duplicates.csv"
    
    # Check and remove duplicates
    check_and_remove_duplicates_chunked(input_path, output_path, chunksize=1_000_000)

