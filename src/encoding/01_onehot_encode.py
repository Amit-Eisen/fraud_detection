# src/encoding/01_onehot_encode.py

import pandas as pd
import numpy as np
import time
import json
from pathlib import Path


def run(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    One-hot encode low-cardinality categorical features.
    
    Encodes: gender, trans_weekday, category, state
    
    Args:
        df: Input DataFrame
    
    Returns:
        Tuple of (encoded DataFrame, encoding mappings dict)
    """
    # Columns to one-hot encode
    onehot_cols = ['gender', 'trans_weekday', 'category', 'state']
    available_cols = [col for col in onehot_cols if col in df.columns]
    
    if not available_cols:
        print("⚠️  No columns to one-hot encode!")
        return df, {}
    
    print(f"  One-hot encoding: {available_cols}")
    
    # Store mappings for documentation
    mappings = {}
    
    # One-hot encode each column
    for col in available_cols:
        # Get unique values before encoding
        unique_values = sorted(df[col].dropna().unique().tolist())
        mappings[col] = {
            'encoding_type': 'one-hot',
            'unique_values': unique_values,
            'num_columns_created': len(unique_values)
        }
        
        # One-hot encode
        encoded = pd.get_dummies(df[col], prefix=col, drop_first=False)
        
        # Add encoded columns to dataframe
        df = pd.concat([df, encoded], axis=1)
        
        # Drop original column
        df = df.drop(columns=[col])
        
        print(f"    {col}: {len(unique_values)} values → {len(unique_values)} binary columns")
    
    return df, mappings


def run_chunked(input_path: Path, output_path: Path, mappings_path: Path, chunksize: int = 1_000_000):
    """
    One-hot encode large CSV files in chunks.
    
    Uses two-pass approach:
    1. First pass: Collect all unique values for each column
    2. Second pass: Apply consistent one-hot encoding across all chunks
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        mappings_path: Path to save encoding mappings JSON
        chunksize: Number of rows to process at a time
    """
    start = time.time()
    
    print(f"🔢 Starting one-hot encoding (chunked)")
    print(f"   Input: {input_path}")
    print(f"   Chunk size: {chunksize:,} rows")
    print()
    
    # Columns to one-hot encode
    onehot_cols = ['gender', 'trans_weekday', 'category', 'state']
    
    # PASS 1: Collect all unique values for each column
    print("📊 Pass 1: Collecting unique values...")
    unique_values = {col: set() for col in onehot_cols}
    
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        for col in onehot_cols:
            if col in chunk.columns:
                unique_values[col].update(chunk[col].dropna().unique())
    
    # Convert sets to sorted lists
    for col in onehot_cols:
        unique_values[col] = sorted(list(unique_values[col]))
    
    # Print summary
    print()
    for col in onehot_cols:
        if unique_values[col]:
            print(f"  {col}: {len(unique_values[col])} unique values")
    print()
    
    # Create mappings for documentation
    mappings = {}
    for col in onehot_cols:
        if unique_values[col]:
            mappings[col] = {
                'encoding_type': 'one-hot',
                'unique_values': unique_values[col],
                'num_columns_created': len(unique_values[col]),
                'column_names': [f"{col}_{val}" for val in unique_values[col]]
            }
    
    # Save mappings
    mappings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mappings_path, 'w') as f:
        json.dump(mappings, f, indent=2)
    print(f"💾 Encoding mappings saved: {mappings_path}")
    print()
    
    # PASS 2: Apply one-hot encoding consistently
    print("🔍 Pass 2: Applying one-hot encoding...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use temporary file to avoid reading/writing same file
    temp_output = output_path.parent / f"{output_path.stem}_temp.csv"
    
    first_chunk = True
    total_rows = 0
    chunk_num = 0
    
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        chunk_num += 1
        rows_in_chunk = len(chunk)
        
        # One-hot encode each column with consistent categories
        for col in onehot_cols:
            if col in chunk.columns and unique_values[col]:
                # Use pd.get_dummies with explicit categories
                encoded = pd.get_dummies(chunk[col], prefix=col, drop_first=False)
                
                # Ensure all expected columns exist (fill missing with 0)
                expected_cols = [f"{col}_{val}" for val in unique_values[col]]
                for exp_col in expected_cols:
                    if exp_col not in encoded.columns:
                        encoded[exp_col] = 0
                
                # Keep only expected columns (in case of unexpected values)
                encoded = encoded[expected_cols]
                
                # Add to chunk
                chunk = pd.concat([chunk, encoded], axis=1)
                
                # Drop original column
                chunk = chunk.drop(columns=[col])
        
        total_rows += rows_in_chunk
        print(f"  Chunk {chunk_num}: {rows_in_chunk:,} rows processed (total: {total_rows:,})")
        
        # Write to CSV
        if first_chunk:
            chunk.to_csv(temp_output, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(temp_output, index=False, mode='a', header=False)
    
    # Replace original file with temp file
    import shutil
    shutil.move(str(temp_output), str(output_path))
    
    end = time.time()
    
    print(f"\n✅ One-hot encoding complete!")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Output: {output_path}")
    print(f"⏱️  Runtime: {end - start:.1f} seconds ({(end - start)/60:.1f} minutes)\n")


# Standalone execution
if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / "data"
    reports_dir = Path(__file__).resolve().parents[2] / "reports"
    
    input_file = data_dir / "cleansed" / "02_clean_columns.csv"
    output_file = data_dir / "encoded" / "01_one_hot_encoded.csv"
    mappings_file = reports_dir / "encoding_mappings_onehot.json"
    
    print("🔢 ONE-HOT ENCODING")
    print("=" * 80)
    print()
    
    # Use chunked processing
    run_chunked(input_file, output_file, mappings_file, chunksize=1_000_000)
    
    print("✅ One-hot encoding complete!")

