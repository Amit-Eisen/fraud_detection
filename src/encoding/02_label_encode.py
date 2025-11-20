# src/encoding/02_label_encode.py

import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def run(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label encode high-cardinality categorical features.
    
    Encodes: job, merchant, city, profile, zip
    
    Args:
        df: Input DataFrame
    
    Returns:
        Tuple of (encoded DataFrame, encoding mappings dict)
    """
    # Columns to label encode
    label_cols = ['job', 'merchant', 'city', 'profile', 'zip']
    available_cols = [col for col in label_cols if col in df.columns]
    
    if not available_cols:
        print("⚠️  No columns to label encode!")
        return df, {}
    
    print(f"  Label encoding: {available_cols}")
    
    # Store mappings for documentation
    mappings = {}
    
    # Label encode each column
    for col in available_cols:
        le = LabelEncoder()
        
        # Handle NaN values
        df[col] = df[col].fillna('unknown')
        
        # Fit and transform
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        
        # Store mapping
        mapping_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        mappings[col] = {
            'encoding_type': 'label',
            'num_unique_values': len(le.classes_),
            'mapping': mapping_dict
        }
        
        # Drop original column
        df = df.drop(columns=[col])
        
        print(f"    {col}: {len(le.classes_)} unique values → {col}_encoded (0-{len(le.classes_)-1})")
    
    return df, mappings


def run_chunked(input_path: Path, output_path: Path, mappings_path: Path, chunksize: int = 1_000_000):
    """
    Label encode large CSV files in chunks.
    
    Uses two-pass approach:
    1. First pass: Collect all unique values and create mappings
    2. Second pass: Apply consistent label encoding across all chunks
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        mappings_path: Path to save encoding mappings JSON
        chunksize: Number of rows to process at a time
    """
    start = time.time()
    
    print(f"🏷️  Starting label encoding (chunked)")
    print(f"   Input: {input_path}")
    print(f"   Chunk size: {chunksize:,} rows")
    print()
    
    # Columns to label encode
    label_cols = ['job', 'merchant', 'city', 'profile', 'zip']
    
    # PASS 1: Collect all unique values for each column
    print("📊 Pass 1: Collecting unique values...")
    unique_values = {col: set() for col in label_cols}
    
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        for col in label_cols:
            if col in chunk.columns:
                # Handle NaN
                chunk[col] = chunk[col].fillna('unknown')
                unique_values[col].update(chunk[col].unique())
    
    # Create label encoders and mappings
    print()
    encoders = {}
    mappings = {}
    
    for col in label_cols:
        if unique_values[col]:
            # Sort for consistency
            sorted_values = sorted(list(unique_values[col]))
            
            # Create mapping (sorted order = label order)
            mapping_dict = {val: idx for idx, val in enumerate(sorted_values)}
            
            encoders[col] = mapping_dict
            mappings[col] = {
                'encoding_type': 'label',
                'num_unique_values': len(sorted_values),
                'mapping': mapping_dict,
                'sample_values': sorted_values[:10]  # First 10 for reference
            }
            
            print(f"  {col}: {len(sorted_values)} unique values → {col}_encoded (0-{len(sorted_values)-1})")
    
    print()
    
    # Save mappings (convert numpy types to native Python types for JSON)
    mappings_path.parent.mkdir(parents=True, exist_ok=True)
    mappings_serializable = {}
    for col, mapping_info in mappings.items():
        # Extract the actual mapping dict and convert to JSON-compatible format
        actual_mapping = mapping_info['mapping']
        mappings_serializable[col] = {
            'encoding_type': mapping_info['encoding_type'],
            'num_unique_values': mapping_info['num_unique_values'],
            'mapping': {str(k): int(v) for k, v in actual_mapping.items()},
            'sample_values': [str(v) for v in mapping_info['sample_values']]
        }
    
    with open(mappings_path, 'w') as f:
        json.dump(mappings_serializable, f, indent=2)
    print(f"💾 Encoding mappings saved: {mappings_path}")
    print()
    
    # PASS 2: Apply label encoding consistently
    print("🔍 Pass 2: Applying label encoding...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use temporary file to avoid reading/writing same file
    temp_output = output_path.parent / f"{output_path.stem}_temp.csv"
    
    first_chunk = True
    total_rows = 0
    chunk_num = 0
    
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        chunk_num += 1
        rows_in_chunk = len(chunk)
        
        # Label encode each column using pre-computed mappings
        for col in label_cols:
            if col in chunk.columns and col in encoders:
                # Handle NaN
                chunk[col] = chunk[col].fillna('unknown')
                
                # Apply mapping (unknown values → -1)
                chunk[f'{col}_encoded'] = chunk[col].map(encoders[col]).fillna(-1).astype(int)
                
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
    
    print(f"\n✅ Label encoding complete!")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Output: {output_path}")
    print(f"⏱️  Runtime: {end - start:.1f} seconds ({(end - start)/60:.1f} minutes)\n")


# Standalone execution
if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / "data"
    reports_dir = Path(__file__).resolve().parents[2] / "reports"
    
    input_file = data_dir / "encoded" / "01_one_hot_encoded.csv"
    output_file = data_dir / "encoded" / "02_final_encoded.csv"
    mappings_file = reports_dir / "encoding_mappings_label.json"
    
    print("🏷️  LABEL ENCODING")
    print("=" * 80)
    print()
    
    # Use chunked processing
    run_chunked(input_file, output_file, mappings_file, chunksize=1_000_000)
    
    print("✅ Label encoding complete!")

