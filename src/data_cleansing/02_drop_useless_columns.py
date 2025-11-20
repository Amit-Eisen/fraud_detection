# src/data_cleansing/02_drop_useless_columns.py

import pandas as pd
import time
from pathlib import Path


def run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop useless columns and create age feature.
    
    Steps:
    1. Create 'age' from 'dob' (before dropping)
    2. Drop multicollinear features (redundant)
    3. Drop PII columns (privacy, not predictive)
    4. Drop already-extracted temporal columns
    5. Drop index column
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with useless columns removed and age feature added
    """
    initial_cols = len(df.columns)
    
    # Step 1: Create age feature from dob (before dropping)
    if 'dob' in df.columns and 'trans_date' in df.columns:
        print("   Creating 'age' feature from 'dob'...")
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['trans_date'] = pd.to_datetime(df['trans_date'], errors='coerce')
        
        # Calculate age at time of transaction
        df['age'] = ((df['trans_date'] - df['dob']).dt.days / 365.25).round().astype('Int64')
        
        # Handle any invalid ages (negative or > 120)
        invalid_ages = (df['age'] < 0) | (df['age'] > 120)
        if invalid_ages.sum() > 0:
            print(f"   ⚠️  Found {invalid_ages.sum():,} invalid ages, setting to median")
            median_age = df.loc[~invalid_ages, 'age'].median()
            df.loc[invalid_ages, 'age'] = median_age
    
    # Step 2: Define columns to drop
    cols_to_drop = [
        # Multicollinear features
        'merch_lat',      # corr 0.994 with lat
        'merch_long',     # corr 0.999 with long
        'unix_time',      # corr 0.867 with trans_year
        
        # PII (not predictive)
        'ssn',            # corr 0.0003
        'cc_num',         # corr 0.0003
        'acct_num',       # corr 0.0009
        'first',          # personal name
        'last',           # personal name
        'street',         # personal address
        'trans_num',      # just transaction ID
        'dob',            # extracted to age
        
        # Already extracted
        'trans_date',     # extracted to year/month/day/weekday
        'trans_time',     # extracted to hour
        
        # Index column
        'Unnamed: 0',     # row numbers
    ]
    
    # Only drop columns that exist
    cols_to_drop_existing = [col for col in cols_to_drop if col in df.columns]
    
    if cols_to_drop_existing:
        df = df.drop(columns=cols_to_drop_existing)
        print(f"   Dropped {len(cols_to_drop_existing)} columns")
    
    final_cols = len(df.columns)
    
    print(f"   Columns: {initial_cols} → {final_cols}")
    
    return df


def run_chunked(input_path: Path, output_path: Path, chunksize: int = 1_000_000):
    """
    Process large CSV files in chunks to drop useless columns.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        chunksize: Number of rows to process at a time (default: 1 million)
    """
    start = time.time()
    
    print(f"🗑️  Dropping useless columns and creating age feature")
    print(f"   Input: {input_path}")
    print(f"   Chunk size: {chunksize:,} rows")
    print()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process in chunks
    first_chunk = True
    total_rows = 0
    chunk_num = 0
    
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        chunk_num += 1
        rows_in_chunk = len(chunk)
        total_rows += rows_in_chunk
        
        print(f"  Processing chunk {chunk_num}: {rows_in_chunk:,} rows (total: {total_rows:,})")
        
        # Process the chunk
        chunk = run(chunk)
        
        # Write to CSV (append mode after first chunk)
        if first_chunk:
            chunk.to_csv(output_path, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(output_path, index=False, mode='a', header=False)
    
    end = time.time()
    
    print(f"\n✅ Column cleanup complete!")
    print(f"   Total rows processed: {total_rows:,}")
    print(f"   Output: {output_path}")
    print(f"⏱️  Runtime: {end - start:.1f} seconds ({(end - start)/60:.1f} minutes)\n")


# Standalone execution
if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / "data"
    input_path = data_dir / "cleansed" / "01_no_duplicates.csv"
    output_path = data_dir / "cleansed" / "02_clean_columns.csv"
    
    # Use chunked processing
    run_chunked(input_path, output_path, chunksize=1_000_000)

