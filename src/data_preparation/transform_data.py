# src/data_preparation/transform_data.py

import pandas as pd
import time
from pathlib import Path


def run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering: extract date/time features and validate coordinates.
    
    Creates new features from transaction date/time:
    - Year, month, day from trans_date
    - Weekday name from trans_date
    - Hour from trans_time
    
    Also filters out rows with invalid GPS coordinates (sanity check).
    
    Args:
        df: Input DataFrame with trans_date, trans_time, and coordinate columns
    
    Returns:
        DataFrame with new time-based features and validated coordinates
    """
    # --- Extract date features ---
    if 'trans_date' in df.columns:
        # Convert to datetime (errors='coerce' handles any malformed dates gracefully)
        df['trans_date'] = pd.to_datetime(df['trans_date'], errors='coerce')
        
        # Extract components
        df['trans_year'] = df['trans_date'].dt.year
        df['trans_month'] = df['trans_date'].dt.month
        df['trans_day'] = df['trans_date'].dt.day
        df['trans_weekday'] = df['trans_date'].dt.day_name()
    
    # --- Extract time features ---
    if 'trans_time' in df.columns:
        # Parse time and extract hour
        df['trans_hour'] = pd.to_datetime(df['trans_time'], format='%H:%M:%S', errors='coerce').dt.hour
    
    # --- Validate GPS coordinates (sanity check) ---
    coord_pairs = [('lat', 'long'), ('merch_lat', 'merch_long')]
    
    for lat_col, lon_col in coord_pairs:
        if lat_col in df.columns and lon_col in df.columns:
            # Valid ranges: latitude [-90, 90], longitude [-180, 180]
            valid_mask = (
                df[lat_col].between(-90, 90) & 
                df[lon_col].between(-180, 180)
            )
            df = df[valid_mask]
    
    return df


def run_chunked(input_path: Path, output_path: Path, chunksize: int = 1_000_000):
    """
    Process large CSV files in chunks for feature engineering.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        chunksize: Number of rows to process at a time (default: 1 million)
    """
    start = time.time()
    
    print(f"⚙️  Starting chunked feature engineering")
    print(f"   Input: {input_path}")
    print(f"   Chunk size: {chunksize:,} rows")
    print()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process in chunks
    first_chunk = True
    total_rows_in = 0
    total_rows_out = 0
    chunk_num = 0
    
    # Track statistics
    null_dates_total = 0
    null_times_total = 0
    invalid_coords_total = 0
    
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        chunk_num += 1
        rows_before = len(chunk)
        total_rows_in += rows_before
        
        # Count nulls before transformation
        if 'trans_date' in chunk.columns:
            null_dates_total += chunk['trans_date'].isna().sum()
        if 'trans_time' in chunk.columns:
            null_times_total += chunk['trans_time'].isna().sum()
        
        # Transform the chunk
        chunk = run(chunk)
        
        rows_after = len(chunk)
        total_rows_out += rows_after
        invalid_coords_chunk = rows_before - rows_after
        invalid_coords_total += invalid_coords_chunk
        
        # Progress update
        if chunk_num % 5 == 0:
            print(f"  Chunk {chunk_num}: {rows_before:,} → {rows_after:,} rows (removed {invalid_coords_chunk:,})")
        
        # Write to CSV (append mode after first chunk)
        if first_chunk:
            chunk.to_csv(output_path, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(output_path, index=False, mode='a', header=False)
    
    end = time.time()
    
    # Final summary
    print(f"\n✅ Feature engineering complete!")
    print(f"   Input rows: {total_rows_in:,}")
    print(f"   Output rows: {total_rows_out:,}")
    
    if invalid_coords_total > 0:
        print(f"   📉 Removed {invalid_coords_total:,} rows with invalid coordinates ({invalid_coords_total/total_rows_in*100:.2f}%)")
    
    if null_dates_total > 0:
        print(f"   ⚠️  {null_dates_total:,} dates could not be parsed")
    
    if null_times_total > 0:
        print(f"   ⚠️  {null_times_total:,} times could not be parsed")
    
    print(f"\n   Output: {output_path}")
    print(f"⏱️  Runtime: {end - start:.1f} seconds ({(end - start)/60:.1f} minutes)\n")


# Standalone execution (for testing/debugging)
if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / "data"
    input_path = data_dir / "processed" / "02_reduced_categories.csv"
    output_path = data_dir / "processed" / "03_transformed.csv"
    
    # Use chunked processing for large files
    run_chunked(input_path, output_path, chunksize=1_000_000)
