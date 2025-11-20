# src/feature_engineering/02_time_features.py

"""
Time-based feature engineering for fraud detection.

Creates temporal features:
- is_night: Binary flag for nighttime transactions (11 PM - 6 AM)
- is_weekend: Binary flag for weekend transactions (Sat/Sun)
- is_business_hours: Binary flag for business hours (9 AM - 5 PM weekdays)
- hour_sin, hour_cos: Cyclical encoding of hour (captures 23h ≈ 0h)
- day_of_month: Day of month (1-31) - spending patterns

Why these features matter:
- Fraudulent transactions are more common at night
- Weekend vs. weekday spending patterns differ
- Cyclical encoding preserves hour proximity (hour 23 is close to hour 0)
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features for fraud detection.
    
    Features created:
    1. is_night: 1 if hour between 23-6, 0 otherwise
    2. is_weekend: 1 if Saturday or Sunday, 0 otherwise
    3. is_business_hours: 1 if 9-17 on weekday, 0 otherwise
    4. hour_sin: sin(2π * hour / 24) - cyclical encoding
    5. hour_cos: cos(2π * hour / 24) - cyclical encoding
    6. day_of_month: Extract day from trans_day (1-31)
    
    Args:
        df: DataFrame with trans_hour, trans_weekday columns
        
    Returns:
        DataFrame with new time features added
    """
    print("   Creating time features...")
    
    # Check required columns
    required_cols = ['trans_hour']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"   ⚠️  Warning: Missing columns: {missing_cols}")
        return df
    
    # 1. is_night: Nighttime transactions (11 PM - 6 AM)
    df['is_night'] = ((df['trans_hour'] >= 23) | (df['trans_hour'] <= 6)).astype(int)
    
    # 2. is_weekend: Weekend transactions
    # Check if we have weekday columns (one-hot encoded)
    if 'trans_weekday_Saturday' in df.columns and 'trans_weekday_Sunday' in df.columns:
        df['is_weekend'] = (df['trans_weekday_Saturday'] | df['trans_weekday_Sunday']).astype(int)
    else:
        print("   ⚠️  Warning: Weekday columns not found, skipping is_weekend")
    
    # 3. is_business_hours: 9 AM - 5 PM on weekdays
    if 'trans_weekday_Saturday' in df.columns and 'trans_weekday_Sunday' in df.columns:
        is_weekday = ~(df['trans_weekday_Saturday'] | df['trans_weekday_Sunday'])
        is_work_hours = (df['trans_hour'] >= 9) & (df['trans_hour'] <= 17)
        df['is_business_hours'] = (is_weekday & is_work_hours).astype(int)
    else:
        print("   ⚠️  Warning: Cannot create is_business_hours without weekday info")
    
    # 4. Cyclical encoding of hour (captures that 23h is close to 0h)
    df['hour_sin'] = np.sin(2 * np.pi * df['trans_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['trans_hour'] / 24)
    
    # 5. Day of month
    if 'trans_day' in df.columns:
        df['day_of_month'] = df['trans_day']
    
    # Count features created
    new_features = ['is_night', 'hour_sin', 'hour_cos']
    if 'is_weekend' in df.columns:
        new_features.append('is_weekend')
    if 'is_business_hours' in df.columns:
        new_features.append('is_business_hours')
    if 'day_of_month' in df.columns:
        new_features.append('day_of_month')
    
    print(f"   ✅ Created {len(new_features)} time features: {', '.join(new_features)}")
    
    return df


def run_chunked(input_path: Path, output_path: Path, chunksize: int = 1_000_000):
    """
    Process large CSV files in chunks to create time features.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        chunksize: Number of rows to process at a time (default: 1 million)
    """
    start = time.time()
    
    print(f"⏰ Creating time features")
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
        
        # Create time features
        chunk = create_time_features(chunk)
        
        # Write to CSV (append mode after first chunk)
        if first_chunk:
            chunk.to_csv(output_path, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(output_path, index=False, mode='a', header=False)
    
    end = time.time()
    
    print(f"\n✅ Time features complete!")
    print(f"   Total rows processed: {total_rows:,}")
    print(f"   Output: {output_path}")
    print(f"⏱️  Runtime: {end - start:.1f} seconds ({(end - start)/60:.1f} minutes)\n")


# Standalone execution
if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / "data"
    
    input_file = data_dir / "encoded" / "02_final_encoded.csv"
    output_file = data_dir / "features" / "01_time_features.csv"
    
    print("⏰ TIME FEATURES")
    print("=" * 80)
    print()
    
    # Run chunked processing
    run_chunked(input_file, output_file, chunksize=1_000_000)
    
    print("✅ Time features complete!")

