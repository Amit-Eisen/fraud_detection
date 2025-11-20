# src/feature_engineering/03_amount_features.py

"""
Amount-based feature engineering for fraud detection.

Creates amount-related features:
- amt_log: Log transformation (reduces skewness)
- amt_zscore: Z-score normalization
- is_high_value: Flag for amounts > 95th percentile
- is_micro_transaction: Flag for amounts < $5
- is_outlier_iqr: Flag for amounts outside IQR bounds (1.5×IQR)

Why these features matter:
- High-value transactions are riskier
- Micro-transactions may be card testing
- Log transformation helps with skewed distributions
- Outliers (from IQR analysis) showed high fraud correlation
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path


def create_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create amount-based features for fraud detection.
    
    Features created:
    1. amt_log: log(amt + 1) - reduces skewness
    2. amt_zscore: (amt - mean) / std - standardized amount
    3. is_high_value: 1 if amt > 95th percentile, 0 otherwise
    4. is_micro_transaction: 1 if amt < $5, 0 otherwise
    5. is_outlier_iqr: 1 if amt outside IQR×1.5 bounds, 0 otherwise
    
    Args:
        df: DataFrame with 'amt' column
        
    Returns:
        DataFrame with new amount features added
        
    Note:
        - IQR bounds calculated per chunk (approximate)
        - For exact IQR, would need full dataset pass
    """
    print("   Creating amount features...")
    
    # Check required columns
    if 'amt' not in df.columns:
        print("   ⚠️  Warning: 'amt' column not found!")
        return df
    
    # 1. Log transformation (reduces skewness)
    df['amt_log'] = np.log1p(df['amt'])  # log(amt + 1) to handle amt=0
    
    # 2. Z-score normalization (per chunk - approximate)
    amt_mean = df['amt'].mean()
    amt_std = df['amt'].std()
    if amt_std > 0:
        df['amt_zscore'] = (df['amt'] - amt_mean) / amt_std
    else:
        df['amt_zscore'] = 0
    
    # 3. High-value flag (95th percentile per chunk)
    threshold_95 = df['amt'].quantile(0.95)
    df['is_high_value'] = (df['amt'] > threshold_95).astype(int)
    
    # 4. Micro-transaction flag (< $5)
    df['is_micro_transaction'] = (df['amt'] < 5).astype(int)
    
    # 5. IQR outlier flag (from our analysis!)
    # This is the insight from your IQR analysis - great idea!
    Q1 = df['amt'].quantile(0.25)
    Q3 = df['amt'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df['is_outlier_iqr'] = ((df['amt'] < lower_bound) | (df['amt'] > upper_bound)).astype(int)
    
    # Count outliers for reporting
    outlier_count = df['is_outlier_iqr'].sum()
    outlier_pct = (outlier_count / len(df)) * 100
    
    print(f"   ✅ Created 5 amount features")
    print(f"      IQR outliers in chunk: {outlier_count:,} ({outlier_pct:.2f}%)")
    
    return df


def run_chunked(input_path: Path, output_path: Path, chunksize: int = 1_000_000):
    """
    Process large CSV files in chunks to create amount features.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        chunksize: Number of rows to process at a time (default: 1 million)
    """
    start = time.time()
    
    print(f"💰 Creating amount features")
    print(f"   Input: {input_path}")
    print(f"   Chunk size: {chunksize:,} rows")
    print()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process in chunks
    first_chunk = True
    total_rows = 0
    total_outliers = 0
    chunk_num = 0
    
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        chunk_num += 1
        rows_in_chunk = len(chunk)
        total_rows += rows_in_chunk
        
        print(f"  Processing chunk {chunk_num}: {rows_in_chunk:,} rows (total: {total_rows:,})")
        
        # Create amount features
        chunk = create_amount_features(chunk)
        
        # Track outliers
        total_outliers += chunk['is_outlier_iqr'].sum()
        
        # Write to CSV (append mode after first chunk)
        if first_chunk:
            chunk.to_csv(output_path, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(output_path, index=False, mode='a', header=False)
    
    end = time.time()
    
    outlier_pct = (total_outliers / total_rows) * 100
    
    print(f"\n✅ Amount features complete!")
    print(f"   Total rows processed: {total_rows:,}")
    print(f"   Total IQR outliers: {total_outliers:,} ({outlier_pct:.2f}%)")
    print(f"   Output: {output_path}")
    print(f"⏱️  Runtime: {end - start:.1f} seconds ({(end - start)/60:.1f} minutes)\n")


# Standalone execution
if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / "data"
    
    input_file = data_dir / "features" / "01_time_features.csv"
    output_file = data_dir / "features" / "02_amount_features.csv"
    
    print("💰 AMOUNT FEATURES")
    print("=" * 80)
    print()
    
    # Run chunked processing
    run_chunked(input_file, output_file, chunksize=1_000_000)
    
    print("✅ Amount features complete!")

