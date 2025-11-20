# src/feature_engineering/04_interaction_features.py

"""
Interaction feature engineering for fraud detection.

Creates interaction features that combine multiple signals:
- amt_per_age: Amount divided by age (young + high spending = suspicious)
- night_x_high_value: Night transaction × high value (red flag!)
- weekend_x_high_value: Weekend × high value
- outlier_x_night: IQR outlier × night (double suspicious!)
- high_value_x_business_hours: High value during business hours (less suspicious)

Why these features matter:
- Interactions capture complex patterns that single features miss
- Multiple risk factors together = stronger fraud signal
- E.g., high amount at night is MORE suspicious than either alone
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features for fraud detection.
    
    Features created:
    1. amt_per_age: amt / age (spending relative to age)
    2. night_x_high_value: is_night × is_high_value
    3. weekend_x_high_value: is_weekend × is_high_value
    4. outlier_x_night: is_outlier_iqr × is_night
    5. high_value_x_business_hours: is_high_value × is_business_hours
    
    Args:
        df: DataFrame with required feature columns
        
    Returns:
        DataFrame with new interaction features added
        
    Note:
        - Missing features will be skipped with warnings
        - Division by zero handled (age=0 → amt_per_age=0)
    """
    print("   Creating interaction features...")
    
    features_created = []
    
    # 1. amt_per_age: Amount relative to age
    if 'amt' in df.columns and 'age' in df.columns:
        # Handle division by zero
        df['amt_per_age'] = df['amt'] / df['age'].replace(0, np.nan)
        df['amt_per_age'] = df['amt_per_age'].fillna(0)
        features_created.append('amt_per_age')
    else:
        print("   ⚠️  Warning: Cannot create amt_per_age (missing amt or age)")
    
    # 2. night_x_high_value: Night + expensive = very suspicious
    if 'is_night' in df.columns and 'is_high_value' in df.columns:
        df['night_x_high_value'] = (df['is_night'] * df['is_high_value']).astype(int)
        features_created.append('night_x_high_value')
    else:
        print("   ⚠️  Warning: Cannot create night_x_high_value")
    
    # 3. weekend_x_high_value: Weekend + expensive
    if 'is_weekend' in df.columns and 'is_high_value' in df.columns:
        df['weekend_x_high_value'] = (df['is_weekend'] * df['is_high_value']).astype(int)
        features_created.append('weekend_x_high_value')
    else:
        print("   ⚠️  Warning: Cannot create weekend_x_high_value")
    
    # 4. outlier_x_night: IQR outlier at night = double red flag!
    if 'is_outlier_iqr' in df.columns and 'is_night' in df.columns:
        df['outlier_x_night'] = (df['is_outlier_iqr'] * df['is_night']).astype(int)
        features_created.append('outlier_x_night')
    else:
        print("   ⚠️  Warning: Cannot create outlier_x_night")
    
    # 5. high_value_x_business_hours: High value during work hours (less suspicious)
    if 'is_high_value' in df.columns and 'is_business_hours' in df.columns:
        df['high_value_x_business_hours'] = (df['is_high_value'] * df['is_business_hours']).astype(int)
        features_created.append('high_value_x_business_hours')
    else:
        print("   ⚠️  Warning: Cannot create high_value_x_business_hours")
    
    print(f"   ✅ Created {len(features_created)} interaction features: {', '.join(features_created)}")
    
    return df


def run_chunked(input_path: Path, output_path: Path, chunksize: int = 1_000_000):
    """
    Process large CSV files in chunks to create interaction features.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file (FINAL features!)
        chunksize: Number of rows to process at a time (default: 1 million)
    """
    start = time.time()
    
    print(f"🤝 Creating interaction features")
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
        
        # Create interaction features
        chunk = create_interaction_features(chunk)
        
        # Write to CSV (append mode after first chunk)
        if first_chunk:
            chunk.to_csv(output_path, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(output_path, index=False, mode='a', header=False)
    
    end = time.time()
    
    print(f"\n✅ Interaction features complete!")
    print(f"   Total rows processed: {total_rows:,}")
    print(f"   Output: {output_path}")
    print(f"⏱️  Runtime: {end - start:.1f} seconds ({(end - start)/60:.1f} minutes)\n")
    
    print("=" * 80)
    print("🎉 FEATURE ENGINEERING COMPLETE!")
    print("=" * 80)
    print()
    print("Final dataset ready for model training:")
    print(f"   📁 {output_path}")
    print()
    print("Next steps:")
    print("   1. ✅ Train baseline models (Random Forest, XGBoost, LightGBM)")
    print("   2. ✅ Evaluate feature importance")
    print("   3. ✅ Select best features")
    print("   4. ✅ Optimize hyperparameters")
    print()
    print("Great work! 🚀")


# Standalone execution
if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / "data"
    
    input_file = data_dir / "features" / "02_amount_features.csv"
    output_file = data_dir / "features" / "03_final_features.csv"
    
    print("🤝 INTERACTION FEATURES")
    print("=" * 80)
    print()
    
    # Run chunked processing
    run_chunked(input_file, output_file, chunksize=1_000_000)
    
    print("✅ All feature engineering complete!")

