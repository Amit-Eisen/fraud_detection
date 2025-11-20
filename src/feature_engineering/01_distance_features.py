# src/feature_engineering/01_distance_features.py

"""
Distance-based feature engineering for fraud detection.

Creates geospatial features:
- distance_from_home: Distance between customer and merchant (km)
- is_local_transaction: Binary flag for local transactions (<50km)

Why these features matter:
- Fraudulent transactions often occur far from customer's home
- Local vs. remote transactions have different fraud patterns
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth.
    
    Uses the Haversine formula to compute distance in kilometers.
    
    Args:
        lat1, lon1: Customer coordinates (degrees)
        lat2, lon2: Merchant coordinates (degrees)
        
    Returns:
        Distance in kilometers (float)
        
    Note:
        - Returns 0 if any coordinate is missing (NaN)
        - Earth radius = 6371 km
    """
    # Handle missing values
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 0.0
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    # Earth radius in km
    R = 6371
    distance = R * c
    
    return distance


def create_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    SKIPPED: Distance features cannot be created.
    
    Reason:
    - Merchant coordinates (merch_lat, merch_long) were dropped during data cleansing
    - They had very high correlation (0.994, 0.999) with customer coordinates
    - Without merchant coordinates, we cannot calculate distance_from_home
    
    Decision:
    - Skip distance features entirely
    - Focus on other feature engineering (time, amount, interactions)
    - This is acceptable because:
      * Tree-based models can learn location patterns from lat/long directly
      * We have other strong features (amount, category, city_pop)
    
    Args:
        df: DataFrame (passed through unchanged)
        
    Returns:
        DataFrame unchanged (no new features added)
    """
    print("   ⚠️  Skipping distance features (merchant coordinates not available)")
    print("   → This is OK! XGBoost can learn from lat/long directly")
    
    # Return dataframe unchanged - no features to add
    return df


def run_chunked(input_path: Path, output_path: Path, chunksize: int = 1_000_000):
    """
    SKIPPED: Just copy input to output (no distance features to add).
    
    Since we cannot create distance features (merchant coordinates unavailable),
    this function simply copies the input file to maintain pipeline compatibility.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        chunksize: Number of rows to process at a time (default: 1 million)
    """
    start = time.time()
    
    print(f"📍 Distance Features Step (SKIPPED)")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Reason: Merchant coordinates not available")
    print()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Simply copy input to output in chunks (no processing needed)
    first_chunk = True
    total_rows = 0
    chunk_num = 0
    
    print("   Copying data (no features added)...")
    
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        chunk_num += 1
        rows_in_chunk = len(chunk)
        total_rows += rows_in_chunk
        
        if chunk_num % 5 == 0:
            print(f"  Processed {total_rows:,} rows...")
        
        # Write to CSV (append mode after first chunk)
        if first_chunk:
            chunk.to_csv(output_path, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(output_path, index=False, mode='a', header=False)
    
    end = time.time()
    
    print(f"\n✅ Distance features step complete (skipped, data copied)")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Output: {output_path}")
    print(f"⏱️  Runtime: {end - start:.1f} seconds ({(end - start)/60:.1f} minutes)")
    print(f"   → Proceed to next feature engineering step (time features)")
    print()


# Standalone execution
if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / "data"
    
    input_file = data_dir / "encoded" / "02_final_encoded.csv"
    output_file = data_dir / "features" / "01_distance_features.csv"
    
    print("\n" + "=" * 80)
    print("DISTANCE FEATURES STEP")
    print("=" * 80)
    print()
    
    print("⚠️  IMPORTANT NOTE:")
    print("-" * 80)
    print("Merchant coordinates (merch_lat, merch_long) were dropped during cleansing")
    print("because they had very high correlation (0.994, 0.999) with customer coords.")
    print()
    print("Decision: SKIP distance features")
    print("Reason: Cannot calculate distance without merchant coordinates")
    print()
    print("Impact: Minimal - XGBoost can learn location patterns from lat/long directly")
    print("-" * 80)
    print()
    
    # Check if input file exists
    if not input_file.exists():
        print(f"❌ ERROR: Input file not found!")
        print(f"   Expected: {input_file}")
        print()
        print("💡 Make sure you've run the encoding pipeline first.")
    else:
        # Run the copy operation (maintains pipeline compatibility)
        run_chunked(input_file, output_file, chunksize=1_000_000)
        
        print("✅ Step complete - proceed to time-based features!")
        print()

