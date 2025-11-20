#!/usr/bin/env python3
"""
Create Stratified Sample from Parquet
======================================

Creates a stratified sample of the dataset and saves it as a new Parquet file.
This is done ONCE to avoid memory issues during model training.

Why?
- Loading 35M rows and then sampling causes memory crashes
- Better to create the sample once and reuse it
- Maintains exact fraud ratio (stratified sampling)

Author: Amit Eisen
Date: 2025-11-07
"""

import sys
from pathlib import Path
import time
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def create_stratified_sample(
    input_parquet: Path,
    output_parquet: Path,
    sample_size: int = 20_000_000,
    chunksize: int = 1_000_000
):
    """
    Create a stratified sample from a large Parquet file.
    
    This function uses a memory-efficient two-pass approach:
    1. First pass: Scan to count fraud/legit cases
    2. Second pass: Sample proportionally from each chunk
    
    Args:
        input_parquet: Path to input Parquet file
        output_parquet: Path to output Parquet file
        sample_size: Total number of rows to sample
        chunksize: Number of rows to process at a time
    """
    print("=" * 70)
    print("CREATING STRATIFIED SAMPLE")
    print("=" * 70)
    print(f"📂 Input:  {input_parquet}")
    print(f"📂 Output: {output_parquet}")
    print(f"🎯 Sample size: {sample_size:,} rows")
    print()
    
    start_time = time.time()
    
    # Step 1: Count total fraud/legit cases
    print("Step 1/2: Counting fraud/legit cases...")
    
    parquet_file = pq.ParquetFile(input_parquet)
    total_fraud = 0
    total_legit = 0
    total_rows = 0
    
    for batch in parquet_file.iter_batches(batch_size=chunksize):
        batch_df = batch.to_pandas()
        total_fraud += (batch_df['is_fraud'] == 1).sum()
        total_legit += (batch_df['is_fraud'] == 0).sum()
        total_rows += len(batch_df)
    
    fraud_ratio = total_fraud / total_rows
    
    print(f"✓ Total rows: {total_rows:,}")
    print(f"  Fraud: {total_fraud:,} ({fraud_ratio*100:.2f}%)")
    print(f"  Legit: {total_legit:,} ({(1-fraud_ratio)*100:.2f}%)")
    print()
    
    # Calculate target counts
    target_fraud = int(sample_size * fraud_ratio)
    target_legit = sample_size - target_fraud
    
    print(f"🎯 Target sample:")
    print(f"  Fraud: {target_fraud:,}")
    print(f"  Legit: {target_legit:,}")
    print(f"  Total: {sample_size:,}")
    print(f"  Ratio: 1:{int(target_legit/target_fraud)}")
    print()
    
    # Calculate sampling rates
    fraud_sample_rate = target_fraud / total_fraud
    legit_sample_rate = target_legit / total_legit
    
    print(f"📊 Sampling rates:")
    print(f"  Fraud: {fraud_sample_rate*100:.2f}% of all fraud cases")
    print(f"  Legit: {legit_sample_rate*100:.2f}% of all legit cases")
    print()
    
    # Step 2: Sample from each chunk
    print("Step 2/2: Sampling data...")
    
    sampled_chunks = []
    sampled_fraud = 0
    sampled_legit = 0
    
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=chunksize)):
        batch_df = batch.to_pandas()
        
        # Separate fraud and legit
        fraud_mask = batch_df['is_fraud'] == 1
        fraud_df = batch_df[fraud_mask]
        legit_df = batch_df[~fraud_mask]
        
        # Sample from each
        if len(fraud_df) > 0 and sampled_fraud < target_fraud:
            # How many fraud to sample from this chunk
            remaining_fraud = target_fraud - sampled_fraud
            n_fraud = min(int(len(fraud_df) * fraud_sample_rate) + 1, remaining_fraud, len(fraud_df))
            
            if n_fraud > 0:
                fraud_sample = fraud_df.sample(n=n_fraud, random_state=42+i)
                sampled_chunks.append(fraud_sample)
                sampled_fraud += n_fraud
        
        if len(legit_df) > 0 and sampled_legit < target_legit:
            # How many legit to sample from this chunk
            remaining_legit = target_legit - sampled_legit
            n_legit = min(int(len(legit_df) * legit_sample_rate) + 1, remaining_legit, len(legit_df))
            
            if n_legit > 0:
                legit_sample = legit_df.sample(n=n_legit, random_state=42+i)
                sampled_chunks.append(legit_sample)
                sampled_legit += n_legit
        
        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Processed {(i+1)*chunksize:,} rows... "
                  f"(sampled: {sampled_fraud:,} fraud + {sampled_legit:,} legit)")
        
        # Stop if we have enough
        if sampled_fraud >= target_fraud and sampled_legit >= target_legit:
            break
    
    print()
    print(f"✓ Sampled {sampled_fraud:,} fraud + {sampled_legit:,} legit = {sampled_fraud + sampled_legit:,} total")
    print()
    
    # Combine and shuffle
    print("Combining and shuffling...")
    df_sample = pd.concat(sampled_chunks, ignore_index=True)
    df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Verify
    final_fraud = (df_sample['is_fraud'] == 1).sum()
    final_legit = (df_sample['is_fraud'] == 0).sum()
    final_ratio = final_fraud / len(df_sample)
    
    print(f"✓ Final sample:")
    print(f"  Rows: {len(df_sample):,}")
    print(f"  Fraud: {final_fraud:,} ({final_ratio*100:.2f}%)")
    print(f"  Legit: {final_legit:,} ({(1-final_ratio)*100:.2f}%)")
    print(f"  Ratio: 1:{int(final_legit/final_fraud)}")
    print()
    
    # Save
    print("Saving to Parquet...")
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_parquet(output_parquet, compression='snappy', index=False)
    
    file_size_mb = output_parquet.stat().st_size / (1024 * 1024)
    
    elapsed = time.time() - start_time
    
    print()
    print("=" * 70)
    print("✅ SAMPLE CREATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"📁 Output file: {output_parquet}")
    print(f"📦 File size: {file_size_mb:.1f} MB")
    print(f"⏱️  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print()


def main():
    """Main execution function."""
    
    # Configuration
    SAMPLE_SIZE = 20_000_000  # 20M rows
    
    # Paths
    project_root = Path(__file__).resolve().parents[2]
    input_parquet = project_root / 'data' / 'features' / 'selected_features_top36.parquet'
    output_parquet = project_root / 'data' / 'features' / 'selected_features_20M.parquet'
    
    # Check input exists
    if not input_parquet.exists():
        print(f"❌ ERROR: Input file not found!")
        print(f"   Expected: {input_parquet}")
        return
    
    # Create sample
    create_stratified_sample(
        input_parquet=input_parquet,
        output_parquet=output_parquet,
        sample_size=SAMPLE_SIZE
    )
    
    print("🎉 Done! You can now use this file for model training:")
    print(f"   {output_parquet}")
    print()


if __name__ == "__main__":
    main()

