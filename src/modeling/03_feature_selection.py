#!/usr/bin/env python3
"""
Feature Selection
=================

Selects top N most important features and creates a reduced dataset.

This script:
1. Loads feature importance rankings from previous analysis
2. Selects top N features (default: 50, capturing 98%+ importance)
3. Filters the full dataset to keep only selected features
4. Saves the reduced dataset as Parquet (for efficient training)
5. Creates a report documenting the selection

Why Feature Selection?
- Reduces memory usage (50 features vs 104 = 52% reduction)
- Enables training on full 35M rows (fits in 16GB RAM)
- Removes noise from unimportant features
- Faster training and prediction

"""

import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def load_feature_importance(reports_dir: Path) -> pd.DataFrame:
    """
    Load feature importance rankings.
    
    Args:
        reports_dir: Directory containing feature importance reports
    
    Returns:
        DataFrame with feature importance rankings
    """
    print("=" * 70)
    print("LOADING FEATURE IMPORTANCE RANKINGS")
    print("=" * 70)
    print()
    
    importance_path = reports_dir / 'feature_importance_all.csv'
    
    if not importance_path.exists():
        print(f"❌ ERROR: Feature importance report not found!")
        print(f"   Expected: {importance_path}")
        print()
        print("💡 Make sure you've run:")
        print("   python src/modeling/02_feature_importance.py")
        return None
    
    df = pd.read_csv(importance_path)
    
    print(f"✓ Loaded importance rankings for {len(df)} features")
    print()
    
    return df


def select_top_features(importance_df: pd.DataFrame, n_features: int = 50) -> tuple:
    """
    Select top N most important features.
    
    Args:
        importance_df: DataFrame with feature importance
        n_features: Number of top features to select
    
    Returns:
        Tuple of (selected_features_list, selection_report_dict)
    """
    print("=" * 70)
    print(f"SELECTING TOP {n_features} FEATURES")
    print("=" * 70)
    print()
    
    # Get top N features
    top_features = importance_df.head(n_features)
    selected_features = top_features['feature'].tolist()
    
    # Calculate statistics
    total_importance = importance_df['Average'].sum()
    selected_importance = top_features['Average'].sum()
    dropped_features = len(importance_df) - n_features
    
    print(f"📊 Selection Statistics:")
    print("-" * 70)
    print(f"Total features available: {len(importance_df)}")
    print(f"Features selected: {n_features}")
    print(f"Features dropped: {dropped_features} ({dropped_features/len(importance_df)*100:.1f}%)")
    print()
    print(f"Importance captured: {selected_importance:.2f}% (of {total_importance:.2f}%)")
    print(f"Importance lost: {total_importance - selected_importance:.2f}%")
    print()
    
    # Show top 10 selected features
    print(f"🏆 Top 10 Selected Features:")
    print("-" * 70)
    for idx, row in top_features.head(10).iterrows():
        print(f"  {row['rank']:2d}. {row['feature']:<30} {row['Average']:>6.2f}%")
    print()
    
    # Show first 10 dropped features (to understand what we're losing)
    if dropped_features > 0:
        dropped = importance_df.iloc[n_features:n_features+10]
        print(f"⚠️  First 10 Dropped Features:")
        print("-" * 70)
        for idx, row in dropped.iterrows():
            print(f"  {row['rank']:2d}. {row['feature']:<30} {row['Average']:>6.2f}%")
        print()
    
    # Create selection report
    report = {
        'n_features_selected': n_features,
        'n_features_dropped': dropped_features,
        'importance_captured': selected_importance,
        'importance_lost': total_importance - selected_importance,
        'selected_features': selected_features
    }
    
    return selected_features, report


def filter_dataset(input_parquet: Path, output_parquet: Path, 
                   selected_features: list, chunksize: int = 1_000_000) -> None:
    """
    Filter dataset to keep only selected features + target.
    
    Uses PyArrow batch processing to handle large datasets efficiently.
    
    Args:
        input_parquet: Path to full feature-engineered Parquet file
        output_parquet: Path to save filtered Parquet file
        selected_features: List of feature names to keep
        chunksize: Number of rows to process per batch
    """
    print("=" * 70)
    print("FILTERING DATASET")
    print("=" * 70)
    print()
    
    print(f"📂 Input:  {input_parquet}")
    print(f"📂 Output: {output_parquet}")
    print()
    
    # Add 'is_fraud' to selected features (we need the target!)
    columns_to_keep = selected_features + ['is_fraud']
    
    print(f"🔍 Columns to keep: {len(columns_to_keep)} ({len(selected_features)} features + target)")
    print()
    
    # Open Parquet file
    parquet_file = pq.ParquetFile(input_parquet)
    total_rows = parquet_file.metadata.num_rows
    
    print(f"📊 Total rows: {total_rows:,}")
    print(f"📦 Processing in batches of {chunksize:,} rows...")
    print()
    
    # Process in batches and write to temporary Parquet files
    temp_files = []
    schema = None
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=chunksize)):
        # Convert to pandas
        batch_df = batch.to_pandas()
        
        # Filter columns
        batch_filtered = batch_df[columns_to_keep]
        
        # Convert back to PyArrow Table
        batch_table = pa.Table.from_pandas(batch_filtered)
        
        # Store schema from first batch
        if schema is None:
            schema = batch_table.schema
        
        # Write to temporary file
        temp_file = output_parquet.parent / f"_temp_batch_{batch_idx}.parquet"
        pq.write_table(batch_table, temp_file)
        temp_files.append(temp_file)
        
        # Progress update
        rows_processed = (batch_idx + 1) * chunksize
        if (batch_idx + 1) % 5 == 0 or rows_processed >= total_rows:
            elapsed = time.time() - start_time
            print(f"   Processed {min(rows_processed, total_rows):,} / {total_rows:,} rows "
                  f"({min(rows_processed/total_rows*100, 100):.1f}%) - {elapsed:.1f}s")
    
    print()
    print(f"✓ Filtered all batches in {time.time() - start_time:.1f} seconds")
    print()
    
    # Combine temporary files into single Parquet
    print("🔗 Combining batches into final Parquet file...")
    print()
    
    tables = []
    for temp_file in temp_files:
        tables.append(pq.read_table(temp_file))
    
    # Concatenate all tables
    final_table = pa.concat_tables(tables)
    
    # Write final Parquet file
    pq.write_table(final_table, output_parquet, compression='snappy')
    
    # Clean up temporary files
    for temp_file in temp_files:
        temp_file.unlink()
    
    # Get file size
    file_size_mb = output_parquet.stat().st_size / (1024 ** 2)
    
    print(f"✓ Saved filtered dataset: {output_parquet}")
    print(f"   Rows: {len(final_table):,}")
    print(f"   Columns: {len(final_table.column_names)} ({len(selected_features)} features + target)")
    print(f"   File size: {file_size_mb:.1f} MB")
    print()


def save_selection_report(report: dict, selected_features_df: pd.DataFrame, 
                          output_dir: Path) -> None:
    """
    Save feature selection report.
    
    Args:
        report: Dictionary with selection statistics
        selected_features_df: DataFrame with selected features and their importance
        output_dir: Directory to save report
    """
    print("=" * 70)
    print("SAVING SELECTION REPORT")
    print("=" * 70)
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save selected features list as CSV
    csv_path = output_dir / 'selected_features.csv'
    selected_features_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✓ Saved selected features: {csv_path}")
    
    # Save detailed text report
    report_path = output_dir / 'feature_selection_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FEATURE SELECTION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SELECTION SUMMARY:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Features selected: {report['n_features_selected']}\n")
        f.write(f"Features dropped: {report['n_features_dropped']}\n")
        f.write(f"Importance captured: {report['importance_captured']:.2f}%\n")
        f.write(f"Importance lost: {report['importance_lost']:.2f}%\n\n")
        
        f.write("SELECTED FEATURES:\n")
        f.write("-" * 70 + "\n")
        for idx, row in selected_features_df.iterrows():
            f.write(f"{row['rank']:3d}. {row['feature']:<35} {row['Average']:>6.2f}%\n")
        f.write("\n")
        
        f.write("RATIONALE:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Selected top {report['n_features_selected']} features to:\n")
        f.write(f"1. Reduce memory usage by {report['n_features_dropped']/104*100:.1f}%\n")
        f.write(f"2. Enable training on full 35M rows (fits in 16GB RAM)\n")
        f.write(f"3. Remove noise from low-importance features\n")
        f.write(f"4. Maintain {report['importance_captured']:.2f}% of predictive power\n\n")
        
        f.write("NEXT STEPS:\n")
        f.write("-" * 70 + "\n")
        f.write("1. Train XGBoost on full 35M rows with selected features\n")
        f.write("2. Compare performance to baseline (10M rows, 104 features)\n")
        f.write("3. If performance is similar/better → proceed to hyperparameter tuning\n")
        f.write("4. If performance drops → try more features (60-70) or sample more rows\n\n")
    
    print(f"✓ Saved selection report: {report_path}")
    print()


def main():
    """Main execution function."""
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print("FEATURE SELECTION")
    print("=" * 70)
    print()
    
    # Configuration
    N_FEATURES = 36  # Top 36 features capture ~95% of importance
    
    print(f"⚙️  Configuration: Selecting top {N_FEATURES} features")
    print()
    
    # Setup paths
    project_root = Path(__file__).resolve().parents[2]
    importance_dir = project_root / 'reports' / 'feature_importance'
    input_parquet = project_root / 'data' / 'features' / 'final_features.parquet'
    output_parquet = project_root / 'data' / 'features' / f'selected_features_top{N_FEATURES}.parquet'
    output_dir = project_root / 'reports' / 'feature_selection'
    
    # Check if input files exist
    if not input_parquet.exists():
        print(f"❌ ERROR: Input Parquet file not found!")
        print(f"   Expected: {input_parquet}")
        print()
        print("💡 Make sure you've run:")
        print("   python src/utils/convert_to_parquet.py")
        return
    
    # 1. Load feature importance
    importance_df = load_feature_importance(importance_dir)
    
    if importance_df is None:
        return
    
    # 2. Select top N features
    selected_features, report = select_top_features(importance_df, n_features=N_FEATURES)
    
    # 3. Filter dataset
    filter_dataset(input_parquet, output_parquet, selected_features)
    
    # 4. Save selection report
    selected_features_df = importance_df.head(N_FEATURES)
    save_selection_report(report, selected_features_df, output_dir)
    
    # Summary
    elapsed = time.time() - start_time
    print("=" * 70)
    print("✅ FEATURE SELECTION COMPLETE!")
    print("=" * 70)
    print()
    print(f"⏱️  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print()
    print("📁 Output files:")
    print(f"   Filtered dataset: {output_parquet}")
    print(f"   Reports: {output_dir}")
    print(f"   - selected_features.csv")
    print(f"   - feature_selection_report.txt")
    print()

if __name__ == "__main__":
    main()

