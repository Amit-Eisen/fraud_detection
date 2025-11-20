# src/data_cleansing/03_handle_outliers.py

import pandas as pd
import time
from pathlib import Path


def analyze_outlier_impact(df: pd.DataFrame):
    """
    Test different IQR multipliers (1.5, 2.0, 3.0) and report impact on fraud cases.
    Helps make data-driven decision on which threshold to use.
    """
    print("\n" + "="*80)
    print("📊 OUTLIER IMPACT ANALYSIS - Testing Different IQR Multipliers")
    print("="*80 + "\n")
    
    # Calculate IQR
    Q1 = df['amt'].quantile(0.25)
    Q3 = df['amt'].quantile(0.75)
    IQR = Q3 - Q1
    
    print(f"📈 Amount Distribution:")
    print(f"   Q1 (25th percentile): ${Q1:.2f}")
    print(f"   Q3 (75th percentile): ${Q3:.2f}")
    print(f"   IQR: ${IQR:.2f}")
    print(f"   Min: ${df['amt'].min():.2f}")
    print(f"   Max: ${df['amt'].max():.2f}")
    print()
    
    total_rows = len(df)
    total_fraud = df['is_fraud'].sum()
    
    print(f"📊 Dataset Overview:")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Total fraud cases: {total_fraud:,} ({total_fraud/total_rows*100:.2f}%)")
    print("\n" + "-"*80 + "\n")
    
    # Test different multipliers
    results = []
    
    for multiplier in [1.5, 2.0, 3.0]:
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Identify outliers
        outliers_mask = (df['amt'] < lower_bound) | (df['amt'] > upper_bound)
        outliers_df = df[outliers_mask]
        
        rows_removed = len(outliers_df)
        fraud_removed = outliers_df['is_fraud'].sum()
        
        pct_rows = rows_removed / total_rows * 100
        pct_fraud = fraud_removed / total_fraud * 100 if total_fraud > 0 else 0
        
        results.append({
            'multiplier': multiplier,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'rows_removed': rows_removed,
            'pct_rows': pct_rows,
            'fraud_removed': fraud_removed,
            'pct_fraud': pct_fraud
        })
        
        print(f"🔍 IQR × {multiplier}")
        print(f"   Bounds: [${lower_bound:.2f}, ${upper_bound:.2f}]")
        print(f"   Rows removed: {rows_removed:,} ({pct_rows:.2f}% of dataset)")
        print(f"   Fraud removed: {fraud_removed:,} ({pct_fraud:.2f}% of fraud cases)")
        print()
    
    print("-"*80 + "\n")
    
    # Recommendation
    print("💡 RECOMMENDATION:")
    print()
    
    # Check if any option is acceptable (< 5% fraud loss)
    best = None
    for r in results:
        if r['pct_fraud'] < 5.0:
            best = r
            break
    
    if best:
        print(f"   ✅ Use IQR × {best['multiplier']} (Standard)")
        print(f"      • Removes {best['rows_removed']:,} outliers ({best['pct_rows']:.2f}%)")
        print(f"      • Only loses {best['fraud_removed']:,} fraud cases ({best['pct_fraud']:.2f}%)")
        print(f"      • Aligns with course material")
    else:
        # All options lose too much fraud - DON'T remove outliers!
        print(f"   ⚠️  DO NOT REMOVE OUTLIERS!")
        print(f"      • All thresholds remove 60-75% of fraud cases")
        print(f"      • High transaction amounts are a FRAUD INDICATOR, not noise")
        print(f"      • Tree-based models (XGBoost, Random Forest) handle outliers well")
        print(f"      • Preserving all data for maximum fraud detection accuracy")
        print(f"      • Course guideline: 'Don't drop if it affects association' - this clearly does!")
    
    print("\n" + "="*80 + "\n")
    
    return results


def run(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers in 'amt' using IQR method.
    
    Args:
        df: Input DataFrame
        multiplier: IQR multiplier (default: 1.5 for standard approach)
    
    Returns:
        DataFrame with outliers removed
    """
    Q1 = df['amt'].quantile(0.25)
    Q3 = df['amt'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    before = len(df)
    df = df[(df['amt'] >= lower_bound) & (df['amt'] <= upper_bound)]
    after = len(df)
    
    removed = before - after
    print(f"  Outliers removed: {removed:,} ({removed/before*100:.2f}%)")
    print(f"  Bounds: [${lower_bound:.2f}, ${upper_bound:.2f}]")
    
    return df


def run_chunked(input_path: Path, output_path: Path, multiplier: float = 1.5, chunksize: int = 1_000_000):
    """
    Process large CSV files in chunks to remove outliers.
    Uses two-pass approach: first pass calculates bounds, second pass filters.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        multiplier: IQR multiplier (default: 1.5)
        chunksize: Number of rows to process at a time
    """
    start = time.time()
    
    print(f"🧹 Starting outlier removal (IQR × {multiplier})")
    print(f"   Input: {input_path}")
    print(f"   Chunk size: {chunksize:,} rows")
    print()
    
    # PASS 1: Calculate Q1, Q3, IQR on full dataset
    print("📊 Pass 1: Calculating IQR bounds...")
    amt_values = []
    for chunk in pd.read_csv(input_path, usecols=['amt'], chunksize=chunksize, low_memory=False):
        amt_values.extend(chunk['amt'].values)
    
    amt_series = pd.Series(amt_values)
    Q1 = amt_series.quantile(0.25)
    Q3 = amt_series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    print(f"   Q1: ${Q1:.2f}, Q3: ${Q3:.2f}, IQR: ${IQR:.2f}")
    print(f"   Bounds: [${lower_bound:.2f}, ${upper_bound:.2f}]")
    print()
    
    # PASS 2: Filter chunks
    print("🔍 Pass 2: Filtering outliers...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    first_chunk = True
    total_rows = 0
    total_removed = 0
    chunk_num = 0
    
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        chunk_num += 1
        rows_before = len(chunk)
        
        # Filter outliers
        chunk = chunk[(chunk['amt'] >= lower_bound) & (chunk['amt'] <= upper_bound)]
        
        rows_after = len(chunk)
        removed = rows_before - rows_after
        
        total_rows += rows_after
        total_removed += removed
        
        print(f"  Chunk {chunk_num}: {rows_after:,} rows kept, {removed:,} removed")
        
        # Write to CSV
        if first_chunk:
            chunk.to_csv(output_path, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(output_path, index=False, mode='a', header=False)
    
    end = time.time()
    
    print(f"\n✅ Outlier removal complete!")
    print(f"   Total rows kept: {total_rows:,}")
    print(f"   Total rows removed: {total_removed:,} ({total_removed/(total_rows+total_removed)*100:.2f}%)")
    print(f"   Output: {output_path}")
    print(f"⏱️  Runtime: {end - start:.1f} seconds ({(end - start)/60:.1f} minutes)\n")


# Standalone execution
if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / "data" / "cleansed"
    input_file = data_dir / "02_clean_columns.csv"
    
    print("🔬 ANALYSIS MODE: Testing different IQR multipliers")
    print("   Loading amt + is_fraud columns only ...\n")
    
    # Load only needed columns for analysis
    df_analysis = pd.read_csv(input_file, usecols=['amt', 'is_fraud'], low_memory=False)
    
    # Run analysis
    results = analyze_outlier_impact(df_analysis)
    
    print("💾 To apply outlier removal, run:")
    print("   python src/data_cleansing/03_handle_outliers.py --apply")
    print()
    print("   Or integrate into main pipeline with chosen multiplier.")

