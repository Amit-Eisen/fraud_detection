# src/eda/basic_statistics.py

import pandas as pd
import numpy as np
import time
from pathlib import Path


def run(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Generate basic statistics for a DataFrame (used when df is already in memory).
    
    Args:
        df: Input DataFrame
        output_dir: Directory to save reports
    
    Returns:
        Same DataFrame (unchanged)
    """
    print(f"📊 Analyzing dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Dataset Overview ---
    overview = {
        'Metric': [
            'Total Rows',
            'Total Columns',
            'Memory Usage (MB)',
            'Numerical Columns',
            'Categorical Columns',
            'Datetime Columns'
        ],
        'Value': [
            f"{df.shape[0]:,}",
            df.shape[1],
            f"{df.memory_usage(deep=True).sum() / 1024**2:.1f}",
            len(df.select_dtypes(include=['number']).columns),
            len(df.select_dtypes(include=['object']).columns),
            len(df.select_dtypes(include=['datetime']).columns)
        ]
    }
    overview_df = pd.DataFrame(overview)
    
    # --- 2. Missing Values Analysis ---
    missing = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percent': (df.isnull().sum().values / len(df) * 100).round(2),
        'Data_Type': df.dtypes.values
    })
    missing = missing[missing['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    # --- 3. Numerical Summary ---
    numerical_cols = df.select_dtypes(include=['number']).columns
    if len(numerical_cols) > 0:
        numerical_summary = df[numerical_cols].describe().T
        numerical_summary['missing'] = df[numerical_cols].isnull().sum()
        numerical_summary = numerical_summary.round(2)
    else:
        numerical_summary = pd.DataFrame()
    
    # --- 4. Categorical Summary ---
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        categorical_summary = pd.DataFrame({
            'Column': categorical_cols,
            'Unique_Values': [df[col].nunique() for col in categorical_cols],
            'Most_Common': [df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A' for col in categorical_cols],
            'Most_Common_Count': [df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0 for col in categorical_cols]
        })
    else:
        categorical_summary = pd.DataFrame()
    
    # --- Save Reports ---
    overview_df.to_csv(output_dir / '01_dataset_overview.csv', index=False)
    
    if not missing.empty:
        missing.to_csv(output_dir / '01_missing_values.csv', index=False)
    
    if not numerical_summary.empty:
        numerical_summary.to_csv(output_dir / '01_numerical_summary.csv')
    
    if not categorical_summary.empty:
        categorical_summary.to_csv(output_dir / '01_categorical_summary.csv', index=False)
    
    # Text report
    report_path = output_dir / '01_basic_statistics_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FRAUD DETECTION - BASIC STATISTICS REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. DATASET OVERVIEW\n")
        f.write("-" * 70 + "\n")
        f.write(overview_df.to_string(index=False) + "\n\n")
        
        f.write("2. MISSING VALUES\n")
        f.write("-" * 70 + "\n")
        if missing.empty:
            f.write("✅ No missing values detected!\n\n")
        else:
            f.write(missing.to_string(index=False) + "\n\n")
        
        f.write("3. NUMERICAL FEATURES SUMMARY\n")
        f.write("-" * 70 + "\n")
        if not numerical_summary.empty:
            f.write(numerical_summary.to_string() + "\n\n")
        else:
            f.write("No numerical columns found.\n\n")
        
        f.write("4. CATEGORICAL FEATURES SUMMARY\n")
        f.write("-" * 70 + "\n")
        if not categorical_summary.empty:
            f.write(categorical_summary.to_string(index=False) + "\n\n")
        else:
            f.write("No categorical columns found.\n\n")
    
    print(f"💾 Reports saved to: {output_dir}")
    
    return df


def run_chunked(input_path: Path, output_dir: Path, chunksize: int = 1_000_000):
    """
    Compute statistics on large CSV files using chunked processing.
    
    Two-pass approach:
    Pass 1: Accumulate statistics across all chunks
    Pass 2: Generate final reports
    
    Args:
        input_path: Path to input CSV file
        output_dir: Directory to save reports
        chunksize: Number of rows to process at a time
    """
    start = time.time()
    
    print(f"📊 Starting chunked statistics analysis")
    print(f"   Input: {input_path}")
    print(f"   Chunk size: {chunksize:,} rows")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- PASS 1: Accumulate statistics ---
    print("  Pass 1/2: Scanning all data...")
    
    total_rows = 0
    chunk_num = 0
    
    # Accumulators
    column_names = None
    dtypes = None
    missing_counts = None
    
    # For numerical stats (we'll use Welford's online algorithm for mean/std)
    numerical_cols = None
    num_means = None
    num_m2 = None  # For variance calculation
    num_mins = None
    num_maxs = None
    num_counts = None
    
    # For categorical stats
    categorical_cols = None
    cat_value_counts = {}
    
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        chunk_num += 1
        chunk_rows = len(chunk)
        total_rows += chunk_rows
        
        # First chunk: initialize
        if column_names is None:
            column_names = chunk.columns.tolist()
            dtypes = chunk.dtypes
            missing_counts = pd.Series(0, index=column_names)
            
            numerical_cols = chunk.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = chunk.select_dtypes(include=['object']).columns.tolist()
            
            # Initialize numerical accumulators
            num_means = {col: 0.0 for col in numerical_cols}
            num_m2 = {col: 0.0 for col in numerical_cols}
            num_mins = {col: float('inf') for col in numerical_cols}
            num_maxs = {col: float('-inf') for col in numerical_cols}
            num_counts = {col: 0 for col in numerical_cols}
            
            # Initialize categorical accumulators
            for col in categorical_cols:
                cat_value_counts[col] = {}
        
        # Accumulate missing values
        missing_counts += chunk.isnull().sum()
        
        # Accumulate numerical statistics (Welford's algorithm)
        for col in numerical_cols:
            valid_data = chunk[col].dropna()
            if len(valid_data) > 0:
                for value in valid_data:
                    num_counts[col] += 1
                    delta = value - num_means[col]
                    num_means[col] += delta / num_counts[col]
                    delta2 = value - num_means[col]
                    num_m2[col] += delta * delta2
                    num_mins[col] = min(num_mins[col], value)
                    num_maxs[col] = max(num_maxs[col], value)
        
        # Accumulate categorical value counts
        for col in categorical_cols:
            chunk_counts = chunk[col].value_counts().to_dict()
            for value, count in chunk_counts.items():
                cat_value_counts[col][value] = cat_value_counts[col].get(value, 0) + count
        
        if chunk_num % 5 == 0:
            print(f"    Scanned {total_rows:,} rows...")
    
    print(f"    Total rows scanned: {total_rows:,}\n")
    
    # --- PASS 2: Generate reports ---
    print("  Pass 2/2: Generating reports...")
    
    # 1. Dataset Overview
    overview = {
        'Metric': [
            'Total Rows',
            'Total Columns',
            'Numerical Columns',
            'Categorical Columns'
        ],
        'Value': [
            f"{total_rows:,}",
            len(column_names),
            len(numerical_cols),
            len(categorical_cols)
        ]
    }
    overview_df = pd.DataFrame(overview)
    overview_df.to_csv(output_dir / '01_dataset_overview.csv', index=False)
    
    # 2. Missing Values
    missing = pd.DataFrame({
        'Column': column_names,
        'Missing_Count': missing_counts.values,
        'Missing_Percent': (missing_counts.values / total_rows * 100).round(2),
        'Data_Type': dtypes.values
    })
    missing = missing[missing['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    if not missing.empty:
        missing.to_csv(output_dir / '01_missing_values.csv', index=False)
    
    # 3. Numerical Summary
    if numerical_cols:
        num_summary = []
        for col in numerical_cols:
            if num_counts[col] > 0:
                mean = num_means[col]
                variance = num_m2[col] / num_counts[col] if num_counts[col] > 1 else 0
                std = np.sqrt(variance)
                
                num_summary.append({
                    'column': col,
                    'count': num_counts[col],
                    'mean': round(mean, 2),
                    'std': round(std, 2),
                    'min': round(num_mins[col], 2),
                    'max': round(num_maxs[col], 2),
                    'missing': int(missing_counts[col])
                })
        
        numerical_summary = pd.DataFrame(num_summary)
        numerical_summary.to_csv(output_dir / '01_numerical_summary.csv', index=False)
    
    # 4. Categorical Summary
    if categorical_cols:
        cat_summary = []
        for col in categorical_cols:
            if cat_value_counts[col]:
                sorted_counts = sorted(cat_value_counts[col].items(), key=lambda x: x[1], reverse=True)
                most_common = sorted_counts[0][0]
                most_common_count = sorted_counts[0][1]
                
                cat_summary.append({
                    'Column': col,
                    'Unique_Values': len(cat_value_counts[col]),
                    'Most_Common': most_common,
                    'Most_Common_Count': most_common_count
                })
        
        categorical_summary = pd.DataFrame(cat_summary)
        categorical_summary.to_csv(output_dir / '01_categorical_summary.csv', index=False)
    
    # 5. Text Report
    report_path = output_dir / '01_basic_statistics_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FRAUD DETECTION - BASIC STATISTICS REPORT (FULL DATASET)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. DATASET OVERVIEW\n")
        f.write("-" * 70 + "\n")
        f.write(overview_df.to_string(index=False) + "\n\n")
        
        f.write("2. MISSING VALUES\n")
        f.write("-" * 70 + "\n")
        if missing.empty:
            f.write("✅ No missing values detected!\n\n")
        else:
            f.write(missing.to_string(index=False) + "\n\n")
        
        if numerical_cols:
            f.write("3. NUMERICAL FEATURES SUMMARY\n")
            f.write("-" * 70 + "\n")
            f.write(numerical_summary.to_string(index=False) + "\n\n")
        
        if categorical_cols:
            f.write("4. CATEGORICAL FEATURES SUMMARY\n")
            f.write("-" * 70 + "\n")
            f.write(categorical_summary.to_string(index=False) + "\n\n")
    
    # Console output
    print("\n📋 Dataset Overview:")
    print(overview_df.to_string(index=False))
    print()
    
    if not missing.empty:
        print("⚠️  Missing Values Detected:")
        print(missing.head(10).to_string(index=False))
        print()
    else:
        print("✅ No missing values detected!")
        print()
    
    end = time.time()
    
    print(f"💾 Reports saved to: {output_dir}")
    print(f"   - 01_dataset_overview.csv")
    print(f"   - 01_missing_values.csv")
    print(f"   - 01_numerical_summary.csv")
    print(f"   - 01_categorical_summary.csv")
    print(f"   - 01_basic_statistics_report.txt")
    print(f"\n⏱️  Runtime: {end - start:.1f} seconds ({(end - start)/60:.1f} minutes)\n")


# Standalone execution
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "processed"
    reports_dir = project_root / "eda_reports"
    
    # Load the final transformed dataset
    input_file = data_dir / "03_transformed.csv"
    
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found!")
        print("   Please run the data preparation pipeline first.")
        exit(1)
    
    print(f"Processing: {input_file}")
    
    # Use chunked processing for full dataset analysis
    run_chunked(input_file, reports_dir, chunksize=1_000_000)
    
    print("✅ Basic statistics analysis complete (full dataset)!")
