# src/eda/categorical_features.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path


def run(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Analyze categorical features and their relationship with fraud.
    
    Key questions:
    - Which jobs have the highest fraud rates?
    - Which merchants are targeted most?
    - Which transaction categories are riskiest?
    
    Args:
        df: Input DataFrame with categorical columns and 'is_fraud'
        output_dir: Directory to save reports and plots
    
    Returns:
        Same DataFrame (unchanged)
    """
    start = time.time()
    
    print(f"📊 Starting categorical features analysis")
    print(f"   Dataset: {df.shape[0]:,} rows")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check required columns
    if 'is_fraud' not in df.columns:
        print("❌ Error: 'is_fraud' column not found!")
        return df
    
    # Categorical columns to analyze
    cat_columns = ['job', 'merchant', 'category']
    available_cols = [col for col in cat_columns if col in df.columns]
    
    if not available_cols:
        print("❌ Error: No categorical columns found!")
        return df
    
    print(f"   Analyzing columns: {available_cols}")
    print()
    
    # --- Analyze each categorical column ---
    all_reports = {}
    
    for col in available_cols:
        print(f"  Analyzing '{col}'...")
        
        # Group by category and calculate fraud statistics
        cat_stats = df.groupby(col).agg({
            'is_fraud': ['sum', 'count', 'mean']
        }).reset_index()
        
        cat_stats.columns = [col, 'fraud_count', 'total_count', 'fraud_rate']
        cat_stats['fraud_rate'] = (cat_stats['fraud_rate'] * 100).round(3)
        
        # Sort by fraud rate descending
        cat_stats = cat_stats.sort_values('fraud_rate', ascending=False)
        
        # Save full report
        cat_stats.to_csv(output_dir / f'05_{col}_fraud_analysis.csv', index=False)
        
        all_reports[col] = cat_stats
    
    # --- Text Report ---
    report_path = output_dir / '05_categorical_features_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FRAUD DETECTION - CATEGORICAL FEATURES ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        for col in available_cols:
            cat_stats = all_reports[col]
            
            f.write(f"{col.upper()} ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total unique {col}s: {len(cat_stats)}\n\n")
            
            # Top 10 by fraud rate
            f.write(f"Top 10 {col}s by Fraud Rate:\n")
            f.write(cat_stats.head(10).to_string(index=False) + "\n\n")
            
            # Top 10 by fraud count
            f.write(f"Top 10 {col}s by Fraud Count:\n")
            top_count = cat_stats.sort_values('fraud_count', ascending=False).head(10)
            f.write(top_count.to_string(index=False) + "\n\n")
            
            # Key insights
            highest_rate = cat_stats.iloc[0]
            highest_count = cat_stats.sort_values('fraud_count', ascending=False).iloc[0]
            
            f.write("KEY INSIGHTS:\n")
            f.write(f"Highest fraud rate:  {highest_rate[col]} ({highest_rate['fraud_rate']:.3f}%)\n")
            f.write(f"Most fraud count:    {highest_count[col]} ({int(highest_count['fraud_count']):,} frauds)\n")
            f.write("\n" + "=" * 80 + "\n\n")
    
    # --- Visualizations ---
    n_cols = len(available_cols)
    fig, axes = plt.subplots(n_cols, 2, figsize=(16, 6 * n_cols))
    
    if n_cols == 1:
        axes = axes.reshape(1, -1)
    
    for idx, col in enumerate(available_cols):
        cat_stats = all_reports[col]
        
        # Plot 1: Top 15 by fraud rate
        ax1 = axes[idx, 0]
        top_rate = cat_stats.head(15)
        
        bars = ax1.barh(range(len(top_rate)), top_rate['fraud_rate'], 
                       color='#e74c3c', alpha=0.7, edgecolor='black')
        ax1.set_yticks(range(len(top_rate)))
        ax1.set_yticklabels(top_rate[col], fontsize=9)
        ax1.set_xlabel('Fraud Rate (%)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Top 15 {col.title()}s by Fraud Rate', fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.3f}%',
                    ha='left', va='center', fontsize=8, fontweight='bold')
        
        # Plot 2: Top 15 by fraud count
        ax2 = axes[idx, 1]
        top_count = cat_stats.sort_values('fraud_count', ascending=False).head(15)
        
        bars = ax2.barh(range(len(top_count)), top_count['fraud_count'],
                       color='#3498db', alpha=0.7, edgecolor='black')
        ax2.set_yticks(range(len(top_count)))
        ax2.set_yticklabels(top_count[col], fontsize=9)
        ax2.set_xlabel('Fraud Count', fontsize=11, fontweight='bold')
        ax2.set_title(f'Top 15 {col.title()}s by Fraud Count', fontsize=13, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(width):,}',
                    ha='left', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '05_categorical_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Console Output ---
    print()
    print("=" * 80)
    print("CATEGORICAL FEATURES ANALYSIS RESULTS")
    print("=" * 80)
    print()
    
    for col in available_cols:
        cat_stats = all_reports[col]
        
        print(f"📊 {col.upper()} ANALYSIS:")
        print("-" * 80)
        print(f"Total unique {col}s: {len(cat_stats)}")
        print()
        
        print(f"Top 10 {col}s by Fraud Rate:")
        print(cat_stats.head(10).to_string(index=False))
        print()
        
        highest_rate = cat_stats.iloc[0]
        highest_count = cat_stats.sort_values('fraud_count', ascending=False).iloc[0]
        
        print(f"⚠️  Highest fraud rate:  {highest_rate[col]} ({highest_rate['fraud_rate']:.3f}%)")
        print(f"📈 Most fraud count:    {highest_count[col]} ({int(highest_count['fraud_count']):,} frauds)")
        print()
        print("=" * 80)
        print()
    
    print(f"💾 Reports saved to: {output_dir}")
    for col in available_cols:
        print(f"   - 05_{col}_fraud_analysis.csv")
    print(f"   - 05_categorical_analysis.png")
    print(f"   - 05_categorical_features_report.txt")
    
    end = time.time()
    print(f"\n⏱️  Runtime: {end - start:.2f} seconds\n")
    
    return df


def run_chunked(input_path: Path, output_dir: Path, chunksize: int = 1_000_000):
    """
    Analyze categorical features on large CSV files using chunked processing.
    
    Args:
        input_path: Path to input CSV file
        output_dir: Directory to save reports
        chunksize: Number of rows to process at a time
    """
    start = time.time()
    
    print(f"📊 Starting chunked categorical features analysis")
    print(f"   Input: {input_path}")
    print(f"   Chunk size: {chunksize:,} rows")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Categorical columns to analyze
    cat_columns = ['job', 'merchant', 'category']
    
    # --- Accumulate counts ---
    print("  Scanning data...")
    
    # Accumulators: {column: {value: {'fraud': count, 'total': count}}}
    cat_counts = {col: {} for col in cat_columns}
    
    chunk_num = 0
    total_rows = 0
    available_cols = []
    
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        chunk_num += 1
        total_rows += len(chunk)
        
        if 'is_fraud' not in chunk.columns:
            continue
        
        # First chunk: determine which columns are available
        if chunk_num == 1:
            available_cols = [col for col in cat_columns if col in chunk.columns]
            print(f"   Analyzing columns: {available_cols}")
            print()
        
        # Accumulate counts for each categorical column
        for col in available_cols:
            for value in chunk[col].unique():
                if pd.isna(value):
                    continue
                
                if value not in cat_counts[col]:
                    cat_counts[col][value] = {'fraud': 0, 'total': 0}
                
                value_data = chunk[chunk[col] == value]
                cat_counts[col][value]['fraud'] += (value_data['is_fraud'] == 1).sum()
                cat_counts[col][value]['total'] += len(value_data)
        
        if chunk_num % 5 == 0:
            print(f"    Scanned {total_rows:,} rows...")
    
    print(f"    Total rows scanned: {total_rows:,}\n")
    
    # --- Generate reports ---
    all_reports = {}
    
    for col in available_cols:
        print(f"  Generating report for '{col}'...")
        
        # Convert to DataFrame
        cat_stats = pd.DataFrame([
            {
                col: value,
                'fraud_count': cat_counts[col][value]['fraud'],
                'total_count': cat_counts[col][value]['total'],
                'fraud_rate': (cat_counts[col][value]['fraud'] / cat_counts[col][value]['total'] * 100)
                             if cat_counts[col][value]['total'] > 0 else 0
            }
            for value in cat_counts[col].keys()
        ])
        
        cat_stats['fraud_rate'] = cat_stats['fraud_rate'].round(3)
        cat_stats = cat_stats.sort_values('fraud_rate', ascending=False)
        
        # Save CSV
        cat_stats.to_csv(output_dir / f'05_{col}_fraud_analysis.csv', index=False)
        
        all_reports[col] = cat_stats
    
    # --- Text Report ---
    report_path = output_dir / '05_categorical_features_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FRAUD DETECTION - CATEGORICAL FEATURES ANALYSIS (FULL DATASET)\n")
        f.write("=" * 80 + "\n\n")
        
        for col in available_cols:
            cat_stats = all_reports[col]
            
            f.write(f"{col.upper()} ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total unique {col}s: {len(cat_stats)}\n\n")
            
            f.write(f"Top 10 {col}s by Fraud Rate:\n")
            f.write(cat_stats.head(10).to_string(index=False) + "\n\n")
            
            f.write(f"Top 10 {col}s by Fraud Count:\n")
            top_count = cat_stats.sort_values('fraud_count', ascending=False).head(10)
            f.write(top_count.to_string(index=False) + "\n\n")
            
            highest_rate = cat_stats.iloc[0]
            highest_count = cat_stats.sort_values('fraud_count', ascending=False).iloc[0]
            
            f.write("KEY INSIGHTS:\n")
            f.write(f"Highest fraud rate:  {highest_rate[col]} ({highest_rate['fraud_rate']:.3f}%)\n")
            f.write(f"Most fraud count:    {highest_count[col]} ({int(highest_count['fraud_count']):,} frauds)\n")
            f.write("\n" + "=" * 80 + "\n\n")
    
    # --- Visualizations ---
    n_cols = len(available_cols)
    fig, axes = plt.subplots(n_cols, 2, figsize=(16, 6 * n_cols))
    
    if n_cols == 1:
        axes = axes.reshape(1, -1)
    
    for idx, col in enumerate(available_cols):
        cat_stats = all_reports[col]
        
        # Plot 1: Top 15 by fraud rate
        ax1 = axes[idx, 0]
        top_rate = cat_stats.head(15)
        
        bars = ax1.barh(range(len(top_rate)), top_rate['fraud_rate'],
                       color='#e74c3c', alpha=0.7, edgecolor='black')
        ax1.set_yticks(range(len(top_rate)))
        ax1.set_yticklabels(top_rate[col], fontsize=9)
        ax1.set_xlabel('Fraud Rate (%)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Top 15 {col.title()}s by Fraud Rate', fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.3f}%',
                    ha='left', va='center', fontsize=8, fontweight='bold')
        
        # Plot 2: Top 15 by fraud count
        ax2 = axes[idx, 1]
        top_count = cat_stats.sort_values('fraud_count', ascending=False).head(15)
        
        bars = ax2.barh(range(len(top_count)), top_count['fraud_count'],
                       color='#3498db', alpha=0.7, edgecolor='black')
        ax2.set_yticks(range(len(top_count)))
        ax2.set_yticklabels(top_count[col], fontsize=9)
        ax2.set_xlabel('Fraud Count', fontsize=11, fontweight='bold')
        ax2.set_title(f'Top 15 {col.title()}s by Fraud Count', fontsize=13, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(width):,}',
                    ha='left', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '05_categorical_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Console Output ---
    print()
    print("=" * 80)
    print("CATEGORICAL FEATURES ANALYSIS RESULTS")
    print("=" * 80)
    print()
    
    for col in available_cols:
        cat_stats = all_reports[col]
        
        print(f"📊 {col.upper()} ANALYSIS:")
        print("-" * 80)
        print(f"Total unique {col}s: {len(cat_stats)}")
        print()
        
        print(f"Top 10 {col}s by Fraud Rate:")
        print(cat_stats.head(10).to_string(index=False))
        print()
        
        highest_rate = cat_stats.iloc[0]
        highest_count = cat_stats.sort_values('fraud_count', ascending=False).iloc[0]
        
        print(f"⚠️  Highest fraud rate:  {highest_rate[col]} ({highest_rate['fraud_rate']:.3f}%)")
        print(f"📈 Most fraud count:    {highest_count[col]} ({int(highest_count['fraud_count']):,} frauds)")
        print()
        print("=" * 80)
        print()
    
    end = time.time()
    
    print(f"💾 Reports saved to: {output_dir}")
    for col in available_cols:
        print(f"   - 05_{col}_fraud_analysis.csv")
    print(f"   - 05_categorical_analysis.png")
    print(f"   - 05_categorical_features_report.txt")
    print(f"\n⏱️  Runtime: {end - start:.1f} seconds ({(end - start)/60:.1f} minutes)\n")


# Standalone execution
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "processed"
    reports_dir = project_root / "reports" / "eda_reports"
    
    input_file = data_dir / "03_transformed.csv"
    
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found!")
        print("   Please run the data preparation pipeline first.")
        exit(1)
    
    print(f"Processing: {input_file}")
    
    # Use chunked processing
    run_chunked(input_file, reports_dir, chunksize=1_000_000)
    
    print("✅ Categorical features analysis complete (full dataset)!")

