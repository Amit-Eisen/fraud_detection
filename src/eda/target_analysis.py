# src/eda/target_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path


def run(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Analyze target variable (fraud) distribution and class imbalance.
    
    This is critical for fraud detection as fraud is typically rare (<1%).
    Understanding class imbalance guides model selection and evaluation metrics.
    
    Args:
        df: Input DataFrame with 'is_fraud' column
        output_dir: Directory to save reports and plots
    
    Returns:
        Same DataFrame (unchanged)
    """
    start = time.time()
    
    print(f"🎯 Starting target variable analysis")
    print(f"   Dataset: {df.shape[0]:,} rows")
    print()
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if target column exists
    if 'is_fraud' not in df.columns:
        print("❌ Error: 'is_fraud' column not found!")
        return df
    
    # --- 1. Fraud Distribution ---
    fraud_counts = df['is_fraud'].value_counts().sort_index()
    fraud_pct = df['is_fraud'].value_counts(normalize=True).sort_index() * 100
    
    distribution = pd.DataFrame({
        'is_fraud': fraud_counts.index,
        'count': fraud_counts.values,
        'percentage': fraud_pct.values.round(2)
    })
    
    non_fraud = fraud_counts.get(0, 0)
    fraud = fraud_counts.get(1, 0)
    total = len(df)
    fraud_rate = (fraud / total * 100) if total > 0 else 0
    imbalance_ratio = (non_fraud / fraud) if fraud > 0 else float('inf')
    
    # --- 2. Save Distribution Report ---
    distribution.to_csv(output_dir / '02_fraud_distribution.csv', index=False)
    
    # Text report
    report_path = output_dir / '02_target_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FRAUD DETECTION - TARGET VARIABLE ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. FRAUD DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Transactions:     {total:,}\n")
        f.write(f"Non-Fraud (0):          {non_fraud:,} ({100-fraud_rate:.2f}%)\n")
        f.write(f"Fraud (1):              {fraud:,} ({fraud_rate:.2f}%)\n")
        f.write(f"\n")
        f.write(f"Fraud Rate:             {fraud_rate:.3f}%\n")
        f.write(f"Imbalance Ratio:        {imbalance_ratio:.1f}:1\n")
        f.write(f"\n")
        
        # Interpretation
        f.write("2. INTERPRETATION\n")
        f.write("-" * 70 + "\n")
        if fraud_rate < 0.5:
            f.write("⚠️  SEVERE CLASS IMBALANCE (<0.5% fraud)\n")
            f.write("   Recommendation: Use SMOTE, class weights, or undersampling\n")
            f.write("   Metrics: Focus on Precision, Recall, F1, ROC-AUC (not accuracy!)\n")
        elif fraud_rate < 1.0:
            f.write("⚠️  HIGH CLASS IMBALANCE (<1% fraud)\n")
            f.write("   Recommendation: Use class weights or balanced sampling\n")
            f.write("   Metrics: Precision, Recall, F1, ROC-AUC\n")
        elif fraud_rate < 5.0:
            f.write("⚠️  MODERATE CLASS IMBALANCE (<5% fraud)\n")
            f.write("   Recommendation: Consider class weights\n")
            f.write("   Metrics: F1, ROC-AUC, Precision-Recall curve\n")
        else:
            f.write("✅ RELATIVELY BALANCED (>5% fraud)\n")
            f.write("   Standard classification techniques should work\n")
        f.write("\n")
    
    # --- 3. Visualizations ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Count bar chart
    ax1 = axes[0]
    bars = ax1.bar(['Non-Fraud', 'Fraud'], [non_fraud, fraud], 
                   color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Fraud vs Non-Fraud Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: Percentage pie chart
    ax2 = axes[1]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0, 0.1)  # Explode fraud slice
    wedges, texts, autotexts = ax2.pie([non_fraud, fraud], 
                                        labels=['Non-Fraud', 'Fraud'],
                                        autopct='%1.2f%%',
                                        colors=colors,
                                        explode=explode,
                                        startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Fraud Percentage', fontsize=14, fontweight='bold')
    
    # Add imbalance ratio annotation
    fig.text(0.5, 0.02, f'Imbalance Ratio: {imbalance_ratio:.1f}:1 | Fraud Rate: {fraud_rate:.3f}%',
             ha='center', fontsize=12, fontweight='bold', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_dir / '02_fraud_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Console Output ---
    print("📊 Fraud Distribution:")
    print(f"   Total Transactions: {total:,}")
    print(f"   Non-Fraud (0):      {non_fraud:,} ({100-fraud_rate:.2f}%)")
    print(f"   Fraud (1):          {fraud:,} ({fraud_rate:.2f}%)")
    print()
    print(f"   Fraud Rate:         {fraud_rate:.3f}%")
    print(f"   Imbalance Ratio:    {imbalance_ratio:.1f}:1")
    print()
    
    if fraud_rate < 1.0:
        print("⚠️  HIGH CLASS IMBALANCE detected!")
        print("   → Use SMOTE, class weights, or balanced sampling")
        print("   → Focus on Precision, Recall, F1, ROC-AUC (not accuracy)")
    
    print()
    print(f"💾 Reports saved to: {output_dir}")
    print(f"   - 02_fraud_distribution.csv")
    print(f"   - 02_fraud_distribution.png")
    print(f"   - 02_target_analysis_report.txt")
    
    end = time.time()
    print(f"\n⏱️  Runtime: {end - start:.2f} seconds\n")
    
    return df


def run_chunked(input_path: Path, output_dir: Path, chunksize: int = 1_000_000):
    """
    Analyze target variable on large CSV files using chunked processing.
    
    Args:
        input_path: Path to input CSV file
        output_dir: Directory to save reports
        chunksize: Number of rows to process at a time
    """
    start = time.time()
    
    print(f"🎯 Starting chunked target analysis")
    print(f"   Input: {input_path}")
    print(f"   Chunk size: {chunksize:,} rows")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Scan all data to count fraud ---
    print("  Scanning data...")
    
    total_rows = 0
    fraud_count = 0
    non_fraud_count = 0
    chunk_num = 0
    
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        chunk_num += 1
        total_rows += len(chunk)
        
        if 'is_fraud' in chunk.columns:
            fraud_count += (chunk['is_fraud'] == 1).sum()
            non_fraud_count += (chunk['is_fraud'] == 0).sum()
        
        if chunk_num % 5 == 0:
            print(f"    Scanned {total_rows:,} rows...")
    
    print(f"    Total rows scanned: {total_rows:,}\n")
    
    # --- Generate reports ---
    fraud_rate = (fraud_count / total_rows * 100) if total_rows > 0 else 0
    imbalance_ratio = (non_fraud_count / fraud_count) if fraud_count > 0 else float('inf')
    
    # Distribution CSV
    distribution = pd.DataFrame({
        'is_fraud': [0, 1],
        'count': [non_fraud_count, fraud_count],
        'percentage': [100 - fraud_rate, fraud_rate]
    })
    distribution.to_csv(output_dir / '02_fraud_distribution.csv', index=False)
    
    # Text report
    report_path = output_dir / '02_target_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FRAUD DETECTION - TARGET VARIABLE ANALYSIS (FULL DATASET)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. FRAUD DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Transactions:     {total_rows:,}\n")
        f.write(f"Non-Fraud (0):          {non_fraud_count:,} ({100-fraud_rate:.2f}%)\n")
        f.write(f"Fraud (1):              {fraud_count:,} ({fraud_rate:.2f}%)\n")
        f.write(f"\n")
        f.write(f"Fraud Rate:             {fraud_rate:.3f}%\n")
        f.write(f"Imbalance Ratio:        {imbalance_ratio:.1f}:1\n")
        f.write(f"\n")
        
        f.write("2. INTERPRETATION\n")
        f.write("-" * 70 + "\n")
        if fraud_rate < 0.5:
            f.write("⚠️  SEVERE CLASS IMBALANCE (<0.5% fraud)\n")
            f.write("   Recommendation: Use SMOTE, class weights, or undersampling\n")
            f.write("   Metrics: Focus on Precision, Recall, F1, ROC-AUC (not accuracy!)\n")
        elif fraud_rate < 1.0:
            f.write("⚠️  HIGH CLASS IMBALANCE (<1% fraud)\n")
            f.write("   Recommendation: Use class weights or balanced sampling\n")
            f.write("   Metrics: Precision, Recall, F1, ROC-AUC\n")
        elif fraud_rate < 5.0:
            f.write("⚠️  MODERATE CLASS IMBALANCE (<5% fraud)\n")
            f.write("   Recommendation: Consider class weights\n")
        else:
            f.write("✅ RELATIVELY BALANCED (>5% fraud)\n")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    bars = ax1.bar(['Non-Fraud', 'Fraud'], [non_fraud_count, fraud_count],
                   color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Fraud vs Non-Fraud Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2 = axes[1]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0, 0.1)
    ax2.pie([non_fraud_count, fraud_count],
            labels=['Non-Fraud', 'Fraud'],
            autopct='%1.2f%%',
            colors=colors,
            explode=explode,
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Fraud Percentage', fontsize=14, fontweight='bold')
    
    fig.text(0.5, 0.02, f'Imbalance Ratio: {imbalance_ratio:.1f}:1 | Fraud Rate: {fraud_rate:.3f}%',
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_dir / '02_fraud_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Console output
    print("📊 Fraud Distribution:")
    print(f"   Total Transactions: {total_rows:,}")
    print(f"   Non-Fraud (0):      {non_fraud_count:,} ({100-fraud_rate:.2f}%)")
    print(f"   Fraud (1):          {fraud_count:,} ({fraud_rate:.2f}%)")
    print()
    print(f"   Fraud Rate:         {fraud_rate:.3f}%")
    print(f"   Imbalance Ratio:    {imbalance_ratio:.1f}:1")
    print()
    
    if fraud_rate < 1.0:
        print("⚠️  HIGH CLASS IMBALANCE detected!")
        print("   → Use SMOTE, class weights, or balanced sampling")
        print("   → Focus on Precision, Recall, F1, ROC-AUC (not accuracy)")
    
    end = time.time()
    
    print()
    print(f"💾 Reports saved to: {output_dir}")
    print(f"   - 02_fraud_distribution.csv")
    print(f"   - 02_fraud_distribution.png")
    print(f"   - 02_target_analysis_report.txt")
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
    
    # Use chunked processing for full dataset
    run_chunked(input_file, reports_dir, chunksize=1_000_000)
    
    print("✅ Target analysis complete (full dataset)!")

