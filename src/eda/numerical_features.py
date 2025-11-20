# src/eda/numerical_features.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path


def run(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Analyze numerical features, focusing on transaction amounts and fraud patterns.
    
    Key insights:
    - Do fraudulent transactions have different amounts?
    - Are there outliers in transaction amounts?
    - Geographic patterns in coordinates
    
    Args:
        df: Input DataFrame with 'amt' and 'is_fraud' columns
        output_dir: Directory to save reports and plots
    
    Returns:
        Same DataFrame (unchanged)
    """
    start = time.time()
    
    print(f"💰 Starting numerical features analysis")
    print(f"   Dataset: {df.shape[0]:,} rows")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check required columns
    if 'amt' not in df.columns or 'is_fraud' not in df.columns:
        print("❌ Error: Required columns 'amt' or 'is_fraud' not found!")
        return df
    
    # --- 1. Amount Analysis by Fraud Status ---
    fraud_df = df[df['is_fraud'] == 1]['amt']
    non_fraud_df = df[df['is_fraud'] == 0]['amt']
    
    amount_stats = pd.DataFrame({
        'Metric': ['Count', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Q25', 'Q75'],
        'Non-Fraud': [
            len(non_fraud_df),
            non_fraud_df.mean(),
            non_fraud_df.median(),
            non_fraud_df.std(),
            non_fraud_df.min(),
            non_fraud_df.max(),
            non_fraud_df.quantile(0.25),
            non_fraud_df.quantile(0.75)
        ],
        'Fraud': [
            len(fraud_df),
            fraud_df.mean(),
            fraud_df.median(),
            fraud_df.std(),
            fraud_df.min(),
            fraud_df.max(),
            fraud_df.quantile(0.25),
            fraud_df.quantile(0.75)
        ]
    })
    
    # Round for readability
    amount_stats['Non-Fraud'] = amount_stats['Non-Fraud'].round(2)
    amount_stats['Fraud'] = amount_stats['Fraud'].round(2)
    
    # Save CSV
    amount_stats.to_csv(output_dir / '03_amount_statistics.csv', index=False)
    
    # --- 2. Text Report ---
    report_path = output_dir / '03_numerical_features_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FRAUD DETECTION - NUMERICAL FEATURES ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. TRANSACTION AMOUNT ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(amount_stats.to_string(index=False) + "\n\n")
        
        f.write("2. KEY INSIGHTS\n")
        f.write("-" * 70 + "\n")
        
        mean_diff = fraud_df.mean() - non_fraud_df.mean()
        median_diff = fraud_df.median() - non_fraud_df.median()
        
        f.write(f"Mean difference:   ${mean_diff:.2f} ")
        f.write("(fraud higher)\n" if mean_diff > 0 else "(fraud lower)\n")
        
        f.write(f"Median difference: ${median_diff:.2f} ")
        f.write("(fraud higher)\n" if median_diff > 0 else "(fraud lower)\n")
        
        f.write("\n")
        if abs(mean_diff) > 10:
            f.write("⚠️  Significant difference in transaction amounts!\n")
            f.write("   → Amount is likely a strong predictor for fraud\n")
        else:
            f.write("✓ Similar transaction amounts between fraud/non-fraud\n")
            f.write("   → Amount alone may not be a strong predictor\n")
        f.write("\n")
    
    # --- 3. Visualizations ---
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Amount distribution comparison (histogram)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Use log scale for better visualization (fraud amounts often skewed)
    bins = np.logspace(np.log10(df['amt'].min() + 0.01), 
                       np.log10(df['amt'].max()), 50)
    
    ax1.hist(non_fraud_df, bins=bins, alpha=0.6, label='Non-Fraud', 
             color='#2ecc71', edgecolor='black', density=True)
    ax1.hist(fraud_df, bins=bins, alpha=0.6, label='Fraud', 
             color='#e74c3c', edgecolor='black', density=True)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Transaction Amount ($) [Log Scale]', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax1.set_title('Transaction Amount Distribution: Fraud vs Non-Fraud', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Box plot comparison
    ax2 = fig.add_subplot(gs[1, 0])
    
    box_data = [non_fraud_df, fraud_df]
    bp = ax2.boxplot(box_data, labels=['Non-Fraud', 'Fraud'],
                     patch_artist=True, showfliers=False)
    
    colors = ['#2ecc71', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.set_ylabel('Transaction Amount ($)', fontsize=11, fontweight='bold')
    ax2.set_title('Amount Distribution (Box Plot)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Violin plot
    ax3 = fig.add_subplot(gs[1, 1])
    
    plot_df = pd.DataFrame({
        'Amount': pd.concat([non_fraud_df.sample(min(10000, len(non_fraud_df))), 
                            fraud_df.sample(min(10000, len(fraud_df)))]),
        'Type': ['Non-Fraud'] * min(10000, len(non_fraud_df)) + 
                ['Fraud'] * min(10000, len(fraud_df))
    })
    
    sns.violinplot(data=plot_df, x='Type', y='Amount', ax=ax3,
                   palette={'Non-Fraud': '#2ecc71', 'Fraud': '#e74c3c'})
    ax3.set_ylabel('Transaction Amount ($)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('')
    ax3.set_title('Amount Distribution (Violin Plot)', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Mean comparison bar chart
    ax4 = fig.add_subplot(gs[2, 0])
    
    means = [non_fraud_df.mean(), fraud_df.mean()]
    bars = ax4.bar(['Non-Fraud', 'Fraud'], means, 
                   color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
    
    ax4.set_ylabel('Mean Amount ($)', fontsize=11, fontweight='bold')
    ax4.set_title('Average Transaction Amount', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 5: Cumulative distribution
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Sample for performance
    non_fraud_sample = non_fraud_df.sample(min(50000, len(non_fraud_df))).sort_values()
    fraud_sample = fraud_df.sample(min(50000, len(fraud_df))).sort_values()
    
    ax5.plot(non_fraud_sample, np.linspace(0, 1, len(non_fraud_sample)),
             label='Non-Fraud', color='#2ecc71', linewidth=2)
    ax5.plot(fraud_sample, np.linspace(0, 1, len(fraud_sample)),
             label='Fraud', color='#e74c3c', linewidth=2)
    
    ax5.set_xlabel('Transaction Amount ($)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax5.set_title('Cumulative Distribution Function (CDF)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3)
    
    plt.savefig(output_dir / '03_amount_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Console Output ---
    print("💰 Transaction Amount Analysis:")
    print()
    print(amount_stats.to_string(index=False))
    print()
    print(f"   Mean difference:   ${mean_diff:.2f}")
    print(f"   Median difference: ${median_diff:.2f}")
    print()
    
    if abs(mean_diff) > 10:
        print("⚠️  Significant difference detected!")
        print("   → Amount is likely a strong predictor")
    
    print()
    print(f"💾 Reports saved to: {output_dir}")
    print(f"   - 03_amount_statistics.csv")
    print(f"   - 03_amount_analysis.png")
    print(f"   - 03_numerical_features_report.txt")
    
    end = time.time()
    print(f"\n⏱️  Runtime: {end - start:.2f} seconds\n")
    
    return df


def run_chunked(input_path: Path, output_dir: Path, chunksize: int = 1_000_000):
    """
    Analyze numerical features on large CSV files using chunked processing.
    
    Args:
        input_path: Path to input CSV file
        output_dir: Directory to save reports
        chunksize: Number of rows to process at a time
    """
    start = time.time()
    
    print(f"💰 Starting chunked numerical features analysis")
    print(f"   Input: {input_path}")
    print(f"   Chunk size: {chunksize:,} rows")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Accumulate statistics ---
    print("  Scanning data...")
    
    # Accumulators for Welford's algorithm
    fraud_n = 0
    fraud_mean = 0.0
    fraud_m2 = 0.0
    fraud_min = float('inf')
    fraud_max = float('-inf')
    fraud_values_for_quantiles = []
    
    non_fraud_n = 0
    non_fraud_mean = 0.0
    non_fraud_m2 = 0.0
    non_fraud_min = float('inf')
    non_fraud_max = float('-inf')
    non_fraud_values_for_quantiles = []
    
    chunk_num = 0
    total_rows = 0
    
    # Sample for quantiles (store up to 100k values)
    max_sample_size = 100000
    
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        chunk_num += 1
        total_rows += len(chunk)
        
        if 'amt' not in chunk.columns or 'is_fraud' not in chunk.columns:
            continue
        
        # Process fraud transactions
        fraud_chunk = chunk[chunk['is_fraud'] == 1]['amt'].dropna()
        for value in fraud_chunk:
            fraud_n += 1
            delta = value - fraud_mean
            fraud_mean += delta / fraud_n
            fraud_m2 += delta * (value - fraud_mean)
            fraud_min = min(fraud_min, value)
            fraud_max = max(fraud_max, value)
            
            # Sample for quantiles
            if len(fraud_values_for_quantiles) < max_sample_size:
                fraud_values_for_quantiles.append(value)
        
        # Process non-fraud transactions
        non_fraud_chunk = chunk[chunk['is_fraud'] == 0]['amt'].dropna()
        for value in non_fraud_chunk:
            non_fraud_n += 1
            delta = value - non_fraud_mean
            non_fraud_mean += delta / non_fraud_n
            non_fraud_m2 += delta * (value - non_fraud_mean)
            non_fraud_min = min(non_fraud_min, value)
            non_fraud_max = max(non_fraud_max, value)
            
            # Sample for quantiles
            if len(non_fraud_values_for_quantiles) < max_sample_size:
                non_fraud_values_for_quantiles.append(value)
        
        if chunk_num % 5 == 0:
            print(f"    Scanned {total_rows:,} rows...")
    
    print(f"    Total rows scanned: {total_rows:,}\n")
    
    # Calculate final statistics
    fraud_std = np.sqrt(fraud_m2 / fraud_n) if fraud_n > 1 else 0
    non_fraud_std = np.sqrt(non_fraud_m2 / non_fraud_n) if non_fraud_n > 1 else 0
    
    fraud_values_arr = np.array(fraud_values_for_quantiles)
    non_fraud_values_arr = np.array(non_fraud_values_for_quantiles)
    
    fraud_median = np.median(fraud_values_arr) if len(fraud_values_arr) > 0 else 0
    fraud_q25 = np.percentile(fraud_values_arr, 25) if len(fraud_values_arr) > 0 else 0
    fraud_q75 = np.percentile(fraud_values_arr, 75) if len(fraud_values_arr) > 0 else 0
    
    non_fraud_median = np.median(non_fraud_values_arr) if len(non_fraud_values_arr) > 0 else 0
    non_fraud_q25 = np.percentile(non_fraud_values_arr, 25) if len(non_fraud_values_arr) > 0 else 0
    non_fraud_q75 = np.percentile(non_fraud_values_arr, 75) if len(non_fraud_values_arr) > 0 else 0
    
    # --- Generate reports ---
    amount_stats = pd.DataFrame({
        'Metric': ['Count', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Q25', 'Q75'],
        'Non-Fraud': [
            non_fraud_n,
            round(non_fraud_mean, 2),
            round(non_fraud_median, 2),
            round(non_fraud_std, 2),
            round(non_fraud_min, 2),
            round(non_fraud_max, 2),
            round(non_fraud_q25, 2),
            round(non_fraud_q75, 2)
        ],
        'Fraud': [
            fraud_n,
            round(fraud_mean, 2),
            round(fraud_median, 2),
            round(fraud_std, 2),
            round(fraud_min, 2),
            round(fraud_max, 2),
            round(fraud_q25, 2),
            round(fraud_q75, 2)
        ]
    })
    
    amount_stats.to_csv(output_dir / '03_amount_statistics.csv', index=False)
    
    # Text report
    mean_diff = fraud_mean - non_fraud_mean
    median_diff = fraud_median - non_fraud_median
    
    report_path = output_dir / '03_numerical_features_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FRAUD DETECTION - NUMERICAL FEATURES ANALYSIS (FULL DATASET)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. TRANSACTION AMOUNT ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(amount_stats.to_string(index=False) + "\n\n")
        
        f.write("2. KEY INSIGHTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean difference:   ${mean_diff:.2f} ")
        f.write("(fraud higher)\n" if mean_diff > 0 else "(fraud lower)\n")
        f.write(f"Median difference: ${median_diff:.2f} ")
        f.write("(fraud higher)\n" if median_diff > 0 else "(fraud lower)\n")
        f.write("\n")
        
        if abs(mean_diff) > 10:
            f.write("⚠️  Significant difference in transaction amounts!\n")
            f.write("   → Amount is likely a strong predictor for fraud\n")
        else:
            f.write("✓ Similar transaction amounts\n")
    
    # Visualization (using sampled data)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Histogram comparison
    ax1 = axes[0, 0]
    bins = np.linspace(0, min(non_fraud_max, fraud_max), 50)
    ax1.hist(non_fraud_values_arr, bins=bins, alpha=0.6, label='Non-Fraud', 
             color='#2ecc71', density=True)
    ax1.hist(fraud_values_arr, bins=bins, alpha=0.6, label='Fraud', 
             color='#e74c3c', density=True)
    ax1.set_xlabel('Amount ($)', fontweight='bold')
    ax1.set_ylabel('Density', fontweight='bold')
    ax1.set_title('Amount Distribution', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Box plot
    ax2 = axes[0, 1]
    bp = ax2.boxplot([non_fraud_values_arr, fraud_values_arr],
                     labels=['Non-Fraud', 'Fraud'],
                     patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_ylabel('Amount ($)', fontweight='bold')
    ax2.set_title('Box Plot Comparison', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Mean comparison
    ax3 = axes[1, 0]
    bars = ax3.bar(['Non-Fraud', 'Fraud'], [non_fraud_mean, fraud_mean],
                   color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Mean Amount ($)', fontweight='bold')
    ax3.set_title('Average Transaction Amount', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: CDF
    ax4 = axes[1, 1]
    sorted_nf = np.sort(non_fraud_values_arr)
    sorted_f = np.sort(fraud_values_arr)
    ax4.plot(sorted_nf, np.linspace(0, 1, len(sorted_nf)),
             label='Non-Fraud', color='#2ecc71', linewidth=2)
    ax4.plot(sorted_f, np.linspace(0, 1, len(sorted_f)),
             label='Fraud', color='#e74c3c', linewidth=2)
    ax4.set_xlabel('Amount ($)', fontweight='bold')
    ax4.set_ylabel('Cumulative Probability', fontweight='bold')
    ax4.set_title('Cumulative Distribution', fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '03_amount_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Console output
    print("💰 Transaction Amount Analysis:")
    print()
    print(amount_stats.to_string(index=False))
    print()
    print(f"   Mean difference:   ${mean_diff:.2f}")
    print(f"   Median difference: ${median_diff:.2f}")
    print()
    
    if abs(mean_diff) > 10:
        print("⚠️  Significant difference detected!")
        print("   → Amount is likely a strong predictor")
    
    end = time.time()
    
    print()
    print(f"💾 Reports saved to: {output_dir}")
    print(f"   - 03_amount_statistics.csv")
    print(f"   - 03_amount_analysis.png")
    print(f"   - 03_numerical_features_report.txt")
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
    
    print("✅ Numerical features analysis complete (full dataset)!")

