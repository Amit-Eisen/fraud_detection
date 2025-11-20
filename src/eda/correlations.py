# src/eda/correlations.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path


def run(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Analyze correlations between features and fraud.
    
    Key insights:
    - Which numerical features correlate with fraud?
    - Feature importance for modeling
    - Multicollinearity detection
    
    Args:
        df: Input DataFrame with numerical features and 'is_fraud'
        output_dir: Directory to save reports and plots
    
    Returns:
        Same DataFrame
    """
    start = time.time()
    
    print(f"🔗 Starting correlation analysis")
    print(f"   Dataset: {df.shape[0]:,} rows")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check required columns
    if 'is_fraud' not in df.columns:
        print("❌ Error: 'is_fraud' column not found!")
        return df
    
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if 'is_fraud' not in numerical_cols:
        print("❌ Error: 'is_fraud' is not numerical!")
        return df
    
    print(f"   Analyzing {len(numerical_cols)} numerical features")
    print()
    
    # --- 1. Correlation with fraud ---
    fraud_corr = df[numerical_cols].corr()['is_fraud'].sort_values(ascending=False)
    fraud_corr = fraud_corr.drop('is_fraud')  # Remove self-correlation
    
    fraud_corr_df = pd.DataFrame({
        'feature': fraud_corr.index,
        'correlation': fraud_corr.values
    })
    fraud_corr_df['abs_correlation'] = fraud_corr_df['correlation'].abs()
    fraud_corr_df = fraud_corr_df.sort_values('abs_correlation', ascending=False)
    
    fraud_corr_df.to_csv(output_dir / '06_fraud_correlations.csv', index=False)
    
    # --- 2. Full correlation matrix ---
    corr_matrix = df[numerical_cols].corr()
    corr_matrix.to_csv(output_dir / '06_correlation_matrix.csv')
    
    # --- 3. Text Report ---
    report_path = output_dir / '06_correlation_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FRAUD DETECTION - CORRELATION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. FEATURES MOST CORRELATED WITH FRAUD\n")
        f.write("-" * 80 + "\n")
        f.write(fraud_corr_df.head(15).to_string(index=False) + "\n\n")
        
        f.write("2. KEY INSIGHTS\n")
        f.write("-" * 80 + "\n")
        
        top_positive = fraud_corr_df[fraud_corr_df['correlation'] > 0].head(3)
        top_negative = fraud_corr_df[fraud_corr_df['correlation'] < 0].head(3)
        
        if len(top_positive) > 0:
            f.write("Strongest POSITIVE correlations (higher value → more fraud):\n")
            for _, row in top_positive.iterrows():
                f.write(f"  • {row['feature']}: {row['correlation']:.4f}\n")
            f.write("\n")
        
        if len(top_negative) > 0:
            f.write("Strongest NEGATIVE correlations (higher value → less fraud):\n")
            for _, row in top_negative.iterrows():
                f.write(f"  • {row['feature']}: {row['correlation']:.4f}\n")
            f.write("\n")
        
        # Multicollinearity check
        f.write("3. MULTICOLLINEARITY CHECK\n")
        f.write("-" * 80 + "\n")
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            f.write("⚠️  High correlations detected (>0.8):\n")
            for pair in high_corr_pairs[:10]:
                f.write(f"  • {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}\n")
            f.write("\n→ Consider removing one feature from each pair to reduce multicollinearity\n\n")
        else:
            f.write("✅ No severe multicollinearity detected (all correlations < 0.8)\n\n")
        
        f.write("4. FEATURE SELECTION RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        
        strong_features = fraud_corr_df[fraud_corr_df['abs_correlation'] > 0.05]
        weak_features = fraud_corr_df[fraud_corr_df['abs_correlation'] < 0.01]
        
        f.write(f"Strong predictors (|corr| > 0.05): {len(strong_features)} features\n")
        if len(strong_features) > 0:
            f.write("  → " + ", ".join(strong_features['feature'].head(10).tolist()) + "\n\n")
        
        f.write(f"Weak predictors (|corr| < 0.01): {len(weak_features)} features\n")
        if len(weak_features) > 0:
            f.write("  → Consider removing: " + ", ".join(weak_features['feature'].head(10).tolist()) + "\n")
    
    # --- 4. Visualizations ---
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Top correlations with fraud (bar chart)
    ax1 = fig.add_subplot(gs[0, :])
    
    top_features = fraud_corr_df.head(20)
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in top_features['correlation']]
    
    bars = ax1.barh(range(len(top_features)), top_features['correlation'],
                    color=colors, alpha=0.7, edgecolor='black')
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'], fontsize=9)
    ax1.set_xlabel('Correlation with Fraud', fontsize=12, fontweight='bold')
    ax1.set_title('Top 20 Features Correlated with Fraud', fontsize=14, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.4f}',
                ha='left' if width > 0 else 'right',
                va='center', fontsize=8, fontweight='bold')
    
    # Plot 2: Correlation heatmap (top features only)
    ax2 = fig.add_subplot(gs[1:, 0])
    
    # Select top correlated features + is_fraud
    top_n = min(15, len(fraud_corr_df))
    top_feature_names = fraud_corr_df.head(top_n)['feature'].tolist() + ['is_fraud']
    top_corr_matrix = df[top_feature_names].corr()
    
    sns.heatmap(top_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax2, vmin=-1, vmax=1)
    ax2.set_title(f'Correlation Heatmap (Top {top_n} Features)', 
                  fontsize=13, fontweight='bold')
    
    # Plot 3: Distribution of correlations
    ax3 = fig.add_subplot(gs[1, 1])
    
    ax3.hist(fraud_corr_df['correlation'], bins=30, color='#9b59b6',
            alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero correlation')
    ax3.set_xlabel('Correlation with Fraud', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of Features', fontsize=11, fontweight='bold')
    ax3.set_title('Distribution of Feature Correlations', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: Absolute correlation ranking
    ax4 = fig.add_subplot(gs[2, 1])
    
    top_abs = fraud_corr_df.head(15)
    bars = ax4.barh(range(len(top_abs)), top_abs['abs_correlation'],
                    color='#f39c12', alpha=0.7, edgecolor='black')
    ax4.set_yticks(range(len(top_abs)))
    ax4.set_yticklabels(top_abs['feature'], fontsize=9)
    ax4.set_xlabel('Absolute Correlation', fontsize=11, fontweight='bold')
    ax4.set_title('Feature Importance (by Absolute Correlation)', 
                  fontsize=12, fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.4f}',
                ha='left', va='center', fontsize=8, fontweight='bold')
    
    plt.savefig(output_dir / '06_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Console Output ---
    print("🔗 Correlation Analysis Results:")
    print()
    print("=" * 80)
    print("TOP 15 FEATURES CORRELATED WITH FRAUD")
    print("=" * 80)
    print(fraud_corr_df.head(15).to_string(index=False))
    print()
    
    print("📊 KEY INSIGHTS:")
    print("-" * 80)
    
    top_positive = fraud_corr_df[fraud_corr_df['correlation'] > 0].head(3)
    if len(top_positive) > 0:
        print("Strongest POSITIVE correlations:")
        for _, row in top_positive.iterrows():
            print(f"  • {row['feature']}: {row['correlation']:.4f}")
        print()
    
    top_negative = fraud_corr_df[fraud_corr_df['correlation'] < 0].head(3)
    if len(top_negative) > 0:
        print("Strongest NEGATIVE correlations:")
        for _, row in top_negative.iterrows():
            print(f"  • {row['feature']}: {row['correlation']:.4f}")
        print()
    
    # Multicollinearity
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append((corr_matrix.columns[i], 
                                       corr_matrix.columns[j], 
                                       corr_matrix.iloc[i, j]))
    
    if high_corr_pairs:
        print("⚠️  Multicollinearity detected:")
        for feat1, feat2, corr in high_corr_pairs[:5]:
            print(f"  • {feat1} ↔ {feat2}: {corr:.3f}")
        print()
    else:
        print("✅ No severe multicollinearity detected")
        print()
    
    print("=" * 80)
    print(f"💾 Reports saved to: {output_dir}")
    print(f"   - 06_fraud_correlations.csv")
    print(f"   - 06_correlation_matrix.csv")
    print(f"   - 06_correlation_analysis.png")
    print(f"   - 06_correlation_analysis_report.txt")
    
    end = time.time()
    print(f"\n⏱️  Runtime: {end - start:.2f} seconds\n")
    
    return df


def run_chunked(input_path: Path, output_dir: Path, sample_size: int = 1_000_000):
    """
    Analyze correlations on large CSV files using sampling.
    
    Note: For correlation analysis, we use a representative sample
    rather than full chunked processing, as correlations are stable
    with large samples.
    
    Args:
        input_path: Path to input CSV file
        output_dir: Directory to save reports
        sample_size: Number of rows to sample for analysis
    """
    start = time.time()
    
    print(f"🔗 Starting correlation analysis (sampled)")
    print(f"   Input: {input_path}")
    print(f"   Sample size: {sample_size:,} rows")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sample
    print("  Loading sample...")
    df = pd.read_csv(input_path, nrows=sample_size, low_memory=False)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print()
    
    # Run analysis on sample
    df = run(df, output_dir)
    
    print("✅ Correlation analysis complete!")
    print(f"💡 Note: Analysis based on {sample_size:,} row sample")
    print(f"   (Correlations are stable with large samples)")


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
    
    # Use random sampling for correlation analysis
    print("  Loading random sample (1M rows)...")
    
    # Count total rows first
    total_rows = sum(1 for _ in open(input_file)) - 1  # -1 for header
    print(f"  Total rows in dataset: {total_rows:,}")
    
    # Random sample
    sample_size = 1_000_000
    skip_prob = 1 - (sample_size / total_rows)
    
    df = pd.read_csv(input_file, 
                     skiprows=lambda i: i > 0 and np.random.random() > (sample_size / total_rows),
                     low_memory=False)
    
    print(f"  Loaded random sample: {len(df):,} rows")
    print()
    
    # Run analysis
    df = run(df, reports_dir)
    
    print("✅ Correlation analysis complete!")
    print("💡 Note: Analysis based on 1M random sample (statistically representative)")

