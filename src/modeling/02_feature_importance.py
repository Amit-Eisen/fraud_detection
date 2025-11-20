#!/usr/bin/env python3
"""
Feature Importance Analysis
============================

Extracts and analyzes feature importance from trained models.

This script:
1. Loads trained baseline models (Random Forest, XGBoost, LightGBM)
2. Extracts feature importance from each model
3. Creates comprehensive reports with ALL features ranked
4. Generates visualizations for top features
5. Provides insights for feature selection

"""

import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def load_models(models_dir: Path) -> dict:
    """
    Load trained models from disk.
    
    Args:
        models_dir: Directory containing saved models
    
    Returns:
        Dictionary of {model_name: model_object}
    """
    print("=" * 70)
    print("LOADING TRAINED MODELS")
    print("=" * 70)
    print(f"📂 Models directory: {models_dir}")
    print()
    
    models = {}
    model_files = {
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl',
        'LightGBM': 'lightgbm.pkl'
    }
    
    for name, filename in model_files.items():
        filepath = models_dir / filename
        if filepath.exists():
            models[name] = joblib.load(filepath)
            print(f"✓ Loaded: {name}")
        else:
            print(f"⚠️  Not found: {name} ({filename})")
    
    print()
    print(f"✓ Loaded {len(models)} models")
    print()
    
    return models


def load_feature_names(models_dir: Path) -> list:
    """
    Load feature names from training split info.
    
    Args:
        models_dir: Directory containing saved models
    
    Returns:
        List of feature names
    """
    print("=" * 70)
    print("LOADING FEATURE NAMES")
    print("=" * 70)
    
    split_info_path = models_dir / 'train_test_split_info.pkl'
    
    if not split_info_path.exists():
        print(f"❌ ERROR: Split info not found!")
        print(f"   Expected: {split_info_path}")
        return []
    
    split_info = joblib.load(split_info_path)
    feature_names = split_info.get('feature_names', [])
    
    print(f"✓ Loaded {len(feature_names)} feature names")
    print()
    
    return feature_names


def extract_feature_importance(models: dict, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importance from all models.
    
    Args:
        models: Dictionary of trained models
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importance from all models
    """
    print("=" * 70)
    print("EXTRACTING FEATURE IMPORTANCE")
    print("=" * 70)
    print()
    
    importance_data = {'feature': feature_names}
    
    for model_name, model in models.items():
        print(f"📊 Extracting from {model_name}...")
        
        # Extract importance based on model type
        if hasattr(model, 'feature_importances_'):
            # Random Forest, XGBoost, LightGBM all have this attribute
            importance = model.feature_importances_
        else:
            print(f"⚠️  {model_name} doesn't have feature_importances_")
            importance = np.zeros(len(feature_names))
        
        # Normalize to percentages (0-100)
        importance_pct = (importance / importance.sum()) * 100
        
        importance_data[model_name] = importance_pct
        
        print(f"   ✓ Extracted {len(importance)} importance values")
        print(f"   ✓ Top feature: {feature_names[np.argmax(importance)]} ({importance_pct.max():.2f}%)")
    
    print()
    
    # Create DataFrame
    df = pd.DataFrame(importance_data)
    
    # Add average importance across all models
    model_cols = [col for col in df.columns if col != 'feature']
    df['Average'] = df[model_cols].mean(axis=1)
    
    # Sort by average importance (descending)
    df = df.sort_values('Average', ascending=False).reset_index(drop=True)
    
    # Add rank
    df.insert(0, 'rank', range(1, len(df) + 1))
    
    print(f"✓ Created importance DataFrame with {len(df)} features")
    print()
    
    return df


def save_importance_report(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save feature importance report to CSV.
    
    Args:
        df: DataFrame with feature importance
        output_dir: Directory to save report
    """
    print("=" * 70)
    print("SAVING FEATURE IMPORTANCE REPORT")
    print("=" * 70)
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full report (ALL features)
    csv_path = output_dir / 'feature_importance_all.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✓ Saved full report: {csv_path}")
    print(f"   Contains ALL {len(df)} features ranked by importance")
    print()
    
    # Save top 50 features (for quick reference)
    top50_path = output_dir / 'feature_importance_top50.csv'
    df.head(50).to_csv(top50_path, index=False, float_format='%.4f')
    print(f"✓ Saved top 50 features: {top50_path}")
    print()
    
    # Print summary statistics
    print("📊 IMPORTANCE STATISTICS:")
    print("-" * 70)
    print(f"Total features: {len(df)}")
    print(f"Top 10 features account for: {df.head(10)['Average'].sum():.2f}% of importance")
    print(f"Top 20 features account for: {df.head(20)['Average'].sum():.2f}% of importance")
    print(f"Top 50 features account for: {df.head(50)['Average'].sum():.2f}% of importance")
    print()
    
    # Features with <0.1% importance (candidates for removal)
    low_importance = df[df['Average'] < 0.1]
    print(f"⚠️  Features with <0.1% importance: {len(low_importance)} ({len(low_importance)/len(df)*100:.1f}%)")
    print(f"   These are candidates for removal in feature selection!")
    print()


def plot_feature_importance(df: pd.DataFrame, output_dir: Path, top_n: int = 20) -> None:
    """
    Create visualizations of feature importance.
    
    Args:
        df: DataFrame with feature importance
        output_dir: Directory to save plots
        top_n: Number of top features to plot
    """
    print("=" * 70)
    print(f"CREATING VISUALIZATIONS (TOP {top_n} FEATURES)")
    print("=" * 70)
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get top N features
    df_top = df.head(top_n)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Plot 1: Comparison of all models (side by side)
    print(f"📊 Creating comparison plot...")
    fig, ax = plt.subplots(figsize=(14, 10))
    
    model_cols = ['Random Forest', 'XGBoost', 'LightGBM']
    x = np.arange(len(df_top))
    width = 0.25
    
    for i, model in enumerate(model_cols):
        if model in df_top.columns:
            ax.barh(x + i * width, df_top[model], width, label=model, alpha=0.8)
    
    ax.set_yticks(x + width)
    ax.set_yticklabels(df_top['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (%)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Top {top_n} Features - Comparison Across Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'feature_importance_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {plot_path}")
    
    # Plot 2: Average importance (single plot)
    print(f"📊 Creating average importance plot...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_top)))
    ax.barh(df_top['feature'], df_top['Average'], color=colors, alpha=0.8)
    ax.invert_yaxis()
    ax.set_xlabel('Average Importance (%)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Top {top_n} Features - Average Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (feature, importance) in enumerate(zip(df_top['feature'], df_top['Average'])):
        ax.text(importance + 0.1, i, f'{importance:.2f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plot_path = output_dir / 'feature_importance_average.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {plot_path}")
    
    # Plot 3: Cumulative importance
    print(f"📊 Creating cumulative importance plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cumulative = df['Average'].cumsum()
    ax.plot(range(1, len(cumulative) + 1), cumulative, linewidth=2, color='#2E86AB')
    ax.axhline(y=80, color='red', linestyle='--', linewidth=1, label='80% threshold')
    ax.axhline(y=90, color='orange', linestyle='--', linewidth=1, label='90% threshold')
    ax.axhline(y=95, color='green', linestyle='--', linewidth=1, label='95% threshold')
    
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Cumulative Importance (%)', fontsize=12)
    ax.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Find how many features needed for 80%, 90%, 95%
    n_80 = (cumulative >= 80).idxmax() + 1
    n_90 = (cumulative >= 90).idxmax() + 1
    n_95 = (cumulative >= 95).idxmax() + 1
    
    ax.annotate(f'{n_80} features', xy=(n_80, 80), xytext=(n_80 + 5, 75),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10)
    ax.annotate(f'{n_90} features', xy=(n_90, 90), xytext=(n_90 + 5, 85),
                arrowprops=dict(arrowstyle='->', color='orange'), fontsize=10)
    ax.annotate(f'{n_95} features', xy=(n_95, 95), xytext=(n_95 + 5, 92),
                arrowprops=dict(arrowstyle='->', color='green'), fontsize=10)
    
    plt.tight_layout()
    plot_path = output_dir / 'feature_importance_cumulative.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {plot_path}")
    
    print()
    print("📈 CUMULATIVE IMPORTANCE INSIGHTS:")
    print("-" * 70)
    print(f"To capture 80% of importance: keep top {n_80} features (drop {len(df) - n_80})")
    print(f"To capture 90% of importance: keep top {n_90} features (drop {len(df) - n_90})")
    print(f"To capture 95% of importance: keep top {n_95} features (drop {len(df) - n_95})")
    print()


def print_top_features(df: pd.DataFrame, n: int = 20) -> None:
    """
    Print top N features to console.
    
    Args:
        df: DataFrame with feature importance
        n: Number of top features to print
    """
    print("=" * 70)
    print(f"TOP {n} MOST IMPORTANT FEATURES")
    print("=" * 70)
    print()
    
    print(f"{'Rank':<6} {'Feature':<30} {'RF %':>8} {'XGB %':>8} {'LGB %':>8} {'Avg %':>8}")
    print("-" * 70)
    
    for idx, row in df.head(n).iterrows():
        print(f"{row['rank']:<6} {row['feature']:<30} "
              f"{row.get('Random Forest', 0):>8.2f} "
              f"{row.get('XGBoost', 0):>8.2f} "
              f"{row.get('LightGBM', 0):>8.2f} "
              f"{row['Average']:>8.2f}")
    
    print()


def main():
    """Main execution function."""
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    print()
    
    # Setup paths
    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / 'models'
    reports_dir = project_root / 'reports' / 'feature_importance'
    
    # Check if models exist
    if not models_dir.exists():
        print(f"❌ ERROR: Models directory not found!")
        print(f"   Expected: {models_dir}")
        print()
        print("💡 Make sure you've run:")
        print("   python src/modeling/01_train_baseline.py")
        return
    
    # 1. Load models
    models = load_models(models_dir)
    
    if not models:
        print("❌ ERROR: No models found!")
        return
    
    # 2. Load feature names
    feature_names = load_feature_names(models_dir)
    
    if not feature_names:
        print("❌ ERROR: No feature names found!")
        return
    
    # 3. Extract feature importance
    importance_df = extract_feature_importance(models, feature_names)
    
    # 4. Save reports
    save_importance_report(importance_df, reports_dir)
    
    # 5. Create visualizations
    plot_feature_importance(importance_df, reports_dir, top_n=20)
    
    # 6. Print top features
    print_top_features(importance_df, n=20)
    
    # Summary
    elapsed = time.time() - start_time
    print("=" * 70)
    print("✅ FEATURE IMPORTANCE ANALYSIS COMPLETE!")
    print("=" * 70)
    print()
    print(f"⏱️  Total time: {elapsed:.1f} seconds")
    print()
    print("📁 Output files:")
    print(f"   Reports: {reports_dir}")
    print(f"   - feature_importance_all.csv (ALL {len(importance_df)} features)")
    print(f"   - feature_importance_top50.csv (top 50 features)")
    print(f"   - feature_importance_comparison.png")
    print(f"   - feature_importance_average.png")
    print(f"   - feature_importance_cumulative.png")
    print()
    print("💡 Next step: Use this data for feature selection!")
    print("   Run: python src/modeling/03_feature_selection.py")
    print()


if __name__ == "__main__":
    main()

