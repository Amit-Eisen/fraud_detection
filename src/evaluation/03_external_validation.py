
"""
External Validation on Unseen Data
===================================

This script validates the trained model on completely unseen data (~15M rows)
that were NOT used during training.

Strategy:
1. Load the 20M rows that were used for training (selected_features_20M.parquet)
2. Load the full 35M dataset (selected_features_top36.parquet)
3. Identify the ~15M rows NOT in training set
4. Run predictions on these unseen rows
5. Calculate performance metrics and compare with test set

This proves the model is robust and generalizes well!

Input:
- data/features/selected_features_top36.parquet (35M rows, 36 features)
- data/features/selected_features_20M.parquet (20M rows used for training)
- models/xgboost_optimized/xgboost_optimized.pkl
- models/xgboost_optimized/evaluation_results.json

Output:
- reports/evaluation/external_validation_report.json
- reports/evaluation/figures/external_validation_comparison.png
"""

import sys
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Set style
sns.set_style("whitegrid")


def load_training_sample_indices(training_parquet: Path) -> set:
    """
    Create a hash set of training data rows for fast lookup.
    
    Uses first 5 columns to create unique identifier.
    """
    print("📂 Loading training sample indices...")
    print(f"   Reading: {training_parquet.name}")
    
    start = time.time()
    
    df_train = pd.read_parquet(training_parquet)
    
    # Use first 5 columns to create hash (fast and unique enough)
    first_cols = df_train.columns[:5].tolist()
    df_train['_hash'] = pd.util.hash_pandas_object(df_train[first_cols], index=False)
    
    training_hashes = set(df_train['_hash'].values)
    
    elapsed = time.time() - start
    print(f"✓ Created hash index for {len(training_hashes):,} training rows")
    print(f"⏱️  Time: {elapsed:.1f} seconds")
    print()
    
    return training_hashes, first_cols


def extract_unseen_data(full_parquet: Path, training_hashes: set, 
                        hash_cols: list, target_size: int = 15_000_000) -> tuple:
    """
    Extract unseen data that was NOT in the training set.
    
    Uses chunked processing for memory efficiency.
    """
    print("📂 Extracting unseen data (NOT in training set)...")
    print(f"   Reading: {full_parquet.name}")
    print(f"   Target: ~{target_size:,} unseen rows")
    print()
    
    start = time.time()
    
    # Read in chunks
    parquet_file = pq.ParquetFile(full_parquet)
    total_rows = parquet_file.metadata.num_rows
    
    print(f"📊 Full dataset: {total_rows:,} rows")
    print(f"📊 Training set: {len(training_hashes):,} rows")
    print(f"📊 Expected unseen: ~{total_rows - len(training_hashes):,} rows")
    print()
    
    chunk_size = 2_000_000
    unseen_chunks = []
    total_unseen = 0
    total_processed = 0
    
    print("Processing chunks...")
    
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        df_chunk = batch.to_pandas()
        total_processed += len(df_chunk)
        
        # Create hash for this chunk
        df_chunk['_hash'] = pd.util.hash_pandas_object(df_chunk[hash_cols], index=False)
        
        # Filter to unseen rows only
        unseen_mask = ~df_chunk['_hash'].isin(training_hashes)
        df_unseen = df_chunk[unseen_mask].drop('_hash', axis=1).copy()
        
        if len(df_unseen) > 0:
            unseen_chunks.append(df_unseen)
            total_unseen += len(df_unseen)
        
        print(f"  Processed: {total_processed:,} / {total_rows:,} | "
              f"Unseen: {total_unseen:,}", end='\r')
        
        # Stop if we have enough
        if total_unseen >= target_size:
            print()
            print(f"✓ Collected {total_unseen:,} unseen rows (target reached)")
            break
    
    print()
    
    if not unseen_chunks:
        print("❌ ERROR: No unseen data found!")
        return None, None
    
    # Combine chunks
    print("Combining chunks...")
    df_unseen = pd.concat(unseen_chunks, ignore_index=True)
    
    # Sample down if needed (stratified)
    if len(df_unseen) > target_size:
        print(f"Sampling down to {target_size:,} rows (stratified)...")
        
        fraud_ratio = df_unseen['is_fraud'].mean()
        n_fraud = int(target_size * fraud_ratio)
        n_legit = target_size - n_fraud
        
        fraud_df = df_unseen[df_unseen['is_fraud'] == 1].sample(
            n=min(n_fraud, (df_unseen['is_fraud'] == 1).sum()), random_state=42
        )
        legit_df = df_unseen[df_unseen['is_fraud'] == 0].sample(
            n=min(n_legit, (df_unseen['is_fraud'] == 0).sum()), random_state=42
        )
        
        df_unseen = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Separate features and target
    X_unseen = df_unseen.drop('is_fraud', axis=1)
    y_unseen = df_unseen['is_fraud']
    
    elapsed = time.time() - start
    memory_gb = df_unseen.memory_usage(deep=True).sum() / 1e9
    
    print()
    print(f"✓ Extracted {len(X_unseen):,} unseen rows")
    print(f"✓ Features: {len(X_unseen.columns)}")
    print(f"✓ Fraud cases: {y_unseen.sum():,} ({y_unseen.mean():.2%})")
    print(f"⏱️  Time: {elapsed/60:.1f} minutes")
    print(f"💾 Memory: ~{memory_gb:.1f} GB")
    print()
    
    return X_unseen, y_unseen


def evaluate_on_unseen(model, X_unseen, y_unseen) -> dict:
    """Evaluate model on unseen data."""
    print("=" * 70)
    print("EVALUATING ON UNSEEN DATA")
    print("=" * 70)
    print()
    
    start = time.time()
    
    # Predictions
    print("Making predictions...")
    y_pred = model.predict(X_unseen)
    y_pred_proba = model.predict_proba(X_unseen)[:, 1]
    
    elapsed = time.time() - start
    print(f"✓ Predictions complete ({elapsed:.1f} seconds)")
    print()
    
    # Calculate metrics
    print("Calculating metrics...")
    
    accuracy = accuracy_score(y_unseen, y_pred)
    precision = precision_score(y_unseen, y_pred, zero_division=0)
    recall = recall_score(y_unseen, y_pred)
    f1 = f1_score(y_unseen, y_pred)
    roc_auc = roc_auc_score(y_unseen, y_pred_proba)
    
    cm = confusion_matrix(y_unseen, y_pred)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    # Business metrics
    total_fraud = y_unseen.sum()
    fraud_caught = tp
    fraud_missed = fn
    false_alarms = fp
    
    fraud_caught_pct = (fraud_caught / total_fraud * 100) if total_fraud > 0 else 0
    false_alarm_rate = (false_alarms / (tn + fp) * 100) if (tn + fp) > 0 else 0
    
    results = {
        'dataset_info': {
            'total_rows': int(len(y_unseen)),
            'fraud_cases': int(total_fraud),
            'fraud_rate': float(y_unseen.mean()),
            'legitimate_cases': int(len(y_unseen) - total_fraud)
        },
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc)
        },
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'business_metrics': {
            'fraud_caught': int(fraud_caught),
            'fraud_missed': int(fraud_missed),
            'fraud_detection_rate': float(fraud_caught_pct),
            'false_alarms': int(false_alarms),
            'false_alarm_rate': float(false_alarm_rate)
        }
    }
    
    # Print results
    print()
    print("=" * 70)
    print("EXTERNAL VALIDATION RESULTS")
    print("=" * 70)
    print()
    print("📊 Dataset:")
    print(f"   Total rows:      {len(y_unseen):,}")
    print(f"   Fraud cases:     {total_fraud:,} ({y_unseen.mean():.2%})")
    print(f"   Legitimate:      {len(y_unseen) - total_fraud:,}")
    print()
    print("📈 Performance Metrics:")
    print(f"   ROC-AUC:         {roc_auc:.4f}")
    print(f"   Accuracy:        {accuracy:.2%}")
    print(f"   Precision:       {precision:.2%}")
    print(f"   Recall:          {recall:.2%}")
    print(f"   F1-Score:        {f1:.4f}")
    print()
    print("🎯 Business Metrics:")
    print(f"   Fraud Caught:    {fraud_caught:,} / {total_fraud:,} ({fraud_caught_pct:.2f}%)")
    print(f"   Fraud Missed:    {fraud_missed:,}")
    print(f"   False Alarms:    {false_alarms:,}")
    print(f"   False Alarm Rate: {false_alarm_rate:.2f}%")
    print()
    
    return results


def compare_with_test_set(external_results: dict, test_results: dict, output_path: Path):
    """Create comparison visualization."""
    print("📊 Creating comparison visualization...")
    
    # Extract metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    test_values = [test_results[m] if m in test_results else test_results.get('f1', 0) 
                   for m in metrics]
    # Handle f1 vs f1_score
    if 'f1_score' in test_results:
        test_values[3] = test_results['f1_score']
    elif 'f1' in test_results:
        test_values[3] = test_results['f1']
        
    external_values = [external_results['metrics'][m] for m in metrics]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, test_values, width, label='Test Set (20M sample)', 
                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, external_values, width, label='External Validation (~15M unseen)', 
                   color='darkorange', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Formatting
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance Comparison\nTest Set vs External Validation (Unseen Data)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=12, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.85, 1.02])
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add interpretation box
    avg_diff = np.mean(np.abs(np.array(test_values) - np.array(external_values)))
    
    interpretation = (
        f"Average Difference: {avg_diff:.4f}\n\n"
        f"Interpretation:\n"
    )
    
    if avg_diff < 0.005:
        interpretation += "✅ Excellent! Nearly identical\nperformance on unseen data.\n\nModel generalizes perfectly!"
        box_color = 'lightgreen'
    elif avg_diff < 0.01:
        interpretation += "✅ Very Good! Minimal differences.\n\nModel is robust and reliable."
        box_color = 'lightyellow'
    elif avg_diff < 0.02:
        interpretation += "⚠️  Acceptable. Some performance\ndrop on unseen data.\n\nModel is reasonably robust."
        box_color = 'lightyellow'
    else:
        interpretation += "❌ Significant difference!\n\nModel may be overfitted.\nConsider retraining."
        box_color = 'lightcoral'
    
    ax.text(0.02, 0.98, interpretation,
           transform=ax.transAxes, fontsize=11, fontweight='bold',
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8, edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def main():
    """Main execution function."""
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print("EXTERNAL VALIDATION ON UNSEEN DATA")
    print("=" * 70)
    print()
    print("🎯 Goal: Validate model on ~15M rows NOT used for training")
    print("   This proves the model generalizes well to new data!")
    print()
    
    # Setup paths
    project_root = Path(__file__).resolve().parents[2]
    full_parquet = project_root / 'data' / 'features' / 'selected_features_top36.parquet'
    training_parquet = project_root / 'data' / 'features' / 'selected_features_20M.parquet'
    model_path = project_root / 'models' / 'xgboost_optimized' / 'xgboost_optimized.pkl'
    test_results_path = project_root / 'models' / 'xgboost_optimized' / 'evaluation_results.json'
    output_dir = project_root / 'reports' / 'evaluation'
    
    # Check files
    if not full_parquet.exists():
        print(f"❌ ERROR: Full dataset not found: {full_parquet}")
        return
    
    if not training_parquet.exists():
        print(f"❌ ERROR: Training dataset not found: {training_parquet}")
        return
    
    if not model_path.exists():
        print(f"❌ ERROR: Model not found: {model_path}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load training indices
    training_hashes, hash_cols = load_training_sample_indices(training_parquet)
    
    # 2. Load model
    print("📂 Loading model...")
    model = joblib.load(model_path)
    print(f"✓ Model loaded")
    print()
    
    # 3. Extract unseen data
    X_unseen, y_unseen = extract_unseen_data(
        full_parquet, training_hashes, hash_cols, target_size=15_000_000
    )
    
    if X_unseen is None:
        return
    
    # 4. Evaluate on unseen data
    external_results = evaluate_on_unseen(model, X_unseen, y_unseen)
    
    # 5. Load test results for comparison
    print("📂 Loading test set results...")
    with open(test_results_path, 'r') as f:
        test_results = json.load(f)
    print("✓ Loaded test set results")
    print()
    
    # 6. Create comparison
    compare_with_test_set(
        external_results, test_results,
        output_dir / 'figures' / 'external_validation_comparison.png'
    )
    
    # 7. Save results
    output_json = output_dir / 'external_validation_report.json'
    with open(output_json, 'w') as f:
        json.dump(external_results, f, indent=2)
    
    print(f"✓ Saved report: {output_json}")
    print()
    
    # Summary
    elapsed = time.time() - start_time
    print("=" * 70)
    print("✅ EXTERNAL VALIDATION COMPLETE!")
    print("=" * 70)
    print()
    print(f"⏱️  Total time: {elapsed/60:.1f} minutes")
    print()
    print("📁 Output files:")
    print(f"   {output_json}")
    print(f"   {output_dir / 'figures' / 'external_validation_comparison.png'}")
    print()
    print("🎉 Model validation on unseen data is complete!")
    print()


if __name__ == "__main__":
    main()

