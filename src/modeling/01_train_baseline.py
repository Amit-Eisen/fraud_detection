#!/usr/bin/env python3
"""
Train Baseline Models for Fraud Detection
==========================================

Trains 3 baseline models and compares their performance:
1. Random Forest - Robust ensemble method
2. XGBoost - State-of-the-art gradient boosting
3. LightGBM - Fast and memory-efficient gradient boosting

All models use class_weight/scale_pos_weight to handle class imbalance.

Usage:
    python 01_train_baseline.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score
)
import xgboost as xgb
import lightgbm as lgb


def load_data(parquet_path: Path, sample_size: int = None) -> tuple:
    """
    Load feature-engineered data from Parquet file.
    
    Args:
        parquet_path: Path to the Parquet file
        sample_size: Number of rows to sample with STRATIFICATION (maintains fraud ratio!)
                     None = load all 35M rows (requires 32GB RAM)
                     15_000_000 = recommended for 16GB RAM (43% of data) ✅
                     10_000_000 = very safe for 16GB RAM (29% of data)
    
    Returns:
        X: Features (DataFrame)
        y: Target (Series)
        feature_names: List of feature column names
    """
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    print(f"📂 Loading: {parquet_path}")
    print()
    
    start_time = time.time()
    
    # STRATEGY: For sampling, use PyArrow to load in batches and sample on-the-fly
    if sample_size:
        print(f"🎯 Stratified sampling: {sample_size:,} rows")
        print(f"⚠️  Using PyArrow batch loading to avoid 11GB memory spike...")
        print()
        
        import pyarrow.parquet as pq
        
        # Step 1: Get total rows and fraud rate (fast metadata read)
        parquet_file = pq.ParquetFile(parquet_path)
        total_rows = parquet_file.metadata.num_rows
        
        print(f"✓ Total rows in dataset: {total_rows:,}")
        print(f"🔍 Scanning for fraud rate (reading in batches)...")
        
        # Read in batches to get fraud indices without loading all data
        fraud_indices = []
        legit_indices = []
        batch_size = 1_000_000
        
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
            batch_df = batch.to_pandas()
            batch_start = batch_idx * batch_size
            
            # Get indices for this batch
            fraud_mask = batch_df['is_fraud'] == 1
            fraud_indices.extend((batch_start + batch_df[fraud_mask].index).tolist())
            legit_indices.extend((batch_start + batch_df[~fraud_mask].index).tolist())
            
            if (batch_idx + 1) % 5 == 0:
                print(f"   Processed {(batch_idx + 1) * batch_size:,} rows...")
        
        fraud_rate_original = len(fraud_indices) / total_rows
        print(f"✓ Original fraud rate: {fraud_rate_original:.4%}")
        print(f"✓ Found {len(fraud_indices):,} fraud cases, {len(legit_indices):,} legit cases")
        print()
        
        # Step 2: Calculate stratified sample
        print(f"🔍 Calculating stratified sample...")
        n_fraud = int(sample_size * fraud_rate_original)
        n_legit = sample_size - n_fraud
        
        import random
        random.seed(42)
        sampled_fraud = set(random.sample(fraud_indices, min(n_fraud, len(fraud_indices))))
        sampled_legit = set(random.sample(legit_indices, min(n_legit, len(legit_indices))))
        sampled_indices_set = sampled_fraud | sampled_legit
        
        print(f"✓ Sampled {len(sampled_fraud):,} fraud cases")
        print(f"✓ Sampled {len(sampled_legit):,} legit cases")
        print(f"✓ Total: {len(sampled_indices_set):,} rows ({len(sampled_indices_set)/total_rows*100:.1f}% of data)")
        print()
        
        # Step 3: Load ONLY sampled rows (batch by batch)
        print(f"📥 Loading sampled rows from Parquet (batch by batch)...")
        sampled_rows = []
        
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
            batch_df = batch.to_pandas()
            batch_start = batch_idx * batch_size
            
            # Filter rows that are in our sample
            batch_indices = set(range(batch_start, batch_start + len(batch_df)))
            keep_indices = batch_indices & sampled_indices_set
            
            if keep_indices:
                # Get local indices within this batch
                local_indices = [idx - batch_start for idx in keep_indices]
                sampled_rows.append(batch_df.iloc[local_indices])
            
            if (batch_idx + 1) % 5 == 0:
                print(f"   Processed {(batch_idx + 1) * batch_size:,} rows, collected {sum(len(r) for r in sampled_rows):,} samples...")
        
        # Combine all sampled rows
        df = pd.concat(sampled_rows, ignore_index=True)
        
        elapsed = time.time() - start_time
        print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        print(f"⏱️  Time: {elapsed:.1f} seconds")
        print(f"💾 Memory: ~{df.memory_usage(deep=True).sum() / 1024**3:.1f} GB")
        print()
        
        # Verify fraud ratio
        fraud_rate_sampled = df['is_fraud'].mean()
        print(f"✓ Fraud ratio in sample: {fraud_rate_sampled:.4%}")
        print(f"✓ Difference from original: {abs(fraud_rate_original - fraud_rate_sampled):.6%}")
        print()
        
    else:
        # Load full dataset (requires 32GB RAM!)
        print(f"⚠️  Loading FULL dataset (35M rows, ~11GB RAM)...")
        print()
        df = pd.read_parquet(parquet_path)
        
        elapsed = time.time() - start_time
        print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        print(f"⏱️  Time: {elapsed:.1f} seconds")
        print(f"💾 Memory: ~{df.memory_usage(deep=True).sum() / 1024**3:.1f} GB")
        print()
    
    # Separate features and target
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    # Check fraud rate
    fraud_rate = y.mean()
    print(f"🎯 Target variable: is_fraud")
    print(f"   Fraud cases: {y.sum():,} ({fraud_rate:.2%})")
    print(f"   Legit cases: {(~y.astype(bool)).sum():,} ({1-fraud_rate:.2%})")
    print(f"   Class imbalance ratio: 1:{int(1/fraud_rate)}")
    print()
    
    return X, y, X.columns.tolist()


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into train and test sets with stratification.
    
    Stratification ensures both sets have the same fraud ratio.
    
    Args:
        X: Features
        y: Target
        test_size: Fraction of data for testing (default: 0.2 = 20%)
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("=" * 70)
    print("SPLITTING DATA")
    print("=" * 70)
    print(f"📊 Split ratio: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
    print(f"🎲 Random state: {random_state} (for reproducibility)")
    print()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,  # CRITICAL: Maintains fraud ratio in both sets!
        random_state=random_state
    )
    
    # Verify split
    print(f"✓ Train set: {len(X_train):,} samples")
    print(f"   Fraud: {y_train.sum():,} ({y_train.mean():.2%})")
    print()
    print(f"✓ Test set:  {len(X_test):,} samples")
    print(f"   Fraud: {y_test.sum():,} ({y_test.mean():.2%})")
    print()
    
    # Sanity check
    if abs(y_train.mean() - y_test.mean()) > 0.001:
        print("⚠️  WARNING: Fraud rates differ between train and test!")
    else:
        print("✅ Fraud rates match perfectly! Stratification worked.")
    print()
    
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, random_state: int = 42):
    """
    Train Random Forest classifier.
    
    Uses class_weight='balanced' to handle class imbalance.
    
    Why Random Forest?
    - Robust to outliers
    - No feature scaling needed
    - Built-in feature importance
    - Good baseline performance
    """
    print("=" * 70)
    print("MODEL 1: RANDOM FOREST")
    print("=" * 70)
    print()
    
    print("🌲 Initializing Random Forest...")
    print("   Parameters:")
    print("   - n_estimators: 100 (number of trees)")
    print("   - max_depth: 20 (prevent overfitting)")
    print("   - min_samples_split: 100 (require 100 samples to split)")
    print("   - class_weight: 'balanced' (handle imbalance)")
    print("   - n_jobs: -1 (use all CPU cores)")
    print()
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=100,
        min_samples_leaf=50,
        class_weight='balanced',  # Handles imbalance!
        random_state=random_state,
        n_jobs=-1,  # Use all cores
        verbose=0
    )
    
    print("⏳ Training Random Forest...")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    print(f"✓ Training complete!")
    print(f"⏱️  Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print()
    
    return model


def train_xgboost(X_train, y_train, random_state: int = 42):
    """
    Train XGBoost classifier.
    
    Uses scale_pos_weight to handle class imbalance.
    
    Why XGBoost?
    - State-of-the-art performance
    - Fast training
    - Excellent for fraud detection
    - Built-in regularization
    """
    print("=" * 70)
    print("MODEL 2: XGBOOST")
    print("=" * 70)
    print()
    
    # Calculate scale_pos_weight for imbalance
    # Formula: (number of negative samples) / (number of positive samples)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    print("🚀 Initializing XGBoost...")
    print("   Parameters:")
    print("   - n_estimators: 100 (number of boosting rounds)")
    print("   - max_depth: 6 (tree depth)")
    print("   - learning_rate: 0.1 (step size)")
    print(f"   - scale_pos_weight: {scale_pos_weight:.1f} (handle imbalance)")
    print("   - eval_metric: 'logloss' (binary classification)")
    print()
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,  # Handles imbalance!
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=-1,
        verbosity=0
    )
    
    print("⏳ Training XGBoost...")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    print(f"✓ Training complete!")
    print(f"⏱️  Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print()
    
    return model


def train_lightgbm(X_train, y_train, random_state: int = 42):
    """
    Train LightGBM classifier.
    
    Uses scale_pos_weight to handle class imbalance.
    
    Why LightGBM?
    - Faster than XGBoost
    - More memory efficient
    - Excellent for large datasets
    - Similar/better performance than XGBoost
    """
    print("=" * 70)
    print("MODEL 3: LIGHTGBM")
    print("=" * 70)
    print()
    
    # Calculate scale_pos_weight for imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    print("⚡ Initializing LightGBM...")
    print("   Parameters:")
    print("   - n_estimators: 100 (number of boosting rounds)")
    print("   - max_depth: 6 (tree depth)")
    print("   - learning_rate: 0.1 (step size)")
    print(f"   - scale_pos_weight: {scale_pos_weight:.1f} (handle imbalance)")
    print("   - num_leaves: 31 (max leaves per tree)")
    print()
    
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,  # Handles imbalance!
        num_leaves=31,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1  # Suppress warnings
    )
    
    print("⏳ Training LightGBM...")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    print(f"✓ Training complete!")
    print(f"⏱️  Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print()
    
    return model


def evaluate_model(model, X_test, y_test, model_name: str):
    """
    Evaluate model on test set and print metrics.
    
    Focus on metrics important for fraud detection:
    - Precision: Of predicted frauds, how many are real?
    - Recall: Of all real frauds, how many did we catch?
    - F1-Score: Balance between precision and recall
    - ROC-AUC: Overall model quality
    """
    print(f"📊 Evaluating {model_name}...")
    print()
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of fraud
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', zero_division=0
    )
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    print(f"🎯 {model_name} Results:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f} (of predicted frauds, {precision:.1%} are real)")
    print(f"   Recall:    {recall:.4f} (caught {recall:.1%} of all frauds)")
    print(f"   F1-Score:  {f1:.4f} (balance of precision & recall)")
    print(f"   ROC-AUC:   {roc_auc:.4f} (overall model quality)")
    print()
    print(f"📋 Confusion Matrix:")
    print(f"   True Negatives (TN):  {tn:,} ✅ Correctly identified legit")
    print(f"   False Positives (FP): {fp:,} ⚠️  False alarms")
    print(f"   False Negatives (FN): {fn:,} ❌ Missed frauds (BAD!)")
    print(f"   True Positives (TP):  {tp:,} ✅ Caught frauds")
    print()
    
    # Calculate fraud catch rate
    total_frauds = tp + fn
    catch_rate = tp / total_frauds if total_frauds > 0 else 0
    print(f"💰 Fraud Detection:")
    print(f"   Total frauds in test: {total_frauds:,}")
    print(f"   Frauds caught: {tp:,} ({catch_rate:.1%})")
    print(f"   Frauds missed: {fn:,} ({1-catch_rate:.1%})")
    print()
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


def save_models_and_results(models_dict: dict, results_list: list, 
                            split_data: dict, project_root: Path):
    """
    Save trained models and evaluation results.
    
    Args:
        models_dict: Dictionary of {model_name: model}
        results_list: List of evaluation result dictionaries
        split_data: Dictionary with train/test split info
        project_root: Project root directory
    """
    print("=" * 70)
    print("SAVING MODELS AND RESULTS")
    print("=" * 70)
    print()
    
    # Create directories
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    reports_dir = project_root / "reports" / "model_training"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    print("💾 Saving models...")
    for model_name, model in models_dict.items():
        model_path = models_dir / f"{model_name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, model_path)
        print(f"   ✓ {model_path.name}")
    
    # Save train/test split info (for reproducibility)
    split_path = models_dir / "train_test_split_info.pkl"
    joblib.dump(split_data, split_path)
    print(f"   ✓ {split_path.name}")
    print()
    
    # Save results as CSV
    print("📊 Saving evaluation results...")
    results_df = pd.DataFrame(results_list)
    results_path = reports_dir / "baseline_models_comparison.csv"
    results_df.to_csv(results_path, index=False)
    print(f"   ✓ {results_path.name}")
    print()
    
    # Save detailed report
    report_path = reports_dir / "training_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FRAUD DETECTION - BASELINE MODELS TRAINING REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        
        f.write("DATASET INFO:\n")
        f.write(f"Total samples: {split_data['total_samples']:,}\n")
        f.write(f"Train samples: {split_data['train_samples']:,}\n")
        f.write(f"Test samples: {split_data['test_samples']:,}\n")
        f.write(f"Fraud rate: {split_data['fraud_rate']:.4f}\n")
        f.write("\n")
        
        f.write("MODEL COMPARISON:\n")
        f.write("-" * 70 + "\n")
        for result in results_list:
            f.write(f"\n{result['model_name']}:\n")
            f.write(f"  Accuracy:  {result['accuracy']:.4f}\n")
            f.write(f"  Precision: {result['precision']:.4f}\n")
            f.write(f"  Recall:    {result['recall']:.4f}\n")
            f.write(f"  F1-Score:  {result['f1_score']:.4f}\n")
            f.write(f"  ROC-AUC:   {result['roc_auc']:.4f}\n")
            f.write(f"  TP: {result['tp']:,}, FP: {result['fp']:,}, ")
            f.write(f"TN: {result['tn']:,}, FN: {result['fn']:,}\n")
        
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("RECOMMENDATION:\n")
        f.write("-" * 70 + "\n")
        
        # Find best model by F1-score
        best_model = max(results_list, key=lambda x: x['f1_score'])
        f.write(f"Best model by F1-Score: {best_model['model_name']}\n")
        f.write(f"F1-Score: {best_model['f1_score']:.4f}\n")
        f.write("\n")
        f.write("Next steps:\n")
        f.write("1. Run 02_evaluate_models.py for detailed visualizations\n")
        f.write("2. Run 03_feature_importance.py to see which features matter\n")
        f.write("3. Consider hyperparameter tuning for the best model\n")
    
    print(f"   ✓ {report_path.name}")
    print()
    
    print(f"📁 All files saved to:")
    print(f"   Models: {models_dir}")
    print(f"   Reports: {reports_dir}")
    print()


def main():
    """
    Main training pipeline.
    """
    print("\n")
    print("=" * 70)
    print("FRAUD DETECTION - BASELINE MODEL TRAINING")
    print("=" * 70)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    
    pipeline_start = time.time()
    
    # Setup paths
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "features" / "final_features.parquet"
    
    # Check if data exists
    if not data_path.exists():
        print(f"❌ ERROR: Data file not found!")
        print(f"   Expected: {data_path}")
        print()
        print("💡 Make sure you've run:")
        print("   python src/utils/convert_to_parquet.py")
        return
    
    # Load data with 10M stratified sample (for 16GB RAM)
    # Change to sample_size=None for full 35M rows (requires 32GB RAM)
    X, y, feature_names = load_data(data_path, sample_size=10_000_000)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Store split info for later use
    split_info = {
        'total_samples': len(X),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'fraud_rate': y.mean(),
        'random_state': 42,
        'test_size': 0.2,
        'feature_names': feature_names
    }
    
    # Train models (one at a time to save memory!)
    models = {}
    results = []
    
    # Model 1: Random Forest
    rf_model = train_random_forest(X_train, y_train)
    models['Random Forest'] = rf_model
    rf_results = evaluate_model(rf_model, X_test, y_test, 'Random Forest')
    results.append(rf_results)
    
    # Model 2: XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    models['XGBoost'] = xgb_model
    xgb_results = evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
    results.append(xgb_results)
    
    # Model 3: LightGBM
    lgb_model = train_lightgbm(X_train, y_train)
    models['LightGBM'] = lgb_model
    lgb_results = evaluate_model(lgb_model, X_test, y_test, 'LightGBM')
    results.append(lgb_results)
    
    # Save everything
    save_models_and_results(models, results, split_info, project_root)
    
    # Final summary
    print("=" * 70)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 70)
    
    # Print comparison table
    print()
    print("📊 MODEL COMPARISON:")
    print("-" * 70)
    print(f"{'Model':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ROC-AUC':>10}")
    print("-" * 70)
    for result in results:
        print(f"{result['model_name']:<15} "
              f"{result['precision']:>10.4f} "
              f"{result['recall']:>10.4f} "
              f"{result['f1_score']:>10.4f} "
              f"{result['roc_auc']:>10.4f}")
    print("-" * 70)
    
    # Highlight best model (by ROC-AUC - best metric for imbalanced data)
    best_model = max(results, key=lambda x: x['roc_auc'])
    print()
    print(f"🏆 Best model: {best_model['model_name']} (ROC-AUC: {best_model['roc_auc']:.4f}, Recall: {best_model['recall']:.4f})")
    print()
    
    # Total time
    total_time = time.time() - pipeline_start
    print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print()
    
    print("✅ Next steps:")
    print("   1. Run: python src/modeling/02_evaluate_models.py")
    print("   2. Check reports in: reports/model_training/")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()


