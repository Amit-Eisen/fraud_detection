#!/usr/bin/env python3
"""
Train XGBoost with Bayesian Hyperparameter Optimization
========================================================

This script trains XGBoost on the selected top features using Optuna
for intelligent hyperparameter tuning.

Strategy:
1. Load selected features (top 36, ~5.8 GB)
2. Stratified train/test split (80/20)
3. Bayesian Optimization with Optuna:
   - 50 trials (intelligent sampling)
   - Optimizes for ROC-AUC (best for imbalanced data)
   - Early stopping for bad trials (pruning)
4. Train final model with best hyperparameters
5. Evaluate on test set
6. Save model and results

Why Bayesian Optimization?
- Learns from previous trials (smarter than random/grid search)
- Finds optimal hyperparameters faster
- Prunes bad trials early (saves time)
- Professional approach for production models

Expected Runtime:
- Full 35M rows: 50 trials × 8-10 min = 7-8 hours
- 25M sample: 50 trials × 5-6 min = 4-5 hours
- 20M sample: 50 trials × 4-5 min = 3-4 hours

"""

import sys
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import optuna
from optuna.pruners import MedianPruner

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def load_data(parquet_path: Path, sample_size: int = None) -> tuple:
    """
    Load the selected features dataset.
    
    Args:
        parquet_path: Path to the Parquet file
        sample_size: Optional - number of rows to sample (stratified)
                    If None, loads all data
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    print(f"📂 Loading: {parquet_path}")
    print()
    
    start = time.time()
    
    # Load full dataset
    df = pd.read_parquet(parquet_path)
    
    elapsed = time.time() - start
    memory_gb = df.memory_usage(deep=True).sum() / 1e9
    
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"⏱️  Time: {elapsed:.1f} seconds")
    print(f"💾 Memory: ~{memory_gb:.1f} GB")
    print()
    
    # Stratified sampling if requested
    if sample_size and sample_size < len(df):
        print(f"🎲 Sampling {sample_size:,} rows (stratified)...")
        fraud_ratio = df['is_fraud'].mean()
        n_fraud = int(sample_size * fraud_ratio)
        n_legit = sample_size - n_fraud
        
        fraud_df = df[df['is_fraud'] == 1].sample(n=n_fraud, random_state=42)
        legit_df = df[df['is_fraud'] == 0].sample(n=n_legit, random_state=42)
        
        df = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        memory_gb = df.memory_usage(deep=True).sum() / 1e9
        print(f"✓ Sampled to {len(df):,} rows")
        print(f"💾 Memory after sampling: ~{memory_gb:.1f} GB")
        print()
    
    # Separate features and target
    print(f"🎯 Target variable: is_fraud")
    
    if 'is_fraud' not in df.columns:
        print("❌ ERROR: 'is_fraud' column not found!")
        return None, None, None
    
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    fraud_count = y.sum()
    legit_count = len(y) - fraud_count
    fraud_pct = fraud_count / len(y) * 100
    
    print(f"   Fraud cases: {fraud_count:,} ({fraud_pct:.2f}%)")
    print(f"   Legit cases: {legit_count:,} ({100-fraud_pct:.2f}%)")
    print(f"   Class imbalance ratio: 1:{int(legit_count/fraud_count)}")
    print()
    
    return X, y, X.columns.tolist()


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test with stratification.
    
    Args:
        X: Features
        y: Target
        test_size: Fraction for test set
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("=" * 70)
    print("SPLITTING DATA")
    print("=" * 70)
    print(f"📊 Split ratio: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
    print(f"🎲 Random state: {random_state} (for reproducibility)")
    print(f"⚠️  Using STRATIFICATION to maintain fraud ratio!")
    print()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✓ Train set: {len(X_train):,} rows ({y_train.sum():,} fraud)")
    print(f"✓ Test set:  {len(X_test):,} rows ({y_test.sum():,} fraud)")
    print()
    
    return X_train, X_test, y_train, y_test


def objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        scale_pos_weight: Class weight for imbalanced data
    
    Returns:
        ROC-AUC score on validation set
    """
    # Suggest hyperparameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': 6,  # Limit parallelism to save memory
        
        # Hyperparameters to optimize
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'scale_pos_weight': scale_pos_weight,
        'max_bin': 256,
    }
    
    # Train model
    model = xgb.XGBClassifier(**params)
    
    # Use early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predict on validation set
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    return roc_auc


def optimize_hyperparameters(X_train, y_train, n_trials=50):
    """
    Use Optuna to find optimal hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of optimization trials
    
    Returns:
        Best hyperparameters dictionary
    """
    print("=" * 70)
    print("BAYESIAN HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"🔍 Method: Optuna (Bayesian Optimization)")
    print(f"🎯 Objective: Maximize ROC-AUC")
    print(f"🔢 Number of trials: {n_trials}")
    print(f"⏱️  Estimated time: {n_trials * 8}-{n_trials * 10} minutes")
    print()
    print("📊 Hyperparameters to optimize:")
    print("   • learning_rate: [0.01, 0.3]")
    print("   • max_depth: [3, 10]")
    print("   • n_estimators: [100, 500]")
    print("   • min_child_weight: [1, 10]")
    print("   • subsample: [0.6, 1.0]")
    print("   • colsample_bytree: [0.6, 1.0]")
    print("   • gamma: [0, 5]")
    print("   • reg_alpha: [0, 10]")
    print("   • reg_lambda: [0, 10]")
    print()
    print("🚀 Starting optimization...")
    print("   (This will take several hours - grab a coffee! ☕)")
    print()
    
    # Split train into train/validation for optimization
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Calculate scale_pos_weight
    scale_pos_weight = (len(y_tr) - y_tr.sum()) / y_tr.sum()
    
    print(f"📊 Optimization split:")
    print(f"   Train: {len(X_tr):,} rows ({y_tr.sum():,} fraud)")
    print(f"   Validation: {len(X_val):,} rows ({y_val.sum():,} fraud)")
    print(f"   scale_pos_weight: {scale_pos_weight:.2f}")
    print()
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Optimize
    start_time = time.time()
    
    study.optimize(
        lambda trial: objective(trial, X_tr, y_tr, X_val, y_val, scale_pos_weight),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    elapsed = time.time() - start_time
    
    print()
    print("=" * 70)
    print("✅ OPTIMIZATION COMPLETE!")
    print("=" * 70)
    print(f"⏱️  Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    print(f"🏆 Best ROC-AUC: {study.best_value:.4f}")
    print()
    print("🎯 Best hyperparameters:")
    for param, value in study.best_params.items():
        print(f"   • {param}: {value}")
    print()
    
    # Add fixed parameters
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': 6,
        'scale_pos_weight': scale_pos_weight,
        'max_bin': 256,
    })
    
    return best_params, study


def train_final_model(X_train, y_train, best_params):
    """
    Train final XGBoost model with best hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training target
        best_params: Best hyperparameters from optimization
    
    Returns:
        Trained XGBoost model
    """
    print("=" * 70)
    print("TRAINING FINAL MODEL")
    print("=" * 70)
    print(f"🎯 Model: XGBoost with optimized hyperparameters")
    print(f"📊 Training on {len(X_train):,} rows")
    print()
    
    start = time.time()
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start
    
    print(f"✓ Training complete!")
    print(f"⏱️  Time: {elapsed/60:.1f} minutes")
    print()
    
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    print(f"📊 Evaluating on {len(X_test):,} test samples")
    print()
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Print results
    print("📈 Performance Metrics:")
    print("-" * 70)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (of predicted frauds, how many are real)")
    print(f"Recall:    {recall:.4f} (of real frauds, how many we caught)")
    print(f"F1-Score:  {f1:.4f} (harmonic mean of precision & recall)")
    print(f"ROC-AUC:   {roc_auc:.4f} (overall discrimination ability)")
    print()
    
    print("🎯 Confusion Matrix:")
    print("-" * 70)
    print(f"True Negatives (TN):  {tn:,} ✓ (correctly identified legit)")
    print(f"False Positives (FP): {fp:,} ✗ (legit flagged as fraud)")
    print(f"False Negatives (FN): {fn:,} ✗ (fraud missed)")
    print(f"True Positives (TP):  {tp:,} ✓ (correctly caught fraud)")
    print()
    
    # Business metrics
    fraud_caught_pct = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    false_alarm_rate = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0
    
    print("💼 Business Metrics:")
    print("-" * 70)
    print(f"Fraud Detection Rate: {fraud_caught_pct:.2f}% (caught {tp:,} of {tp+fn:,} frauds)")
    print(f"False Alarm Rate:     {false_alarm_rate:.2f}% ({fp:,} false alarms)")
    print()
    
    # Store results
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        },
        'business_metrics': {
            'fraud_caught_pct': float(fraud_caught_pct),
            'false_alarm_rate': float(false_alarm_rate)
        }
    }
    
    return results


def save_results(model, best_params, results, study, output_dir: Path):
    """
    Save trained model, hyperparameters, and evaluation results.
    
    Args:
        model: Trained model
        best_params: Best hyperparameters
        results: Evaluation results dictionary
        study: Optuna study object
        output_dir: Directory to save outputs
    """
    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / 'xgboost_optimized.pkl'
    joblib.dump(model, model_path)
    print(f"✓ Saved model: {model_path}")
    
    # Save hyperparameters
    params_path = output_dir / 'best_hyperparameters.json'
    with open(params_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        params_serializable = {}
        for k, v in best_params.items():
            if isinstance(v, (np.integer, np.floating)):
                params_serializable[k] = float(v)
            else:
                params_serializable[k] = v
        json.dump(params_serializable, f, indent=2)
    print(f"✓ Saved hyperparameters: {params_path}")
    
    # Save evaluation results
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved evaluation results: {results_path}")
    
    # Save optimization history
    history_path = output_dir / 'optimization_history.csv'
    trials_df = study.trials_dataframe()
    trials_df.to_csv(history_path, index=False)
    print(f"✓ Saved optimization history: {history_path}")
    
    # Save text report
    report_path = output_dir / 'training_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("XGBoost Training Report (Bayesian Optimization)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("BEST HYPERPARAMETERS:\n")
        f.write("-" * 70 + "\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
        f.write("\n")
        
        f.write("EVALUATION RESULTS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy:  {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall:    {results['recall']:.4f}\n")
        f.write(f"F1-Score:  {results['f1_score']:.4f}\n")
        f.write(f"ROC-AUC:   {results['roc_auc']:.4f}\n")
        f.write("\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 70 + "\n")
        cm = results['confusion_matrix']
        f.write(f"True Negatives:  {cm['tn']:,}\n")
        f.write(f"False Positives: {cm['fp']:,}\n")
        f.write(f"False Negatives: {cm['fn']:,}\n")
        f.write(f"True Positives:  {cm['tp']:,}\n")
        f.write("\n")
        
        f.write("BUSINESS METRICS:\n")
        f.write("-" * 70 + "\n")
        bm = results['business_metrics']
        f.write(f"Fraud Detection Rate: {bm['fraud_caught_pct']:.2f}%\n")
        f.write(f"False Alarm Rate:     {bm['false_alarm_rate']:.2f}%\n")
    
    print(f"✓ Saved text report: {report_path}")
    print()


def main():
    """Main execution function."""
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print("XGBOOST TRAINING WITH BAYESIAN OPTIMIZATION")
    print("=" * 70)
    print()
    
    # Configuration
    N_TRIALS = 50  # Number of Optuna trials
    
    print(f"⚙️  Configuration: Training on 20M row sample (pre-created)")
    print(f"⚙️  Optimization trials: {N_TRIALS}")
    print()
    
    # Setup paths
    project_root = Path(__file__).resolve().parents[2]
    input_parquet = project_root / 'data' / 'features' / 'selected_features_20M.parquet'
    output_dir = project_root / 'models' / 'xgboost_optimized'
    
    # Check if input exists
    if not input_parquet.exists():
        print(f"❌ ERROR: Input file not found!")
        print(f"   Expected: {input_parquet}")
        print()
        print("💡 Make sure you've run:")
        print("   python src/modeling/03_feature_selection.py")
        return
    
    # 1. Load data (already sampled, no need for sample_size parameter)
    X, y, feature_names = load_data(input_parquet, sample_size=None)
    
    if X is None:
        return
    
    # 2. Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 3. Optimize hyperparameters
    best_params, study = optimize_hyperparameters(X_train, y_train, n_trials=N_TRIALS)
    
    # 4. Train final model
    model = train_final_model(X_train, y_train, best_params)
    
    # 5. Evaluate
    results = evaluate_model(model, X_test, y_test)
    
    # 6. Save everything
    save_results(model, best_params, results, study, output_dir)
    
    # Summary
    elapsed = time.time() - start_time
    print("=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print()
    print(f"⏱️  Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    print(f"🏆 Final ROC-AUC: {results['roc_auc']:.4f}")
    print(f"🎯 Fraud Detection Rate: {results['business_metrics']['fraud_caught_pct']:.2f}%")
    print()
    print("📁 Output files:")
    print(f"   Model: {output_dir / 'xgboost_optimized.pkl'}")
    print(f"   Hyperparameters: {output_dir / 'best_hyperparameters.json'}")
    print(f"   Results: {output_dir / 'evaluation_results.json'}")
    print(f"   Report: {output_dir / 'training_report.txt'}")
    print()


if __name__ == "__main__":
    main()
