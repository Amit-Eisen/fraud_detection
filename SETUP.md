# 🔧 Credit Card Fraud Detection - Setup Guide

## Project Overview
- **Dataset**: 35M credit card transactions (~10GB CSV)
- **Goal**: Detect fraudulent transactions using XGBoost
- **Challenge**: Extreme class imbalance (0.61% fraud)
- **Result**: ROC-AUC 0.9996, 97% fraud detection rate

---

## First Time Setup (10 minutes)

### 1. Create Virtual Environment
```bash
cd fraud_detection
python3 -m venv .venv
```

### 2. Activate Virtual Environment
```bash
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows
```
You should see `(.venv)` in your terminal prompt.

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- pandas, numpy (data processing)
- scikit-learn (ML framework)
- xgboost, lightgbm (models)
- optuna (Bayesian optimization)
- matplotlib (visualization)
- pyarrow (Parquet files)

### 4. Verify Installation
```bash
python -c "import pandas, xgboost, optuna; print('✅ All packages installed!')"
```

---

## Data Preparation

### Step 1: Place Raw Data
```
data/raw/credit_card_fraud.csv  (35M rows, ~10GB)
```

### Step 2: Run Data Pipeline
```bash
# Convert CSV to Parquet (memory efficient)
python src/utils/convert_to_parquet.py

# Data cleansing
python src/data_cleansing/01_remove_duplicates.py
python src/data_cleansing/02_drop_useless_columns.py
# Note: 03_handle_outliers.py - analysis showed outliers are fraud signals, so we keep them!

# Feature engineering
python src/feature_engineering/02_time_features.py
python src/feature_engineering/03_amount_features.py
python src/feature_engineering/04_interaction_features.py
# Note: 01_distance_features.py - skipped (merchant coordinates unavailable)
```

---

## Model Training

### Baseline Models (10M sample)
```bash
python src/modeling/01_train_baseline.py
```
Trains 3 models: Random Forest, XGBoost, LightGBM

### Feature Importance & Selection
```bash
python src/modeling/02_feature_importance.py
python src/modeling/03_feature_selection.py
```
Selects top 36 features using permutation importance

### Optimized Model (20M sample)
```bash
python src/modeling/04_train_selected_features.py
```
- Bayesian optimization (50 trials, 3.3 hours)
- Final ROC-AUC: 0.9996
- Precision: 88.7%, Recall: 97.2%

### External Validation (12.9M unseen)
```bash
python src/evaluation/03_external_validation.py
```
Validates on completely unseen data - proves no overfitting!

---

## Important Notes

### Always Activate the Virtual Environment!
Before running any Python scripts:
```bash
source .venv/bin/activate
```
You should see `(.venv)` in your prompt.

### Memory Requirements
- **Minimum**: 16GB RAM (with 10M-20M samples)
- **Recommended**: 32GB RAM (for full 35M dataset)
- **Solution**: Use Parquet format + chunked processing

### Using the Correct Python
- ✅ `.venv/bin/python` - Virtual environment (has all packages)
- ❌ `/usr/bin/python3` - System Python (missing packages)

### Deactivate Virtual Environment
When done:
```bash
deactivate
```

---

## Project Structure

```
fraud_detection/
├── data/
│   ├── raw/                    # Original CSV (35M rows)
│   ├── processed/              # Cleaned data
│   ├── features/               # Engineered features (Parquet)
│   └── cleansed/               # After cleansing
├── src/
│   ├── data_cleansing/         # Remove duplicates, outliers
│   ├── feature_engineering/    # Time, amount, interaction features
│   ├── modeling/               # Baseline, optimization, training
│   └── evaluation/             # Metrics, visualizations, validation
├── models/
│   └── xgboost_optimized/      # Final model + hyperparameters
├── reports/
│   ├── evaluation/             # Performance metrics, plots
│   ├── feature_importance/     # Feature analysis
│   └── model_training/         # Baseline comparison
└── requirements.txt            # Python dependencies
```

---

## Troubleshooting

**Problem**: `ModuleNotFoundError: No module named 'pandas'`  
**Solution**: Activate virtual environment: `source .venv/bin/activate`

**Problem**: `MemoryError` during training  
**Solution**: Reduce sample size in training scripts (10M → 5M)

**Problem**: Parquet files not found  
**Solution**: Run `python src/utils/convert_to_parquet.py` first

**Problem**: Slow training  
**Solution**: Reduce Optuna trials (50 → 20) or use smaller sample

---

## Key Results

### Final Model Performance (Test Set)
- **ROC-AUC**: 0.9996 (near perfect!)
- **Precision**: 88.7% (low false alarms)
- **Recall**: 97.2% (catches 97% of fraud)
- **F1-Score**: 0.928

### External Validation (12.9M unseen)
- **ROC-AUC**: 0.9996 (identical!)
- **Recall**: 97.3%
- **Precision**: 88.7%
- **Conclusion**: Model generalizes perfectly!

### Business Impact
- **Fraud caught**: 97.3% (76,498 / 78,597)
- **False alarms**: 0.076% (9,764 / 12.9M)
- **Production-ready**: ✅

---

## Next Steps

1. ✅ **Model is trained and validated**
2. 🎯 Deploy to production environment
3. 📊 Monitor performance on live data
4. 🔄 Retrain periodically with new fraud patterns

---

## Project Status

✅ Data pipeline complete (35M rows processed)  
✅ Feature engineering complete (36 features)  
✅ Baseline models trained (3 models compared)  
✅ Hyperparameter optimization complete (Bayesian, 50 trials)  
✅ Final model trained (ROC-AUC 0.9996)  
✅ External validation complete (12.9M unseen rows)  
✅ Production-ready! 🚀

---

**Author**: Amit Eisen  
**Project**: Credit Card Fraud Detection  
**Date**: November 2024  
**Achievement**: Successfully processed and modeled 35M transactions - full dataset!
