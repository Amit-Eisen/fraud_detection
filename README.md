# 💳 Credit Card Fraud Detection

**A production-ready fraud detection system using XGBoost on 35M transactions**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 Project Overview

This project implements a **state-of-the-art fraud detection system** that processes and analyzes **35 million credit card transactions** to identify fraudulent activity with near-perfect accuracy.

### Key Achievements
- ✅ **ROC-AUC: 0.9996** (near perfect discrimination)
- ✅ **97.3% Fraud Detection Rate** (catches 97 out of 100 fraud cases)
- ✅ **88.7% Precision** (only 11% false alarms)
- ✅ **Validated on 12.9M unseen transactions** (no overfitting!)
- ✅ **Production-ready** with optimized hyperparameters

### The Challenge
- **Dataset Size**: 35M transactions (~10GB)
- **Class Imbalance**: Only 0.61% fraud (1:163 ratio)
- **Memory Constraints**: Requires efficient chunked processing
- **Real-world Impact**: Minimize false alarms while catching fraud

---

## 📊 Dataset

**Source**: [Kaggle - Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)

**Statistics**:
- 35,180,112 transactions
- 214,597 fraud cases (0.61%)
- 23 features (transaction, customer, temporal, geospatial)
- Date range: 2019-2020

**📥 Download Instructions**: See [`DATA_DOWNLOAD.md`](DATA_DOWNLOAD.md)

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/fraud_detection.git
cd fraud_detection
```

### 2. Download Dataset
Follow instructions in [`DATA_DOWNLOAD.md`](DATA_DOWNLOAD.md) to download the dataset from Kaggle.

Place the file in: `data/raw/credit_card_fraud.csv`

### 3. Setup Environment
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 4. Run Pipeline
See [`SETUP.md`](SETUP.md) for detailed instructions on running the complete pipeline.

---

## 🏗️ Project Structure

```
fraud_detection/
├── data/
│   ├── raw/                    # Original dataset (download here!)
│   ├── processed/              # Cleaned data
│   ├── features/               # Engineered features (Parquet)
│   └── cleansed/               # After data cleansing
├── src/
│   ├── data_cleansing/         # Duplicate removal, feature cleanup
│   ├── feature_engineering/    # Time, amount, interaction features
│   ├── modeling/               # Baseline, optimization, training
│   ├── evaluation/             # Metrics, visualizations, validation
│   └── utils/                  # Helper functions, converters
├── models/
│   └── xgboost_optimized/      # Final model + hyperparameters
├── reports/
│   ├── evaluation/             # Performance metrics, plots
│   ├── feature_importance/     # Feature analysis
│   └── model_training/         # Baseline comparison
├── requirements.txt            # Python dependencies
├── SETUP.md                    # Detailed setup guide
├── DATA_DOWNLOAD.md            # Dataset download instructions
└── README.md                   # This file
```

---

## 🔬 Methodology

### 1. Data Preparation
- **Cleansing**: Remove duplicates, drop multicollinear features
- **Outlier Analysis**: Keep outliers (they're fraud signals!)
- **Format**: Convert CSV → Parquet (3x faster, 70% smaller)

### 2. Feature Engineering
- **Temporal**: Hour, day of week, is_night, is_weekend
- **Amount**: Log transformation, high-value flags, outlier detection
- **Interaction**: night × high_value, weekend × high_value, amt_per_age
- **Total**: 36 carefully selected features

### 3. Model Development
- **Baseline**: Trained 3 models (Random Forest, XGBoost, LightGBM)
- **Selection**: XGBoost performed best (ROC-AUC 0.9984)
- **Optimization**: Bayesian hyperparameter tuning (50 trials, 3.3 hours)
- **Final**: ROC-AUC 0.9996 on 20M training sample

### 4. Validation
- **Test Set**: 30% holdout (stratified split)
- **External Validation**: 12.9M completely unseen transactions
- **Result**: Identical performance → no overfitting!

---

## 📈 Results

### Model Performance

| Metric | Baseline XGBoost | Optimized XGBoost | Improvement |
|--------|------------------|-------------------|-------------|
| **Precision** | 23.6% | 88.7% | **+65 points** |
| **Recall** | 97.9% | 97.2% | -0.7 points |
| **F1-Score** | 0.381 | 0.928 | **+144%** |
| **ROC-AUC** | 0.9984 | 0.9996 | +0.12 points |

### Business Impact (External Validation - 12.9M transactions)
- ✅ **Fraud Caught**: 76,498 / 78,597 (97.33%)
- ✅ **Fraud Missed**: 2,099 (2.67%)
- ✅ **False Alarms**: 9,764 (0.076% of legitimate transactions)
- ✅ **False Alarm Rate**: Only 1 in 1,316 legitimate transactions flagged

### Key Insights
1. **Amount is the strongest predictor** (18.2% importance)
2. **Merchant category matters** (gas_transport: 15.4%)
3. **Location signals** (city_pop: 12.8%)
4. **Time patterns** (hour, day: 12% combined)
5. **Engineered features add value** (amt_log, is_night, interactions)

---

## 🛠️ Technologies Used

- **Python 3.8+**: Core language
- **pandas**: Data manipulation (35M rows!)
- **pyarrow**: Efficient Parquet I/O
- **scikit-learn**: ML framework, metrics
- **XGBoost**: Main model (gradient boosting)
- **Optuna**: Bayesian hyperparameter optimization
- **matplotlib**: Visualization
- **numpy**: Numerical operations

---

## 📊 Visualizations

The project generates comprehensive visualizations:

### Model Performance
- ROC Curve (AUC: 0.9996)
- Precision-Recall Curve
- Confusion Matrix
- Threshold Analysis

### Optimization
- Optimization History (50 trials)
- Convergence Plot
- Parameter Importance

### Feature Analysis
- Feature Importance (top 36 features)
- Cumulative Importance
- Feature Comparison

### Validation
- External Validation Comparison (Test vs Unseen)

All visualizations saved in `reports/evaluation/figures/`

---

## 🎓 What Makes This Project Special

### 1. **Scale**
- Processed **full 35M dataset** (most projects use samples)
- Efficient memory management (chunking + Parquet)

### 2. **Validation**
- **External validation** on 12.9M unseen transactions
- Proves model generalizes (no overfitting!)

### 3. **Optimization**
- **Bayesian optimization** (not grid search)
- Smart, efficient hyperparameter tuning

### 4. **Data-Driven Decisions**
- Outlier analysis: Kept outliers (they're fraud signals!)
- Feature selection: Permutation importance
- Evidence-based choices throughout

### 5. **Production-Ready**
- Clean, documented code
- Reproducible pipeline
- Comprehensive testing
- Ready for deployment

---

## 📝 Documentation

- **[SETUP.md](SETUP.md)**: Complete setup and pipeline guide
- **[DATA_DOWNLOAD.md](DATA_DOWNLOAD.md)**: Dataset download instructions
- **Code Comments**: Every script is well-documented
- **Reports**: Detailed analysis in `reports/` directory

---

## 🔮 Future Improvements

- [ ] Real-time prediction API (Flask/FastAPI)
- [ ] Model monitoring dashboard
- [ ] A/B testing framework
- [ ] Ensemble methods (stacking)
- [ ] Deep learning comparison (LSTM, Transformer)
- [ ] Explainability (SHAP values)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [Kaggle - Credit Card Fraud Detection 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
- **Inspiration**: Real-world fraud detection challenges
- **Tools**: XGBoost, Optuna, scikit-learn communities

---

## 👤 Author

**Amit Eisen**

- 📧 Email: eisenamit96@gmail.com
- 💼 LinkedIn: https://www.linkedin.com/in/amit-eisen/

---

## 🚀 Getting Started

Ready to dive in? Start with:
1. 📥 [`DATA_DOWNLOAD.md`](DATA_DOWNLOAD.md) - Download the dataset
2. 🔧 [`SETUP.md`](SETUP.md) - Setup and run the pipeline
3. 📊 Explore the `reports/` directory for insights!

**Questions?** Open an issue or reach out!
