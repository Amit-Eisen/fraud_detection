# src/feature_engineering/__init__.py

"""
Feature Engineering Module

This module creates new features from existing data to improve fraud detection.

Modules:
    - 01_distance_features: Geospatial features (SKIPPED - no merchant coords)
    - 02_time_features: Time-based features (is_night, is_weekend, cyclical hour)
    - 03_amount_features: Amount transformations (log, z-score, flags)
    - 04_interaction_features: Feature interactions (distance×amount, etc.)

Usage:
    # Run individual feature engineering scripts
    python src/feature_engineering/02_time_features.py
    python src/feature_engineering/03_amount_features.py
    
    # Or import functions
    from src.feature_engineering.time_features import create_time_features

Pipeline:
    data/encoded/02_final_encoded.csv
        ↓
    02_time_features.py → data/features/01_time_features.csv
        ↓
    03_amount_features.py → data/features/02_amount_features.csv
        ↓
    04_interaction_features.py → data/features/03_final_features.csv
        ↓
    Ready for model training!

Note:
    - All scripts process data in chunks (1M rows) for memory efficiency
    - Distance features skipped (merchant coordinates dropped in cleansing)
    - Each script adds features incrementally
"""

__version__ = "1.0.0"
__author__ = "Fraud Detection Team"

# Feature engineering functions can be imported here if needed
# from .time_features import create_time_features
# from .amount_features import create_amount_features

