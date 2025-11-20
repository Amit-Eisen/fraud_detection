# src/eda/__init__.py
"""
Exploratory Data Analysis (EDA) Module
=======================================

Contains all EDA scripts for fraud detection analysis.

Modules:
    - basic_statistics: Dataset overview and basic stats
    - target_analysis: Fraud distribution and class imbalance (coming soon)
    - numerical_features: Amount and coordinate analysis (coming soon)
    - temporal_analysis: Time-based patterns (coming soon)
    - categorical_features: Job, merchant, category analysis (coming soon)
    - correlations: Feature correlations (coming soon)
"""

from . import basic_statistics
from . import target_analysis
from . import numerical_features
from . import temporal_analysis
from . import categorical_features
from . import correlations

__all__ = ['basic_statistics', 'target_analysis', 'numerical_features', 'temporal_analysis', 'categorical_features', 'correlations']

