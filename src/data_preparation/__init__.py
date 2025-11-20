# src/data_preparation/__init__.py
"""
Data Preparation Module
========================

Contains all data cleaning and preprocessing functions for the fraud detection pipeline.

Modules:
    - clean_text: Text cleaning and normalization
    - reduce_categories: High-cardinality category reduction
    - transform_data: Feature engineering and data transformation
"""

# Make modules easily importable
from . import clean_text
from . import reduce_categories
from . import transform_data

__all__ = ['clean_text', 'reduce_categories', 'transform_data']

