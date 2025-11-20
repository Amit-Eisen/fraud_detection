# src/data_cleansing/__init__.py

"""
Data Cleansing Module

This module contains scripts for cleaning the dataset:
1. Remove duplicates (logical duplicates based on key columns)
2. Drop useless columns (PII, redundant data) and create age feature
3. Handle outliers (analysis only - decided NOT to remove for fraud detection)
"""

from . import remove_duplicates
from . import drop_useless_columns
from . import handle_outliers

__all__ = [
    'remove_duplicates',
    'drop_useless_columns', 
    'handle_outliers'
]

