# src/encoding/__init__.py

"""
Encoding Module

This module contains scripts for encoding categorical features:
1. One-hot encoding for low-cardinality features (gender, weekday, category, state)
2. Label encoding for high-cardinality features (job, merchant, city, profile, zip)

All encoding mappings are saved to reports/ directory as JSON files for reference.
"""

from . import onehot_encode
from . import label_encode

__all__ = [
    'onehot_encode',
    'label_encode'
]

