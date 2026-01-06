"""
Test data generation for CCM validation.

Provides synthetic time series with known ground truth causal relationships.
"""

from .generators import (
    make_test_dataframe,
    get_ground_truth_network,
)

__all__ = [
    'make_test_dataframe',
    'get_ground_truth_network',
]
