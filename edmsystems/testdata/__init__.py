"""
Test data generation for EDM validation and demonstration.

This module provides synthetic time series with known ground truth
causal relationships for validating CCM and other EDM methods.
"""

from .generators import (
    make_independent_series,
    make_correlated_series,
    make_correlated_autocorrelated_series,
    make_pure_autocorrelated_series,
    make_seasonal_series,
    make_causal_series,
    make_bidirectional_causal_series,
    make_indirect_causal_series,
    make_test_dataframe,
    get_ground_truth_network,
)

__all__ = [
    'make_independent_series',
    'make_correlated_series',
    'make_correlated_autocorrelated_series',
    'make_pure_autocorrelated_series',
    'make_seasonal_series',
    'make_causal_series',
    'make_bidirectional_causal_series',
    'make_indirect_causal_series',
    'make_test_dataframe',
    'get_ground_truth_network',
]
