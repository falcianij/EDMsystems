"""
Preprocessing module for EDM time series analysis.

This module provides normalization, detrending, and utility functions
for preparing time series data for empirical dynamical modeling.
"""

from .normalization import (
    Normalizer,
    ZScoreNormalizer,
    RobustNormalizer,
    MinMaxNormalizer,
    RankNormalizer,
    SqrtMinMaxNormalizer,
    Log1pNormalizer,
    WinsorizingNormalizer,
    get_normalizer,
    normalize_dataframe,
    inverse_normalize_dataframe,
)

from .detrending import (
    detrend_linear,
    detrend_polynomial,
    detrend_stl,
    detrend_hp,
    detrend_loess,
    detrend_rolling_mean,
    detrend_savgol,
    detrend_seasonal_diff,
    get_detrending_method,
    detrend_dataframe,
)

from .utils import (
    add_time_column,
    reset_to_quarter_start,
    aggregate_to_seasonal,
    aggregate_to_annual,
    check_for_nans,
    interpolate_missing,
    remove_autocorrelation_lag,
    split_train_test,
)

__all__ = [
    # Normalizers
    'Normalizer',
    'ZScoreNormalizer',
    'RobustNormalizer',
    'MinMaxNormalizer',
    'RankNormalizer',
    'SqrtMinMaxNormalizer',
    'Log1pNormalizer',
    'WinsorizingNormalizer',
    'get_normalizer',
    'normalize_dataframe',
    'inverse_normalize_dataframe',
    # Detrending
    'detrend_linear',
    'detrend_polynomial',
    'detrend_stl',
    'detrend_hp',
    'detrend_loess',
    'detrend_rolling_mean',
    'detrend_savgol',
    'detrend_seasonal_diff',
    'get_detrending_method',
    'detrend_dataframe',
    # Utils
    'add_time_column',
    'reset_to_quarter_start',
    'aggregate_to_seasonal',
    'aggregate_to_annual',
    'check_for_nans',
    'interpolate_missing',
    'remove_autocorrelation_lag',
    'split_train_test',
]
