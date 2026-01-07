"""
Preprocessing utilities for standardized CCM workflow.

Provides normalization, detrending, and aggregation functions for time series data.
"""

from .aggregation import (
    aggregate_temporal,
    fill_missing_values,
    align_time_series,
    create_lagged_features,
)

from .transforms import (
    SeriesTransformer,
    TransformParams,
    normalize,
    denormalize,
    detrend,
    retrend,
    preprocess_for_ccm,
)

__all__ = [
    # Aggregation
    'aggregate_temporal',
    'fill_missing_values',
    'align_time_series',
    'create_lagged_features',
    # Transformations
    'SeriesTransformer',
    'TransformParams',
    'normalize',
    'denormalize',
    'detrend',
    'retrend',
    'preprocess_for_ccm',
]
