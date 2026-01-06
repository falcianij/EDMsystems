"""
EDMsystems: A modular package for Empirical Dynamical Modeling (EDM) analysis.

This package provides standardized tools for:
- Preprocessing time series data (normalization, detrending)
- Convergent Cross Mapping (CCM) analysis
- Parameter optimization (E, tau, theta)
- Surrogate-based significance testing
"""

__version__ = "0.1.0"

# Import main modules for convenient access
from . import preprocessing
from . import ccm
from . import surrogates

# Import key functions for direct access
from .preprocessing import (
    get_normalizer,
    normalize_dataframe,
    inverse_normalize_dataframe,
    get_detrending_method,
    detrend_dataframe,
)

from .ccm import (
    ccm_analysis,
    ccm_with_significance,
    find_optimal_E_tau,
    auto_optimize_parameters,
)

from .surrogates import (
    generate_seasonal_pair_surrogates,
    empirical_p,
    test_significance,
)

__all__ = [
    'preprocessing',
    'ccm',
    'surrogates',
    # Preprocessing
    'get_normalizer',
    'normalize_dataframe',
    'inverse_normalize_dataframe',
    'get_detrending_method',
    'detrend_dataframe',
    # CCM
    'ccm_analysis',
    'ccm_with_significance',
    'find_optimal_E_tau',
    'auto_optimize_parameters',
    # Surrogates
    'generate_seasonal_pair_surrogates',
    'empirical_p',
    'test_significance',
]
