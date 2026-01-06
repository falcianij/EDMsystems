"""
Surrogate time series generation and significance testing for EDM.

This module provides methods for generating null models that preserve
different statistical properties while destroying causal temporal structure.
"""

from .generators import (
    generate_random_surrogates,
    generate_iaaft_surrogates,
    generate_seasonal_surrogates,
    generate_seasonal_pair_surrogates,
    generate_random_pair_surrogates,
)

from .testing import (
    empirical_p,
    compute_significance_thresholds,
    test_significance,
    fdr_correction,
    bonferroni_correction,
    summarize_surrogate_test,
    batch_empirical_p,
)

__all__ = [
    # Generators
    'generate_random_surrogates',
    'generate_iaaft_surrogates',
    'generate_seasonal_surrogates',
    'generate_seasonal_pair_surrogates',
    'generate_random_pair_surrogates',
    # Testing
    'empirical_p',
    'compute_significance_thresholds',
    'test_significance',
    'fdr_correction',
    'bonferroni_correction',
    'summarize_surrogate_test',
    'batch_empirical_p',
]
