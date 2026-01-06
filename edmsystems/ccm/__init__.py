"""
Convergent Cross Mapping module for EDM analysis.

This module provides standardized CCM analysis with convergence testing
and parameter optimization.
"""

from .core import (
    saturating_curve,
    fit_ccm_curve,
    has_converged,
    ccm_analysis,
    ccm_with_significance,
)

from .optimization import (
    find_optimal_E_tau,
    find_tau_autocorrelation,
    optimize_simplex_E,
    grid_search_parameters,
    auto_optimize_parameters,
    optimize_parameters_edm_standard,
)

__all__ = [
    # Core CCM
    'saturating_curve',
    'fit_ccm_curve',
    'has_converged',
    'ccm_analysis',
    'ccm_with_significance',
    # Optimization
    'find_optimal_E_tau',
    'find_tau_autocorrelation',
    'optimize_simplex_E',
    'grid_search_parameters',
    'auto_optimize_parameters',
    'optimize_parameters_edm_standard',
]
