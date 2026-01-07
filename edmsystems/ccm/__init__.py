"""
Convergent Cross Mapping module for EDM analysis.

Provides CCM workflow, parameter optimization, and analysis functions.
"""

from .parameters import (
    optimize_parameters,
    find_optimal_tau,
    find_optimal_E,
    find_optimal_Tp,
    acf_nan,
)

from .workflow import (
    compute_ccm_pair,
    run_ccm_workflow,
    fit_ccm_curve,
    has_converged,
)

from .analysis import (
    compare_to_ground_truth,
    compute_performance_metrics,
    summarize_results,
    create_adjacency_matrix,
)

__all__ = [
    # Parameters
    'optimize_parameters',
    'find_optimal_tau',
    'find_optimal_E',
    'find_optimal_Tp',
    'acf_nan',
    # Workflow
    'compute_ccm_pair',
    'run_ccm_workflow',
    'fit_ccm_curve',
    'has_converged',
    # Analysis
    'compare_to_ground_truth',
    'compute_performance_metrics',
    'summarize_results',
    'create_adjacency_matrix',
]
