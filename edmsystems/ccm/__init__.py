"""
Convergent Cross Mapping (CCM) for causal inference.

This module provides a standardized workflow for CCM analysis including:
- Parameter optimization (tau, E, Tp, theta)
- Random library sampling for robust convergence testing
- Multiple surrogate testing methods
- Parallel processing support
- Ground truth comparison
"""

from .core import (
    compute_xmap,
    compute_ccm,
    compute_auc,
)

from .parameters import (
    find_optimal_tau,
    find_optimal_E,
    find_optimal_Tp,
    find_optimal_theta_smap,
    optimize_parameters,
)

from .workflow import (
    run_ccm_workflow,
    test_ccm_pair,
)

from .analysis import (
    compare_to_ground_truth,
    compute_performance_metrics,
    summarize_results,
)

__all__ = [
    # Core functions
    'compute_xmap',
    'compute_ccm',
    'compute_auc',
    # Parameter optimization
    'find_optimal_tau',
    'find_optimal_E',
    'find_optimal_Tp',
    'find_optimal_theta_smap',
    'optimize_parameters',
    # Workflow
    'run_ccm_workflow',
    'test_ccm_pair',
    # Analysis
    'compare_to_ground_truth',
    'compute_performance_metrics',
    'summarize_results',
]
