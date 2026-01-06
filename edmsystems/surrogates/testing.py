"""
Statistical significance testing using surrogate time series.

This module provides functions for computing empirical p-values and
significance thresholds from surrogate distributions.
"""

import numpy as np
from typing import Literal, Tuple, Optional
import pandas as pd


def empirical_p(observed: float,
               surrogates: np.ndarray,
               tail: Literal["greater", "less", "two-sided"] = "greater") -> float:
    """
    Calculate empirical p-value from surrogate distribution.

    Parameters
    ----------
    observed : float
        Observed test statistic
    surrogates : np.ndarray
        Array of surrogate test statistics
    tail : {'greater', 'less', 'two-sided'}, default 'greater'
        Type of test:
        - 'greater': Test if observed > surrogates (causal signal)
        - 'less': Test if observed < surrogates
        - 'two-sided': Test if observed differs from surrogates

    Returns
    -------
    float
        Empirical p-value

    Notes
    -----
    Uses +1 trick: p = (k + 1) / (n + 1) to avoid p=0
    """
    surrogates = np.asarray(surrogates)
    n = len(surrogates)

    if tail == "greater":
        k = np.sum(surrogates >= observed)
    elif tail == "less":
        k = np.sum(surrogates <= observed)
    elif tail == "two-sided":
        # Distance from median, two-sided
        med = np.median(surrogates)
        k = np.sum(np.abs(surrogates - med) >= np.abs(observed - med))
    else:
        raise ValueError("tail must be 'greater', 'less', or 'two-sided'")

    # +1 trick to avoid p=0
    return (k + 1) / (n + 1)


def compute_significance_thresholds(surrogates: np.ndarray,
                                   percentiles: Tuple[float, ...] = (95, 99)) -> dict:
    """
    Compute significance thresholds from surrogate distribution.

    Parameters
    ----------
    surrogates : np.ndarray
        Array of surrogate test statistics
    percentiles : tuple of float, default (95, 99)
        Percentiles for thresholds

    Returns
    -------
    dict
        Dictionary mapping percentile to threshold value
    """
    thresholds = {}

    for p in percentiles:
        thresholds[f'p{int(p)}'] = np.percentile(surrogates, p)

    return thresholds


def test_significance(observed: float,
                     surrogates: np.ndarray,
                     alpha: float = 0.05,
                     tail: str = "greater") -> dict:
    """
    Test significance of observed statistic against surrogates.

    Parameters
    ----------
    observed : float
        Observed test statistic
    surrogates : np.ndarray
        Surrogate distribution
    alpha : float, default 0.05
        Significance level
    tail : str, default 'greater'
        Type of test

    Returns
    -------
    dict
        Results with p-value, significance, and summary statistics
    """
    p_value = empirical_p(observed, surrogates, tail=tail)
    significant = p_value < alpha

    result = {
        'observed': observed,
        'p_value': p_value,
        'significant': significant,
        'alpha': alpha,
        'n_surrogates': len(surrogates),
        'surr_mean': np.mean(surrogates),
        'surr_std': np.std(surrogates),
        'surr_median': np.median(surrogates),
        'surr_95p': np.percentile(surrogates, 95),
        'surr_99p': np.percentile(surrogates, 99),
    }

    return result


def fdr_correction(p_values: np.ndarray,
                  alpha: float = 0.05,
                  method: Literal["bh", "by"] = "bh") -> Tuple[np.ndarray, float]:
    """
    False Discovery Rate correction for multiple testing.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values
    alpha : float, default 0.05
        Family-wise error rate
    method : {'bh', 'by'}, default 'bh'
        - 'bh': Benjamini-Hochberg procedure
        - 'by': Benjamini-Yekutieli procedure (more conservative)

    Returns
    -------
    significant : np.ndarray
        Boolean array indicating significance after correction
    threshold : float
        Adjusted p-value threshold

    References
    ----------
    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate:
    a practical and powerful approach to multiple testing.
    Journal of the Royal Statistical Society, Series B, 57(1), 289-300.
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # Compute thresholds
    if method == "bh":
        # Benjamini-Hochberg
        thresholds = (np.arange(1, n + 1) / n) * alpha
    elif method == "by":
        # Benjamini-Yekutieli (accounts for dependency)
        c = np.sum(1.0 / np.arange(1, n + 1))
        thresholds = (np.arange(1, n + 1) / (n * c)) * alpha
    else:
        raise ValueError("method must be 'bh' or 'by'")

    # Find largest i where p(i) <= threshold(i)
    significant_sorted = sorted_p <= thresholds
    if np.any(significant_sorted):
        max_i = np.where(significant_sorted)[0][-1]
        threshold = thresholds[max_i]

        # Create boolean array for original order
        significant = np.zeros(n, dtype=bool)
        significant[sorted_indices[:max_i + 1]] = True
    else:
        threshold = 0.0
        significant = np.zeros(n, dtype=bool)

    return significant, threshold


def bonferroni_correction(p_values: np.ndarray,
                         alpha: float = 0.05) -> Tuple[np.ndarray, float]:
    """
    Bonferroni correction for multiple testing.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values
    alpha : float, default 0.05
        Family-wise error rate

    Returns
    -------
    significant : np.ndarray
        Boolean array indicating significance after correction
    threshold : float
        Adjusted p-value threshold
    """
    n = len(p_values)
    threshold = alpha / n
    significant = p_values < threshold

    return significant, threshold


def summarize_surrogate_test(observed: float,
                            surrogates: np.ndarray,
                            alpha: float = 0.05,
                            tail: str = "greater") -> pd.DataFrame:
    """
    Create a summary table for surrogate testing results.

    Parameters
    ----------
    observed : float
        Observed statistic
    surrogates : np.ndarray
        Surrogate distribution
    alpha : float, default 0.05
        Significance level
    tail : str, default 'greater'
        Test type

    Returns
    -------
    pd.DataFrame
        Summary table
    """
    result = test_significance(observed, surrogates, alpha=alpha, tail=tail)

    summary = pd.DataFrame({
        'Metric': ['Observed', 'Surr Mean', 'Surr Median', 'Surr Std',
                  '95th Percentile', '99th Percentile', 'P-value', 'Significant'],
        'Value': [
            result['observed'],
            result['surr_mean'],
            result['surr_median'],
            result['surr_std'],
            result['surr_95p'],
            result['surr_99p'],
            result['p_value'],
            result['significant']
        ]
    })

    return summary


def batch_empirical_p(observed: np.ndarray,
                     surrogates: np.ndarray,
                     tail: str = "greater") -> np.ndarray:
    """
    Compute empirical p-values for multiple observations.

    Parameters
    ----------
    observed : np.ndarray, shape (n_tests,)
        Array of observed statistics
    surrogates : np.ndarray, shape (n_tests, n_surr) or (n_surr,)
        Surrogate distributions (can be same for all or specific to each)
    tail : str, default 'greater'
        Test type

    Returns
    -------
    np.ndarray, shape (n_tests,)
        Array of p-values
    """
    observed = np.asarray(observed)
    surrogates = np.asarray(surrogates)

    if surrogates.ndim == 1:
        # Same surrogate distribution for all tests
        p_values = np.array([empirical_p(obs, surrogates, tail) for obs in observed])
    elif surrogates.ndim == 2:
        # Specific surrogate distribution for each test
        assert len(observed) == len(surrogates), \
            "Number of observations must match number of surrogate distributions"
        p_values = np.array([empirical_p(obs, surr, tail)
                           for obs, surr in zip(observed, surrogates)])
    else:
        raise ValueError("surrogates must be 1D or 2D array")

    return p_values
