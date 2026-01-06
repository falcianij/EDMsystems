"""
Test data generation functions for EDM validation and demonstration.

This module provides functions to generate synthetic time series with known
ground truth relationships for testing CCM and other EDM methods.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def make_independent_series(n: int = 500, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate pure white noise with no relationship.

    Ground truth: No causality, no correlation.

    Parameters
    ----------
    n : int, default 500
        Number of time points
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    X, Y : np.ndarray, shape (n,)
        Independent white noise time series
    """
    if seed is not None:
        np.random.seed(seed)
    X = np.random.randn(n)
    Y = np.random.randn(n)
    return X, Y


def make_correlated_series(n: int = 500, rho: float = 0.8, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate correlated series without causal dynamics.

    Ground truth: Correlation but no causality (instantaneous correlation only).

    Parameters
    ----------
    n : int, default 500
        Number of time points
    rho : float, default 0.8
        Correlation coefficient between X and Y
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    X, Y : np.ndarray, shape (n,)
        Correlated but non-causal time series
    """
    if seed is not None:
        np.random.seed(seed)
    X = np.random.randn(n)
    noise = np.random.randn(n)
    Y = rho * X + np.sqrt(1 - rho**2) * noise
    return X, Y


def make_correlated_autocorrelated_series(n: int = 500,
                                          rho_cross: float = 0.8,
                                          rho_auto: float = 0.7,
                                          seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate X and Y that are cross-correlated and Y is autocorrelated.

    Ground truth: Y has autocorrelation, X drives Y (instantaneous correlation),
    but X has no autocorrelation.

    Parameters
    ----------
    n : int, default 500
        Number of time points
    rho_cross : float, default 0.8
        Cross-correlation between X and Y
    rho_auto : float, default 0.7
        Autocorrelation in Y (AR(1) coefficient)
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    X, Y : np.ndarray, shape (n,)
        Time series where X is white noise and Y is autocorrelated + cross-correlated
    """
    if seed is not None:
        np.random.seed(seed)

    # X is white noise (no autocorrelation)
    X = np.random.randn(n)

    # Y is AR(1) process influenced by X
    Y = np.zeros(n)
    Y[0] = np.random.randn()

    for t in range(1, n):
        # Y(t) depends on:
        # 1. Its own past (autocorrelation)
        # 2. Current X value (cross-correlation)
        # 3. Independent noise
        Y[t] = (rho_auto * Y[t-1] +
                rho_cross * X[t] +
                np.sqrt(1 - rho_auto**2 - rho_cross**2) * np.random.randn())

    return X, Y


def make_pure_autocorrelated_series(n: int = 500,
                                    rho_auto: float = 0.7,
                                    seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Y that is autocorrelated, X is independent white noise.

    Ground truth: No relationship between X and Y at all.
    Y has internal dynamics, X does not.

    Parameters
    ----------
    n : int, default 500
        Number of time points
    rho_auto : float, default 0.7
        Autocorrelation coefficient in Y (AR(1))
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    X, Y : np.ndarray, shape (n,)
        X is white noise, Y is AR(1) process
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.random.randn(n)  # Independent
    Y = np.zeros(n)
    Y[0] = np.random.randn()

    for t in range(1, n):
        Y[t] = rho_auto * Y[t-1] + np.sqrt(1 - rho_auto**2) * np.random.randn()

    return X, Y


def make_seasonal_series(n: int = 500,
                        period: int = 24,
                        phase_diff: float = 0,
                        seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate series driven by same seasonal signal.

    Ground truth: Both driven by external forcing (common cause).
    No direct causal relationship between X and Y.

    Parameters
    ----------
    n : int, default 500
        Number of time points
    period : int, default 24
        Period of seasonal cycle
    phase_diff : float, default 0
        Phase difference in radians (0 = synchronous)
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    X, Y : np.ndarray, shape (n,)
        Seasonally forced time series
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(n)
    seasonal = np.sin(2 * np.pi * t / period)
    X = seasonal + 0.3 * np.random.randn(n)
    Y = np.sin(2 * np.pi * t / period + phase_diff) + 0.3 * np.random.randn(n)

    return X, Y


def make_causal_series(n: int = 500,
                      coupling: float = 0.2,
                      noise: float = 0.01,
                      seed: Optional[int] = None,
                      rX: float = 3.7,
                      rY: float = 3.6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate unidirectional causal system X -> Y.

    Ground truth: X causally drives Y via multiplicative (parameter) coupling.
    X is autonomous chaotic logistic map.
    Y dynamics depend on X.

    Parameters
    ----------
    n : int, default 500
        Number of time points
    coupling : float, default 0.2
        Coupling strength from X to Y
    noise : float, default 0.01
        Dynamical noise level
    seed : int or None
        Random seed for reproducibility
    rX : float, default 3.7
        Logistic map parameter for X
    rY : float, default 3.6
        Base logistic map parameter for Y

    Returns
    -------
    X, Y : np.ndarray, shape (n,)
        Coupled logistic maps with X -> Y causality
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.zeros(n)
    Y = np.zeros(n)
    X[0], Y[0] = np.random.rand(), np.random.rand()

    for t in range(1, n):
        X[t] = rX * X[t-1] * (1 - X[t-1]) + noise * np.random.randn()
        rY_eff = rY * (1 + coupling * (X[t-1] - 0.5))
        Y[t] = rY_eff * Y[t-1] * (1 - Y[t-1]) + noise * np.random.randn()

        # Tiny numeric safety bounds (not hard clipping)
        X[t] = np.clip(X[t], 1e-12, 1 - 1e-12)
        Y[t] = np.clip(Y[t], 1e-12, 1 - 1e-12)

    return X, Y


def make_bidirectional_causal_series(n: int = 500,
                                     coupling_XY: float = 0.4,
                                     coupling_YX: float = 0.4,
                                     noise: float = 0.05,
                                     seed: Optional[int] = None,
                                     rX: float = 3.7,
                                     rY: float = 3.6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate bidirectional causal system X <-> Y.

    Ground truth: X drives Y AND Y drives X (bidirectional causality).
    Uses multiplicative parameter coupling so the system stays bounded
    and remains chaotic without hard clipping.

    Parameters
    ----------
    n : int, default 500
        Number of time points
    coupling_XY : float, default 0.4
        Coupling strength from X to Y
    coupling_YX : float, default 0.4
        Coupling strength from Y to X
    noise : float, default 0.05
        Dynamical noise level
    seed : int or None
        Random seed for reproducibility
    rX : float, default 3.7
        Base logistic map parameter for X
    rY : float, default 3.6
        Base logistic map parameter for Y

    Returns
    -------
    X, Y : np.ndarray, shape (n,)
        Bidirectionally coupled logistic maps
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.zeros(n)
    Y = np.zeros(n)
    X[0] = np.random.rand()
    Y[0] = np.random.rand()

    for t in range(1, n):
        # Multiplicative parameter coupling:
        # The driver modulates the effective growth rate
        rX_eff = rX * (1 + coupling_YX * (Y[t-1] - 0.5))
        rY_eff = rY * (1 + coupling_XY * (X[t-1] - 0.5))

        # Update with effective parameters
        X[t] = rX_eff * X[t-1] * (1 - X[t-1]) + noise * np.random.randn()
        Y[t] = rY_eff * Y[t-1] * (1 - Y[t-1]) + noise * np.random.randn()

        # Keep values inside the logistic basin without hard clipping
        X[t] = np.clip(X[t], 1e-6, 1 - 1e-6)
        Y[t] = np.clip(Y[t], 1e-6, 1 - 1e-6)

    return X, Y


def make_indirect_causal_series(n: int = 500,
                               coupling_XZ: float = 0.2,
                               coupling_ZY: float = 0.2,
                               noise: float = 0.02,
                               seed: Optional[int] = None,
                               rX: float = 3.7,
                               rZ: float = 3.8,
                               rY: float = 3.6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate indirect causal chain X -> Z -> Y.

    Ground truth: X drives Z, Z drives Y (indirect causality).
    No direct causal link from X to Y.
    Uses multiplicative (parameter) coupling.

    Parameters
    ----------
    n : int, default 500
        Number of time points
    coupling_XZ : float, default 0.2
        Coupling strength from X to Z
    coupling_ZY : float, default 0.2
        Coupling strength from Z to Y
    noise : float, default 0.02
        Dynamical noise level
    seed : int or None
        Random seed for reproducibility
    rX : float, default 3.7
        Logistic map parameter for X
    rZ : float, default 3.8
        Base logistic map parameter for Z
    rY : float, default 3.6
        Base logistic map parameter for Y

    Returns
    -------
    X, Z, Y : np.ndarray, shape (n,)
        Causal chain time series
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.zeros(n)
    Z = np.zeros(n)
    Y = np.zeros(n)
    X[0], Z[0], Y[0] = np.random.rand(), np.random.rand(), np.random.rand()

    for t in range(1, n):
        # Autonomous X (logistic)
        X[t] = rX * X[t-1] * (1 - X[t-1]) + noise * np.random.randn()

        # Effective r for Z is modulated by X[t-1]
        rZ_eff = rZ * (1 + coupling_XZ * (X[t-1] - 0.5))
        Z[t] = rZ_eff * Z[t-1] * (1 - Z[t-1]) + noise * np.random.randn()

        # Effective r for Y is modulated by Z[t-1] (no direct X->Y)
        rY_eff = rY * (1 + coupling_ZY * (Z[t-1] - 0.5))
        Y[t] = rY_eff * Y[t-1] * (1 - Y[t-1]) + noise * np.random.randn()

        # Tiny numerical safety clamp (not hard clipping)
        X[t] = np.clip(X[t], 1e-12, 1 - 1e-12)
        Z[t] = np.clip(Z[t], 1e-12, 1 - 1e-12)
        Y[t] = np.clip(Y[t], 1e-12, 1 - 1e-12)

    return X, Z, Y


def make_test_dataframe(n: int = 500, seed: Optional[int] = None, start_date: str = '2000-01-01') -> pd.DataFrame:
    """
    Create a comprehensive test dataframe with all scenario types.

    This creates a single dataframe containing multiple time series with
    known ground truth relationships for testing CCM and network detection.

    Parameters
    ----------
    n : int, default 500
        Number of time points
    seed : int or None
        Random seed for reproducibility
    start_date : str, default '2000-01-01'
        Starting date for the datetime index

    Returns
    -------
    df : pd.DataFrame
        Dataframe with columns:
        - datetime: Fake datetime index
        - independent_X, independent_Y: No relationship
        - correlated_X, correlated_Y: Correlated but no causality
        - crosscorr_X, crosscorr_Y: Cross-correlated with autocorrelation
        - autocorr_X, autocorr_Y: Pure autocorrelation, no relationship
        - seasonal_X, seasonal_Y: Common seasonal forcing
        - causal_X, causal_Y: X -> Y unidirectional
        - bidirect_X, bidirect_Y: X <-> Y bidirectional
        - indirect_X, indirect_Z, indirect_Y: X -> Z -> Y chain

    Notes
    -----
    Ground truth network:
    - independent: No edges
    - correlated: No edges (correlation is not causation)
    - crosscorr: No causal edges (instantaneous correlation only)
    - autocorr: No edges
    - seasonal: No edges (common cause, not direct causation)
    - causal: X -> Y
    - bidirect: X <-> Y
    - indirect: X -> Z -> Y (no direct X -> Y)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate all scenarios
    ind_X, ind_Y = make_independent_series(n, seed)
    corr_X, corr_Y = make_correlated_series(n, seed=seed+1 if seed else None)
    cross_X, cross_Y = make_correlated_autocorrelated_series(n, seed=seed+2 if seed else None)
    auto_X, auto_Y = make_pure_autocorrelated_series(n, seed=seed+3 if seed else None)
    seas_X, seas_Y = make_seasonal_series(n, seed=seed+4 if seed else None)
    caus_X, caus_Y = make_causal_series(n, seed=seed+5 if seed else None)
    bid_X, bid_Y = make_bidirectional_causal_series(n, seed=seed+6 if seed else None)
    indir_X, indir_Z, indir_Y = make_indirect_causal_series(n, seed=seed+7 if seed else None)

    # Create datetime index
    datetime_index = pd.date_range(start=start_date, periods=n, freq='D')

    # Assemble dataframe
    df = pd.DataFrame({
        'datetime': datetime_index,
        'independent_X': ind_X,
        'independent_Y': ind_Y,
        'correlated_X': corr_X,
        'correlated_Y': corr_Y,
        'crosscorr_X': cross_X,
        'crosscorr_Y': cross_Y,
        'autocorr_X': auto_X,
        'autocorr_Y': auto_Y,
        'seasonal_X': seas_X,
        'seasonal_Y': seas_Y,
        'causal_X': caus_X,
        'causal_Y': caus_Y,
        'bidirect_X': bid_X,
        'bidirect_Y': bid_Y,
        'indirect_X': indir_X,
        'indirect_Z': indir_Z,
        'indirect_Y': indir_Y,
    })

    return df


def get_ground_truth_network() -> pd.DataFrame:
    """
    Return the ground truth causal network for the test dataframe.

    Returns
    -------
    df : pd.DataFrame
        Adjacency matrix where rows are drivers and columns are targets.
        Value of 1 indicates true causal relationship, 0 indicates no causation.

    Notes
    -----
    The ground truth is:
    - independent: No causation
    - correlated: No causation (correlation ≠ causation)
    - crosscorr: No causation (instantaneous correlation only)
    - autocorr: No causation
    - seasonal: No causation (common external driver)
    - causal: X -> Y (unidirectional)
    - bidirect: X <-> Y (bidirectional)
    - indirect: X -> Z -> Y (no direct X -> Y edge)
    """
    # List all variables
    variables = [
        'independent_X', 'independent_Y',
        'correlated_X', 'correlated_Y',
        'crosscorr_X', 'crosscorr_Y',
        'autocorr_X', 'autocorr_Y',
        'seasonal_X', 'seasonal_Y',
        'causal_X', 'causal_Y',
        'bidirect_X', 'bidirect_Y',
        'indirect_X', 'indirect_Z', 'indirect_Y',
    ]

    # Initialize adjacency matrix (all zeros)
    n_vars = len(variables)
    adjacency = np.zeros((n_vars, n_vars), dtype=int)

    # Create variable index mapping
    var_idx = {var: i for i, var in enumerate(variables)}

    # Add true causal edges
    # causal: X -> Y
    adjacency[var_idx['causal_X'], var_idx['causal_Y']] = 1

    # bidirect: X <-> Y
    adjacency[var_idx['bidirect_X'], var_idx['bidirect_Y']] = 1
    adjacency[var_idx['bidirect_Y'], var_idx['bidirect_X']] = 1

    # indirect: X -> Z -> Y
    adjacency[var_idx['indirect_X'], var_idx['indirect_Z']] = 1
    adjacency[var_idx['indirect_Z'], var_idx['indirect_Y']] = 1

    # Create dataframe
    df = pd.DataFrame(adjacency, index=variables, columns=variables)

    return df
