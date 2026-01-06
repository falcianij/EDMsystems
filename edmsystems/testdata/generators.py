"""
Test data generators with known ground truth for CCM validation.

Based on "Surrogate Test 11-20-25.ipynb" logic.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy.ndimage import shift


def make_test_dataframe(n: int = 500,
                       seed: Optional[int] = None,
                       start_date: str = '2000-01-01',
                       freq: str = 'D') -> pd.DataFrame:
    """
    Create comprehensive test dataframe with known causal relationships.

    Generates multiple scenarios including:
    - Independent (no relationship)
    - Correlated (non-causal)
    - Lag-correlated (non-causal)
    - Correlated & autocorrelated (non-causal)
    - Pure autocorrelated (non-causal)
    - Seasonal synchronous (non-causal)
    - Seasonal lagged (non-causal)
    - Unidirectional causal (X -> Y)
    - Bidirectional causal (X <-> Y)
    - Weak causal (X -> Y, weak)

    Parameters
    ----------
    n : int, default 500
        Number of time points
    seed : int or None
        Random seed for reproducibility
    start_date : str, default '2000-01-01'
        Start date for datetime index
    freq : str, default 'D'
        Frequency for datetime index ('D'=daily, 'M'=monthly, etc.)

    Returns
    -------
    pd.DataFrame
        Dataframe with datetime column and all test scenarios

    Notes
    -----
    Ground truth causal edges:
    - unidirectional: X -> Y
    - bidirectional: X <-> Y
    - weak: X -> Y (weak coupling)
    All other pairs have NO causal relationship.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate datetime index
    datetime_index = pd.date_range(start=start_date, periods=n, freq=freq)

    # 1. Independent
    X_indep = np.random.randn(n)
    Y_indep = np.random.randn(n)

    # 2. Correlated (non-causal)
    X_corr = np.random.randn(n)
    rho = 0.8
    noise = np.random.randn(n)
    Y_corr = rho * X_corr + np.sqrt(1 - rho**2) * noise

    # 3. Lag-correlated (non-causal)
    X_lagcorr = X_corr.copy()
    Y_lagcorr = shift(Y_corr, shift=3, cval=np.nan)

    # 4. Correlated AND autocorrelated (non-causal)
    rho_cross = 0.7
    rho_auto = 0.6
    X_auto = np.random.randn(n)
    Y_auto = np.zeros(n)
    Y_auto[0] = np.random.randn()
    for t in range(1, n):
        Y_auto[t] = (rho_auto * Y_auto[t-1] +
                     rho_cross * X_auto[t] +
                     np.sqrt(1 - rho_auto**2 - rho_cross**2) * np.random.randn())

    # 5. Pure autocorrelated (non-causal)
    rho_auto = 0.8
    X_pure_auto = np.random.randn(n)
    Y_pure_auto = np.zeros(n)
    Y_pure_auto[0] = np.random.randn()
    for t in range(1, n):
        Y_pure_auto[t] = rho_auto * Y_pure_auto[t-1] + np.sqrt(1 - rho_auto**2) * np.random.randn()

    # 6. Seasonal synchronous (non-causal)
    period = 24
    t = np.arange(n)
    seasonal = np.sin(2 * np.pi * t / period)
    X_seasonal_sync = seasonal + 0.3 * np.random.randn(n)
    Y_seasonal_sync = seasonal + 0.3 * np.random.randn(n)

    # 7. Seasonal lagged (non-causal)
    phase_diff = np.pi / 6
    X_seasonal_lag = seasonal + 0.3 * np.random.randn(n)
    Y_seasonal_lag = np.sin(2 * np.pi * t / period + phase_diff) + 0.3 * np.random.randn(n)

    # 8. Unidirectional causal (X -> Y)
    coupling = 0.6
    noise_level = 0.05
    rX = 3.7
    rY = 3.6
    X_causal = np.zeros(n)
    Y_causal = np.zeros(n)
    X_causal[0], Y_causal[0] = np.random.rand(), np.random.rand()
    for t in range(1, n):
        X_causal[t] = rX * X_causal[t-1] * (1 - X_causal[t-1]) + noise_level * np.random.randn()
        rY_eff = rY * (1 + coupling * (X_causal[t-1] - 0.5))
        Y_causal[t] = rY_eff * Y_causal[t-1] * (1 - Y_causal[t-1]) + noise_level * np.random.randn()
        X_causal[t] = np.clip(X_causal[t], 1e-12, 1-1e-12)
        Y_causal[t] = np.clip(Y_causal[t], 1e-12, 1-1e-12)

    # 9. Bidirectional causal (X <-> Y)
    coupling_XY = 0.4
    coupling_YX = 0.4
    noise_level = 0.05
    X_bidir = np.zeros(n)
    Y_bidir = np.zeros(n)
    X_bidir[0] = np.random.rand()
    Y_bidir[0] = np.random.rand()
    for t in range(1, n):
        rX_eff = rX * (1 + coupling_YX * (Y_bidir[t-1] - 0.5))
        rY_eff = rY * (1 + coupling_XY * (X_bidir[t-1] - 0.5))
        X_bidir[t] = rX_eff * X_bidir[t-1] * (1 - X_bidir[t-1]) + noise_level * np.random.randn()
        Y_bidir[t] = rY_eff * Y_bidir[t-1] * (1 - Y_bidir[t-1]) + noise_level * np.random.randn()
        X_bidir[t] = np.clip(X_bidir[t], 1e-6, 1 - 1e-6)
        Y_bidir[t] = np.clip(Y_bidir[t], 1e-6, 1 - 1e-6)

    # 10. Weak causal (X -> Y, weak)
    coupling = 0.2
    noise_level = 0.05
    X_weak = np.zeros(n)
    Y_weak = np.zeros(n)
    X_weak[0], Y_weak[0] = np.random.rand(), np.random.rand()
    for t in range(1, n):
        X_weak[t] = rX * X_weak[t-1] * (1 - X_weak[t-1]) + noise_level * np.random.randn()
        rY_eff = rY * (1 + coupling * (X_weak[t-1] - 0.5))
        Y_weak[t] = rY_eff * Y_weak[t-1] * (1 - Y_weak[t-1]) + noise_level * np.random.randn()
        X_weak[t] = np.clip(X_weak[t], 1e-12, 1-1e-12)
        Y_weak[t] = np.clip(Y_weak[t], 1e-12, 1-1e-12)

    # Assemble dataframe
    df = pd.DataFrame({
        'datetime': datetime_index,
        'independent_X': X_indep,
        'independent_Y': Y_indep,
        'correlated_X': X_corr,
        'correlated_Y': Y_corr,
        'lagcorr_X': X_lagcorr,
        'lagcorr_Y': Y_lagcorr,
        'crosscorr_X': X_auto,
        'crosscorr_Y': Y_auto,
        'autocorr_X': X_pure_auto,
        'autocorr_Y': Y_pure_auto,
        'seasonal_sync_X': X_seasonal_sync,
        'seasonal_sync_Y': Y_seasonal_sync,
        'seasonal_lag_X': X_seasonal_lag,
        'seasonal_lag_Y': Y_seasonal_lag,
        'unidirectional_X': X_causal,
        'unidirectional_Y': Y_causal,
        'bidirectional_X': X_bidir,
        'bidirectional_Y': Y_bidir,
        'weak_X': X_weak,
        'weak_Y': Y_weak,
    })

    return df


def get_ground_truth_network() -> pd.DataFrame:
    """
    Get ground truth causal network for test dataframe.

    Returns
    -------
    pd.DataFrame
        Adjacency matrix where rows=drivers, columns=targets.
        Value of 1 indicates true causal edge, 0 indicates no causation.

    Notes
    -----
    True causal edges:
    - unidirectional_X -> unidirectional_Y
    - bidirectional_X -> bidirectional_Y
    - bidirectional_Y -> bidirectional_X
    - weak_X -> weak_Y

    All other pairs have NO true causal relationship.
    """
    variables = [
        'independent_X', 'independent_Y',
        'correlated_X', 'correlated_Y',
        'lagcorr_X', 'lagcorr_Y',
        'crosscorr_X', 'crosscorr_Y',
        'autocorr_X', 'autocorr_Y',
        'seasonal_sync_X', 'seasonal_sync_Y',
        'seasonal_lag_X', 'seasonal_lag_Y',
        'unidirectional_X', 'unidirectional_Y',
        'bidirectional_X', 'bidirectional_Y',
        'weak_X', 'weak_Y',
    ]

    # Initialize adjacency matrix (all zeros)
    n_vars = len(variables)
    adjacency = np.zeros((n_vars, n_vars), dtype=int)

    # Create variable index mapping
    var_idx = {var: i for i, var in enumerate(variables)}

    # Add true causal edges
    # Unidirectional: X -> Y
    adjacency[var_idx['unidirectional_X'], var_idx['unidirectional_Y']] = 1

    # Bidirectional: X <-> Y
    adjacency[var_idx['bidirectional_X'], var_idx['bidirectional_Y']] = 1
    adjacency[var_idx['bidirectional_Y'], var_idx['bidirectional_X']] = 1

    # Weak: X -> Y
    adjacency[var_idx['weak_X'], var_idx['weak_Y']] = 1

    # Create dataframe
    df = pd.DataFrame(adjacency, index=variables, columns=variables)

    return df
