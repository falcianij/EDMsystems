"""
Parameter optimization for EDM following rigorous standards.

This module implements the EDM-standard parameter selection procedure:
1. Find τ where ACF crosses threshold (0.1)
2. Find E that best reconstructs target attractor (within n_std of max)
3. Find Tp that maximizes cross-mapping (within n_std of max)

Based on: Surrogate Test 11-20-25.ipynb
"""

import numpy as np
import pandas as pd
from typing import Tuple, Iterable, Optional, Dict
from statsmodels.tsa.stattools import acf as statsmodels_acf

try:
    from pyEDM import SMap, Simplex
    PYEDM_AVAILABLE = True
except ImportError:
    PYEDM_AVAILABLE = False


def acf_nan(y: np.ndarray, nlags: int = 20) -> np.ndarray:
    """
    Compute autocorrelation with NaN handling.

    Uses pairwise-complete observations for each lag.

    Parameters
    ----------
    y : np.ndarray
        Input time series with possible NaNs
    nlags : int, default 20
        Maximum lag to compute

    Returns
    -------
    np.ndarray
        ACF values of length nlags+1
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    acf_vals = []

    # Centered series (mean computed on non-NaN)
    y_mean = np.nanmean(y)
    y_centered = y - y_mean

    # Denominator: variance using non-NaN values
    denom = np.nansum(y_centered ** 2)

    if denom == 0:
        return np.full(nlags + 1, np.nan)

    for lag in range(nlags + 1):
        # Indices that are valid for this lag
        valid = (~np.isnan(y_centered[:n-lag])) & (~np.isnan(y_centered[lag:]))

        if not np.any(valid):
            acf_vals.append(np.nan)
        else:
            num = np.sum(y_centered[:n-lag][valid] * y_centered[lag:][valid])
            acf_vals.append(num / denom)

    return np.array(acf_vals)


def find_optimal_tau(Y: np.ndarray,
                    max_lag: int = 20,
                    threshold: float = 0.1,
                    verbose: bool = True) -> Tuple[int, np.ndarray]:
    """
    Step 1: Find τ where ACF(Y) drops below threshold.

    Parameters
    ----------
    Y : np.ndarray
        Target time series
    max_lag : int, default 20
        Maximum lag to test
    threshold : float, default 0.1
        ACF threshold for tau selection (EDM standard)
    verbose : bool, default True
        Print selection details

    Returns
    -------
    tau_optimal : int
        Optimal time lag
    acf_values : np.ndarray
        ACF values for diagnostics
    """
    acf_values = acf_nan(Y, nlags=max_lag)

    # Find first lag where |ACF| < threshold
    tau_optimal = None
    for lag in range(1, len(acf_values)):
        if np.abs(acf_values[lag]) < threshold:
            tau_optimal = lag
            break

    if tau_optimal is None:
        tau_optimal = max_lag
        if verbose:
            print(f"  Warning: ACF did not drop below {threshold}, using max_lag={max_lag}")

    if verbose:
        print(f"Step 1: Found τ = {tau_optimal} where ACF ≈ {acf_values[tau_optimal]:.3f}")

    return tau_optimal, acf_values


def find_optimal_E(df: pd.DataFrame,
                  target: str,
                  tau: int,
                  E_range: Iterable[int] = range(2, 11),
                  n_std: float = 1.0,
                  verbose: bool = True) -> Tuple[int, pd.DataFrame]:
    """
    Step 2: Find E that best reconstructs target attractor.

    Choose smallest E within n_std of maximum performance.

    Parameters
    ----------
    df : pd.DataFrame
        Time series dataframe with 'time' and target columns
    target : str
        Target variable column name
    tau : int
        Time lag (from find_optimal_tau)
    E_range : iterable of int, default range(2, 11)
        Embedding dimensions to test
    n_std : float, default 1.0
        Number of standard deviations below max for threshold
    verbose : bool, default True
        Print selection details

    Returns
    -------
    E_optimal : int
        Optimal embedding dimension
    E_df : pd.DataFrame
        Results for all E values tested
    """
    if not PYEDM_AVAILABLE:
        raise ImportError("pyEDM is required. Install with: pip install pyEDM")

    lib = f'1 {len(df)}'
    pred = lib

    E_results = []
    for E in E_range:
        try:
            simplex = Simplex(
                dataFrame=df,
                lib=lib,
                pred=pred,
                columns=target,
                target=target,
                E=E,
                tau=-tau,  # Negative for lag
                Tp=1,
                exclusionRadius=0  # Don't use exclusion for E selection
            )
            rho = simplex[['Observations', 'Predictions']].corr().iloc[0, 1]
            E_results.append({'E': E, 'rho': rho})
        except Exception:
            E_results.append({'E': E, 'rho': np.nan})

    E_df = pd.DataFrame(E_results)

    # Remove NaN results
    E_df_valid = E_df[~E_df['rho'].isna()].copy()

    if len(E_df_valid) == 0:
        if verbose:
            print("  Warning: All E values failed, using E=2")
        return 2, E_df

    max_rho = E_df_valid['rho'].max()
    std_rho = E_df_valid['rho'].std()

    # Threshold: within n standard deviations of max
    threshold = max_rho - n_std * std_rho

    # Find smallest E exceeding threshold
    candidates = E_df_valid[E_df_valid['rho'] >= threshold].copy()
    E_optimal = int(candidates['E'].min())

    optimal_rho = candidates[candidates['E'] == E_optimal]['rho'].iloc[0]

    if verbose:
        print(f"Step 2: Found E = {E_optimal} with ρ = {optimal_rho:.3f}")
        print(f"         Max ρ = {max_rho:.3f}, threshold = {threshold:.3f} ({n_std}σ below max)")

    return E_optimal, E_df


def find_optimal_Tp(df: pd.DataFrame,
                   driver: str,
                   target: str,
                   E: int,
                   tau: int,
                   Tp_range: Iterable[int] = range(-8, 1),
                   n_std: float = 1.0,
                   verbose: bool = True) -> Tuple[int, pd.DataFrame]:
    """
    Step 3: Find Tp that shows greatest causal effect.

    Test target xmap driver at different Tp values.
    Choose smallest |Tp| within n_std of maximum performance.

    Parameters
    ----------
    df : pd.DataFrame
        Time series dataframe
    driver : str
        Driver variable column
    target : str
        Target variable column
    E : int
        Embedding dimension (from find_optimal_E)
    tau : int
        Time lag (from find_optimal_tau)
    Tp_range : iterable of int, default range(-8, 1)
        Prediction horizons to test
    n_std : float, default 1.0
        Number of standard deviations below max for threshold
    verbose : bool, default True
        Print selection details

    Returns
    -------
    Tp_optimal : int
        Optimal prediction horizon
    Tp_df : pd.DataFrame
        Results for all Tp values tested
    """
    if not PYEDM_AVAILABLE:
        raise ImportError("pyEDM is required. Install with: pip install pyEDM")

    lib = f'1 {len(df)}'

    Tp_results = []
    for Tp in Tp_range:
        try:
            simplex = Simplex(
                dataFrame=df,
                columns=target,
                target=driver,
                lib=lib,
                pred=lib,
                E=E,
                Tp=Tp,
                tau=-tau,
                exclusionRadius=0
            )
            rho = simplex[['Observations', 'Predictions']].corr().iloc[0, 1]
            Tp_results.append({'Tp': Tp, 'rho': rho})
        except Exception:
            Tp_results.append({'Tp': Tp, 'rho': np.nan})

    Tp_df = pd.DataFrame(Tp_results)

    # Remove NaN results
    Tp_df_valid = Tp_df[~Tp_df['rho'].isna()].copy()

    if len(Tp_df_valid) == 0:
        if verbose:
            print("  Warning: All Tp values failed, using Tp=0")
        return 0, Tp_df

    max_rho = Tp_df_valid['rho'].max()
    std_rho = Tp_df_valid['rho'].std()

    # Threshold: within n standard deviations of max
    threshold = max_rho - n_std * std_rho

    # Find smallest |Tp| exceeding threshold
    candidates = Tp_df_valid[Tp_df_valid['rho'] >= threshold].copy()
    candidates['abs_Tp'] = candidates['Tp'].abs()
    candidates = candidates.sort_values(['abs_Tp', 'Tp'])

    Tp_optimal = int(candidates.iloc[0]['Tp'])
    optimal_rho = candidates.iloc[0]['rho']

    if verbose:
        print(f"Step 3: Found Tp = {Tp_optimal} with ρ = {optimal_rho:.3f}")
        print(f"         Max ρ = {max_rho:.3f}, threshold = {threshold:.3f} ({n_std}σ below max)")

    return Tp_optimal, Tp_df


def optimize_parameters(df: pd.DataFrame,
                       driver: str,
                       target: str,
                       max_lag: int = 20,
                       acf_threshold: float = 0.1,
                       E_range: Iterable[int] = range(2, 11),
                       Tp_range: Iterable[int] = range(-8, 1),
                       n_std: float = 1.0,
                       verbose: bool = True) -> Dict:
    """
    Complete EDM parameter optimization workflow.

    Implements rigorous 3-step procedure:
    1. Find τ where ACF(target) < threshold
    2. Find E within n_std of max (target reconstruction)
    3. Find Tp within n_std of max (cross-mapping)

    Parameters
    ----------
    df : pd.DataFrame
        Time series dataframe with 'time', driver, and target columns
    driver : str
        Driver variable column
    target : str
        Target variable column
    max_lag : int, default 20
        Maximum lag for ACF
    acf_threshold : float, default 0.1
        ACF threshold for tau selection
    E_range : iterable of int, default range(2, 11)
        Embedding dimensions to test
    Tp_range : iterable of int, default range(-8, 1)
        Prediction horizons to test
    n_std : float, default 1.0
        Standard deviations below max for selection threshold
    verbose : bool, default True
        Print optimization steps

    Returns
    -------
    dict
        Parameters and diagnostics:
        - tau, E, Tp: optimal values
        - acf_values: ACF for diagnostics
        - E_results, Tp_results: full optimization results
    """
    if verbose:
        print("="*70)
        print("PARAMETER OPTIMIZATION")
        print("="*70)

    # Ensure df has required columns
    Y = df[target].values

    # Step 1: Find optimal tau
    tau, acf_values = find_optimal_tau(
        Y, max_lag=max_lag, threshold=acf_threshold, verbose=verbose
    )

    # Step 2: Find optimal E
    E, E_results = find_optimal_E(
        df, target, tau, E_range=E_range, n_std=n_std, verbose=verbose
    )

    # Step 3: Find optimal Tp
    Tp, Tp_results = find_optimal_Tp(
        df, driver, target, E, tau,
        Tp_range=Tp_range, n_std=n_std, verbose=verbose
    )

    if verbose:
        print("="*70)
        print()

    return {
        'tau': tau,
        'E': E,
        'Tp': Tp,
        'acf_values': acf_values,
        'acf_threshold': acf_threshold,
        'E_results': E_results,
        'Tp_results': Tp_results,
        'n_std': n_std
    }
