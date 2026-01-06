"""
Parameter optimization for CCM.

Implements the standard EDM workflow for finding optimal embedding parameters:
1. Find tau where ACF crosses threshold
2. Find E that unfolds the target attractor
3. Find Tp that maximizes cross-mapping skill
4. (Optional) Find theta for S-map nonlinearity parameter
"""

import numpy as np
import pandas as pd
import pyEDM
from typing import Tuple, Optional, List
from statsmodels.tsa.stattools import acf


def acf_nan(y: np.ndarray, nlags: int = 20) -> np.ndarray:
    """
    Compute autocorrelation with NaN handling.

    Uses pairwise-complete observations for each lag.

    Parameters
    ----------
    y : np.ndarray
        1D array with possible NaNs
    nlags : int, default 20
        Maximum lag to compute

    Returns
    -------
    np.ndarray
        ACF values of length nlags+1
    """
    y = np.asarray(y, float)
    n = len(y)
    acf_vals = []

    # Centered series (mean computed on non-NaN)
    y_mean = np.nanmean(y)
    y_centered = y - y_mean

    # Denominator: variance using non-NaN values
    denom = np.nansum(y_centered ** 2)

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
                    method: str = 'first_crossing') -> Tuple[int, np.ndarray]:
    """
    Find optimal time delay (tau) based on ACF.

    Parameters
    ----------
    Y : np.ndarray
        Target variable time series
    max_lag : int, default 20
        Maximum lag to consider
    threshold : float, default 0.1
        ACF threshold for determining decorrelation
    method : str, default 'first_crossing'
        Method for choosing tau:
        - 'first_crossing': First lag where |ACF| < threshold
        - 'first_minimum': First lag where ACF reaches local minimum

    Returns
    -------
    tau_optimal : int
        Optimal time delay
    acf_values : np.ndarray
        ACF values for all lags

    Notes
    -----
    Standard EDM practice is to use tau where ACF crosses 0.1 to ensure
    decorrelated delay embeddings while avoiding unnecessary embedding
    dimension inflation.
    """
    acf_values = acf_nan(Y, nlags=max_lag)

    tau_optimal = None

    if method == 'first_crossing':
        # Find first lag where |ACF| < threshold
        for lag in range(1, len(acf_values)):
            if np.abs(acf_values[lag]) < threshold:
                tau_optimal = lag
                break
    elif method == 'first_minimum':
        # Find first local minimum
        for lag in range(1, len(acf_values) - 1):
            if acf_values[lag] < acf_values[lag - 1] and acf_values[lag] < acf_values[lag + 1]:
                tau_optimal = lag
                break
    else:
        raise ValueError(f"Unknown method: {method}")

    if tau_optimal is None:
        tau_optimal = max_lag

    return tau_optimal, acf_values


def find_optimal_E(Y: np.ndarray,
                  tau: int,
                  max_E: int = 10,
                  n_std: float = 1,
                  Tp: int = 1,
                  exclusionRadius: int = 0) -> Tuple[int, pd.DataFrame]:
    """
    Find optimal embedding dimension (E) using Simplex projection.

    Tests reconstruction of Y from itself (Y -> Y) to determine
    how many dimensions are needed to unfold the attractor.

    Parameters
    ----------
    Y : np.ndarray
        Target variable time series
    tau : int
        Time delay (typically positive for this step)
    max_E : int, default 10
        Maximum embedding dimension to test
    n_std : float, default 1
        Number of standard deviations below maximum for threshold
    Tp : int, default 1
        Prediction horizon for simplex
    exclusionRadius : int, default 0
        Exclusion radius for nearest neighbors

    Returns
    -------
    E_optimal : int
        Optimal embedding dimension
    E_results : pd.DataFrame
        Results for all tested E values

    Notes
    -----
    Chooses the smallest E that achieves performance within n_std
    of the maximum, following the principle of parsimony.
    """
    df = pd.DataFrame({'time': range(1, len(Y) + 1), 'Y': Y})
    lib = f'1 {len(df)}'
    pred = lib

    E_results = []
    for E in range(2, max_E + 1):
        simplex = pyEDM.Simplex(
            dataFrame=df,
            lib=lib,
            pred=pred,
            columns='Y',
            target='Y',
            E=E,
            tau=-tau,  # Negative tau for pyEDM
            Tp=Tp,
            exclusionRadius=exclusionRadius
        )
        rho = simplex[['Observations', 'Predictions']].corr().iloc[0, 1]
        E_results.append({'E': E, 'rho': rho})

    E_df = pd.DataFrame(E_results)

    max_rho = E_df['rho'].max()
    std_rho = E_df['rho'].std()

    # Threshold: within n standard deviations of max
    threshold = max_rho - n_std * std_rho

    # Find smallest E exceeding threshold
    candidates = E_df[E_df['rho'] >= threshold].copy()
    E_optimal = int(candidates['E'].min())

    return E_optimal, E_df


def find_optimal_Tp(X: np.ndarray,
                   Y: np.ndarray,
                   E: int,
                   tau: int,
                   max_Tp: int = 8,
                   n_std: float = 1,
                   exclusionRadius: int = 0,
                   theta: float = 0) -> Tuple[int, pd.DataFrame]:
    """
    Find optimal prediction horizon (Tp) for cross-mapping.

    Tests Y xmap X at different Tp values to find the lag that
    shows the strongest causal signal.

    Parameters
    ----------
    X : np.ndarray
        Driver variable (cause)
    Y : np.ndarray
        Target variable (effect)
    E : int
        Embedding dimension (from find_optimal_E)
    tau : int
        Time delay (from find_optimal_tau)
    max_Tp : int, default 8
        Maximum Tp to test (tests from -max_Tp to 0)
    n_std : float, default 1
        Number of standard deviations below maximum for threshold
    exclusionRadius : int, default 0
        Exclusion radius for nearest neighbors
    theta : float, default 0
        S-map localization parameter

    Returns
    -------
    Tp_optimal : int
        Optimal prediction horizon (typically negative)
    Tp_results : pd.DataFrame
        Results for all tested Tp values

    Notes
    -----
    For causal detection, Tp is typically negative (predicting into the past)
    to test if current state of Y can reconstruct past states of X.
    Chooses smallest |Tp| that achieves performance within n_std of maximum.
    """
    to_xmap = pd.DataFrame({'X': X, 'Y': Y, 'time': range(1, len(X) + 1)})

    Tp_results = []
    for Tp in range(-max_Tp, 1):
        lib = f'1 {len(to_xmap)}'
        try:
            if theta == 0:
                xmap = pyEDM.Simplex(
                    dataFrame=to_xmap[['time', 'X', 'Y']],
                    columns='Y',
                    target='X',
                    lib=lib,
                    pred=lib,
                    E=E,
                    Tp=Tp,
                    tau=-tau,
                    exclusionRadius=exclusionRadius
                )
            else:
                xmap = pyEDM.SMap(
                    dataFrame=to_xmap[['time', 'X', 'Y']],
                    columns='Y',
                    target='X',
                    lib=lib,
                    pred=lib,
                    E=E,
                    Tp=Tp,
                    tau=-tau,
                    theta=theta,
                    exclusionRadius=exclusionRadius
                )
            rho = xmap[['Observations', 'Predictions']].corr().iloc[0, 1]
            Tp_results.append({'Tp': Tp, 'rho': rho})
        except:
            Tp_results.append({'Tp': Tp, 'rho': np.nan})

    Tp_df = pd.DataFrame(Tp_results).dropna()

    max_rho = Tp_df['rho'].max()
    std_rho = Tp_df['rho'].std()

    # Threshold: within n standard deviations of max
    threshold = max_rho - n_std * std_rho

    # Find smallest |Tp| exceeding threshold
    candidates = Tp_df[Tp_df['rho'] >= threshold].copy()
    candidates['abs_Tp'] = candidates['Tp'].abs()
    candidates = candidates.sort_values(['abs_Tp', 'Tp'])

    Tp_optimal = int(candidates.iloc[0]['Tp'])

    return Tp_optimal, Tp_df


def find_optimal_theta_smap(X: np.ndarray,
                           Y: np.ndarray,
                           E: int,
                           tau: int,
                           Tp: int,
                           theta_range: Optional[List[float]] = None,
                           exclusionRadius: int = 0) -> Tuple[float, pd.DataFrame]:
    """
    Find optimal theta (nonlinearity parameter) for S-map.

    Tests different theta values to find the best localization parameter.
    theta = 0 corresponds to Simplex (linear), higher theta values allow
    more nonlinear dynamics.

    Parameters
    ----------
    X : np.ndarray
        Driver variable
    Y : np.ndarray
        Target variable
    E : int
        Embedding dimension
    tau : int
        Time delay
    Tp : int
        Prediction horizon
    theta_range : list of float or None
        Theta values to test. If None, uses [0, 0.01, 0.1, 0.3, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]
    exclusionRadius : int, default 0
        Exclusion radius for nearest neighbors

    Returns
    -------
    theta_optimal : float
        Optimal theta value
    theta_results : pd.DataFrame
        Results for all tested theta values

    Notes
    -----
    This is an optional step that can improve performance for highly nonlinear systems.
    For many systems, Simplex (theta=0) performs well.
    """
    if theta_range is None:
        theta_range = [0, 0.01, 0.1, 0.3, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]

    to_xmap = pd.DataFrame({'X': X, 'Y': Y, 'time': range(1, len(X) + 1)})

    theta_results = []
    for theta in theta_range:
        lib = f'1 {len(to_xmap)}'
        try:
            xmap = pyEDM.SMap(
                dataFrame=to_xmap[['time', 'X', 'Y']],
                columns='Y',
                target='X',
                lib=lib,
                pred=lib,
                E=E,
                Tp=Tp,
                tau=-tau,
                theta=theta,
                exclusionRadius=exclusionRadius
            )
            rho = xmap[['Observations', 'Predictions']].corr().iloc[0, 1]
            theta_results.append({'theta': theta, 'rho': rho})
        except:
            theta_results.append({'theta': theta, 'rho': np.nan})

    theta_df = pd.DataFrame(theta_results).dropna()

    # Choose theta with maximum rho
    best_idx = theta_df['rho'].idxmax()
    theta_optimal = theta_df.loc[best_idx, 'theta']

    return theta_optimal, theta_df


def optimize_parameters(X: np.ndarray,
                       Y: np.ndarray,
                       max_tau_lag: int = 20,
                       tau_threshold: float = 0.1,
                       max_E: int = 10,
                       max_Tp: int = 8,
                       optimize_theta: bool = False,
                       theta_range: Optional[List[float]] = None,
                       exclusionRadius: int = 0,
                       n_std: float = 1,
                       verbose: bool = True) -> dict:
    """
    Full parameter optimization workflow for CCM.

    Executes the standard EDM procedure:
    1. Find tau from ACF of Y
    2. Find E from Y -> Y reconstruction
    3. Find Tp from Y xmap X performance
    4. (Optional) Find theta for S-map

    Parameters
    ----------
    X : np.ndarray
        Driver variable (cause)
    Y : np.ndarray
        Target variable (effect)
    max_tau_lag : int, default 20
        Maximum lag for ACF
    tau_threshold : float, default 0.1
        ACF threshold for tau selection
    max_E : int, default 10
        Maximum embedding dimension to test
    max_Tp : int, default 8
        Maximum prediction horizon to test
    optimize_theta : bool, default False
        Whether to optimize theta using S-map
    theta_range : list of float or None
        Theta values to test (if optimize_theta=True)
    exclusionRadius : int, default 0
        Exclusion radius for nearest neighbors
    n_std : float, default 1
        Number of std deviations below max for threshold
    verbose : bool, default True
        Print progress information

    Returns
    -------
    dict
        Dictionary containing:
        - tau, E, Tp, theta: optimal parameters
        - acf_values, E_results, Tp_results, theta_results: detailed results
    """
    if verbose:
        print("=" * 70)
        print("PARAMETER OPTIMIZATION")
        print("=" * 70)
        print()

    # Step 1: Find optimal tau
    if verbose:
        print("Step 1: Finding optimal tau from ACF...")
    tau, acf_values = find_optimal_tau(Y, max_tau_lag, tau_threshold)
    if verbose:
        print(f"  → tau = {tau} (ACF = {acf_values[tau]:.3f})")
        print()

    # Step 2: Find optimal E
    if verbose:
        print("Step 2: Finding optimal E from target reconstruction...")
    E, E_results = find_optimal_E(Y, tau, max_E, n_std, exclusionRadius=exclusionRadius)
    optimal_rho_E = E_results[E_results['E'] == E]['rho'].iloc[0]
    if verbose:
        print(f"  → E = {E} (rho = {optimal_rho_E:.3f})")
        print()

    # Step 3: Find optimal Tp
    if verbose:
        print("Step 3: Finding optimal Tp from cross-mapping...")
    Tp, Tp_results = find_optimal_Tp(X, Y, E, tau, max_Tp, n_std, exclusionRadius)
    optimal_rho_Tp = Tp_results[Tp_results['Tp'] == Tp]['rho'].iloc[0]
    if verbose:
        print(f"  → Tp = {Tp} (rho = {optimal_rho_Tp:.3f})")
        print()

    # Step 4: (Optional) Find optimal theta
    theta = 0
    theta_results = None
    if optimize_theta:
        if verbose:
            print("Step 4: Finding optimal theta using S-map...")
        theta, theta_results = find_optimal_theta_smap(
            X, Y, E, tau, Tp, theta_range, exclusionRadius
        )
        optimal_rho_theta = theta_results[theta_results['theta'] == theta]['rho'].iloc[0]
        if verbose:
            print(f"  → theta = {theta} (rho = {optimal_rho_theta:.3f})")
            print()

    if verbose:
        print("=" * 70)
        print()

    return {
        'tau': tau,
        'E': E,
        'Tp': Tp,
        'theta': theta,
        'exclusionRadius': exclusionRadius,
        'acf_values': acf_values,
        'E_results': E_results,
        'Tp_results': Tp_results,
        'theta_results': theta_results,
    }
