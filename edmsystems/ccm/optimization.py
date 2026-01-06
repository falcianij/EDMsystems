"""
Parameter optimization for EDM analysis.

This module provides functions to find optimal embedding dimension (E)
and time lag (tau) that avoid autocorrelation artifacts.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Iterable, Optional
from statsmodels.tsa.stattools import acf

try:
    from pyEDM import SMap, Simplex
    PYEDM_AVAILABLE = True
except ImportError:
    PYEDM_AVAILABLE = False


def find_optimal_E_tau(df: pd.DataFrame,
                      driver: str,
                      target: str,
                      E_range: Iterable[int] = range(2, 7),
                      tau_range: Iterable[int] = range(-3, 0),
                      theta_range: Iterable[float] = [0],
                      Tp: int = 1,
                      metric: str = 'rho') -> Tuple[int, int, float]:
    """
    Find optimal (E, tau, theta) parameters via grid search using S-Map.

    Optimizes for prediction skill (correlation between observations and predictions).

    Parameters
    ----------
    df : pd.DataFrame
        Time series dataframe with required columns
    driver : str
        Column name of driver variable (predictor)
    target : str
        Column name of target variable
    E_range : iterable of int, default range(2, 7)
        Embedding dimensions to test
    tau_range : iterable of int, default range(-3, 0)
        Time lags to test (negative = past values)
    theta_range : iterable of float, default [0]
        Nonlinearity parameters to test (0 = simplex, >0 = S-map)
    Tp : int, default 1
        Prediction horizon
    metric : str, default 'rho'
        Metric to optimize ('rho' for correlation)

    Returns
    -------
    best_E : int
        Optimal embedding dimension
    best_tau : int
        Optimal time lag
    best_theta : float
        Optimal nonlinearity parameter
    """
    if not PYEDM_AVAILABLE:
        raise ImportError("pyEDM is required. Install with: pip install pyEDM")

    best_E = list(E_range)[0]
    best_tau = list(tau_range)[0]
    best_theta = list(theta_range)[0]
    best_score = -np.inf

    lib_all = f'1 {len(df)}'
    pred = lib_all

    for E in E_range:
        for tau in tau_range:
            exclusion_radius = abs(E * tau)

            for theta in theta_range:
                try:
                    smap = SMap(
                        dataFrame=df,
                        columns=driver,
                        target=target,
                        lib=lib_all,
                        pred=pred,
                        exclusionRadius=exclusion_radius,
                        E=E,
                        tau=tau,
                        Tp=Tp,
                        theta=theta
                    )

                    # Compute correlation between observations and predictions
                    preds = smap['predictions']
                    rho = preds[['Observations', 'Predictions']].corr().iloc[0, 1]

                    if rho is not None and np.isfinite(rho) and rho > best_score:
                        best_score = rho
                        best_E = E
                        best_tau = tau
                        best_theta = theta

                except Exception as e:
                    # Skip parameter combinations that fail
                    continue

    return best_E, best_tau, best_theta


def find_tau_autocorrelation(x: pd.Series,
                             max_lag: int = 10,
                             threshold: float = 0.1) -> int:
    """
    Find time lag where autocorrelation drops below threshold.

    Following EDM best practices, tau should be chosen where ACF crosses
    a low threshold (typically 0.1) to avoid autocorrelation artifacts.

    Parameters
    ----------
    x : pd.Series
        Input time series
    max_lag : int, default 10
        Maximum lag to consider
    threshold : float, default 0.1
        ACF threshold (EDM standard: 0.1)

    Returns
    -------
    int
        Recommended tau (first lag where ACF < threshold)
    """
    # Compute autocorrelation function
    x_clean = x.dropna()
    if len(x_clean) < max_lag + 1:
        return 1

    autocorr = acf(x_clean, nlags=max_lag, fft=True)

    # Find first lag where ACF drops below threshold
    for lag in range(1, len(autocorr)):
        if autocorr[lag] < threshold:
            return lag

    # If no drop found, return max_lag
    return max_lag


def optimize_parameters_edm_standard(df: pd.DataFrame,
                                     driver: str,
                                     target: str,
                                     max_lag: int = 10,
                                     acf_threshold: float = 0.1,
                                     E_range: Iterable[int] = range(2, 11),
                                     Tp_range: Iterable[int] = range(0, 3),
                                     theta_range: Iterable[float] = np.linspace(0, 8, 17)) -> dict:
    """
    Optimize EDM parameters following standard procedure.

    Procedure:
    1. Choose τ > lag where ACF crosses threshold (default 0.1)
    2. Given τ, choose E that best unfolds the attractor of target (Y)
    3. Choose Tp that maximizes cross-mapping skill between target and driver

    Parameters
    ----------
    df : pd.DataFrame
        Time series dataframe with required columns
    driver : str
        Column name of driver variable (X)
    target : str
        Column name of target variable (Y)
    max_lag : int, default 10
        Maximum lag to consider for ACF
    acf_threshold : float, default 0.1
        ACF threshold for tau selection
    E_range : iterable of int, default range(2, 11)
        Embedding dimensions to test
    Tp_range : iterable of int, default range(0, 3)
        Prediction horizons to test
    theta_range : iterable of float
        Nonlinearity parameters to test

    Returns
    -------
    dict
        Dictionary with:
        - 'tau': optimal time lag
        - 'E': optimal embedding dimension
        - 'Tp': optimal prediction horizon
        - 'theta': optimal nonlinearity parameter
        - 'acf_values': autocorrelation values for diagnostics
    """
    if not PYEDM_AVAILABLE:
        raise ImportError("pyEDM is required. Install with: pip install pyEDM")

    # Step 1: Find tau based on ACF of target (Y)
    tau = -find_tau_autocorrelation(df[target], max_lag=max_lag, threshold=acf_threshold)

    # Get ACF values for diagnostics
    target_clean = df[target].dropna()
    acf_values = acf(target_clean, nlags=max_lag, fft=True)

    print(f"Step 1: Selected τ = {tau} (ACF < {acf_threshold} at lag {-tau})")

    # Step 2: Given tau, find E that best unfolds target attractor
    # Use Simplex projection on target to predict itself
    best_E = 2
    best_E_score = -np.inf
    lib_all = f'1 {len(df)}'

    print(f"Step 2: Optimizing E for target attractor...")
    for E in E_range:
        try:
            simplex = Simplex(
                dataFrame=df,
                columns=target,
                target=target,
                lib=lib_all,
                pred=lib_all,
                E=E,
                tau=tau,
                Tp=1,
                exclusionRadius=abs(E * tau)
            )

            preds = simplex['predictions']
            rho = preds[['Observations', 'Predictions']].corr().iloc[0, 1]

            if rho is not None and np.isfinite(rho) and rho > best_E_score:
                best_E_score = rho
                best_E = E

        except Exception:
            continue

    print(f"  Selected E = {best_E} (target reconstruction ρ = {best_E_score:.3f})")

    # Step 3: Given tau and E, find Tp and theta that maximize cross-mapping
    best_Tp = 1
    best_theta = 0.0
    best_ccm_score = -np.inf

    print(f"Step 3: Optimizing Tp and θ for cross-mapping...")
    for Tp in Tp_range:
        for theta in theta_range:
            try:
                smap = SMap(
                    dataFrame=df,
                    columns=driver,
                    target=target,
                    lib=lib_all,
                    pred=lib_all,
                    exclusionRadius=abs(best_E * tau),
                    E=best_E,
                    tau=tau,
                    Tp=Tp,
                    theta=theta
                )

                preds = smap['predictions']
                rho = preds[['Observations', 'Predictions']].corr().iloc[0, 1]

                if rho is not None and np.isfinite(rho) and rho > best_ccm_score:
                    best_ccm_score = rho
                    best_Tp = Tp
                    best_theta = theta

            except Exception:
                continue

    print(f"  Selected Tp = {best_Tp}, θ = {best_theta:.2f} (CCM ρ = {best_ccm_score:.3f})")

    return {
        'tau': tau,
        'E': best_E,
        'Tp': best_Tp,
        'theta': best_theta,
        'E_score': best_E_score,
        'ccm_score': best_ccm_score,
        'acf_values': acf_values,
        'acf_threshold': acf_threshold
    }


def optimize_simplex_E(df: pd.DataFrame,
                      column: str,
                      target: str,
                      E_range: Iterable[int] = range(1, 11),
                      tau: int = -1,
                      Tp: int = 1) -> Tuple[int, pd.DataFrame]:
    """
    Optimize embedding dimension using Simplex projection.

    Parameters
    ----------
    df : pd.DataFrame
        Time series dataframe
    column : str
        Column to use for embedding
    target : str
        Target column to predict
    E_range : iterable of int, default range(1, 11)
        Embedding dimensions to test
    tau : int, default -1
        Time lag for embedding
    Tp : int, default 1
        Prediction horizon

    Returns
    -------
    best_E : int
        Optimal embedding dimension
    results : pd.DataFrame
        Results for all E values tested
    """
    if not PYEDM_AVAILABLE:
        raise ImportError("pyEDM is required. Install with: pip install pyEDM")

    results = []
    lib_all = f'1 {len(df)}'

    for E in E_range:
        try:
            simplex = Simplex(
                dataFrame=df,
                columns=column,
                target=target,
                lib=lib_all,
                pred=lib_all,
                E=E,
                tau=tau,
                Tp=Tp,
                exclusionRadius=abs(E * tau)
            )

            # Get prediction skill
            preds = simplex['predictions']
            rho = preds[['Observations', 'Predictions']].corr().iloc[0, 1]

            results.append({
                'E': E,
                'rho': rho,
                'mae': np.mean(np.abs(preds['Observations'] - preds['Predictions']))
            })

        except Exception as e:
            results.append({
                'E': E,
                'rho': np.nan,
                'mae': np.nan
            })

    results_df = pd.DataFrame(results)

    # Find E with maximum rho
    best_idx = results_df['rho'].idxmax()
    best_E = results_df.loc[best_idx, 'E']

    return int(best_E), results_df


def grid_search_parameters(df: pd.DataFrame,
                          driver: str,
                          target: str,
                          E_range: Iterable[int] = range(2, 7),
                          tau_range: Iterable[int] = range(-3, 0),
                          theta_range: Iterable[float] = np.linspace(0, 8, 17),
                          Tp: int = 1,
                          return_full_results: bool = False) -> dict:
    """
    Comprehensive grid search over E, tau, and theta parameters.

    Parameters
    ----------
    df : pd.DataFrame
        Time series dataframe
    driver : str
        Driver variable column
    target : str
        Target variable column
    E_range : iterable of int
        Embedding dimensions to test
    tau_range : iterable of int
        Time lags to test
    theta_range : iterable of float
        Nonlinearity parameters to test
    Tp : int
        Prediction horizon
    return_full_results : bool, default False
        If True, return full grid search results

    Returns
    -------
    dict
        Results with optimal parameters and optionally full grid
    """
    results = []
    best_score = -np.inf
    best_params = None

    lib_all = f'1 {len(df)}'
    pred = lib_all

    for E in E_range:
        for tau in tau_range:
            exclusion_radius = abs(E * tau)

            for theta in theta_range:
                try:
                    smap = SMap(
                        dataFrame=df,
                        columns=driver,
                        target=target,
                        lib=lib_all,
                        pred=pred,
                        exclusionRadius=exclusion_radius,
                        E=E,
                        tau=tau,
                        Tp=Tp,
                        theta=theta
                    )

                    preds = smap['predictions']
                    rho = preds[['Observations', 'Predictions']].corr().iloc[0, 1]

                    result = {
                        'E': E,
                        'tau': tau,
                        'theta': theta,
                        'rho': rho if np.isfinite(rho) else np.nan
                    }

                    results.append(result)

                    if rho is not None and np.isfinite(rho) and rho > best_score:
                        best_score = rho
                        best_params = result

                except Exception:
                    results.append({
                        'E': E,
                        'tau': tau,
                        'theta': theta,
                        'rho': np.nan
                    })

    output = {
        'best_E': best_params['E'] if best_params else list(E_range)[0],
        'best_tau': best_params['tau'] if best_params else list(tau_range)[0],
        'best_theta': best_params['theta'] if best_params else list(theta_range)[0],
        'best_rho': best_score
    }

    if return_full_results:
        output['full_results'] = pd.DataFrame(results)

    return output


def auto_optimize_parameters(df: pd.DataFrame,
                            driver: str,
                            target: str,
                            use_autocorr_tau: bool = True,
                            **kwargs) -> dict:
    """
    Automatic parameter optimization with sensible defaults.

    Parameters
    ----------
    df : pd.DataFrame
        Time series dataframe
    driver : str
        Driver variable column
    target : str
        Target variable column
    use_autocorr_tau : bool, default True
        Use autocorrelation to determine tau range
    **kwargs
        Additional arguments for grid_search_parameters

    Returns
    -------
    dict
        Optimal parameters
    """
    # Determine tau range based on autocorrelation if requested
    if use_autocorr_tau:
        max_tau = find_tau_autocorrelation(df[driver], max_lag=10)
        tau_range = range(-max_tau, 0)
    else:
        tau_range = kwargs.pop('tau_range', range(-3, 0))

    # Set defaults
    E_range = kwargs.pop('E_range', range(2, 7))
    theta_range = kwargs.pop('theta_range', np.linspace(0, 8, 17))

    # Run grid search
    result = grid_search_parameters(
        df, driver, target,
        E_range=E_range,
        tau_range=tau_range,
        theta_range=theta_range,
        **kwargs
    )

    return result
