"""
Core Convergent Cross Mapping (CCM) functions for EDM analysis.

This module provides standardized CCM analysis with convergence testing
using variable library sizes.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any
from scipy.optimize import curve_fit

try:
    from pyEDM import CCM, SMap
    PYEDM_AVAILABLE = True
except ImportError:
    PYEDM_AVAILABLE = False
    print("Warning: pyEDM not available. CCM functionality will be limited.")


def saturating_curve(L: np.ndarray, a: float, K: float, b: float) -> np.ndarray:
    """
    Saturating curve model for CCM convergence.

    Model: y = a*L / (K + L) + b

    Parameters
    ----------
    L : np.ndarray
        Library sizes
    a : float
        Amplitude parameter (asymptotic increase)
    K : float
        Half-saturation constant (library size at half of asymptote)
    b : float
        Baseline/offset parameter

    Returns
    -------
    np.ndarray
        Predicted values
    """
    return a * L / (K + L) + b


def fit_ccm_curve(df: pd.DataFrame,
                  rho_col: str,
                  lib_col: str = 'LibSize',
                  plot: bool = False) -> Optional[Dict[str, float]]:
    """
    Fit a saturating curve to CCM library size convergence.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with library sizes and CCM rho values
    rho_col : str
        Column name for CCM rho values
    lib_col : str, default 'LibSize'
        Column name for library sizes
    plot : bool, default False
        Whether to plot the fit

    Returns
    -------
    dict or None
        Dictionary with fit parameters:
        - 'a': amplitude
        - 'K': half-saturation constant
        - 'b': baseline
        - 'R2': R-squared of fit
        - 'Lmax': maximum library size
        - 'slope_tail': slope in last third of points
        - 'rho_conv': mean rho in convergence region (last third)
    """
    # Drop NaN
    sub = df[[lib_col, rho_col]].dropna()
    if len(sub) < 3:
        return None

    x = sub[lib_col].to_numpy(float)
    y = sub[rho_col].to_numpy(float)
    Lmax = float(np.max(x))

    # Initial guesses
    p0 = [
        max(1e-6, np.nanmax(y) - np.nanmin(y)),  # a: amplitude
        float(np.median(x)),                      # K: half-saturation
        float(np.nanmin(y))                       # b: baseline
    ]

    try:
        popt, pcov = curve_fit(saturating_curve, x, y, p0=p0, maxfev=10000)
        a, K, b = map(float, popt)

        # Goodness of fit (R²)
        yfit = saturating_curve(x, a, K, b)
        resid = y - yfit
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot

    except Exception as e:
        print(f"Fit failed for {rho_col}: {e}")
        return None

    # Tail-slope test (plateau region = last third of points)
    tail_n = max(3, int(np.ceil(len(x) / 3)))
    x_tail = x[-tail_n:]
    y_tail = y[-tail_n:]

    if len(x_tail) >= 2:
        slope_tail = np.polyfit(x_tail, y_tail, 1)[0]
    else:
        slope_tail = np.nan

    # Convergence rho (mean of tail region)
    rho_conv = float(np.nanmean(y_tail))

    # Optional plotting
    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5, 3))
        plt.scatter(x, y, s=40, alpha=0.6, label='Observed')
        L_grid = np.linspace(np.min(x), np.max(x), 200)
        plt.plot(L_grid, saturating_curve(L_grid, a, K, b), 'r-', lw=2, label='Fit')

        if K < Lmax:  # Show K only if within observed range
            plt.axvline(K, color='gray', ls='--', lw=1, label=f"K={K:.1f}")

        plt.xlabel('Library size')
        plt.ylabel(f'ρ({rho_col})')
        plt.title(f'Saturating fit: a={a:.3f}, K={K:.1f}, b={b:.3f}, R²={r2:.2f}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        'a': a,
        'K': K,
        'b': b,
        'R2': r2,
        'Lmax': Lmax,
        'slope_tail': slope_tail,
        'rho_conv': rho_conv
    }


def has_converged(fit_result: Optional[Dict[str, float]],
                 r2_threshold: float = 0.7,
                 slope_threshold: float = 1e-3,
                 k_threshold_ratio: float = 0.8) -> bool:
    """
    Test whether CCM has converged based on curve fit.

    Parameters
    ----------
    fit_result : dict or None
        Result from fit_ccm_curve
    r2_threshold : float, default 0.7
        Minimum R² for good fit
    slope_threshold : float, default 1e-3
        Maximum slope in tail region (plateau criterion)
    k_threshold_ratio : float, default 0.8
        Maximum K as fraction of Lmax (convergence before end)

    Returns
    -------
    bool
        True if converged, False otherwise
    """
    if fit_result is None:
        return False

    converged = (
        fit_result['a'] > 0 and
        fit_result['R2'] > r2_threshold and
        fit_result['rho_conv'] > 0 and
        fit_result['slope_tail'] < slope_threshold and
        fit_result['K'] < k_threshold_ratio * fit_result['Lmax']
    )

    return converged


def ccm_analysis(df: pd.DataFrame,
                driver: str,
                target: str,
                E: int = 3,
                tau: int = -1,
                theta: float = 0,
                Tp: int = 1,
                exclusion_radius: Optional[int] = None,
                lib_frac_range: Tuple[float, float] = (0.05, 0.8),
                n_lib_sizes: int = 10,
                n_samples: int = 100) -> Dict[str, Any]:
    """
    Perform CCM analysis with convergence testing.

    Parameters
    ----------
    df : pd.DataFrame
        Time series dataframe with 'time' column
    driver : str
        Column name of putative causal driver (library variable)
    target : str
        Column name of target variable
    E : int, default 3
        Embedding dimension
    tau : int, default -1
        Time lag for embedding
    theta : float, default 0
        Nonlinearity parameter (0 = simplex, >0 = S-map)
    Tp : int, default 1
        Prediction horizon
    exclusion_radius : int or None
        Temporal exclusion radius (default: |E * tau|)
    lib_frac_range : tuple, default (0.05, 0.8)
        Min and max library size as fraction of data length
    n_lib_sizes : int, default 10
        Number of library sizes to test
    n_samples : int, default 100
        Number of random samples per library size

    Returns
    -------
    dict
        Results dictionary with:
        - 'rho': final CCM correlation
        - 'rho_conv': converged rho (from curve fit)
        - 'convergence': boolean convergence status
        - 'fit': curve fit parameters
        - 'lib_means': dataframe with library size results
    """
    if not PYEDM_AVAILABLE:
        raise ImportError("pyEDM is required for CCM analysis. Install with: pip install pyEDM")

    # Set exclusion radius
    if exclusion_radius is None:
        exclusion_radius = abs(E * tau)

    # Library sizes
    n = len(df)
    lib_small = int(round(lib_frac_range[0] * n))
    lib_large = int(round(lib_frac_range[1] * n))
    lib_inc = max(1, int(round((lib_large - lib_small) / (n_lib_sizes - 1))))
    libsizes = f"{lib_small} {lib_large} {lib_inc}"

    # Run CCM
    result = CCM(
        dataFrame=df,
        columns=driver,
        target=target,
        libSizes=libsizes,
        sample=n_samples,
        exclusionRadius=exclusion_radius,
        E=E,
        tau=tau,
        Tp=Tp,
        includeData=True
    )

    # Extract results
    lib_means = result['LibMeans']
    rho = lib_means.iloc[-1, 1]  # Last row, correlation column

    # Fit convergence curve
    rho_col = f"{driver}:{target}"
    fit = fit_ccm_curve(lib_means, rho_col=rho_col, plot=False)
    convergence = has_converged(fit)

    if fit is not None:
        rho_conv = fit['rho_conv']
    else:
        rho_conv = np.nan

    return {
        'driver': driver,
        'target': target,
        'E': E,
        'tau': tau,
        'theta': theta,
        'Tp': Tp,
        'rho': rho,
        'rho_conv': rho_conv,
        'convergence': convergence,
        'fit': fit,
        'lib_means': lib_means
    }


def ccm_with_significance(df: pd.DataFrame,
                         driver: str,
                         target: str,
                         E: int = 3,
                         tau: int = -1,
                         theta: float = 0,
                         Tp: int = 1,
                         exclusion_radius: Optional[int] = None,
                         surrogate_generator=None,
                         n_surrogates: int = 100,
                         **ccm_kwargs) -> Dict[str, Any]:
    """
    Perform CCM analysis with significance testing using surrogates.

    Parameters
    ----------
    df : pd.DataFrame
        Time series dataframe
    driver : str
        Column name of putative causal driver
    target : str
        Column name of target variable
    E, tau, theta, Tp : int/float
        EDM parameters
    exclusion_radius : int or None
        Temporal exclusion radius
    surrogate_generator : callable
        Function that generates surrogate pairs: (x, y, n_surr) -> (xs, ys)
    n_surrogates : int, default 100
        Number of surrogate realizations
    **ccm_kwargs
        Additional arguments for ccm_analysis

    Returns
    -------
    dict
        Results with significance testing
    """
    from ..surrogates.testing import empirical_p

    # Original CCM
    result = ccm_analysis(df, driver, target, E=E, tau=tau, theta=theta,
                         Tp=Tp, exclusion_radius=exclusion_radius, **ccm_kwargs)

    # Only test significance if converged
    if not result['convergence']:
        result['p_value'] = np.nan
        result['significant'] = False
        result['surr_mean'] = np.nan
        result['surr_95p'] = np.nan
        result['surr_99p'] = np.nan
        return result

    # Generate surrogates and compute null distribution
    if surrogate_generator is None:
        from ..surrogates.generators import generate_seasonal_pair_surrogates
        surrogate_generator = generate_seasonal_pair_surrogates

    rho_surr = []

    for _ in range(n_surrogates):
        xs, ys = surrogate_generator(df[driver].values, df[target].values,
                                     n_surr=1, period=12, mode='within_phase')

        df_surr = df.copy()
        df_surr[driver] = xs[0]
        df_surr[target] = ys[0]

        # Run CCM on surrogate (single library size for speed)
        n = len(df_surr)
        lib_size = int(0.8 * n)

        ccm_surr = CCM(
            dataFrame=df_surr,
            columns=driver,
            target=target,
            libSizes=str(lib_size),
            sample=100,
            exclusionRadius=exclusion_radius if exclusion_radius else abs(E * tau),
            E=E,
            tau=tau,
            Tp=Tp
        )

        rho_surr.append(ccm_surr['LibMeans'].iloc[0, 1])

    rho_surr = np.array(rho_surr)

    # Compute p-value
    p_value = empirical_p(result['rho'], rho_surr, tail='greater')

    result['p_value'] = p_value
    result['significant'] = (p_value < 0.05)
    result['surr_mean'] = np.mean(rho_surr)
    result['surr_95p'] = np.percentile(rho_surr, 95)
    result['surr_99p'] = np.percentile(rho_surr, 99)
    result['ccm_norm'] = result['rho'] - result['surr_mean']

    return result
