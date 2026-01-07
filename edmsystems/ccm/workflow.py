"""
CCM workflow for pairwise causal analysis.

Implements the workflow from 011_ccm_analysis_norm_v3.ipynb with:
- Parallel processing across pairs (multiprocessing)
- Twin surrogate testing
- Convergence checking
- Result DataFrame with significance determination
"""

import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
from scipy.optimize import curve_fit

try:
    from pyEDM import CCM, Simplex
    PYEDM_AVAILABLE = True
except ImportError:
    PYEDM_AVAILABLE = False

from .parameters import optimize_parameters
from ..surrogates import generate_multivariate_fourier_surrogates


def _process_pair_wrapper(args):
    """
    Wrapper function for parallel processing of CCM pairs.

    This function must be at module level (not nested) to be picklable
    for multiprocessing.

    Parameters
    ----------
    args : tuple
        (pair, df, auto_optimize, E_range, libSizes, sample, n_surrogates, seed)

    Returns
    -------
    dict
        CCM result for the pair
    """
    pair, df, auto_optimize, E_range, libSizes, sample, n_surrogates, seed = args
    driver, target = pair

    # Optimize parameters if requested
    if auto_optimize:
        params = optimize_parameters(
            df[['time', driver, target]],
            driver=driver,
            target=target,
            E_range=E_range,
            verbose=False
        )
        E, tau, Tp, theta = params['E'], params['tau'], params['Tp'], 0
        exclusionRadius = abs(E * tau)
    else:
        # Use default or provided parameters
        E = list(E_range)[0]
        tau = -1  # Default tau
        Tp = 1
        theta = 0
        exclusionRadius = abs(E * tau)

    # Compute CCM
    result = compute_ccm_pair(
        df, driver, target,
        E=E, tau=tau, Tp=Tp, theta=theta,
        exclusionRadius=exclusionRadius,
        libSizes=libSizes,
        sample=sample,
        n_surrogates=n_surrogates,
        seed=seed,
        verbose=False
    )

    return result


def saturating_curve(L: np.ndarray, a: float, K: float, b: float) -> np.ndarray:
    """Saturating curve for CCM convergence fitting."""
    return a * L / (K + L) + b


def fit_ccm_curve(lib_sizes: np.ndarray,
                 rho_values: np.ndarray) -> Optional[Dict]:
    """
    Fit saturating curve to CCM convergence.

    Parameters
    ----------
    lib_sizes : np.ndarray
        Library sizes
    rho_values : np.ndarray
        Corresponding rho values

    Returns
    -------
    dict or None
        Fit parameters and diagnostics
    """
    # Drop NaN
    valid = ~(np.isnan(lib_sizes) | np.isnan(rho_values))
    if np.sum(valid) < 3:
        return None

    x = lib_sizes[valid]
    y = rho_values[valid]
    Lmax = float(np.max(x))

    # Initial guesses
    p0 = [max(1e-6, np.nanmax(y) - np.nanmin(y)),
          float(np.median(x)),
          float(np.nanmin(y))]

    try:
        popt, _ = curve_fit(saturating_curve, x, y, p0=p0, maxfev=10000)
        a, K, b = map(float, popt)
        yfit = saturating_curve(x, a, K, b)
        resid = y - yfit
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Tail slope
        tail_n = max(3, int(np.ceil(len(x) / 3)))
        x_tail = x[-tail_n:]
        y_tail = y[-tail_n:]
        slope_tail = np.polyfit(x_tail, y_tail, 1)[0] if len(x_tail) >= 2 else np.nan

        # Convergence rho
        rho_conv = float(np.nanmean(y_tail))

        return {
            'a': a,
            'K': K,
            'b': b,
            'R2': r2,
            'Lmax': Lmax,
            'slope_tail': slope_tail,
            'rho_conv': rho_conv
        }
    except Exception:
        return None


def has_converged(fit: Optional[Dict]) -> bool:
    """
    Check if CCM has converged based on fit.

    Criteria:
    - Good fit (R² > 0.7)
    - Positive amplitude (a > 0)
    - Positive convergence rho
    - Flat tail (slope < 0.001)
    - K within range
    """
    if fit is None:
        return False

    return (
        fit['a'] > 0 and
        fit['R2'] > 0.7 and
        fit['rho_conv'] > 0 and
        fit['slope_tail'] < 1e-3 and
        fit['K'] < 0.8 * fit['Lmax']
    )


def compute_ccm_pair(df: pd.DataFrame,
                    driver: str,
                    target: str,
                    E: int,
                    tau: int,
                    Tp: int,
                    theta: float = 0,
                    exclusionRadius: int = 0,
                    libSizes: str = "50 500 25",
                    sample: int = 100,
                    n_surrogates: int = 99,
                    seed: Optional[int] = None,
                    verbose: bool = False) -> Dict:
    """
    Test CCM for a single pair with surrogate testing.

    Parameters
    ----------
    df : pd.DataFrame
        Time series dataframe
    driver : str
        Driver variable (X)
    target : str
        Target variable (Y) - xmaps driver
    E, tau, Tp, theta : int/float
        EDM parameters
    exclusionRadius : int
        Exclusion radius
    libSizes : str
        Library sizes "start end increment"
    sample : int
        Random samples per library size
    n_surrogates : int
        Number of surrogates
    seed : int or None
        Random seed
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Results dictionary
    """
    if not PYEDM_AVAILABLE:
        raise ImportError("pyEDM required")

    if seed is not None:
        np.random.seed(seed)

    X = df[driver].values
    Y = df[target].values

    # Compute original CCM
    ccm_df = pd.DataFrame({'time': range(1, len(X)+1), driver: X, target: Y})

    try:
        ccm_result = CCM(
            dataFrame=ccm_df,
            columns=target,  # Y xmap X (testing X -> Y)
            target=driver,
            libSizes=libSizes,
            sample=sample,
            E=E,
            Tp=Tp,
            tau=tau,
            exclusionRadius=exclusionRadius,
            includeData=False
        )

        lib_sizes = ccm_result['LibMeans']['LibSize'].values
        rho_values = ccm_result['LibMeans'][f'{target}:{driver}'].values

        # Fit convergence curve
        fit = fit_ccm_curve(lib_sizes, rho_values)
        convergent = has_converged(fit)

        # Get rho at max lib size
        rho_original = rho_values[-1] if len(rho_values) > 0 else np.nan

    except Exception as e:
        if verbose:
            print(f"  CCM failed: {e}")
        return {
            'lib': driver,
            'target': target,
            'E': E,
            'tau': tau,
            'Tp': Tp,
            'theta': theta,
            'ccm_rho': np.nan,
            'convergence': False,
            'ccm_norm': np.nan,
            'trial_rho95p': np.nan,
            'trial_rho99p': np.nan,
            'p_rho': np.nan
        }

    # Surrogate testing if convergent
    if convergent:
        surrogate_rhos = []

        for _ in range(n_surrogates):
            # Generate twin surrogates
            X_surr, Y_surr = generate_multivariate_fourier_surrogates(
                X, Y, n_surr=1, verbose=False
            )
            X_surr = X_surr[0]
            Y_surr = Y_surr[0]

            df_surr = pd.DataFrame({
                'time': range(1, len(X_surr)+1),
                driver: X_surr,
                target: Y_surr
            })

            try:
                # Use only max library size for surrogates (faster)
                max_lib = lib_sizes[-1]
                ccm_surr = CCM(
                    dataFrame=df_surr,
                    columns=target,
                    target=driver,
                    libSizes=str(int(max_lib)),
                    sample=sample,
                    E=E,
                    Tp=Tp,
                    tau=tau,
                    exclusionRadius=exclusionRadius,
                    includeData=False
                )
                rho_surr = ccm_surr['LibMeans'][f'{target}:{driver}'].values[0]
                surrogate_rhos.append(rho_surr)
            except Exception:
                surrogate_rhos.append(np.nan)

        surrogate_rhos = np.array(surrogate_rhos)
        valid_surr = surrogate_rhos[~np.isnan(surrogate_rhos)]

        if len(valid_surr) > 0:
            rho_trial_mean = np.mean(valid_surr)
            rho_trial_p95 = np.percentile(valid_surr, 95)
            rho_trial_p99 = np.percentile(valid_surr, 99)
            p_rho = (np.sum(valid_surr >= rho_original) + 1) / (len(valid_surr) + 1)
        else:
            rho_trial_mean = np.nan
            rho_trial_p95 = np.nan
            rho_trial_p99 = np.nan
            p_rho = np.nan
    else:
        rho_trial_mean = np.nan
        rho_trial_p95 = np.nan
        rho_trial_p99 = np.nan
        p_rho = np.nan

    return {
        'lib': driver,
        'target': target,
        'E': E,
        'tau': tau,
        'Tp': Tp,
        'theta': theta,
        'ccm_rho': rho_original,
        'convergence': convergent,
        'ccm_norm': rho_original - rho_trial_mean if not np.isnan(rho_trial_mean) else np.nan,
        'trial_rho95p': rho_trial_p95,
        'trial_rho99p': rho_trial_p99,
        'p_rho': p_rho
    }


def run_ccm_workflow(df: pd.DataFrame,
                    pairs: Optional[List[Tuple[str, str]]] = None,
                    auto_optimize: bool = True,
                    E_range: range = range(2, 7),
                    tau_range: range = range(-3, 0),
                    theta_range: np.ndarray = np.linspace(0, 8, 17),
                    libSizes: str = "50 500 25",
                    sample: int = 100,
                    n_surrogates: int = 99,
                    n_jobs: int = 64,
                    seed: Optional[int] = None,
                    verbose: bool = True) -> pd.DataFrame:
    """
    Run CCM workflow on multiple pairs with parallel processing.

    Matches structure from 011_ccm_analysis_norm_v3.ipynb.

    Parameters
    ----------
    df : pd.DataFrame
        Time series dataframe with 'time' column
    pairs : list of tuples or None
        (driver, target) pairs to test. If None, test all permutations.
    auto_optimize : bool, default True
        Auto-optimize parameters per pair
    E_range, tau_range, theta_range : iterables
        Parameter ranges if not auto-optimizing
    libSizes : str
        Library sizes
    sample : int
        Random samples per library size
    n_surrogates : int
        Number of surrogates
    n_jobs : int, default 64
        Number of parallel jobs
    seed : int or None
        Random seed
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame
        Results with columns: lib, target, E, tau, Tp, theta, ccm_rho,
        convergence, ccm_norm, trial_rho95p, trial_rho99p, p_rho
    """
    # Get variable names (exclude 'time')
    variables = [col for col in df.columns if col != 'time']

    # Generate pairs if not provided
    if pairs is None:
        from itertools import permutations
        pairs = list(permutations(variables, 2))

    if verbose:
        print(f"Testing {len(pairs)} pairs...")

    # Prepare arguments for each pair
    args_list = [
        (pair, df, auto_optimize, E_range, libSizes, sample, n_surrogates, seed)
        for pair in pairs
    ]

    # Run in parallel
    with multiprocessing.Pool(processes=n_jobs) as pool:
        results = list(tqdm(
            pool.imap(_process_pair_wrapper, args_list),
            total=len(pairs),
            desc="CCM pairs",
            disable=not verbose
        ))

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Add resolved column
    results_df['resolved_nonlinear'] = (
        (results_df['convergence'] == True) &
        (results_df['p_rho'] < 0.05)
    )

    return results_df
