"""
Standardized CCM workflow for causal inference.

Provides high-level functions for running complete CCM analysis
with parameter optimization, surrogate testing, and result formatting.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Literal, Union
from joblib import Parallel, delayed
from tqdm import tqdm

from .core import compute_ccm, compute_auc
from .parameters import optimize_parameters
from ..surrogates import (
    generate_twin_surrogates,
    generate_random_surrogates,
    generate_random_paired_surrogates,
    generate_circular_surrogates,
    generate_within_phase_surrogates,
)


def test_ccm_pair(X: np.ndarray,
                 Y: np.ndarray,
                 driver_name: str,
                 target_name: str,
                 tau: Optional[int] = None,
                 E: Optional[int] = None,
                 Tp: Optional[int] = None,
                 theta: float = 0,
                 exclusionRadius: int = 0,
                 libSizes: str = "50 500 25",
                 sample: int = 100,
                 n_surrogates: int = 99,
                 surrogate_method: Literal['twin', 'random', 'random_paired', 'circular', 'within_phase'] = 'twin',
                 period: int = 12,
                 optimize_params: bool = True,
                 optimize_theta: bool = False,
                 n_jobs: int = -1,
                 seed: Optional[int] = None,
                 verbose: bool = True) -> dict:
    """
    Test a single CCM pair with surrogate testing.

    Parameters
    ----------
    X : np.ndarray
        Driver variable (cause)
    Y : np.ndarray
        Target variable (effect)
    driver_name : str
        Name of driver variable
    target_name : str
        Name of target variable
    tau, E, Tp : int or None
        Embedding parameters. If None, will be optimized.
    theta : float, default 0
        S-map localization parameter
    exclusionRadius : int, default 0
        Exclusion radius for nearest neighbors
    libSizes : str, default "50 500 25"
        Library sizes in format "start end increment"
    sample : int, default 100
        Number of random library samples per library size
    n_surrogates : int, default 99
        Number of surrogate realizations
    surrogate_method : str, default 'twin'
        Surrogate method: 'twin', 'random', 'random_paired', 'circular', 'within_phase'
    period : int, default 12
        Period for within_phase surrogates
    optimize_params : bool, default True
        Whether to optimize parameters (tau, E, Tp)
    optimize_theta : bool, default False
        Whether to optimize theta using S-map
    n_jobs : int, default -1
        Number of parallel jobs (-1 = all cores)
    seed : int or None
        Random seed
    verbose : bool, default True
        Print progress information

    Returns
    -------
    dict
        Results dictionary containing:
        - driver, target: variable names
        - tau, E, Tp, theta: parameters used
        - rho_mean: mean cross-mapping skill
        - auc_original: AUC of original convergence curve
        - auc_surrogates: array of surrogate AUCs
        - auc_surrogate_mean, auc_surrogate_std: summary statistics
        - p_value: empirical p-value
        - is_significant: boolean (p < 0.01)
        - convergent: boolean (positive slope in convergence curve)
        - summary_original: convergence curve data
        - percentiles: 95th and 99th percentile of surrogate distribution
    """
    if seed is not None:
        np.random.seed(seed)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Testing: {driver_name} -> {target_name}")
        print(f"{'='*70}")

    # Step 1: Optimize parameters if needed
    if optimize_params or tau is None or E is None or Tp is None:
        if verbose:
            print("\nOptimizing parameters...")
        params = optimize_parameters(
            X, Y,
            optimize_theta=optimize_theta,
            verbose=verbose
        )
        tau = params['tau'] if tau is None else tau
        E = params['E'] if E is None else E
        Tp = params['Tp'] if Tp is None else Tp
        if optimize_theta:
            theta = params['theta']
    elif verbose:
        print(f"\nUsing provided parameters: tau={tau}, E={E}, Tp={Tp}, theta={theta}")

    # Step 2: Compute original CCM
    if verbose:
        print(f"\nComputing original CCM (libSizes={libSizes}, sample={sample})...")

    df = pd.DataFrame({
        'time': range(1, len(X) + 1),
        driver_name: X,
        target_name: Y
    })

    summary_orig, details_orig = compute_ccm(
        df,
        columns=target_name,  # Y xmap X (testing if X causes Y)
        target=driver_name,
        E=E,
        Tp=Tp,
        tau=-tau,
        libSizes=libSizes,
        sample=sample,
        exclusionRadius=exclusionRadius,
        theta=theta,
        seed=seed,
        verbose=verbose
    )

    auc_orig = compute_auc(summary_orig)
    rho_mean = summary_orig['rho_mean'].iloc[-1]  # Final library size rho

    if verbose:
        print(f"  Original AUC: {auc_orig:.2f}")
        print(f"  Final rho: {rho_mean:.3f}")

    # Step 3: Test convergence (positive slope)
    if len(summary_orig) >= 2:
        # Simple linear regression on last half of convergence curve
        mid_point = len(summary_orig) // 2
        lib_sizes = summary_orig['LibSize'].values[mid_point:]
        rhos = summary_orig['rho_mean'].values[mid_point:]

        # Remove NaNs
        valid = ~np.isnan(rhos)
        if np.sum(valid) >= 2:
            slope = np.polyfit(lib_sizes[valid], rhos[valid], 1)[0]
            convergent = slope > 0
        else:
            convergent = False
    else:
        convergent = False

    if verbose:
        print(f"  Convergent: {convergent}")

    # Step 4: Generate and test surrogates
    if verbose:
        print(f"\nGenerating {n_surrogates} {surrogate_method} surrogates...")

    surrogate_seeds = np.random.randint(0, 2**31, size=n_surrogates)

    def compute_single_surrogate(surr_seed):
        np.random.seed(surr_seed)

        # Generate surrogate
        if surrogate_method == 'twin':
            X_surr, Y_surr = generate_twin_surrogates(X, Y)
        elif surrogate_method == 'random_paired':
            X_surr, Y_surr = generate_random_paired_surrogates(X, Y)
        elif surrogate_method == 'circular':
            X_surr, Y_surr = generate_circular_surrogates(X, Y)
        elif surrogate_method == 'within_phase':
            X_surr, Y_surr = generate_within_phase_surrogates(X, Y, period=period)
        elif surrogate_method == 'random':
            # For 'random', shuffle both independently
            X_surr = generate_random_surrogates(X)
            Y_surr = generate_random_surrogates(Y)
        else:
            raise ValueError(f"Unknown surrogate method: {surrogate_method}")

        # Compute CCM on surrogate
        df_surr = pd.DataFrame({
            'time': range(1, len(X_surr) + 1),
            driver_name: X_surr,
            target_name: Y_surr
        })

        summary_surr, _ = compute_ccm(
            df_surr,
            columns=target_name,
            target=driver_name,
            E=E,
            Tp=Tp,
            tau=-tau,
            libSizes=libSizes,
            sample=sample,
            exclusionRadius=exclusionRadius,
            theta=theta,
            verbose=False
        )

        return compute_auc(summary_surr)

    # Parallel surrogate computation
    auc_surrogates = Parallel(n_jobs=n_jobs)(
        delayed(compute_single_surrogate)(seed)
        for seed in tqdm(surrogate_seeds, desc="Surrogates", disable=not verbose)
    )

    auc_surrogates = np.array(auc_surrogates)

    # Remove NaN surrogates
    valid_surrogates = auc_surrogates[~np.isnan(auc_surrogates)]

    if len(valid_surrogates) == 0:
        if verbose:
            print("  WARNING: All surrogates failed!")
        p_value = np.nan
        is_significant = False
        percentile_95 = np.nan
        percentile_99 = np.nan
    else:
        # Compute p-value
        p_value = (np.sum(valid_surrogates >= auc_orig) + 1) / (len(valid_surrogates) + 1)
        is_significant = p_value < 0.01

        # Compute percentiles
        percentile_95 = np.percentile(valid_surrogates, 95)
        percentile_99 = np.percentile(valid_surrogates, 99)

        if verbose:
            print(f"\n  Surrogate AUC: {np.mean(valid_surrogates):.2f} ± {np.std(valid_surrogates):.2f}")
            print(f"  95th percentile: {percentile_95:.2f}")
            print(f"  99th percentile: {percentile_99:.2f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Significant: {is_significant}")

    return {
        'driver': driver_name,
        'target': target_name,
        'tau': tau,
        'E': E,
        'Tp': Tp,
        'theta': theta,
        'exclusionRadius': exclusionRadius,
        'rho_mean': rho_mean,
        'auc_original': auc_orig,
        'auc_surrogates': auc_surrogates,
        'auc_surrogate_mean': np.nanmean(auc_surrogates),
        'auc_surrogate_std': np.nanstd(auc_surrogates),
        'p_value': p_value,
        'is_significant': is_significant,
        'convergent': convergent,
        'percentile_95': percentile_95,
        'percentile_99': percentile_99,
        'summary_original': summary_orig,
        'n_failed_surrogates': np.sum(np.isnan(auc_surrogates)),
    }


def run_ccm_workflow(df: pd.DataFrame,
                    pairs: Optional[List[Tuple[str, str]]] = None,
                    datetime_col: str = 'datetime',
                    **kwargs) -> pd.DataFrame:
    """
    Run complete CCM workflow on multiple variable pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with datetime column and variables
    pairs : list of tuples or None
        List of (driver, target) pairs to test.
        If None, tests all possible directed pairs.
    datetime_col : str, default 'datetime'
        Name of datetime column (excluded from analysis)
    **kwargs
        Additional arguments passed to test_ccm_pair()

    Returns
    -------
    pd.DataFrame
        Results dataframe with one row per tested pair

    Examples
    --------
    >>> # Test all pairs
    >>> results = run_ccm_workflow(df)

    >>> # Test specific pairs
    >>> results = run_ccm_workflow(
    ...     df,
    ...     pairs=[('X', 'Y'), ('Y', 'X')],
    ...     n_surrogates=99,
    ...     surrogate_method='twin'
    ... )
    """
    # Get variable names
    variables = [col for col in df.columns if col != datetime_col]

    # Generate pairs if not provided
    if pairs is None:
        pairs = [(driver, target) for driver in variables
                 for target in variables if driver != target]

    # Extract common kwargs
    verbose = kwargs.get('verbose', True)
    n_jobs_pairs = kwargs.get('n_jobs_pairs', 1)  # Parallelization across pairs

    # Remove n_jobs_pairs from kwargs (not used by test_ccm_pair)
    kwargs_for_test = {k: v for k, v in kwargs.items() if k != 'n_jobs_pairs'}

    def test_single_pair(driver, target):
        X = df[driver].values
        Y = df[target].values
        result = test_ccm_pair(X, Y, driver, target, **kwargs_for_test)
        return result

    # Test pairs (optionally in parallel)
    if n_jobs_pairs == 1:
        # Sequential
        results = [test_single_pair(driver, target) for driver, target in pairs]
    else:
        # Parallel across pairs
        if verbose:
            print(f"\nTesting {len(pairs)} pairs in parallel (n_jobs={n_jobs_pairs})...")
        results = Parallel(n_jobs=n_jobs_pairs)(
            delayed(test_single_pair)(driver, target)
            for driver, target in pairs
        )

    # Format results as dataframe
    results_df = pd.DataFrame([
        {
            'driver': r['driver'],
            'target': r['target'],
            'tau': r['tau'],
            'E': r['E'],
            'Tp': r['Tp'],
            'theta': r['theta'],
            'rho_mean': r['rho_mean'],
            'auc_original': r['auc_original'],
            'auc_surrogate_mean': r['auc_surrogate_mean'],
            'auc_surrogate_std': r['auc_surrogate_std'],
            'percentile_95': r['percentile_95'],
            'percentile_99': r['percentile_99'],
            'p_value': r['p_value'],
            'is_significant': r['is_significant'],
            'convergent': r['convergent'],
            'n_failed_surrogates': r['n_failed_surrogates'],
        }
        for r in results
    ])

    return results_df
