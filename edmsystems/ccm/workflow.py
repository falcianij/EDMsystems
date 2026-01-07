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
                 surrogate_method: Union[str, List[str]] = 'twin',
                 period: int = 12,
                 optimize_params: bool = True,
                 optimize_theta: bool = False,
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
    surrogate_method : str or list of str, default 'twin'
        Surrogate method(s): 'twin', 'random', 'random_paired', 'circular', 'within_phase'
        Can be a single method or a list of methods to test multiple null models
    period : int, default 12
        Period for within_phase surrogates
    optimize_params : bool, default True
        Whether to optimize parameters (tau, E, Tp)
    optimize_theta : bool, default False
        Whether to optimize theta using S-map
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
        - rho_original: cross-mapping skill at largest library size
        - rho_surrogates_{method}: dict with surrogate rho values per method
        - rho_surrogate_mean_{method}: mean surrogate rho per method
        - rho_surrogate_std_{method}: std dev of surrogate rho per method
        - p_value_{method}: empirical p-value per method
        - is_significant_{method}: boolean (p < 0.01) per method
        - convergent: boolean (positive slope in convergence curve)
        - summary_original: convergence curve data
        - percentiles_{method}: 95th and 99th percentile per method
        - surrogate_methods: list of methods tested
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

    # Use rho at largest library size instead of AUC
    rho_original = summary_orig['rho_mean'].iloc[-1]  # Final library size rho

    if verbose:
        print(f"  Original rho (max lib): {rho_original:.3f}")

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
    # Convert surrogate_method to list if it's a single string
    if isinstance(surrogate_method, str):
        surrogate_methods = [surrogate_method]
    else:
        surrogate_methods = surrogate_method

    surrogate_seeds = np.random.randint(0, 2**31, size=n_surrogates)

    # Store results for each method
    results_by_method = {}

    for method in surrogate_methods:
        if verbose:
            print(f"\nGenerating {n_surrogates} '{method}' surrogates...")

        def compute_single_surrogate(surr_seed, surrogate_type):
            np.random.seed(surr_seed)

            # Generate surrogate based on method
            if surrogate_type == 'twin':
                X_surr, Y_surr = generate_twin_surrogates(X, Y)
            elif surrogate_type == 'random_paired':
                X_surr, Y_surr = generate_random_paired_surrogates(X, Y)
            elif surrogate_type == 'circular':
                X_surr, Y_surr = generate_circular_surrogates(X, Y)
            elif surrogate_type == 'within_phase':
                X_surr, Y_surr = generate_within_phase_surrogates(X, Y, period=period)
            elif surrogate_type == 'random':
                # For 'random', shuffle both independently
                X_surr = generate_random_surrogates(X)
                Y_surr = generate_random_surrogates(Y)
            else:
                raise ValueError(f"Unknown surrogate method: {surrogate_type}")

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

            # Return rho at largest library size
            return summary_surr['rho_mean'].iloc[-1]

        # Sequential surrogate computation (no parallel processing within pair)
        rho_surrogates = []
        for seed in surrogate_seeds:
            rho = compute_single_surrogate(seed, method)
            rho_surrogates.append(rho)

        rho_surrogates = np.array(rho_surrogates)

        # Remove NaN surrogates
        valid_surrogates = rho_surrogates[~np.isnan(rho_surrogates)]

        if len(valid_surrogates) == 0:
            if verbose:
                print(f"  WARNING: All {method} surrogates failed!")
            p_value = np.nan
            is_significant = False
            percentile_95 = np.nan
            percentile_99 = np.nan
        else:
            # Compute p-value (rho_original should be GREATER than surrogates for significance)
            p_value = (np.sum(valid_surrogates >= rho_original) + 1) / (len(valid_surrogates) + 1)
            is_significant = p_value < 0.01

            # Compute percentiles
            percentile_95 = np.percentile(valid_surrogates, 95)
            percentile_99 = np.percentile(valid_surrogates, 99)

            if verbose:
                print(f"  Surrogate rho: {np.mean(valid_surrogates):.3f} ± {np.std(valid_surrogates):.3f}")
                print(f"  95th percentile: {percentile_95:.3f}")
                print(f"  99th percentile: {percentile_99:.3f}")
                print(f"  p-value: {p_value:.4f}")
                print(f"  Significant: {is_significant}")

        # Store results for this method
        results_by_method[method] = {
            'rho_surrogates': rho_surrogates,
            'rho_surrogate_mean': np.nanmean(rho_surrogates),
            'rho_surrogate_std': np.nanstd(rho_surrogates),
            'p_value': p_value,
            'is_significant': is_significant,
            'percentile_95': percentile_95,
            'percentile_99': percentile_99,
            'n_failed': np.sum(np.isnan(rho_surrogates)),
        }

    # Build result dictionary with method-specific statistics
    result = {
        'driver': driver_name,
        'target': target_name,
        'tau': tau,
        'E': E,
        'Tp': Tp,
        'theta': theta,
        'exclusionRadius': exclusionRadius,
        'rho_original': rho_original,
        'convergent': convergent,
        'summary_original': summary_orig,
        'surrogate_methods': surrogate_methods,
    }

    # Add method-specific results
    for method, stats in results_by_method.items():
        result[f'rho_surrogates_{method}'] = stats['rho_surrogates']
        result[f'rho_surrogate_mean_{method}'] = stats['rho_surrogate_mean']
        result[f'rho_surrogate_std_{method}'] = stats['rho_surrogate_std']
        result[f'p_value_{method}'] = stats['p_value']
        result[f'is_significant_{method}'] = stats['is_significant']
        result[f'percentile_95_{method}'] = stats['percentile_95']
        result[f'percentile_99_{method}'] = stats['percentile_99']
        result[f'n_failed_surrogates_{method}'] = stats['n_failed']

    return result


def run_ccm_workflow(df: pd.DataFrame,
                    pairs: Optional[List[Tuple[str, str]]] = None,
                    datetime_col: str = 'datetime',
                    n_jobs: int = 1,
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
    n_jobs : int, default 1
        Number of parallel jobs for processing multiple pairs.
        Use -1 for all cores. Set to 1 for sequential processing.
    **kwargs
        Additional arguments passed to test_ccm_pair()

    Returns
    -------
    pd.DataFrame
        Results dataframe with one row per tested pair

    Examples
    --------
    >>> # Test all pairs sequentially
    >>> results = run_ccm_workflow(df)

    >>> # Test specific pairs in parallel
    >>> results = run_ccm_workflow(
    ...     df,
    ...     pairs=[('X', 'Y'), ('Y', 'X')],
    ...     n_jobs=-1,  # Parallel across pairs
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

    # Extract verbose flag for progress bar control
    verbose = kwargs.get('verbose', True)

    def test_single_pair(driver, target):
        X = df[driver].values
        Y = df[target].values
        result = test_ccm_pair(X, Y, driver, target, **kwargs)
        return result

    # Test pairs (optionally in parallel) with progress bar
    if n_jobs == 1:
        # Sequential with progress bar
        results = []
        for driver, target in tqdm(pairs, desc="Testing pairs", disable=not verbose):
            results.append(test_single_pair(driver, target))
    else:
        # Parallel across pairs with progress bar
        if verbose:
            print(f"\nTesting {len(pairs)} pairs in parallel (n_jobs={n_jobs})...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(test_single_pair)(driver, target)
            for driver, target in tqdm(pairs, desc="Testing pairs", disable=not verbose)
        )

    # Format results as dataframe
    # Extract all unique surrogate methods used
    all_methods = set()
    for r in results:
        all_methods.update(r['surrogate_methods'])
    all_methods = sorted(list(all_methods))

    # Build dataframe rows
    rows = []
    for r in results:
        row = {
            'driver': r['driver'],
            'target': r['target'],
            'tau': r['tau'],
            'E': r['E'],
            'Tp': r['Tp'],
            'theta': r['theta'],
            'rho_original': r['rho_original'],
            'convergent': r['convergent'],
        }

        # Add method-specific columns
        for method in all_methods:
            if method in r['surrogate_methods']:
                row[f'rho_surrogate_mean_{method}'] = r[f'rho_surrogate_mean_{method}']
                row[f'rho_surrogate_std_{method}'] = r[f'rho_surrogate_std_{method}']
                row[f'p_value_{method}'] = r[f'p_value_{method}']
                row[f'is_significant_{method}'] = r[f'is_significant_{method}']
                row[f'percentile_95_{method}'] = r[f'percentile_95_{method}']
                row[f'percentile_99_{method}'] = r[f'percentile_99_{method}']
                row[f'n_failed_surrogates_{method}'] = r[f'n_failed_surrogates_{method}']
            else:
                # Method not used for this pair, fill with NaN
                row[f'rho_surrogate_mean_{method}'] = np.nan
                row[f'rho_surrogate_std_{method}'] = np.nan
                row[f'p_value_{method}'] = np.nan
                row[f'is_significant_{method}'] = False
                row[f'percentile_95_{method}'] = np.nan
                row[f'percentile_99_{method}'] = np.nan
                row[f'n_failed_surrogates_{method}'] = np.nan

        rows.append(row)

    results_df = pd.DataFrame(rows)

    return results_df
