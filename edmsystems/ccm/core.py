"""
Core CCM computation functions.

Implements convergent cross mapping with random library sampling
following EDM best practices.
"""

import numpy as np
import pandas as pd
import pyEDM
from typing import Optional, Tuple, List


def compute_xmap(dataFrame: pd.DataFrame,
                columns: str,
                target: str,
                lib: str,
                pred: str,
                E: int,
                Tp: int,
                tau: int,
                exclusionRadius: int = 0,
                theta: float = 0,
                method: str = 'simplex') -> pd.DataFrame:
    """
    Compute cross-mapping from columns to target using pyEDM.

    Parameters
    ----------
    dataFrame : pd.DataFrame
        Input dataframe with time column
    columns : str
        Library variable (predictor)
    target : str
        Target variable (to be predicted)
    lib : str
        Library indices in pyEDM format (e.g., "1 100")
    pred : str
        Prediction indices in pyEDM format
    E : int
        Embedding dimension
    Tp : int
        Prediction horizon (negative for causal lag)
    tau : int
        Time delay (typically negative)
    exclusionRadius : int, default 0
        Exclusion radius for nearest neighbors
    theta : float, default 0
        S-map localization parameter (0 = simplex)
    method : str, default 'simplex'
        Method to use ('simplex' or 'smap')

    Returns
    -------
    pd.DataFrame
        pyEDM results with Observations and Predictions columns
    """
    # Set up validLib for NaN handling
    to_xmap = dataFrame.copy()
    to_xmap['validLib'] = to_xmap[target].shift(-Tp).notna()

    if method == 'simplex' or theta == 0:
        result = pyEDM.Simplex(
            dataFrame=to_xmap,
            columns=columns,
            target=target,
            lib=lib,
            pred=pred,
            E=E,
            Tp=Tp,
            tau=tau,
            exclusionRadius=exclusionRadius,
            validLib=to_xmap['validLib']
        )
    elif method == 'smap':
        result = pyEDM.SMap(
            dataFrame=to_xmap,
            columns=columns,
            target=target,
            lib=lib,
            pred=pred,
            E=E,
            Tp=Tp,
            tau=tau,
            theta=theta,
            exclusionRadius=exclusionRadius,
            validLib=to_xmap['validLib']
        )
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'simplex' or 'smap'")

    return result


def compute_ccm(dataFrame: pd.DataFrame,
               columns: str,
               target: str,
               E: int,
               Tp: int,
               tau: int,
               libSizes: str = "50 500 25",
               sample: int = 100,
               exclusionRadius: int = 0,
               theta: float = 0,
               method: str = 'simplex',
               seed: Optional[int] = None,
               verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute CCM with random library sampling.

    Tests if target causes columns by cross-mapping from columns to target
    across multiple library sizes.

    Parameters
    ----------
    dataFrame : pd.DataFrame
        Input dataframe with time column and variables
    columns : str
        Library variable (what we're using to predict)
    target : str
        Target variable (what we're predicting)
    E : int
        Embedding dimension
    Tp : int
        Prediction horizon
    tau : int
        Time delay
    libSizes : str, default "50 500 25"
        Library sizes in format "start end increment"
    sample : int, default 100
        Number of random samples per library size
    exclusionRadius : int, default 0
        Exclusion radius for nearest neighbors
    theta : float, default 0
        S-map localization parameter
    method : str, default 'simplex'
        Method to use ('simplex' or 'smap')
    seed : int or None
        Random seed for reproducibility
    verbose : bool, default False
        Print progress information

    Returns
    -------
    summary : pd.DataFrame
        Summary statistics (mean and std of rho) per library size
    details : pd.DataFrame
        Detailed results for each random library sample

    Notes
    -----
    Random library sampling ensures robust convergence testing by:
    - Avoiding temporal bias from fixed library windows
    - Averaging over many realizations to reduce sampling variance
    - Testing convergence across a range of library sizes
    """
    if seed is not None:
        np.random.seed(seed)

    # Parse libSizes string (format: "start end increment")
    lib_params = libSizes.split()
    start_lib = int(lib_params[0])
    end_lib = int(lib_params[1])
    increment = int(lib_params[2])

    lib_sizes = list(range(start_lib, end_lib + 1, increment))

    # Generate random libraries
    results = []
    max_index = len(dataFrame)

    for libSize in lib_sizes:
        if verbose:
            print(f"  LibSize {libSize}: ", end='', flush=True)

        for i in range(sample):
            # Randomly sample an end index
            end_idx = np.random.randint(libSize, max_index + 1)
            start_idx = end_idx - libSize

            # Create lib string in pyEDM format (1-based indexing)
            lib = f"{start_idx + 1} {end_idx}"
            pred = lib  # Use same range for prediction

            # Get cross-map results
            try:
                xmap = compute_xmap(
                    dataFrame, columns, target, lib, pred,
                    E, Tp, tau, exclusionRadius, theta, method
                )
                rho = xmap[['Observations', 'Predictions']].corr().iloc[0, 1]
            except Exception as e:
                if verbose and i == 0:  # Only print first failure
                    print(f"[Failed: {e}]", end='', flush=True)
                xmap = None
                rho = np.nan

            results.append({
                'LibSize': libSize,
                'lib': lib,
                'pred': pred,
                'rho': rho,
            })

        if verbose:
            print("✓")

    # Create detailed results dataframe
    details = pd.DataFrame(results)

    # Create summary table
    summary = (
        details
        .groupby('LibSize')['rho']
        .agg(rho_mean=lambda x: x.mean(skipna=True),
             rho_std=lambda x: x.std(skipna=True, ddof=0))
        .reset_index()
    )

    return summary, details


def compute_auc(summary: pd.DataFrame,
               lib_col: str = 'LibSize',
               rho_col: str = 'rho_mean') -> float:
    """
    Compute area under the convergence curve (AUC).

    Parameters
    ----------
    summary : pd.DataFrame
        CCM summary dataframe with library sizes and rho values
    lib_col : str, default 'LibSize'
        Name of library size column
    rho_col : str, default 'rho_mean'
        Name of rho column

    Returns
    -------
    float
        Area under curve using trapezoidal integration
    """
    valid_data = summary.dropna(subset=[rho_col])
    if len(valid_data) == 0:
        return np.nan

    lib_sizes = valid_data[lib_col].values
    rho_values = valid_data[rho_col].values

    auc = np.trapz(rho_values, lib_sizes)
    return auc
