"""
Surrogate time series generators for CCM null model testing.

Implements multiple surrogate generation methods that preserve
different statistical properties while destroying causal structure.
"""

import numpy as np
from typing import Tuple, Optional
from numpy.fft import fft, ifft


def generate_twin_surrogates(X: np.ndarray,
                            Y: np.ndarray,
                            seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate twin (multivariate Fourier) surrogates.

    Applies the SAME random phase shift to both X and Y, which:

    PRESERVES:
    - Power spectrum of X (and thus ACF of X via Wiener-Khinchin)
    - Power spectrum of Y (and thus ACF of Y via Wiener-Khinchin)
    - Cross-spectrum (and thus CCF between X and Y)
    - Amplitude distributions

    DESTROYS:
    - Temporal phase structure
    - Causal predictive relationships
    - Nonlinear dependencies

    Parameters
    ----------
    X, Y : np.ndarray
        Input time series (1D arrays, same length)
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    X_surr, Y_surr : np.ndarray
        Surrogate time series with NaNs preserved in original positions

    Notes
    -----
    This is the GOLD STANDARD for CCM surrogate testing because it preserves
    all linear correlations (ACF and CCF) while destroying only the causal
    temporal structure.

    The key insight is that applying the SAME phase randomization to both
    series preserves the cross-spectrum S_xy(f) = X(f) * conj(Y(f)).

    References
    ----------
    Prichard, D., & Theiler, J. (1994). Generating surrogate data for time
    series with several simultaneously measured variables. Physical Review
    Letters, 73(7), 951.
    """
    if seed is not None:
        np.random.seed(seed)

    # Record NaN positions
    nan_mask_X = np.isnan(X)
    nan_mask_Y = np.isnan(Y)

    # Replace NaNs with 0 for FFT
    X_clean = np.where(nan_mask_X, 0.0, X)
    Y_clean = np.where(nan_mask_Y, 0.0, Y)

    n = len(X_clean)

    # FFT of both series
    X_fft = fft(X_clean)
    Y_fft = fft(Y_clean)

    # Generate symmetric random phase sequence
    n_freqs = n // 2 + 1
    random_phases = np.random.uniform(-np.pi, np.pi, n_freqs)

    # DC (0 freq) must stay real
    random_phases[0] = 0

    # Nyquist freq (for even n) must stay real
    if n % 2 == 0:
        random_phases[-1] = 0

    # Build full symmetric phase vector for real IFFT
    if n % 2 == 0:
        full_phases = np.concatenate([random_phases, -random_phases[-2:0:-1]])
    else:
        full_phases = np.concatenate([random_phases, -random_phases[-1:0:-1]])

    # Apply the SAME random phase shift to X and Y (preserves cross-spectrum)
    X_fft_surr = np.abs(X_fft) * np.exp(1j * (np.angle(X_fft) + full_phases))
    Y_fft_surr = np.abs(Y_fft) * np.exp(1j * (np.angle(Y_fft) + full_phases))

    # Inverse FFT
    X_surr = np.real(ifft(X_fft_surr))
    Y_surr = np.real(ifft(Y_fft_surr))

    # Restore NaN positions
    X_surr[nan_mask_X] = np.nan
    Y_surr[nan_mask_Y] = np.nan

    return X_surr, Y_surr


def generate_random_surrogates(X: np.ndarray,
                              seed: Optional[int] = None) -> np.ndarray:
    """
    Generate random shuffle surrogates (single variable).

    PRESERVES:
    - Amplitude distribution

    DESTROYS:
    - All temporal structure
    - Autocorrelation
    - Any relationships with other variables

    Parameters
    ----------
    X : np.ndarray
        Input time series
    seed : int or None
        Random seed

    Returns
    -------
    np.ndarray
        Randomly shuffled time series
    """
    if seed is not None:
        np.random.seed(seed)

    # Handle NaNs
    nan_mask = np.isnan(X)
    X_clean = X[~nan_mask]

    # Shuffle non-NaN values
    X_surr_clean = np.random.permutation(X_clean)

    # Reconstruct with NaNs in original positions
    X_surr = np.empty_like(X)
    X_surr[nan_mask] = np.nan
    X_surr[~nan_mask] = X_surr_clean

    return X_surr


def generate_random_paired_surrogates(X: np.ndarray,
                                     Y: np.ndarray,
                                     seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random paired surrogates (shuffle together).

    PRESERVES:
    - Amplitude distributions
    - Instantaneous correlation between X and Y

    DESTROYS:
    - Temporal order
    - Autocorrelation
    - Lagged cross-correlation

    Parameters
    ----------
    X, Y : np.ndarray
        Input time series (same length)
    seed : int or None
        Random seed

    Returns
    -------
    X_surr, Y_surr : np.ndarray
        Jointly shuffled time series

    Notes
    -----
    This applies the SAME permutation to both X and Y, preserving their
    instantaneous pairing while destroying temporal structure.
    """
    if seed is not None:
        np.random.seed(seed)

    assert len(X) == len(Y), "X and Y must have same length"

    # Get valid indices (where both are non-NaN)
    valid_mask = (~np.isnan(X)) & (~np.isnan(Y))

    # Generate random permutation of valid indices
    valid_indices = np.where(valid_mask)[0]
    shuffled_indices = np.random.permutation(valid_indices)

    # Create surrogates
    X_surr = X.copy()
    Y_surr = Y.copy()

    # Apply same permutation to both
    X_surr[valid_indices] = X[shuffled_indices]
    Y_surr[valid_indices] = Y[shuffled_indices]

    return X_surr, Y_surr


def generate_circular_surrogates(X: np.ndarray,
                                Y: np.ndarray,
                                seed: Optional[int] = None,
                                exclude_zero_shift: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate circular shift surrogates (paired).

    PRESERVES:
    - All temporal structure within each variable
    - ACF of X and Y
    - CCF shape (but shifted in time)

    DESTROYS:
    - Temporal alignment between X and Y
    - Causal relationships

    Parameters
    ----------
    X, Y : np.ndarray
        Input time series
    seed : int or None
        Random seed
    exclude_zero_shift : bool, default True
        If True, avoid zero shift (which would be identity)

    Returns
    -------
    X_surr, Y_surr : np.ndarray
        Circularly shifted time series

    Notes
    -----
    Applies a random circular shift to one variable, breaking the
    temporal alignment with the other while preserving internal structure.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(X)

    if exclude_zero_shift and n > 1:
        shift = np.random.randint(1, n)
    else:
        shift = np.random.randint(0, n)

    # Apply circular shift to Y (keep X unchanged)
    X_surr = X.copy()
    Y_surr = np.roll(Y, shift)

    return X_surr, Y_surr


def generate_within_phase_surrogates(X: np.ndarray,
                                    Y: np.ndarray,
                                    period: int = 12,
                                    seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate within-phase (seasonal) surrogates.

    PRESERVES:
    - Seasonal patterns
    - Within-season relationships
    - Amplitude distributions

    DESTROYS:
    - Year-to-year temporal order
    - Long-term trends

    Parameters
    ----------
    X, Y : np.ndarray
        Input time series
    period : int, default 12
        Seasonal period (e.g., 12 for monthly data)
    seed : int or None
        Random seed

    Returns
    -------
    X_surr, Y_surr : np.ndarray
        Within-phase shuffled time series

    Notes
    -----
    For each phase (e.g., month), permutes across cycles (e.g., years)
    with the SAME permutation for X and Y. This preserves seasonal
    relationships while destroying interannual structure.

    Example:
    For monthly data with period=12:
    - All Januaries are permuted together (same permutation for X and Y)
    - All Februaries are permuted together
    - etc.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(X)
    n_cycles = n // period
    head = n_cycles * period

    if n_cycles == 0:
        # Not enough data for even one cycle, return copies
        return X.copy(), Y.copy()

    # Reshape into [cycles, phases] (e.g., [years, months])
    X_core = X[:head].reshape(n_cycles, period)
    Y_core = Y[:head].reshape(n_cycles, period)

    X_surr_core = X_core.copy()
    Y_surr_core = Y_core.copy()

    # For each phase (month), permute cycles with SAME permutation for X and Y
    for phase in range(period):
        perm = np.random.permutation(n_cycles)
        X_surr_core[:, phase] = X_core[perm, phase]
        Y_surr_core[:, phase] = Y_core[perm, phase]

    # Reconstruct flat arrays
    X_surr = np.empty_like(X)
    Y_surr = np.empty_like(Y)

    X_surr[:head] = X_surr_core.reshape(-1)
    Y_surr[:head] = Y_surr_core.reshape(-1)

    # Copy remainder (partial cycle at end)
    if head < n:
        X_surr[head:] = X[head:]
        Y_surr[head:] = Y[head:]

    return X_surr, Y_surr
