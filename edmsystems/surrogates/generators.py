"""
Surrogate time series generation for EDM null model testing.

This module provides multiple methods for generating surrogate time series
that preserve different properties of the original data while destroying
temporal causal structure.
"""

import numpy as np
from typing import Tuple, Optional, Literal
from tqdm import tqdm


def generate_random_surrogates(x: np.ndarray,
                               n_surr: int,
                               verbose: bool = False) -> np.ndarray:
    """
    Generate random-shuffled surrogates.

    Destroys all temporal structure while preserving amplitude distribution.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Input time series
    n_surr : int
        Number of surrogates to generate
    verbose : bool, default False
        Show progress bar

    Returns
    -------
    np.ndarray, shape (n_surr, N)
        Surrogate time series
    """
    x = np.asarray(x)
    n = x.shape[0]
    surrogates = np.zeros((n_surr, n), dtype=float)

    iterator = tqdm(range(n_surr), desc="Random shuffle", disable=not verbose)

    for i in iterator:
        surrogates[i] = np.random.permutation(x)

    return surrogates


def generate_iaaft_surrogates(x: np.ndarray,
                              n_surr: int,
                              tol_pc: float = 5.0,
                              max_iter: int = 10000,
                              sorttype: str = "quicksort",
                              verbose: bool = True) -> np.ndarray:
    """
    Generate Iterative Amplitude Adjusted Fourier Transform (IAAFT) surrogates.

    Preserves both power spectrum AND amplitude distribution.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Input time series
    n_surr : int
        Number of surrogates to generate
    tol_pc : float, default 5.0
        Tolerance percentage for power spectrum matching
    max_iter : int, default 10000
        Maximum iterations for convergence
    sorttype : str, default 'quicksort'
        Sorting algorithm for numpy.argsort
    verbose : bool, default True
        Show progress bar

    Returns
    -------
    np.ndarray, shape (n_surr, N)
        IAAFT surrogate time series
    """
    x = np.asarray(x)
    n = x.shape[0]
    surrogates = np.zeros((n_surr, n), dtype=float)

    # Original spectrum and sorted values
    x_fft_amp = np.abs(np.fft.fft(x))
    x_sorted = np.sort(x)
    r_orig = np.argsort(x)

    iterator = tqdm(range(n_surr), desc="IAAFT surrogates", disable=not verbose)

    for k in iterator:
        # Initialize with random permutation
        count = 0
        r_prev = np.random.permutation(n)
        r_curr = r_orig
        z_n = x[r_prev]
        percent_unequal = 100.0

        # Iterative adjustment
        while (percent_unequal > tol_pc) and (count < max_iter):
            r_prev = r_curr

            # FFT and replace amplitudes
            y_prev = z_n
            fft_prev = np.fft.fft(y_prev)
            phi_prev = np.angle(fft_prev)
            e_i_phi = np.exp(phi_prev * 1j)
            z_n = np.fft.ifft(x_fft_amp * e_i_phi)

            # Rescale to original distribution
            r_curr = np.argsort(z_n, kind=sorttype)
            z_n[r_curr] = x_sorted.copy()

            percent_unequal = ((r_curr != r_prev).sum() * 100.0) / n
            count += 1

        if count >= (max_iter - 1) and verbose:
            print(f"Warning: max iterations reached for surrogate {k}")

        surrogates[k] = np.real(z_n)

    return surrogates


def generate_seasonal_surrogates(x: np.ndarray,
                                 n_surr: int,
                                 period: int = 12,
                                 mode: Literal["circular", "cycle_perm", "within_phase"] = "within_phase",
                                 seed: Optional[int] = None,
                                 exclude_zero_shift: bool = True,
                                 verbose: bool = False) -> np.ndarray:
    """
    Generate seasonal/recurrence-aware surrogate time series.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Input time series (e.g., monthly data)
    n_surr : int
        Number of surrogates to generate
    period : int, default 12
        Seasonal period (e.g., 12 for monthly)
    mode : {'circular', 'cycle_perm', 'within_phase'}, default 'within_phase'
        - 'circular': Random circular shift of full series
        - 'cycle_perm': Permute complete cycles, keep intra-cycle order
        - 'within_phase': Shuffle values within same phase across cycles
    seed : int or None
        Random seed for reproducibility
    exclude_zero_shift : bool, default True
        For 'circular', avoid zero shift
    verbose : bool, default False
        Show progress bar

    Returns
    -------
    np.ndarray, shape (n_surr, N)
        Surrogate time series
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = x.size
    surrogates = np.zeros((n_surr, n), dtype=float)

    # Circular mode
    if mode == "circular":
        for k in range(n_surr):
            if n == 0:
                surrogates[k] = x
                continue

            if exclude_zero_shift and n > 1:
                shift = rng.integers(1, n)
            else:
                shift = rng.integers(0, n)

            surrogates[k] = np.r_[x[-shift:], x[:-shift]] if shift else x.copy()

        return surrogates

    # Cycle-based modes
    n_cycles = n // period
    head = n_cycles * period
    remainder = x[head:]

    if mode == "cycle_perm":
        if n_cycles <= 1:
            return np.tile(x, (n_surr, 1))

        core = x[:head].reshape(n_cycles, period)

        for k in range(n_surr):
            order = rng.permutation(n_cycles)
            surrogates[k, :head] = core[order, :].reshape(-1)
            surrogates[k, head:] = remainder

        return surrogates

    if mode == "within_phase":
        if n_cycles == 0:
            return np.tile(x, (n_surr, 1))

        core = x[:head].reshape(n_cycles, period)

        for k in range(n_surr):
            y = core.copy()
            # For each phase (month), permute across cycles (years)
            for j in range(period):
                y[:, j] = y[rng.permutation(n_cycles), j]

            surrogates[k, :head] = y.reshape(-1)
            surrogates[k, head:] = remainder

        return surrogates

    raise ValueError(f"Unknown mode: {mode}. Must be 'circular', 'cycle_perm', or 'within_phase'")


def generate_seasonal_pair_surrogates(x: np.ndarray,
                                     y: np.ndarray,
                                     n_surr: int,
                                     period: int = 12,
                                     mode: Literal["within_phase", "circular"] = "within_phase",
                                     seed: Optional[int] = None,
                                     exclude_zero_shift: bool = True,
                                     verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate seasonal surrogates for two paired series, shuffling them together.

    IMPORTANT: Both variables are shuffled together to preserve their
    instantaneous correlation while destroying causal temporal structure.

    Parameters
    ----------
    x, y : np.ndarray, shape (N,)
        Input time series (must have same length)
    n_surr : int
        Number of surrogate pairs to generate
    period : int, default 12
        Seasonal period (e.g., 12 for monthly)
    mode : {'within_phase', 'circular'}, default 'within_phase'
        - 'within_phase': For each phase, permute years jointly for x,y
        - 'circular': Random circular shift applied to both x,y
    seed : int or None
        Random seed for reproducibility
    exclude_zero_shift : bool, default True
        For 'circular', avoid zero shift
    verbose : bool, default False
        Show progress bar

    Returns
    -------
    x_surr : np.ndarray, shape (n_surr, N)
        Surrogate series for x
    y_surr : np.ndarray, shape (n_surr, N)
        Surrogate series for y (shuffled with same permutation as x)
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    assert x.shape == y.shape, "x and y must have same shape"

    n = x.size
    x_surr = np.zeros((n_surr, n), dtype=float)
    y_surr = np.zeros((n_surr, n), dtype=float)

    iterator = tqdm(range(n_surr), desc=f"{mode} (paired)",
                   disable=not verbose)

    # Circular paired shift
    if mode == "circular":
        for k in iterator:
            if n == 0:
                x_surr[k] = x
                y_surr[k] = y
                continue

            if exclude_zero_shift and n > 1:
                shift = rng.integers(1, n)
            else:
                shift = rng.integers(0, n)

            if shift:
                x_surr[k] = np.r_[x[-shift:], x[:-shift]]
                y_surr[k] = np.r_[y[-shift:], y[:-shift]]
            else:
                x_surr[k] = x.copy()
                y_surr[k] = y.copy()

        return x_surr, y_surr

    # Within-phase paired shuffling
    if mode == "within_phase":
        if n == 0:
            return np.tile(x, (n_surr, 1)), np.tile(y, (n_surr, 1))

        n_cycles = n // period
        head = n_cycles * period
        remainder_x = x[head:]
        remainder_y = y[head:]

        if n_cycles == 0:
            return np.tile(x, (n_surr, 1)), np.tile(y, (n_surr, 1))

        # Reshape into [cycles, phases] (e.g., [years, months])
        core_x = x[:head].reshape(n_cycles, period)
        core_y = y[:head].reshape(n_cycles, period)

        for k in iterator:
            Xk = core_x.copy()
            Yk = core_y.copy()

            # For each phase (month), permute cycles with SAME permutation for x and y
            for j in range(period):
                perm = rng.permutation(n_cycles)
                Xk[:, j] = Xk[perm, j]
                Yk[:, j] = Yk[perm, j]

            x_surr[k, :head] = Xk.reshape(-1)
            y_surr[k, :head] = Yk.reshape(-1)
            x_surr[k, head:] = remainder_x
            y_surr[k, head:] = remainder_y

        return x_surr, y_surr

    raise ValueError(f"Unknown mode: {mode}. Must be 'within_phase' or 'circular'")


def generate_random_pair_surrogates(x: np.ndarray,
                                   y: np.ndarray,
                                   n_surr: int,
                                   verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate randomly shuffled paired surrogates.

    Preserves pairing (and correlation) between x and y while randomizing
    their temporal order.

    Parameters
    ----------
    x, y : np.ndarray, shape (N,)
        Input time series (must have same length)
    n_surr : int
        Number of surrogate pairs
    verbose : bool, default False
        Show progress bar

    Returns
    -------
    x_surr : np.ndarray, shape (n_surr, N)
        Surrogate series for x
    y_surr : np.ndarray, shape (n_surr, N)
        Surrogate series for y
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    assert x.shape == y.shape, "x and y must have same shape"

    n = x.shape[0]
    x_surr = np.zeros((n_surr, n), dtype=float)
    y_surr = np.zeros((n_surr, n), dtype=float)

    iterator = tqdm(range(n_surr), desc="Random (paired)",
                   disable=not verbose)

    for k in iterator:
        perm = np.random.permutation(n)
        x_surr[k] = x[perm]
        y_surr[k] = y[perm]

    return x_surr, y_surr
