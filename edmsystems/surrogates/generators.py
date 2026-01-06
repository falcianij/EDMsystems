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


def generate_twin_iaaft_surrogates(x: np.ndarray,
                                   y: np.ndarray,
                                   n_surr: int,
                                   tol_pc: float = 5.0,
                                   max_iter: int = 10000,
                                   sorttype: str = "quicksort",
                                   verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate twin IAAFT surrogates for two time series.

    Creates surrogates that preserve the autocorrelation function (ACF)
    of each variable independently. This preserves the internal dynamics
    of each time series while destroying their cross-correlation and
    causal coupling.

    IMPORTANT: This generates surrogates INDEPENDENTLY for X and Y.
    The ACF of each is preserved, but the CCF (cross-correlation) is
    destroyed, which is appropriate for testing causal influence while
    controlling for autocorrelation.

    Parameters
    ----------
    x, y : np.ndarray, shape (N,)
        Input time series (must have same length)
    n_surr : int
        Number of surrogate pairs to generate
    tol_pc : float, default 5.0
        Tolerance percentage for IAAFT convergence
    max_iter : int, default 10000
        Maximum iterations for IAAFT
    sorttype : str, default 'quicksort'
        Sorting algorithm for numpy.argsort
    verbose : bool, default True
        Show progress bar

    Returns
    -------
    x_surr : np.ndarray, shape (n_surr, N)
        IAAFT surrogates for x (preserves ACF of x)
    y_surr : np.ndarray, shape (n_surr, N)
        IAAFT surrogates for y (preserves ACF of y)

    Notes
    -----
    Twin surrogates preserve:
    - ACF of X (autocorrelation of driver)
    - ACF of Y (autocorrelation of target)
    - Amplitude distributions of X and Y

    Twin surrogates destroy:
    - CCF between X and Y (cross-correlation)
    - Causal coupling between X and Y
    - Phase relationships between X and Y

    This makes them ideal for testing causal relationships in CCM while
    accounting for autocorrelation artifacts.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    assert x.shape == y.shape, "x and y must have same shape"

    # Generate independent IAAFT surrogates for each time series
    x_surr = generate_iaaft_surrogates(x, n_surr, tol_pc, max_iter,
                                       sorttype, verbose)
    y_surr = generate_iaaft_surrogates(y, n_surr, tol_pc, max_iter,
                                       sorttype, verbose)

    return x_surr, y_surr


def generate_block_bootstrap_surrogates(x: np.ndarray,
                                        y: np.ndarray,
                                        n_surr: int,
                                        block_length: Optional[int] = None,
                                        circular: bool = True,
                                        seed: Optional[int] = None,
                                        verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate block bootstrap surrogates for two time series.

    Creates surrogates by randomly resampling blocks of consecutive time points.
    This preserves short-range correlations (ACF and CCF) up to the block length
    while destroying long-range temporal structure.

    PRESERVES (within block length):
    - ACF of X (autocorrelation of driver)
    - ACF of Y (autocorrelation of target)
    - CCF between X and Y (cross-correlation)
    - Amplitude distributions

    DESTROYS:
    - Long-range temporal order beyond block length
    - Year-to-year predictive structure
    - Causal lag relationships across blocks

    This is ideal for testing causality while preserving local correlations
    and the instantaneous relationship between X and Y.

    Parameters
    ----------
    x, y : np.ndarray, shape (N,)
        Input time series (must have same length)
    n_surr : int
        Number of surrogate pairs to generate
    block_length : int or None, default None
        Length of blocks to resample. If None, uses optimal block length
        based on autocorrelation structure (approximately 3 * ACF decay time)
    circular : bool, default True
        If True, use circular block bootstrap (wrap around at boundaries)
        If False, use non-overlapping blocks
    seed : int or None
        Random seed for reproducibility
    verbose : bool, default False
        Show progress bar

    Returns
    -------
    x_surr : np.ndarray, shape (n_surr, N)
        Block bootstrap surrogates for x
    y_surr : np.ndarray, shape (n_surr, N)
        Block bootstrap surrogates for y

    References
    ----------
    Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary
    observations. The Annals of Statistics, 17(3), 1217-1241.

    Notes
    -----
    The block length should be chosen to:
    - Preserve correlations at relevant timescales
    - Be large enough to capture dependencies
    - Be small enough to provide sufficient randomization

    For monthly data with annual cycles, block_length=12 preserves
    within-year structure while randomizing year order.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    assert x.shape == y.shape, "x and y must have same shape"

    n = x.shape[0]

    # Determine optimal block length if not specified
    if block_length is None:
        # Simple heuristic: find where ACF drops below 0.1, multiply by 3
        from statsmodels.tsa.stattools import acf
        acf_x = acf(x, nlags=min(50, n//4), fft=True)
        acf_decay = np.where(acf_x < 0.1)[0]
        if len(acf_decay) > 0:
            block_length = min(max(3 * acf_decay[0], 5), n // 4)
        else:
            block_length = max(int(np.sqrt(n)), 5)

        if verbose:
            print(f"Optimal block length: {block_length}")

    block_length = int(block_length)

    # Pre-allocate output arrays
    x_surr = np.zeros((n_surr, n), dtype=float)
    y_surr = np.zeros((n_surr, n), dtype=float)

    iterator = tqdm(range(n_surr), desc="Block bootstrap (paired)",
                   disable=not verbose)

    for k in iterator:
        if circular:
            # Circular block bootstrap: sample with wraparound
            x_boot = np.zeros(n)
            y_boot = np.zeros(n)

            pos = 0
            while pos < n:
                # Random starting position (circular)
                start = rng.integers(0, n)

                # Extract block (with wraparound)
                block_end = min(block_length, n - pos)

                if start + block_end <= n:
                    # Block doesn't wrap
                    x_boot[pos:pos + block_end] = x[start:start + block_end]
                    y_boot[pos:pos + block_end] = y[start:start + block_end]
                else:
                    # Block wraps around
                    first_part = n - start
                    x_boot[pos:pos + first_part] = x[start:]
                    y_boot[pos:pos + first_part] = y[start:]

                    if pos + first_part < n:
                        second_part = min(block_end - first_part, n - pos - first_part)
                        x_boot[pos + first_part:pos + first_part + second_part] = x[:second_part]
                        y_boot[pos + first_part:pos + first_part + second_part] = y[:second_part]

                pos += block_end

            x_surr[k] = x_boot
            y_surr[k] = y_boot

        else:
            # Non-overlapping block bootstrap
            n_blocks = n // block_length
            remainder = n % block_length

            # Sample block indices with replacement
            block_indices = rng.choice(n_blocks, size=n_blocks, replace=True)

            x_boot = []
            y_boot = []

            for idx in block_indices:
                start = idx * block_length
                end = start + block_length
                x_boot.append(x[start:end])
                y_boot.append(y[start:end])

            # Handle remainder
            if remainder > 0:
                start = rng.integers(0, n - remainder + 1)
                x_boot.append(x[start:start + remainder])
                y_boot.append(y[start:start + remainder])

            x_surr[k] = np.concatenate(x_boot)
            y_surr[k] = np.concatenate(y_boot)

    return x_surr, y_surr
