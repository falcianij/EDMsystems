import pandas as pd
import numpy as np
from itertools import permutations, combinations
import scipy.stats as sps
from pyunicorn.timeseries import Surrogates
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.optimize import curve_fit

def saturating_curve(L, a, K, b):
    return a * L / (K + L) + b

def fit_ccm_curve(df, ab_col, lib_col='LibSize', plot=True):
    # Drop NaN
    sub = df[[lib_col, ab_col]].dropna()
    x = sub[lib_col].to_numpy(float)
    y = sub[ab_col].to_numpy(float)
    Lmax = float(np.max(x))

    # Initial guesses
    p0 = [max(1e-6, np.nanmax(y) - np.nanmin(y)), float(np.median(x)), float(np.nanmin(y))]

    try:
        popt, pcov = curve_fit(saturating_curve, x, y, p0=p0, maxfev=10000)
        a, K, b = map(float, popt)
        yfit = saturating_curve(x, a, K, b)
        resid = y - yfit
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot
    except Exception as e:
        print(f"Fit failed for {ab_col}: {e}")
        return None

    # Tail-slope (plateau) test on the last third of points
    tail_n = max(3, int(np.ceil(len(x) / 3)))
    x_tail = x[-tail_n:]
    y_tail = y[-tail_n:]
    slope_tail = np.polyfit(x_tail, y_tail, 1)[0] if len(x_tail) >= 2 else np.nan

    # Convergence rho
    rho_conv = float(np.nanmean(y_tail))

    if plot:
        plt.figure(figsize=(5,3))
        plt.scatter(x, y, s=40, alpha=0.6, label='Observed')
        L_grid = np.linspace(np.min(x), np.max(x), 200)
        plt.plot(L_grid, saturating_curve(L_grid, a, K, b), 'r-', lw=2, label='Fit')
        if K < Lmax:  # show K only if it lies within observed range
            plt.axvline(K, color='gray', ls='--', lw=1, label=f"K={K:.1f}")
        plt.xlabel('Library size')
        plt.ylabel(f'ρ({ab_col})')
        plt.title(f'Saturating fit: a={a:.3f}, K={K:.1f}, b={b:.3f}, R²={r2:.2f}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        'a': a, 'K': K, 'b': b, 'R2': r2,
        'Lmax': Lmax, 'slope_tail': slope_tail,
        'rho_conv': rho_conv
        #'plateau_ok': plateau_ok, 'K_ok': K_ok
    }

def has_converged(fit):
    return (
        fit is not None and
        fit['a'] > 0 and
        fit['R2'] > 0.7 and
        fit['rho_conv'] > 0 and
        fit['slope_tail'] < 1e-3 and
        fit['K'] < 0.8 * fit['Lmax']
    )


def empirical_p(obs, surr, tail="greater"):
    surr = np.asarray(surr)
    n = len(surr)
    
    if tail == "greater":
        k = np.sum(surr >= obs)
    elif tail == "less":
        k = np.sum(surr <= obs)
    elif tail == "two-sided":
        # distance from median, two-sided
        med = np.median(surr)
        k = np.sum(np.abs(surr - med) >= np.abs(obs - med))
    else:
        raise ValueError("tail must be 'greater', 'less', or 'two-sided'")
    
    # +1 trick to avoid p=0
    return (k + 1) / (n + 1)


def surrogates_iaaft(x, ns, tol_pc=5., verbose=True, maxiter=1E6, sorttype="quicksort"):
    """
    Returns iAAFT surrogates of given time series.

    Refer to: https://github.com/mlcs/iaaft/blob/fc9c622d15829a5fafe95b48b14b8f3e4bda0655/iaaft.py

    Parameter
    ---------
    x : numpy.ndarray, with shape (N,)
        Input time series for which IAAFT surrogates are to be estimated.
    ns : int
        Number of surrogates to be generated.
    tol_pc : float
        Tolerance (in percent) level which decides the extent to which the
        difference in the power spectrum of the surrogates to the original
        power spectrum is allowed (default = 5).
    verbose : bool
        Show progress bar (default = `True`).
    maxiter : int
        Maximum number of iterations before which the algorithm should
        converge. If the algorithm does not converge until this iteration
        number is reached, the while loop breaks.
    sorttype : string
        Type of sorting algorithm to be used when the amplitudes of the newly
        generated surrogate are to be adjusted to the original data. This
        argument is passed on to `numpy.argsort`. Options include: 'quicksort',
        'mergesort', 'heapsort', 'stable'. See `numpy.argsort` for further
        information. Note that although quick sort can be a bit faster than 
        merge sort or heap sort, it can, depending on the data, have worse case
        spends that are much slower.

    Returns
    -------
    xs : numpy.ndarray, with shape (ns, N)
        Array containing the IAAFT surrogates of `x` such that each row of `xs`
        is an individual surrogate time series.

    See Also
    --------
    numpy.argsort

    """
    # as per the steps given in Lancaster et al., Phys. Rep (2018)
    nx = x.shape[0]
    xs = np.zeros((ns, nx))
    maxiter = 10000
    ii = np.arange(nx)

    # get the fft of the original array
    x_amp = np.abs(np.fft.fft(x))
    x_srt = np.sort(x)
    r_orig = np.argsort(x)

    # loop over surrogate number
    pb_fmt = "{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}"
    pb_desc = "Estimating IAAFT surrogates ..."
    for k in tqdm(range(ns), bar_format=pb_fmt, desc=pb_desc,
                  disable=not verbose):

        # 1) Generate random shuffle of the data
        count = 0
        r_prev = np.random.permutation(ii)
        r_curr = r_orig
        z_n = x[r_prev]
        percent_unequal = 100.

        # core iterative loop
        while (percent_unequal > tol_pc) and (count < maxiter):
            r_prev = r_curr

            # 2) FFT current iteration yk, and then invert it but while
            # replacing the amplitudes with the original amplitudes but
            # keeping the angles from the FFT-ed version of the random
            y_prev = z_n
            fft_prev = np.fft.fft(y_prev)
            phi_prev = np.angle(fft_prev)
            e_i_phi = np.exp(phi_prev * 1j)
            z_n = np.fft.ifft(x_amp * e_i_phi)

            # 3) rescale zk to the original distribution of x
            r_curr = np.argsort(z_n, kind=sorttype)
            z_n[r_curr] = x_srt.copy()
            percent_unequal = ((r_curr != r_prev).sum() * 100.) / nx

            # 4) repeat until number of unequal entries between r_curr and 
            # r_prev is less than tol_pc percent
            count += 1

        if count >= (maxiter - 1):
            print("maximum number of iterations reached!")

        xs[k] = np.real(z_n)

    return xs


def surrogates_random(x, ns, tol_pc=None, verbose=True, maxiter=None, sorttype=None):
    """
    Returns random-shuffled surrogates of a given time series.

    Parameters
    ----------
    x : numpy.ndarray, shape (N,)
        Input time series.
    ns : int
        Number of random surrogates to generate.
    tol_pc, maxiter, sorttype : ignored (present for signature compatibility)
    verbose : bool
        Whether to show a progress bar.

    Returns
    -------
    xs : numpy.ndarray, shape (ns, N)
        Array of random surrogates (each row is a shuffled version of `x`).
    """
    nx = x.shape[0]
    xs = np.zeros((ns, nx))
    pb_fmt = "{desc:<8}{percentage:3.0f}%|{bar:30}{r_bar}"
    pb_desc = "Shuffling"
    for k in tqdm(range(ns), bar_format=pb_fmt, desc=pb_desc, disable=not verbose):
        xs[k] = np.random.permutation(x)
    return xs


def surrogates_seasonal(x, ns, *, period=12, mode="circular", seed=None, exclude_zero_shift=True):
    """
    Generate seasonal/recurrence-aware surrogate time series.

    Parameters
    ----------
    x : array-like, shape (N,)
        Input time series (e.g., monthly data).
    ns : int
        Number of surrogates to generate.
    period : int, default 12
        Recurrence period (e.g., 12 for monthly).
    mode : {"circular", "cycle_perm", "within_phase"}, default "circular"
        - "circular": random circular shift over the full series -> Shifts months down by period e.g., [Jan, Feb, …, Dec] → [Jun, Jul, …, May]
        - "cycle_perm": permute full cycles of length `period`, keeping intra-cycle order -> Permutes years but holds within years e.g., year1, year2, year3 → year3, year1, year2
        - "within_phase": shuffle values within the same phase across cycles (e.g., all Januaries permuted) -> Randomizes months for seasonal shuffling
    seed : int or None
        Random seed for reproducibility.
    exclude_zero_shift : bool, default True
        For "circular", avoid k = 0 (i.e., make sure we actually shift).

    Returns
    -------
    xs : ndarray, shape (ns, N)
        Surrogate series; each row is one surrogate.

    Notes
    -----
    - If len(x) is not a multiple of `period`, the trailing remainder is handled as follows:
      * "circular": unaffected (shift over full length).
      * "cycle_perm": only complete cycles are permuted; the remainder tail stays in place at the end.
      * "within_phase": only complete cycles are phase-shuffled; the remainder tail stays in place.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    N = x.size
    xs = np.zeros((ns, N), dtype=float)

    if mode == "circular":
        for k in range(ns):
            if N == 0:
                xs[k] = x
                continue
            if exclude_zero_shift and N > 1:
                shift = rng.integers(1, N)  # 1..N-1
            else:
                shift = rng.integers(0, N)  # 0..N-1
            xs[k] = np.r_[x[-shift:], x[:-shift]] if shift else x.copy()
        return xs

    # Helpers for cycle-based modes
    n_cycles = N // period
    head = n_cycles * period
    remainder = x[head:]  # possibly empty

    if mode == "cycle_perm":
        if n_cycles <= 1:
            # nothing to permute
            return np.tile(x, (ns, 1))
        core = x[:head].reshape(n_cycles, period)  # rows = cycles (e.g., years)
        for k in range(ns):
            order = rng.permutation(n_cycles)
            xs[k, :head] = core[order, :].reshape(-1)
            xs[k, head:] = remainder
        return xs

    if mode == "within_phase":
        if n_cycles == 0:
            return np.tile(x, (ns, 1))
        core = x[:head].reshape(n_cycles, period)  # rows=cycles, cols=phase (months)
        for k in range(ns):
            y = core.copy()
            # For each phase (column), permute across cycles (rows)
            for j in range(period):
                y[:, j] = y[rng.permutation(n_cycles), j]
            xs[k, :head] = y.reshape(-1)
            xs[k, head:] = remainder
        return xs

    raise ValueError("mode must be one of {'circular','cycle_perm','within_phase'}")


import numpy as np
from tqdm import tqdm

def surrogates_seasonal_pair(x, y, ns, *, period=12, mode="within_phase",
                             seed=None, exclude_zero_shift=True, verbose=True):
    """
    Seasonal surrogates for two paired series, shuffling them together.

    Parameters
    ----------
    x, y : array-like, shape (N,)
        Input time series (e.g., monthly data). Must have the same length.
    ns : int
        Number of surrogates to generate.
    period : int, default 12
        Recurrence period (e.g., 12 for monthly).
    mode : {"within_phase", "circular"}, default "within_phase"
        - "within_phase": for each phase (month), permute years jointly for x,y.
          Preserves monthly climatology & x–y correlation, destroys sequential
          seasonal causal structure across years.
        - "circular": random circular shift over full series, applied to both x,y.
    seed : int or None
        Random seed for reproducibility.
    exclude_zero_shift : bool, default True
        For "circular", avoid k = 0 (ensure an actual shift).
    verbose : bool
        Whether to show a progress bar.

    Returns
    -------
    xs, ys : ndarray, shape (ns, N)
        Surrogate series; each row is one surrogate for x and y, shuffled together.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert x.shape == y.shape, "x and y must have the same shape"

    N = x.size
    xs = np.zeros((ns, N), dtype=float)
    ys = np.zeros((ns, N), dtype=float)

    pb_fmt = "{desc:<14}{percentage:3.0f}%|{bar:30}{r_bar}"
    desc = f"{mode} (paired)"

    # --- circular paired shift ---
    if mode == "circular":
        for k in tqdm(range(ns), bar_format=pb_fmt, desc=desc, disable=not verbose):
            if N == 0:
                xs[k] = x
                ys[k] = y
                continue

            if exclude_zero_shift and N > 1:
                shift = rng.integers(1, N)  # 1..N-1
            else:
                shift = rng.integers(0, N)  # 0..N-1

            if shift:
                xs[k] = np.r_[x[-shift:], x[:-shift]]
                ys[k] = np.r_[y[-shift:], y[:-shift]]
            else:
                xs[k] = x.copy()
                ys[k] = y.copy()
        return xs, ys

    # --- within_phase (paired) ---
    if mode == "within_phase":
        if N == 0:
            return np.tile(x, (ns, 1)), np.tile(y, (ns, 1))

        n_cycles = N // period
        head = n_cycles * period
        remainder_x = x[head:]  # tail, if N not multiple of period
        remainder_y = y[head:]

        if n_cycles == 0:
            # not enough data for even one full cycle
            return np.tile(x, (ns, 1)), np.tile(y, (ns, 1))

        # reshape into [years, months]
        core_x = x[:head].reshape(n_cycles, period)  # rows = years, cols = months
        core_y = y[:head].reshape(n_cycles, period)

        for k in tqdm(range(ns), bar_format=pb_fmt, desc=desc, disable=not verbose):
            Xk = core_x.copy()
            Yk = core_y.copy()
            # For each month (phase), permute years with the SAME permutation for x and y
            for j in range(period):
                perm = rng.permutation(n_cycles)
                Xk[:, j] = Xk[perm, j]
                Yk[:, j] = Yk[perm, j]

            xs[k, :head] = Xk.reshape(-1)
            ys[k, :head] = Yk.reshape(-1)
            xs[k, head:] = remainder_x
            ys[k, head:] = remainder_y

        return xs, ys

    raise ValueError("mode must be one of {'within_phase','circular'}")


def surrogates_random_pair(x, y, ns, verbose=False):
    """
    Randomly shuffle two paired time series together.
    Preserves the pairing (and thus correlation) between x and y,
    but randomizes their temporal order.

    Parameters
    ----------
    x, y : array-like, shape (N,)
        Input time series (must have same length).
    ns : int
        Number of surrogates to generate.
    verbose : bool
        Whether to show a progress bar.

    Returns
    -------
    xs, ys : ndarray, shape (ns, N)
        Surrogates of x and y; each row is a jointly shuffled series.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape == y.shape, "x and y must have same shape"

    nx = x.shape[0]
    xs = np.zeros((ns, nx), dtype=float)
    ys = np.zeros((ns, nx), dtype=float)

    pb_fmt = "{desc:<8}{percentage:3.0f}%|{bar:30}{r_bar}"
    pb_desc = "Shuffling (paired)"
    for k in tqdm(range(ns), bar_format=pb_fmt, desc=pb_desc, disable=not verbose):
        perm = np.random.permutation(nx)
        xs[k] = x[perm]
        ys[k] = y[perm]

    return xs, ys