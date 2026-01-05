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

# ---- Common utility: apply a per-column function with NaN safety ----
# Add time step column (not equivalent to the date)
def add_time(data, datecolumn):
    data[datecolumn] = data.index
    cols = data.columns.tolist()
    data = data[[cols[-1]] + cols[:-1]]
    return data

# Change month in datetime value to be the first month of the quarter/season
def reset_to_quarter_start(dt):
    month = ((dt.month - 1) // 3) * 3 + 1
    return pd.Timestamp(year=dt.year, month=month, day=1)

def _apply_cols(df, fn):
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        x = pd.to_numeric(df[c], errors='coerce')
        out[c] = fn(x)
    return out


# ---- DETREND ----
# 1) Linear detrend (remove best-fit line over time)
def detrend_linear(df):
    t = np.arange(len(df), dtype=float)
    def _lin(y):
        ok = np.isfinite(y)
        if ok.sum() < 3: return y - np.nanmean(y)
        X = np.c_[np.ones(ok.sum()), t[ok]]
        beta = np.linalg.pinv(X) @ y[ok]
        trend = np.full_like(y, np.nan, dtype=float)
        trend[ok] = (X @ beta)
        return y - trend
    return _apply_cols(df, _lin)

# 2) Polynomial detrend (degree ≥ 2 for gentle curvature)
def detrend_poly(df, degree=2):
    t = np.arange(len(df), dtype=float)
    T = np.vstack([t**k for k in range(degree+1)]).T  # [1, t, t^2, ...]
    def _poly(y):
        ok = np.isfinite(y)
        if ok.sum() < degree+2: return y - np.nanmean(y)
        beta = np.linalg.pinv(T[ok]) @ y[ok]
        trend = np.full_like(y, np.nan, dtype=float)
        trend[ok] = T[ok] @ beta
        return y - trend
    return _apply_cols(df, _poly)

# 3) STL trend removal (keeps or removes seasonality)
#    period = 12 for monthly; 4 for quarterly; 52 for weekly, etc.
def detrend_stl(df, period, robust=True, keep_seasonal=True, stl_kwargs=None):
    stl_kwargs = stl_kwargs or {}
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        x = pd.to_numeric(df[c], errors='coerce')
        if np.isfinite(x).sum() < (2*period+5):
            out[c] = x - np.nanmean(x)
            continue
        stl = STL(x, period=period, robust=robust, **stl_kwargs).fit()
        trend = stl.trend
        if keep_seasonal:
            # remove only trend, keep seasonal component
            out[c] = x - trend
        else:
            # remove trend + seasonality (pure residual)
            out[c] = x - trend - stl.seasonal
    return out

# 4) HP filter (smooth trend via Hodrick–Prescott)
#    lambda choice: ~129600 for monthly, 1600 for quarterly, 100 for annual
def detrend_hp(df, lamb=129600):
    def _hp(y):
        if np.isfinite(y).sum() < 10: return y - np.nanmean(y)
        cycle, trend = hpfilter(pd.Series(y).interpolate(limit_direction='both'), lamb=lamb)
        return y - np.asarray(trend)
    return _apply_cols(df, _hp)

# 5) LOESS/LOWESS smoother as trend (nonparametric)
def detrend_loess(df, frac=0.25):
    t = np.arange(len(df), dtype=float)
    def _lo(y):
        ok = np.isfinite(y)
        if ok.sum() < 10: return y - np.nanmean(y)
        z = lowess(y[ok], t[ok], frac=frac, return_sorted=False)
        trend = np.full_like(y, np.nan, dtype=float)
        trend[ok] = z
        return y - trend
    return _apply_cols(df, _lo)

# 6) Rolling-mean detrend (very simple)
def detrend_rolling(df, window=12, center=True, min_periods=None):
    trend = df.rolling(window=window, center=center, min_periods=min_periods).mean()
    return df - trend

# 7) Savitzky–Golay smooth trend (poly fit in a moving window)
def detrend_savgol(df, window_length=13, polyorder=2, mode='interp'):
    def _sg(y):
        y2 = pd.Series(y).interpolate(limit_direction='both').to_numpy()
        if np.count_nonzero(np.isfinite(y2)) < window_length:
            return y - np.nanmean(y)
        tr = savgol_filter(y2, window_length=window_length, polyorder=polyorder, mode=mode)
        return y - tr
    return _apply_cols(df, _sg)

# 8) Seasonal differencing (keeps interannual variability)
def seasonal_diff(df, period=12):
    return df - df.shift(period)


# ---- NORMALIZATION ----
def normalize_series(x: pd.Series, method="zscore", clip_quantiles=None):
    """
    Normalize a 1D series with NaN safety.
    method: 'zscore' (mean/std), 'robust' (median/MAD), 'minmax', 'rank'
    clip_quantiles: e.g., (0.01, 0.99) to winsorize before scaling
    """
    s = pd.to_numeric(x, errors="coerce").astype(float).copy()

    # optional winsorization to tame outliers
    if clip_quantiles is not None:
        lo, hi = np.nanquantile(s, clip_quantiles)
        s = s.clip(lo, hi)

    if method == "zscore":
        mu = np.nanmean(s)
        sd = np.nanstd(s, ddof=1)
        return (s - mu) / sd if sd > 0 else s * 0.0
    elif method == "robust":  # median/MAD (more outlier-resistant)
        med = np.nanmedian(s)
        mad = np.nanmedian(np.abs(s - med))
        scale = 1.4826 * mad  # MAD -> sigma
        return (s - med) / scale if scale > 0 else s * 0.0
    elif method == "minmax":
        mn = np.nanmin(s)
        mx = np.nanmax(s)
        return (s - mn) / (mx - mn) if mx > mn else pd.Series(0.5, index=s.index)
    elif method == "rank":
        # rank to uniform (0,1), then to ~N(0,1) by inverse normal (no ties handling fanciness)
        r = s.rank(method="average", na_option="keep")
        n = float(r.count())
        u = (r - 0.5) / n
        u = u.clip(1e-6, 1 - 1e-6)
        from scipy.special import erfinv
        return np.sqrt(2.0) * erfinv(2.0 * u - 1.0)
    elif method == "sqrt_minmax":
        mn, mx = np.nanmin(s), np.nanmax(s)
        if mx > mn:
            v = (s - mn) / (mx - mn)
            v = np.clip(v, 0.0, None)
            return np.sqrt(v)
    else:
        raise ValueError("Unknown method")

def normalize_df(df: pd.DataFrame, method="zscore", exclude=("time","Date","temp_PC1"),
                 log1p_cols=None, clip_quantiles=None):
    """
    Apply normalization column-wise, skipping id/time columns.
    Optionally apply log1p to selected columns before scaling.
    """
    out = df.copy()
    cols = [c for c in out.columns if c not in exclude]
    if log1p_cols:
        for c in cols:
            if (log1p_cols == "all") or (c in log1p_cols):
                out[c] = np.log1p(out[c].astype(float))
    for c in cols:
        out[c] = normalize_series(out[c], method=method, clip_quantiles=clip_quantiles)
    return out