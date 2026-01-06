"""
Detrending methods for EDM time series preprocessing.

This module provides multiple detrending strategies to remove trends
while preserving the dynamics relevant for EDM analysis.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.filters.hp_filter import hpfilter
from typing import Union, Optional, Dict, Any


def _apply_to_columns(df: pd.DataFrame, func) -> pd.DataFrame:
    """
    Apply a function to each column of a DataFrame with NaN safety.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    func : callable
        Function to apply to each column (takes array, returns array)

    Returns
    -------
    pd.DataFrame
        Result of applying func to each column
    """
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        x = pd.to_numeric(df[col], errors='coerce')
        out[col] = func(x)
    return out


def detrend_linear(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linear trend from time series.

    Fits a line y = a + b*t and returns residuals.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series data

    Returns
    -------
    pd.DataFrame
        Detrended data (residuals from linear fit)
    """
    t = np.arange(len(df), dtype=float)

    def _lin(y):
        ok = np.isfinite(y)
        if ok.sum() < 3:
            return y - np.nanmean(y)

        X = np.c_[np.ones(ok.sum()), t[ok]]
        beta = np.linalg.pinv(X) @ y[ok]
        trend = np.full_like(y, np.nan, dtype=float)
        trend[ok] = (X @ beta)
        return y - trend

    return _apply_to_columns(df, _lin)


def detrend_polynomial(df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
    """
    Remove polynomial trend from time series.

    Fits a polynomial of specified degree and returns residuals.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series data
    degree : int, default 2
        Polynomial degree (2 for quadratic, 3 for cubic, etc.)

    Returns
    -------
    pd.DataFrame
        Detrended data (residuals from polynomial fit)
    """
    t = np.arange(len(df), dtype=float)
    T = np.vstack([t**k for k in range(degree + 1)]).T  # [1, t, t^2, ...]

    def _poly(y):
        ok = np.isfinite(y)
        if ok.sum() < degree + 2:
            return y - np.nanmean(y)

        beta = np.linalg.pinv(T[ok]) @ y[ok]
        trend = np.full_like(y, np.nan, dtype=float)
        trend[ok] = T[ok] @ beta
        return y - trend

    return _apply_to_columns(df, _poly)


def detrend_stl(df: pd.DataFrame,
                period: int,
                robust: bool = True,
                keep_seasonal: bool = True,
                seasonal: Optional[int] = None,
                trend: Optional[int] = None,
                **stl_kwargs) -> pd.DataFrame:
    """
    STL (Seasonal-Trend decomposition using LOESS) detrending.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series data
    period : int
        Seasonal period (12 for monthly, 4 for quarterly, 52 for weekly, etc.)
    robust : bool, default True
        Use robust fitting (resistant to outliers)
    keep_seasonal : bool, default True
        If True, remove only trend (keep seasonal component)
        If False, remove both trend and seasonal (return pure residuals)
    seasonal : int or None
        Length of seasonal smoother (odd integer). If None, auto-selected.
    trend : int or None
        Length of trend smoother (odd integer). If None, auto-selected.
    **stl_kwargs
        Additional arguments passed to statsmodels.tsa.seasonal.STL

    Returns
    -------
    pd.DataFrame
        Detrended data
    """
    out = pd.DataFrame(index=df.index)

    for col in df.columns:
        x = pd.to_numeric(df[col], errors='coerce')

        if np.isfinite(x).sum() < (2 * period + 5):
            out[col] = x - np.nanmean(x)
            continue

        # Prepare STL arguments
        stl_args = {'period': period, 'robust': robust}
        if seasonal is not None:
            stl_args['seasonal'] = seasonal
        if trend is not None:
            stl_args['trend'] = trend
        stl_args.update(stl_kwargs)

        stl = STL(x, **stl_args).fit()

        if keep_seasonal:
            # Remove only trend, keep seasonal component
            out[col] = x - stl.trend
        else:
            # Remove trend + seasonality (pure residual)
            out[col] = x - stl.trend - stl.seasonal

    return out


def detrend_hp(df: pd.DataFrame, lamb: float = 129600) -> pd.DataFrame:
    """
    Hodrick-Prescott filter detrending.

    Separates series into trend and cycle components via penalized smoothing.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series data
    lamb : float, default 129600
        Smoothing parameter. Common values:
        - 129600 for monthly data
        - 1600 for quarterly data
        - 100 for annual data

    Returns
    -------
    pd.DataFrame
        Detrended data (cycle component)
    """
    def _hp(y):
        if np.isfinite(y).sum() < 10:
            return y - np.nanmean(y)

        # Interpolate missing values for HP filter
        y_series = pd.Series(y).interpolate(limit_direction='both')
        cycle, trend = hpfilter(y_series, lamb=lamb)
        return y - np.asarray(trend)

    return _apply_to_columns(df, _hp)


def detrend_loess(df: pd.DataFrame, frac: float = 0.25) -> pd.DataFrame:
    """
    LOESS/LOWESS (Locally Weighted Scatterplot Smoothing) detrending.

    Non-parametric trend removal using local polynomial regression.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series data
    frac : float, default 0.25
        Fraction of data to use in local regression (0 < frac <= 1)

    Returns
    -------
    pd.DataFrame
        Detrended data (residuals from LOESS fit)
    """
    t = np.arange(len(df), dtype=float)

    def _lo(y):
        ok = np.isfinite(y)
        if ok.sum() < 10:
            return y - np.nanmean(y)

        z = lowess(y[ok], t[ok], frac=frac, return_sorted=False)
        trend = np.full_like(y, np.nan, dtype=float)
        trend[ok] = z
        return y - trend

    return _apply_to_columns(df, _lo)


def detrend_rolling_mean(df: pd.DataFrame,
                         window: int = 12,
                         center: bool = True,
                         min_periods: Optional[int] = None) -> pd.DataFrame:
    """
    Rolling mean detrending.

    Simple moving average trend removal.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series data
    window : int, default 12
        Size of moving window
    center : bool, default True
        Whether to center the window
    min_periods : int or None
        Minimum number of observations required (default = window)

    Returns
    -------
    pd.DataFrame
        Detrended data (residuals from rolling mean)
    """
    trend = df.rolling(window=window, center=center, min_periods=min_periods).mean()
    return df - trend


def detrend_savgol(df: pd.DataFrame,
                   window_length: int = 13,
                   polyorder: int = 2,
                   mode: str = 'interp') -> pd.DataFrame:
    """
    Savitzky-Golay filter detrending.

    Polynomial fit in a moving window for smooth trend estimation.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series data
    window_length : int, default 13
        Length of filter window (must be odd)
    polyorder : int, default 2
        Order of polynomial to fit
    mode : str, default 'interp'
        How to handle edges ('interp', 'mirror', 'nearest', 'wrap')

    Returns
    -------
    pd.DataFrame
        Detrended data (residuals from Savitzky-Golay fit)
    """
    def _sg(y):
        # Interpolate NaNs for filter
        y2 = pd.Series(y).interpolate(limit_direction='both').to_numpy()

        if np.count_nonzero(np.isfinite(y2)) < window_length:
            return y - np.nanmean(y)

        trend = savgol_filter(y2, window_length=window_length,
                            polyorder=polyorder, mode=mode)
        return y - trend

    return _apply_to_columns(df, _sg)


def detrend_seasonal_diff(df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
    """
    Seasonal differencing.

    Removes repeating seasonal cycles by subtracting lagged values.
    Preserves interannual variability and regime shifts.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series data
    period : int, default 12
        Seasonal period to difference (12 for monthly, 4 for quarterly)

    Returns
    -------
    pd.DataFrame
        Seasonally differenced data
    """
    return df - df.shift(period)


# Factory function for easy access
def get_detrending_method(method: str, **kwargs):
    """
    Get a detrending function by name.

    Parameters
    ----------
    method : str
        Detrending method name: 'linear', 'polynomial', 'stl', 'hp',
        'loess', 'rolling_mean', 'savgol', 'seasonal_diff'
    **kwargs
        Additional arguments for the specific detrending method

    Returns
    -------
    callable
        Detrending function that takes a DataFrame and returns detrended DataFrame
    """
    methods = {
        'linear': detrend_linear,
        'polynomial': detrend_polynomial,
        'stl': detrend_stl,
        'hp': detrend_hp,
        'loess': detrend_loess,
        'rolling_mean': detrend_rolling_mean,
        'savgol': detrend_savgol,
        'seasonal_diff': detrend_seasonal_diff,
    }

    if method not in methods:
        raise ValueError(f"Unknown detrending method: {method}. "
                        f"Available: {list(methods.keys())}")

    func = methods[method]

    # Return a partial function with kwargs if provided
    if kwargs:
        return lambda df: func(df, **kwargs)
    return func


def detrend_dataframe(df: pd.DataFrame, method: str, **kwargs) -> pd.DataFrame:
    """
    Convenience function to detrend a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series data
    method : str
        Detrending method name
    **kwargs
        Additional arguments for the specific detrending method

    Returns
    -------
    pd.DataFrame
        Detrended data
    """
    detrend_func = get_detrending_method(method, **kwargs)
    return detrend_func(df)
