"""
Utility functions for time series preprocessing in EDM analysis.
"""

import pandas as pd
import numpy as np
from typing import Union


def add_time_column(df: pd.DataFrame, date_column: str = 'Date') -> pd.DataFrame:
    """
    Add a time step column to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (index should be dates)
    date_column : str, default 'Date'
        Name for the date column

    Returns
    -------
    pd.DataFrame
        Dataframe with time column as first column
    """
    result = df.copy()
    result[date_column] = result.index
    cols = result.columns.tolist()
    result = result[[cols[-1]] + cols[:-1]]
    return result


def reset_to_quarter_start(dt: pd.Timestamp) -> pd.Timestamp:
    """
    Reset datetime to the first month of the quarter/season.

    Parameters
    ----------
    dt : pd.Timestamp
        Input datetime

    Returns
    -------
    pd.Timestamp
        Datetime adjusted to quarter start (Jan, Apr, Jul, Oct)
    """
    month = ((dt.month - 1) // 3) * 3 + 1
    return pd.Timestamp(year=dt.year, month=month, day=1)


def aggregate_to_seasonal(df: pd.DataFrame,
                         agg_func: str = 'mean',
                         date_index: bool = True) -> pd.DataFrame:
    """
    Aggregate time series to seasonal (quarterly) resolution.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with datetime index
    agg_func : str, default 'mean'
        Aggregation function ('mean', 'sum', 'median', etc.)
    date_index : bool, default True
        If True, reset dates to quarter start

    Returns
    -------
    pd.DataFrame
        Seasonally aggregated dataframe
    """
    df_copy = df.copy()

    if date_index:
        df_copy.index = df_copy.index.map(reset_to_quarter_start)

    # Group by quarter and aggregate
    result = df_copy.groupby(df_copy.index).agg(agg_func)

    return result


def aggregate_to_annual(df: pd.DataFrame,
                       agg_func: str = 'mean') -> pd.DataFrame:
    """
    Aggregate time series to annual resolution.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with datetime index
    agg_func : str, default 'mean'
        Aggregation function ('mean', 'sum', 'median', etc.)

    Returns
    -------
    pd.DataFrame
        Annually aggregated dataframe
    """
    df_copy = df.copy()
    df_copy['year'] = df_copy.index.year
    result = df_copy.groupby('year').agg(agg_func)
    return result


def check_for_nans(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Check for NaN values in dataframe columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool, default True
        Print summary of NaN counts

    Returns
    -------
    pd.DataFrame
        Summary of NaN counts per column
    """
    nan_counts = df.isna().sum()
    nan_pct = 100 * nan_counts / len(df)

    summary = pd.DataFrame({
        'NaN_count': nan_counts,
        'NaN_percent': nan_pct
    })

    if verbose:
        print("NaN Summary:")
        print(summary[summary['NaN_count'] > 0])

    return summary


def interpolate_missing(df: pd.DataFrame,
                       method: str = 'linear',
                       limit: Union[int, None] = None,
                       limit_direction: str = 'both') -> pd.DataFrame:
    """
    Interpolate missing values in time series.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    method : str, default 'linear'
        Interpolation method ('linear', 'polynomial', 'spline', etc.)
    limit : int or None
        Maximum number of consecutive NaNs to fill
    limit_direction : str, default 'both'
        Direction to fill ('forward', 'backward', 'both')

    Returns
    -------
    pd.DataFrame
        Dataframe with interpolated values
    """
    return df.interpolate(method=method, limit=limit,
                         limit_direction=limit_direction)


def remove_autocorrelation_lag(x: pd.Series, max_lag: int = 10) -> int:
    """
    Find the lag where autocorrelation drops below significance threshold.

    Useful for determining tau parameter in EDM to avoid autocorrelation artifacts.

    Parameters
    ----------
    x : pd.Series
        Input time series
    max_lag : int, default 10
        Maximum lag to consider

    Returns
    -------
    int
        Recommended lag to avoid autocorrelation (first lag where ACF < 1/e)
    """
    from statsmodels.tsa.stattools import acf

    # Compute autocorrelation
    autocorr = acf(x.dropna(), nlags=max_lag, fft=True)

    # Find first lag where autocorrelation drops below 1/e ≈ 0.368
    threshold = 1.0 / np.e

    for lag in range(1, len(autocorr)):
        if autocorr[lag] < threshold:
            return lag

    # If no drop found, return max_lag
    return max_lag


def split_train_test(df: pd.DataFrame,
                     train_frac: float = 0.8,
                     gap: int = 0) -> tuple:
    """
    Split time series into training and testing sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    train_frac : float, default 0.8
        Fraction of data to use for training
    gap : int, default 0
        Number of time steps to skip between train and test

    Returns
    -------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Testing data
    """
    n = len(df)
    train_end = int(n * train_frac)

    train_df = df.iloc[:train_end]
    test_df = df.iloc[train_end + gap:]

    return train_df, test_df
