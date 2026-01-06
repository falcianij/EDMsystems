"""
Time series aggregation and alignment utilities for EDM.

This module provides functions for:
- Temporal aggregation (daily, weekly, monthly, etc.)
- Column aggregation (mean, sum, etc.)
- Missing data handling
- Time series alignment
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Literal, Callable


def aggregate_temporal(df: pd.DataFrame,
                       datetime_col: str = 'datetime',
                       freq: str = 'D',
                       agg_func: Union[str, Callable, dict] = 'mean',
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Aggregate time series to a specified temporal frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with datetime column
    datetime_col : str, default 'datetime'
        Name of datetime column
    freq : str, default 'D'
        Pandas frequency string:
        - 'D': daily
        - 'W': weekly
        - 'M': monthly (month end)
        - 'MS': monthly (month start)
        - 'Q': quarterly
        - 'Y' or 'A': yearly
        - 'H': hourly
        - '15min', '30min', etc.
    agg_func : str, callable, or dict, default 'mean'
        Aggregation function(s):
        - 'mean', 'sum', 'min', 'max', 'median', 'std', 'count'
        - Custom function: lambda x: x.mean()
        - Dict mapping columns to functions: {'col1': 'mean', 'col2': 'sum'}
    columns : list of str or None
        Columns to aggregate. If None, aggregates all numeric columns.

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe with datetime index

    Examples
    --------
    >>> # Aggregate hourly data to daily means
    >>> df_daily = aggregate_temporal(df, freq='D', agg_func='mean')

    >>> # Aggregate to monthly with different functions per column
    >>> df_monthly = aggregate_temporal(
    ...     df,
    ...     freq='MS',
    ...     agg_func={'temperature': 'mean', 'precipitation': 'sum'}
    ... )
    """
    # Make copy to avoid modifying original
    df = df.copy()

    # Ensure datetime column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Set datetime as index for resampling
    df = df.set_index(datetime_col)

    # Determine columns to aggregate
    if columns is None:
        # Get all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Select only the columns to aggregate
    df_to_agg = df[columns]

    # Perform resampling
    df_agg = df_to_agg.resample(freq).agg(agg_func)

    # Reset index to make datetime a column again
    df_agg = df_agg.reset_index()

    return df_agg


def fill_missing_values(df: pd.DataFrame,
                        method: Literal['ffill', 'bfill', 'interpolate', 'drop'] = 'interpolate',
                        columns: Optional[List[str]] = None,
                        limit: Optional[int] = None,
                        **kwargs) -> pd.DataFrame:
    """
    Fill or remove missing values in time series.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    method : {'ffill', 'bfill', 'interpolate', 'drop'}
        Method for handling missing values:
        - 'ffill': Forward fill (propagate last valid observation)
        - 'bfill': Backward fill  (propagate next valid observation)
        - 'interpolate': Linear interpolation
        - 'drop': Drop rows with any NaN
    columns : list of str or None
        Columns to apply method to. If None, applies to all columns.
    limit : int or None
        Maximum number of consecutive NaNs to fill. If None, fill all.
    **kwargs
        Additional arguments passed to pandas interpolate method

    Returns
    -------
    pd.DataFrame
        Dataframe with missing values handled

    Examples
    --------
    >>> # Linear interpolation (default)
    >>> df_filled = fill_missing_values(df)

    >>> # Forward fill with limit
    >>> df_filled = fill_missing_values(df, method='ffill', limit=3)

    >>> # Drop any rows with NaN
    >>> df_clean = fill_missing_values(df, method='drop')
    """
    df = df.copy()

    if columns is None:
        columns = df.columns.tolist()

    if method == 'ffill':
        df[columns] = df[columns].fillna(method='ffill', limit=limit)
    elif method == 'bfill':
        df[columns] = df[columns].fillna(method='bfill', limit=limit)
    elif method == 'interpolate':
        df[columns] = df[columns].interpolate(limit=limit, **kwargs)
    elif method == 'drop':
        df = df.dropna(subset=columns)
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'ffill', 'bfill', 'interpolate', or 'drop'")

    return df


def align_time_series(df1: pd.DataFrame,
                     df2: pd.DataFrame,
                     datetime_col: str = 'datetime',
                     how: Literal['inner', 'outer', 'left', 'right'] = 'inner') -> tuple:
    """
    Align two time series by their datetime columns.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        Input dataframes with datetime columns
    datetime_col : str, default 'datetime'
        Name of datetime column
    how : {'inner', 'outer', 'left', 'right'}, default 'inner'
        Type of merge:
        - 'inner': Only keep dates present in both
        - 'outer': Keep all dates from both (fill with NaN)
        - 'left': Keep all dates from df1
        - 'right': Keep all dates from df2

    Returns
    -------
    df1_aligned, df2_aligned : pd.DataFrame
        Aligned dataframes with matching datetime indices

    Examples
    --------
    >>> df1_aligned, df2_aligned = align_time_series(temp_df, precip_df, how='inner')
    """
    # Ensure datetime columns are datetime type
    df1 = df1.copy()
    df2 = df2.copy()

    if not pd.api.types.is_datetime64_any_dtype(df1[datetime_col]):
        df1[datetime_col] = pd.to_datetime(df1[datetime_col])
    if not pd.api.types.is_datetime64_any_dtype(df2[datetime_col]):
        df2[datetime_col] = pd.to_datetime(df2[datetime_col])

    # Merge on datetime column
    merged = pd.merge(df1, df2, on=datetime_col, how=how, suffixes=('_1', '_2'))

    # Split back into two dataframes
    cols1 = [datetime_col] + [c for c in merged.columns if c.endswith('_1')]
    cols2 = [datetime_col] + [c for c in merged.columns if c.endswith('_2')]

    # Get original column names (remove suffixes)
    df1_aligned = merged[cols1].copy()
    df2_aligned = merged[cols2].copy()

    df1_aligned.columns = [c.replace('_1', '') for c in df1_aligned.columns]
    df2_aligned.columns = [c.replace('_2', '') for c in df2_aligned.columns]

    return df1_aligned, df2_aligned


def create_lagged_features(df: pd.DataFrame,
                          columns: Union[str, List[str]],
                          lags: Union[int, List[int]],
                          drop_na: bool = True) -> pd.DataFrame:
    """
    Create lagged versions of specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : str or list of str
        Column(s) to create lags for
    lags : int or list of int
        Lag values (positive = lag, negative = lead)
    drop_na : bool, default True
        Whether to drop rows with NaN from lagging

    Returns
    -------
    pd.DataFrame
        Dataframe with original and lagged columns

    Examples
    --------
    >>> # Create lags 1-3 for temperature
    >>> df_lagged = create_lagged_features(df, 'temperature', lags=[1, 2, 3])
    """
    df = df.copy()

    if isinstance(columns, str):
        columns = [columns]
    if isinstance(lags, int):
        lags = [lags]

    for col in columns:
        for lag in lags:
            lag_col_name = f"{col}_lag{lag}"
            df[lag_col_name] = df[col].shift(lag)

    if drop_na:
        df = df.dropna()

    return df
