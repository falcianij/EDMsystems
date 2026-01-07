"""
Time series transformations with inverse capability.

Provides simple, lean transformation functions that store parameters
for inverse transformation - critical for interpreting predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class TransformParams:
    """Store transformation parameters for inverse operations."""

    def __init__(self, method: str, params: Dict):
        self.method = method
        self.params = params


def normalize_with_params(data: np.ndarray,
                         method: str = 'zscore') -> Tuple[np.ndarray, Dict]:
    """
    Normalize data and return parameters for inverse transform.

    Parameters
    ----------
    data : np.ndarray
        Input data
    method : str, default 'zscore'
        Normalization method: 'zscore', 'minmax', 'robust'

    Returns
    -------
    normalized : np.ndarray
        Normalized data
    params : dict
        Parameters for inverse transform
    """
    data = np.asarray(data).flatten()

    if method == 'zscore':
        mean = np.nanmean(data)
        std = np.nanstd(data, ddof=1)
        if std == 0:
            std = 1.0
        normalized = (data - mean) / std
        params = {'mean': mean, 'std': std, 'method': method}

    elif method == 'minmax':
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        if max_val == min_val:
            normalized = np.zeros_like(data)
        else:
            normalized = (data - min_val) / (max_val - min_val)
        params = {'min': min_val, 'max': max_val, 'method': method}

    elif method == 'robust':
        median = np.nanmedian(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        if iqr == 0:
            iqr = 1.0
        normalized = (data - median) / iqr
        params = {'median': median, 'iqr': iqr, 'method': method}

    else:
        raise ValueError(f"Unknown method: {method}")

    return normalized, params


def inverse_normalize(data: np.ndarray, params: Dict) -> np.ndarray:
    """
    Inverse normalize using stored parameters.

    Parameters
    ----------
    data : np.ndarray
        Normalized data
    params : dict
        Parameters from normalize_with_params

    Returns
    -------
    np.ndarray
        Data in original scale
    """
    data = np.asarray(data).flatten()
    method = params['method']

    if method == 'zscore':
        return data * params['std'] + params['mean']
    elif method == 'minmax':
        return data * (params['max'] - params['min']) + params['min']
    elif method == 'robust':
        return data * params['iqr'] + params['median']
    else:
        raise ValueError(f"Unknown method: {method}")


def detrend_with_params(data: np.ndarray,
                       method: str = 'linear') -> Tuple[np.ndarray, Dict]:
    """
    Detrend data and return parameters for inverse transform.

    Parameters
    ----------
    data : np.ndarray
        Input data
    method : str, default 'linear'
        Detrending method: 'linear', 'polynomial'

    Returns
    -------
    detrended : np.ndarray
        Detrended data
    params : dict
        Parameters for inverse transform
    """
    data = np.asarray(data).flatten()
    t = np.arange(len(data))

    if method == 'linear':
        # Fit linear trend
        valid = ~np.isnan(data)
        if np.sum(valid) < 2:
            return data - np.nanmean(data), {'method': method, 'slope': 0, 'intercept': np.nanmean(data)}

        coeffs = np.polyfit(t[valid], data[valid], 1)
        slope, intercept = coeffs
        trend = slope * t + intercept
        detrended = data - trend
        params = {'method': method, 'slope': slope, 'intercept': intercept}

    elif method == 'polynomial':
        degree = 2
        valid = ~np.isnan(data)
        if np.sum(valid) < degree + 1:
            return data - np.nanmean(data), {'method': method, 'coeffs': [0, 0, np.nanmean(data)]}

        coeffs = np.polyfit(t[valid], data[valid], degree)
        trend = np.polyval(coeffs, t)
        detrended = data - trend
        params = {'method': method, 'coeffs': coeffs}

    else:
        raise ValueError(f"Unknown method: {method}")

    return detrended, params


def inverse_detrend(data: np.ndarray, params: Dict) -> np.ndarray:
    """
    Add trend back using stored parameters.

    Parameters
    ----------
    data : np.ndarray
        Detrended data
    params : dict
        Parameters from detrend_with_params

    Returns
    -------
    np.ndarray
        Data with trend restored
    """
    data = np.asarray(data).flatten()
    t = np.arange(len(data))
    method = params['method']

    if method == 'linear':
        trend = params['slope'] * t + params['intercept']
        return data + trend
    elif method == 'polynomial':
        trend = np.polyval(params['coeffs'], t)
        return data + trend
    else:
        raise ValueError(f"Unknown method: {method}")


def preprocess_for_ccm(data: np.ndarray,
                      detrend_method: Optional[str] = 'linear',
                      normalize_method: Optional[str] = 'zscore') -> Tuple[np.ndarray, Dict]:
    """
    Standard preprocessing pipeline for CCM (detrend + normalize).

    Parameters
    ----------
    data : np.ndarray
        Input time series
    detrend_method : str or None
        Detrending method or None to skip
    normalize_method : str or None
        Normalization method or None to skip

    Returns
    -------
    processed : np.ndarray
        Processed data
    transform_params : dict
        All transformation parameters for inverse
    """
    transform_params = {'steps': []}
    result = data.copy()

    # Detrend first
    if detrend_method is not None:
        result, detrend_params = detrend_with_params(result, method=detrend_method)
        transform_params['steps'].append(('detrend', detrend_params))

    # Then normalize
    if normalize_method is not None:
        result, norm_params = normalize_with_params(result, method=normalize_method)
        transform_params['steps'].append(('normalize', norm_params))

    return result, transform_params


def inverse_preprocess(data: np.ndarray, transform_params: Dict) -> np.ndarray:
    """
    Inverse preprocessing (apply in reverse order).

    Parameters
    ----------
    data : np.ndarray
        Processed data
    transform_params : dict
        Parameters from preprocess_for_ccm

    Returns
    -------
    np.ndarray
        Data in original scale
    """
    result = data.copy()

    # Apply inverse transforms in reverse order
    for step_type, params in reversed(transform_params['steps']):
        if step_type == 'normalize':
            result = inverse_normalize(result, params)
        elif step_type == 'detrend':
            result = inverse_detrend(result, params)

    return result
