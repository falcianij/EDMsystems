"""
Time series transformation and normalization utilities for EDM.

This module provides functions for:
- Normalization (z-score, min-max, robust scaling)
- Detrending (linear, polynomial, differencing)
- Transformation storage and inversion
- Batch transformation of multiple series

All transformations store parameters to enable inverse transformation,
which is critical for interpreting predictions in the original scale.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Literal, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class TransformParams:
    """
    Container for transformation parameters to enable inverse transformation.

    Attributes
    ----------
    method : str
        Transformation method name
    params : dict
        Method-specific parameters for inverse transformation
    """
    method: str
    params: Dict = field(default_factory=dict)


class SeriesTransformer:
    """
    Transform time series with automatic parameter storage for inversion.

    This class handles normalization and detrending of time series while
    storing all parameters needed to inverse transform predictions back
    to the original scale.

    Examples
    --------
    >>> # Basic usage
    >>> transformer = SeriesTransformer()
    >>> X_normalized = transformer.fit_transform(X, method='zscore')
    >>> X_original = transformer.inverse_transform(X_normalized)

    >>> # Chain multiple transformations
    >>> transformer = SeriesTransformer()
    >>> X_detrended = transformer.fit_transform(X, method='linear_detrend')
    >>> X_normalized = transformer.transform(X_detrended, method='zscore')
    >>> # Inverse in reverse order
    >>> X_back = transformer.inverse_transform(X_normalized, method='zscore')
    >>> X_original = transformer.inverse_transform(X_back, method='linear_detrend')

    >>> # Or use chained transformations
    >>> X_transformed = transformer.fit_transform_chain(X, ['linear_detrend', 'zscore'])
    >>> X_original = transformer.inverse_transform_chain(X_transformed)
    """

    def __init__(self):
        self.transforms: Dict[str, TransformParams] = {}
        self.transform_chain: List[str] = []

    def fit_transform(self,
                     data: np.ndarray,
                     method: str = 'zscore',
                     **kwargs) -> np.ndarray:
        """
        Fit transformation parameters and transform data.

        Parameters
        ----------
        data : np.ndarray
            Input time series (1D array)
        method : str
            Transformation method:
            - 'zscore': Z-score normalization (mean=0, std=1)
            - 'minmax': Min-max normalization (range [0, 1])
            - 'robust': Robust scaling using median and IQR
            - 'linear_detrend': Remove linear trend
            - 'polynomial_detrend': Remove polynomial trend
            - 'difference': First-order differencing
            - 'log': Log transformation
        **kwargs
            Method-specific parameters

        Returns
        -------
        np.ndarray
            Transformed data
        """
        data = np.asarray(data).flatten()

        if method == 'zscore':
            return self._fit_zscore(data)
        elif method == 'minmax':
            return self._fit_minmax(data, **kwargs)
        elif method == 'robust':
            return self._fit_robust(data)
        elif method == 'linear_detrend':
            return self._fit_linear_detrend(data)
        elif method == 'polynomial_detrend':
            return self._fit_polynomial_detrend(data, **kwargs)
        elif method == 'difference':
            return self._fit_difference(data, **kwargs)
        elif method == 'log':
            return self._fit_log(data, **kwargs)
        else:
            raise ValueError(f"Unknown transformation method: {method}")

    def transform(self,
                 data: np.ndarray,
                 method: Optional[str] = None) -> np.ndarray:
        """
        Transform data using previously fitted parameters.

        Parameters
        ----------
        data : np.ndarray
            Input data to transform
        method : str or None
            Method to use. If None, uses last fitted method.

        Returns
        -------
        np.ndarray
            Transformed data
        """
        if method is None:
            if not self.transforms:
                raise ValueError("No transformation has been fitted yet")
            method = list(self.transforms.keys())[-1]

        if method not in self.transforms:
            raise ValueError(f"Method '{method}' has not been fitted")

        data = np.asarray(data).flatten()
        params = self.transforms[method].params

        if method == 'zscore':
            return (data - params['mean']) / params['std']
        elif method == 'minmax':
            return (data - params['min']) / (params['max'] - params['min'])
        elif method == 'robust':
            return (data - params['median']) / params['iqr']
        elif method == 'linear_detrend':
            t = np.arange(len(data))
            trend = params['slope'] * t + params['intercept']
            return data - trend
        elif method == 'polynomial_detrend':
            t = np.arange(len(data))
            trend = np.polyval(params['coefficients'], t)
            return data - trend
        elif method == 'difference':
            return np.diff(data, n=params['order'])
        elif method == 'log':
            return np.log(data + params['offset'])
        else:
            raise ValueError(f"Unknown method: {method}")

    def inverse_transform(self,
                         data: np.ndarray,
                         method: Optional[str] = None) -> np.ndarray:
        """
        Inverse transform data back to original scale.

        Parameters
        ----------
        data : np.ndarray
            Transformed data
        method : str or None
            Method to invert. If None, uses last fitted method.

        Returns
        -------
        np.ndarray
            Data in original scale
        """
        if method is None:
            if not self.transforms:
                raise ValueError("No transformation has been fitted yet")
            method = list(self.transforms.keys())[-1]

        if method not in self.transforms:
            raise ValueError(f"Method '{method}' has not been fitted")

        data = np.asarray(data).flatten()
        params = self.transforms[method].params

        if method == 'zscore':
            return data * params['std'] + params['mean']
        elif method == 'minmax':
            return data * (params['max'] - params['min']) + params['min']
        elif method == 'robust':
            return data * params['iqr'] + params['median']
        elif method == 'linear_detrend':
            t = np.arange(len(data))
            trend = params['slope'] * t + params['intercept']
            return data + trend
        elif method == 'polynomial_detrend':
            t = np.arange(len(data))
            trend = np.polyval(params['coefficients'], t)
            return data + trend
        elif method == 'difference':
            # Note: Inverse of differencing requires initial value(s)
            if 'initial_values' not in params:
                raise ValueError("Cannot inverse difference without initial values")
            return self._inverse_difference(data, params['order'], params['initial_values'])
        elif method == 'log':
            return np.exp(data) - params['offset']
        else:
            raise ValueError(f"Unknown method: {method}")

    def fit_transform_chain(self,
                           data: np.ndarray,
                           methods: List[str],
                           **kwargs) -> np.ndarray:
        """
        Apply multiple transformations in sequence.

        Parameters
        ----------
        data : np.ndarray
            Input data
        methods : list of str
            Transformation methods to apply in order
        **kwargs
            Method-specific parameters (use method name as key)

        Returns
        -------
        np.ndarray
            Transformed data

        Examples
        --------
        >>> X_trans = transformer.fit_transform_chain(
        ...     X,
        ...     ['linear_detrend', 'zscore']
        ... )
        """
        self.transform_chain = methods
        result = data.copy()

        for method in methods:
            method_kwargs = kwargs.get(method, {})
            result = self.fit_transform(result, method=method, **method_kwargs)

        return result

    def inverse_transform_chain(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform using stored transformation chain (in reverse).

        Parameters
        ----------
        data : np.ndarray
            Transformed data

        Returns
        -------
        np.ndarray
            Data in original scale
        """
        if not self.transform_chain:
            raise ValueError("No transformation chain has been fitted")

        result = data.copy()

        # Apply inverse transformations in reverse order
        for method in reversed(self.transform_chain):
            result = self.inverse_transform(result, method=method)

        return result

    # Private methods for each transformation type

    def _fit_zscore(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalization: (X - mean) / std"""
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        if std == 0:
            std = 1.0  # Avoid division by zero

        self.transforms['zscore'] = TransformParams(
            method='zscore',
            params={'mean': mean, 'std': std}
        )

        return (data - mean) / std

    def _fit_minmax(self, data: np.ndarray,
                   feature_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """Min-max normalization to specified range"""
        min_val = np.min(data)
        max_val = np.max(data)

        if max_val == min_val:
            # Constant series
            normalized = np.full_like(data, feature_range[0])
        else:
            # Scale to [0, 1] first, then to feature_range
            normalized = (data - min_val) / (max_val - min_val)
            normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]

        self.transforms['minmax'] = TransformParams(
            method='minmax',
            params={
                'min': min_val,
                'max': max_val,
                'feature_range': feature_range
            }
        )

        return normalized

    def _fit_robust(self, data: np.ndarray) -> np.ndarray:
        """Robust scaling using median and IQR"""
        median = np.median(data)
        q25 = np.percentile(data, 25)
        q75 = np.percentile(data, 75)
        iqr = q75 - q25

        if iqr == 0:
            iqr = 1.0  # Avoid division by zero

        self.transforms['robust'] = TransformParams(
            method='robust',
            params={'median': median, 'iqr': iqr}
        )

        return (data - median) / iqr

    def _fit_linear_detrend(self, data: np.ndarray) -> np.ndarray:
        """Remove linear trend"""
        t = np.arange(len(data))
        coeffs = np.polyfit(t, data, 1)
        slope, intercept = coeffs

        trend = slope * t + intercept
        detrended = data - trend

        self.transforms['linear_detrend'] = TransformParams(
            method='linear_detrend',
            params={'slope': slope, 'intercept': intercept}
        )

        return detrended

    def _fit_polynomial_detrend(self, data: np.ndarray, degree: int = 2) -> np.ndarray:
        """Remove polynomial trend"""
        t = np.arange(len(data))
        coeffs = np.polyfit(t, data, degree)

        trend = np.polyval(coeffs, t)
        detrended = data - trend

        self.transforms['polynomial_detrend'] = TransformParams(
            method='polynomial_detrend',
            params={'coefficients': coeffs, 'degree': degree}
        )

        return detrended

    def _fit_difference(self, data: np.ndarray, order: int = 1) -> np.ndarray:
        """Apply differencing"""
        # Store initial values for inverse transformation
        initial_values = [data[i] for i in range(order)]

        differenced = np.diff(data, n=order)

        self.transforms['difference'] = TransformParams(
            method='difference',
            params={'order': order, 'initial_values': initial_values}
        )

        return differenced

    def _fit_log(self, data: np.ndarray, offset: float = 0.0) -> np.ndarray:
        """Log transformation with optional offset"""
        # Auto-determine offset if data contains non-positive values
        if offset == 0.0 and np.any(data <= 0):
            offset = abs(np.min(data)) + 1.0

        transformed = np.log(data + offset)

        self.transforms['log'] = TransformParams(
            method='log',
            params={'offset': offset}
        )

        return transformed

    def _inverse_difference(self,
                           data: np.ndarray,
                           order: int,
                           initial_values: List[float]) -> np.ndarray:
        """Inverse differencing (cumulative sum with initial values)"""
        result = data.copy()

        for i in range(order):
            result = np.cumsum(np.concatenate([[initial_values[order - 1 - i]], result]))

        return result


# Convenience functions for quick transformations

def normalize(data: np.ndarray,
             method: Literal['zscore', 'minmax', 'robust'] = 'zscore',
             return_params: bool = False,
             **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Normalize time series data.

    Parameters
    ----------
    data : np.ndarray
        Input time series
    method : {'zscore', 'minmax', 'robust'}
        Normalization method
    return_params : bool, default False
        If True, return (normalized_data, parameters) for inverse transform
    **kwargs
        Method-specific parameters

    Returns
    -------
    normalized : np.ndarray
        Normalized data
    params : dict (if return_params=True)
        Parameters for inverse transformation

    Examples
    --------
    >>> X_norm = normalize(X, method='zscore')
    >>> X_norm, params = normalize(X, method='zscore', return_params=True)
    >>> X_original = denormalize(X_norm, params)
    """
    transformer = SeriesTransformer()
    normalized = transformer.fit_transform(data, method=method, **kwargs)

    if return_params:
        return normalized, transformer.transforms[method].params
    return normalized


def denormalize(data: np.ndarray,
               params: dict,
               method: Optional[str] = None) -> np.ndarray:
    """
    Inverse normalize using stored parameters.

    Parameters
    ----------
    data : np.ndarray
        Normalized data
    params : dict
        Parameters from normalize() with return_params=True
    method : str or None
        Method used (inferred from params if not provided)

    Returns
    -------
    np.ndarray
        Data in original scale

    Examples
    --------
    >>> X_norm, params = normalize(X, return_params=True)
    >>> X_original = denormalize(X_norm, params)
    """
    if method is None:
        # Try to infer from params
        if 'mean' in params and 'std' in params:
            method = 'zscore'
        elif 'min' in params and 'max' in params:
            method = 'minmax'
        elif 'median' in params and 'iqr' in params:
            method = 'robust'
        else:
            raise ValueError("Cannot infer method from params")

    transformer = SeriesTransformer()
    transformer.transforms[method] = TransformParams(method=method, params=params)

    return transformer.inverse_transform(data, method=method)


def detrend(data: np.ndarray,
           method: Literal['linear', 'polynomial', 'difference'] = 'linear',
           return_params: bool = False,
           **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Remove trend from time series.

    Parameters
    ----------
    data : np.ndarray
        Input time series
    method : {'linear', 'polynomial', 'difference'}
        Detrending method
    return_params : bool, default False
        If True, return (detrended_data, parameters)
    **kwargs
        Method-specific parameters (e.g., degree for polynomial)

    Returns
    -------
    detrended : np.ndarray
        Detrended data
    params : dict (if return_params=True)
        Parameters for retrending

    Examples
    --------
    >>> X_detrended = detrend(X, method='linear')
    >>> X_detrended, params = detrend(X, method='polynomial', degree=2, return_params=True)
    >>> X_original = retrend(X_detrended, params)
    """
    method_map = {
        'linear': 'linear_detrend',
        'polynomial': 'polynomial_detrend',
        'difference': 'difference'
    }

    transformer = SeriesTransformer()
    full_method = method_map[method]
    detrended = transformer.fit_transform(data, method=full_method, **kwargs)

    if return_params:
        return detrended, transformer.transforms[full_method].params
    return detrended


def retrend(data: np.ndarray,
           params: dict) -> np.ndarray:
    """
    Add trend back using stored parameters.

    Parameters
    ----------
    data : np.ndarray
        Detrended data
    params : dict
        Parameters from detrend() with return_params=True

    Returns
    -------
    np.ndarray
        Data with trend restored

    Examples
    --------
    >>> X_detrended, params = detrend(X, return_params=True)
    >>> X_original = retrend(X_detrended, params)
    """
    # Infer method from params
    if 'slope' in params:
        method = 'linear_detrend'
    elif 'coefficients' in params:
        method = 'polynomial_detrend'
    elif 'initial_values' in params:
        method = 'difference'
    else:
        raise ValueError("Cannot infer method from params")

    transformer = SeriesTransformer()
    transformer.transforms[method] = TransformParams(method=method, params=params)

    return transformer.inverse_transform(data, method=method)


def preprocess_for_ccm(data: np.ndarray,
                      detrend_method: Optional[str] = 'linear',
                      normalize_method: Optional[str] = 'zscore',
                      return_transformer: bool = False,
                      **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, SeriesTransformer]]:
    """
    Standard preprocessing pipeline for CCM analysis.

    Applies detrending followed by normalization, which is the recommended
    preprocessing for most EDM applications.

    Parameters
    ----------
    data : np.ndarray
        Input time series
    detrend_method : str or None
        Detrending method: 'linear', 'polynomial', 'difference', or None
    normalize_method : str or None
        Normalization method: 'zscore', 'minmax', 'robust', or None
    return_transformer : bool, default False
        If True, return transformer object for inverse transformation
    **kwargs
        Method-specific parameters

    Returns
    -------
    processed : np.ndarray
        Preprocessed data
    transformer : SeriesTransformer (if return_transformer=True)
        Fitted transformer for inverse operations

    Examples
    --------
    >>> # Standard preprocessing
    >>> X_processed = preprocess_for_ccm(X)

    >>> # With transformer for inverse
    >>> X_processed, transformer = preprocess_for_ccm(X, return_transformer=True)
    >>> # After CCM predictions
    >>> predictions_original = transformer.inverse_transform_chain(predictions)
    """
    transformer = SeriesTransformer()

    methods = []
    if detrend_method is not None:
        method_map = {
            'linear': 'linear_detrend',
            'polynomial': 'polynomial_detrend',
            'difference': 'difference'
        }
        methods.append(method_map[detrend_method])

    if normalize_method is not None:
        methods.append(normalize_method)

    if not methods:
        # No transformation
        if return_transformer:
            return data, transformer
        return data

    processed = transformer.fit_transform_chain(data, methods, **kwargs)

    if return_transformer:
        return processed, transformer
    return processed
