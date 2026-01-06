"""
Normalization methods for EDM time series preprocessing.

This module provides multiple normalization strategies with inverse transforms
for unnormalizing predictions back to original scale.
"""

import numpy as np
import pandas as pd
from scipy.special import erfinv, erf
from typing import Union, Tuple, Optional, Dict, Any


class Normalizer:
    """Base class for normalization with inverse transform."""

    def __init__(self):
        self.params = {}

    def fit(self, x: Union[pd.Series, np.ndarray]) -> 'Normalizer':
        """Fit normalizer to data (compute parameters)."""
        raise NotImplementedError

    def transform(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """Apply normalization transform."""
        raise NotImplementedError

    def inverse_transform(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """Inverse transform to recover original scale."""
        raise NotImplementedError

    def fit_transform(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """Fit and transform in one step."""
        return self.fit(x).transform(x)


class ZScoreNormalizer(Normalizer):
    """Z-score normalization: (x - mean) / std."""

    def fit(self, x: Union[pd.Series, np.ndarray]) -> 'ZScoreNormalizer':
        """Compute mean and std from data."""
        is_series = isinstance(x, pd.Series)
        x_array = pd.to_numeric(x, errors='coerce').values if is_series else np.asarray(x)

        self.params['mean'] = np.nanmean(x_array)
        self.params['std'] = np.nanstd(x_array, ddof=1)
        self.params['is_series'] = is_series

        if is_series:
            self.params['index'] = x.index
            self.params['name'] = x.name

        return self

    def transform(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """Apply z-score normalization."""
        is_series = isinstance(x, pd.Series)
        x_array = pd.to_numeric(x, errors='coerce').values if is_series else np.asarray(x)

        mean = self.params['mean']
        std = self.params['std']

        if std > 0:
            result = (x_array - mean) / std
        else:
            result = x_array * 0.0

        if is_series:
            return pd.Series(result, index=x.index, name=x.name)
        return result

    def inverse_transform(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """Inverse z-score: x * std + mean."""
        is_series = isinstance(x, pd.Series)
        x_array = x.values if is_series else np.asarray(x)

        mean = self.params['mean']
        std = self.params['std']

        result = x_array * std + mean

        if is_series:
            return pd.Series(result, index=x.index, name=x.name)
        return result


class RobustNormalizer(Normalizer):
    """Robust normalization using median and MAD (median absolute deviation)."""

    def fit(self, x: Union[pd.Series, np.ndarray]) -> 'RobustNormalizer':
        """Compute median and MAD from data."""
        is_series = isinstance(x, pd.Series)
        x_array = pd.to_numeric(x, errors='coerce').values if is_series else np.asarray(x)

        self.params['median'] = np.nanmedian(x_array)
        mad = np.nanmedian(np.abs(x_array - self.params['median']))
        self.params['scale'] = 1.4826 * mad  # MAD to sigma conversion
        self.params['is_series'] = is_series

        if is_series:
            self.params['index'] = x.index
            self.params['name'] = x.name

        return self

    def transform(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """Apply robust normalization."""
        is_series = isinstance(x, pd.Series)
        x_array = pd.to_numeric(x, errors='coerce').values if is_series else np.asarray(x)

        median = self.params['median']
        scale = self.params['scale']

        if scale > 0:
            result = (x_array - median) / scale
        else:
            result = x_array * 0.0

        if is_series:
            return pd.Series(result, index=x.index, name=x.name)
        return result

    def inverse_transform(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """Inverse robust normalization."""
        is_series = isinstance(x, pd.Series)
        x_array = x.values if is_series else np.asarray(x)

        median = self.params['median']
        scale = self.params['scale']

        result = x_array * scale + median

        if is_series:
            return pd.Series(result, index=x.index, name=x.name)
        return result


class MinMaxNormalizer(Normalizer):
    """Min-Max normalization to [0, 1] range."""

    def fit(self, x: Union[pd.Series, np.ndarray]) -> 'MinMaxNormalizer':
        """Compute min and max from data."""
        is_series = isinstance(x, pd.Series)
        x_array = pd.to_numeric(x, errors='coerce').values if is_series else np.asarray(x)

        self.params['min'] = np.nanmin(x_array)
        self.params['max'] = np.nanmax(x_array)
        self.params['is_series'] = is_series

        if is_series:
            self.params['index'] = x.index
            self.params['name'] = x.name

        return self

    def transform(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """Apply min-max normalization."""
        is_series = isinstance(x, pd.Series)
        x_array = pd.to_numeric(x, errors='coerce').values if is_series else np.asarray(x)

        min_val = self.params['min']
        max_val = self.params['max']

        if max_val > min_val:
            result = (x_array - min_val) / (max_val - min_val)
        else:
            result = np.full_like(x_array, 0.5, dtype=float)

        if is_series:
            return pd.Series(result, index=x.index, name=x.name)
        return result

    def inverse_transform(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """Inverse min-max normalization."""
        is_series = isinstance(x, pd.Series)
        x_array = x.values if is_series else np.asarray(x)

        min_val = self.params['min']
        max_val = self.params['max']

        result = x_array * (max_val - min_val) + min_val

        if is_series:
            return pd.Series(result, index=x.index, name=x.name)
        return result


class RankNormalizer(Normalizer):
    """Rank-based normalization to ~N(0,1) via inverse normal transform."""

    def fit(self, x: Union[pd.Series, np.ndarray]) -> 'RankNormalizer':
        """Store data properties for ranking."""
        is_series = isinstance(x, pd.Series)
        self.params['is_series'] = is_series

        if is_series:
            self.params['index'] = x.index
            self.params['name'] = x.name

        return self

    def transform(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """Apply rank-based normalization."""
        is_series = isinstance(x, pd.Series)

        if is_series:
            x_series = pd.to_numeric(x, errors='coerce')
            r = x_series.rank(method="average", na_option="keep")
            n = float(r.count())
            u = (r - 0.5) / n
            u = u.clip(1e-6, 1 - 1e-6)
            result = np.sqrt(2.0) * erfinv(2.0 * u - 1.0)
            return result
        else:
            x_series = pd.Series(x)
            r = x_series.rank(method="average", na_option="keep")
            n = float(r.count())
            u = (r - 0.5) / n
            u = u.clip(1e-6, 1 - 1e-6)
            return (np.sqrt(2.0) * erfinv(2.0 * u - 1.0)).values

    def inverse_transform(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """
        Inverse rank transform.

        Note: This is approximate and will not perfectly recover original values
        since rank transformation is not one-to-one (multiple values can have same rank).
        This returns the cumulative normal distribution values.
        """
        is_series = isinstance(x, pd.Series)
        x_array = x.values if is_series else np.asarray(x)

        # Convert back from N(0,1) to uniform [0,1]
        u = 0.5 * (1 + erf(x_array / np.sqrt(2.0)))

        if is_series:
            return pd.Series(u, index=x.index, name=x.name)
        return u


class SqrtMinMaxNormalizer(Normalizer):
    """Combined square-root and min-max normalization."""

    def fit(self, x: Union[pd.Series, np.ndarray]) -> 'SqrtMinMaxNormalizer':
        """Compute min and max from sqrt-transformed data."""
        is_series = isinstance(x, pd.Series)
        x_array = pd.to_numeric(x, errors='coerce').values if is_series else np.asarray(x)

        # Store original min for inverse
        self.params['orig_min'] = np.nanmin(x_array)

        # Shift to non-negative
        x_shifted = x_array - self.params['orig_min']
        x_sqrt = np.sqrt(np.clip(x_shifted, 0.0, None))

        self.params['sqrt_min'] = np.nanmin(x_sqrt)
        self.params['sqrt_max'] = np.nanmax(x_sqrt)
        self.params['is_series'] = is_series

        if is_series:
            self.params['index'] = x.index
            self.params['name'] = x.name

        return self

    def transform(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """Apply sqrt-minmax normalization."""
        is_series = isinstance(x, pd.Series)
        x_array = pd.to_numeric(x, errors='coerce').values if is_series else np.asarray(x)

        # Shift and sqrt
        x_shifted = x_array - self.params['orig_min']
        x_sqrt = np.sqrt(np.clip(x_shifted, 0.0, None))

        # Min-max on sqrt values
        sqrt_min = self.params['sqrt_min']
        sqrt_max = self.params['sqrt_max']

        if sqrt_max > sqrt_min:
            result = (x_sqrt - sqrt_min) / (sqrt_max - sqrt_min)
        else:
            result = np.full_like(x_sqrt, 0.5, dtype=float)

        if is_series:
            return pd.Series(result, index=x.index, name=x.name)
        return result

    def inverse_transform(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """Inverse sqrt-minmax normalization."""
        is_series = isinstance(x, pd.Series)
        x_array = x.values if is_series else np.asarray(x)

        sqrt_min = self.params['sqrt_min']
        sqrt_max = self.params['sqrt_max']

        # Inverse min-max
        x_sqrt = x_array * (sqrt_max - sqrt_min) + sqrt_min

        # Inverse sqrt
        x_shifted = x_sqrt ** 2

        # Inverse shift
        result = x_shifted + self.params['orig_min']

        if is_series:
            return pd.Series(result, index=x.index, name=x.name)
        return result


class Log1pNormalizer(Normalizer):
    """Log1p transformation followed by optional normalization."""

    def __init__(self, post_normalize: Optional[str] = 'zscore'):
        """
        Initialize Log1p normalizer.

        Parameters
        ----------
        post_normalize : str or None
            Apply additional normalization after log1p: 'zscore', 'minmax', or None
        """
        super().__init__()
        self.post_normalize = post_normalize
        self.post_normalizer = None

    def fit(self, x: Union[pd.Series, np.ndarray]) -> 'Log1pNormalizer':
        """Fit log1p and optional post-normalization."""
        is_series = isinstance(x, pd.Series)
        x_array = pd.to_numeric(x, errors='coerce').values if is_series else np.asarray(x)

        # Apply log1p
        x_log = np.log1p(x_array)

        # Fit post-normalizer if requested
        if self.post_normalize == 'zscore':
            self.post_normalizer = ZScoreNormalizer().fit(x_log)
        elif self.post_normalize == 'minmax':
            self.post_normalizer = MinMaxNormalizer().fit(x_log)

        self.params['is_series'] = is_series
        if is_series:
            self.params['index'] = x.index
            self.params['name'] = x.name

        return self

    def transform(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """Apply log1p transformation."""
        is_series = isinstance(x, pd.Series)
        x_array = pd.to_numeric(x, errors='coerce').values if is_series else np.asarray(x)

        result = np.log1p(x_array)

        if self.post_normalizer is not None:
            result = self.post_normalizer.transform(result)
            if is_series:
                return pd.Series(result, index=x.index, name=x.name)
            return result

        if is_series:
            return pd.Series(result, index=x.index, name=x.name)
        return result

    def inverse_transform(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """Inverse log1p transformation."""
        is_series = isinstance(x, pd.Series)
        x_array = x.values if is_series else np.asarray(x)

        # Inverse post-normalization first
        if self.post_normalizer is not None:
            x_array = self.post_normalizer.inverse_transform(x_array)
            if isinstance(x_array, pd.Series):
                x_array = x_array.values

        # Inverse log1p
        result = np.expm1(x_array)

        if is_series:
            return pd.Series(result, index=x.index, name=x.name)
        return result


class WinsorizingNormalizer(Normalizer):
    """Winsorization (clipping extreme quantiles) followed by normalization."""

    def __init__(self, quantiles: Tuple[float, float] = (0.01, 0.99),
                 post_normalize: str = 'zscore'):
        """
        Initialize Winsorizing normalizer.

        Parameters
        ----------
        quantiles : tuple of float
            Lower and upper quantiles to clip (e.g., (0.01, 0.99))
        post_normalize : str
            Normalization method after clipping: 'zscore', 'robust', 'minmax'
        """
        super().__init__()
        self.quantiles = quantiles
        self.post_normalize = post_normalize
        self.post_normalizer = None

    def fit(self, x: Union[pd.Series, np.ndarray]) -> 'WinsorizingNormalizer':
        """Compute quantiles and fit post-normalizer."""
        is_series = isinstance(x, pd.Series)
        x_array = pd.to_numeric(x, errors='coerce').values if is_series else np.asarray(x)

        # Compute clipping bounds
        self.params['lower'] = np.nanquantile(x_array, self.quantiles[0])
        self.params['upper'] = np.nanquantile(x_array, self.quantiles[1])

        # Clip and fit post-normalizer
        x_clipped = np.clip(x_array, self.params['lower'], self.params['upper'])

        if self.post_normalize == 'zscore':
            self.post_normalizer = ZScoreNormalizer().fit(x_clipped)
        elif self.post_normalize == 'robust':
            self.post_normalizer = RobustNormalizer().fit(x_clipped)
        elif self.post_normalize == 'minmax':
            self.post_normalizer = MinMaxNormalizer().fit(x_clipped)
        else:
            raise ValueError(f"Unknown post_normalize method: {self.post_normalize}")

        self.params['is_series'] = is_series
        if is_series:
            self.params['index'] = x.index
            self.params['name'] = x.name

        return self

    def transform(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """Apply winsorization and normalization."""
        is_series = isinstance(x, pd.Series)
        x_array = pd.to_numeric(x, errors='coerce').values if is_series else np.asarray(x)

        # Clip
        x_clipped = np.clip(x_array, self.params['lower'], self.params['upper'])

        # Post-normalize
        result = self.post_normalizer.transform(x_clipped)

        if is_series and not isinstance(result, pd.Series):
            return pd.Series(result, index=x.index, name=x.name)
        return result

    def inverse_transform(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """
        Inverse winsorization and normalization.

        Note: Values beyond original quantiles cannot be perfectly recovered.
        """
        return self.post_normalizer.inverse_transform(x)


# Factory function for easy access
def get_normalizer(method: str, **kwargs) -> Normalizer:
    """
    Get a normalizer instance by name.

    Parameters
    ----------
    method : str
        Normalization method: 'zscore', 'robust', 'minmax', 'rank',
        'sqrt_minmax', 'log1p', 'winsorize'
    **kwargs
        Additional arguments for specific normalizers

    Returns
    -------
    Normalizer
        Instance of the requested normalizer
    """
    normalizers = {
        'zscore': ZScoreNormalizer,
        'robust': RobustNormalizer,
        'minmax': MinMaxNormalizer,
        'rank': RankNormalizer,
        'sqrt_minmax': SqrtMinMaxNormalizer,
        'log1p': Log1pNormalizer,
        'winsorize': WinsorizingNormalizer,
    }

    if method not in normalizers:
        raise ValueError(f"Unknown normalization method: {method}. "
                        f"Available: {list(normalizers.keys())}")

    return normalizers[method](**kwargs)


def normalize_dataframe(df: pd.DataFrame,
                        method: str = 'zscore',
                        exclude_cols: Optional[list] = None,
                        **kwargs) -> Tuple[pd.DataFrame, Dict[str, Normalizer]]:
    """
    Normalize all columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    method : str
        Normalization method to apply
    exclude_cols : list or None
        Column names to exclude from normalization
    **kwargs
        Additional arguments for the normalizer

    Returns
    -------
    df_normalized : pd.DataFrame
        Normalized dataframe
    normalizers : dict
        Dictionary mapping column names to fitted normalizer instances
    """
    exclude_cols = exclude_cols or []
    df_norm = df.copy()
    normalizers = {}

    for col in df.columns:
        if col in exclude_cols:
            continue

        normalizer = get_normalizer(method, **kwargs)
        df_norm[col] = normalizer.fit_transform(df[col])
        normalizers[col] = normalizer

    return df_norm, normalizers


def inverse_normalize_dataframe(df: pd.DataFrame,
                                normalizers: Dict[str, Normalizer]) -> pd.DataFrame:
    """
    Inverse normalize a DataFrame using fitted normalizers.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized dataframe
    normalizers : dict
        Dictionary mapping column names to fitted normalizer instances

    Returns
    -------
    pd.DataFrame
        Dataframe in original scale
    """
    df_orig = df.copy()

    for col, normalizer in normalizers.items():
        if col in df.columns:
            df_orig[col] = normalizer.inverse_transform(df[col])

    return df_orig
