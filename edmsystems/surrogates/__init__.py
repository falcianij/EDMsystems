"""
Surrogate time series generation for CCM null model testing.

Provides multiple surrogate methods that preserve different properties:
- Twin surrogates: Preserve ACF and CCF (multivariate Fourier)
- Random surrogates: Preserve amplitude distribution
- Random paired: Preserve correlation between variables
- Circular/within_phase: Preserve seasonal structure
"""

from .generators import (
    generate_twin_surrogates,
    generate_random_surrogates,
    generate_random_paired_surrogates,
    generate_circular_surrogates,
    generate_within_phase_surrogates,
)

__all__ = [
    'generate_twin_surrogates',
    'generate_random_surrogates',
    'generate_random_paired_surrogates',
    'generate_circular_surrogates',
    'generate_within_phase_surrogates',
]
