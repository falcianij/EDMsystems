# EDMsystems

A modular Python package for Empirical Dynamical Modeling (EDM) analysis with standardized preprocessing, Convergent Cross Mapping (CCM), and significance testing.

## Features

### 📊 Preprocessing Module
- **Multiple normalization methods** with inverse transforms:
  - Z-score normalization
  - Robust normalization (median/MAD)
  - Min-max normalization
  - Rank-based normalization
  - Square-root min-max normalization
  - Log1p transformation
  - Winsorization

- **8 detrending methods**:
  - Linear detrending
  - Polynomial detrending
  - STL (Seasonal-Trend decomposition)
  - HP-filter (Hodrick-Prescott)
  - LOESS/LOWESS smoothing
  - Rolling mean
  - Savitzky-Golay filter
  - Seasonal differencing

### 🔄 CCM Analysis Module
- Standardized CCM procedure with library size convergence
- Automatic convergence testing using saturating curve fits
- Parameter optimization (E, tau, theta) via grid search
- Autocorrelation-aware tau selection
- Integration with pyEDM library

### 🎲 Surrogate Testing Module
- **Multiple surrogate generation methods**:
  - Random shuffling
  - IAAFT (Iterative Amplitude Adjusted Fourier Transform)
  - Seasonal surrogates (circular shift, cycle permutation, within-phase)

- **Paired surrogate generation** (critical for CCM):
  - Shuffles both variables together to preserve correlation
  - Destroys temporal causal structure while maintaining statistical properties

- **Significance testing**:
  - Empirical p-values
  - Multiple testing correction (FDR, Bonferroni)
  - Customizable thresholds

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/EDMsystems.git
cd EDMsystems

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### Basic CCM Analysis

```python
import pandas as pd
from edmsystems.preprocessing import detrend_stl, normalize_dataframe
from edmsystems.ccm import auto_optimize_parameters, ccm_analysis
from edmsystems.surrogates import generate_seasonal_pair_surrogates, test_significance

# Load your data
data = pd.read_csv('your_data.csv')

# Preprocess
detrended = detrend_stl(data[['driver', 'target']], period=12, keep_seasonal=True)
normalized, normalizers = normalize_dataframe(detrended, method='zscore')
normalized['time'] = range(len(normalized))

# Optimize parameters
params = auto_optimize_parameters(
    df=normalized,
    driver='driver',
    target='target'
)

# Run CCM
result = ccm_analysis(
    df=normalized,
    driver='driver',
    target='target',
    E=params['best_E'],
    tau=params['best_tau'],
    theta=params['best_theta']
)

print(f"CCM rho: {result['rho']:.3f}")
print(f"Converged: {result['convergence']}")

# Test significance with paired surrogates
x_surr, y_surr = generate_seasonal_pair_surrogates(
    normalized['driver'].values,
    normalized['target'].values,
    n_surr=100,
    period=12,
    mode='within_phase'
)

# Compute surrogate CCM values and test significance
# (see examples for full implementation)
```

### Full Pipeline Example

```bash
# Run the complete pipeline example
python examples/full_pipeline.py
```

This will:
1. Load or generate synthetic data
2. Preprocess with detrending and normalization
3. Optimize parameters for each interaction
4. Perform CCM analysis with convergence testing
5. Test significance using seasonal surrogates
6. Generate visualization plots
7. Save results to CSV

## Module Structure

```
EDMsystems/
├── edmsystems/              # Main package
│   ├── preprocessing/       # Preprocessing module
│   │   ├── normalization.py    # Normalization methods with inverse transforms
│   │   ├── detrending.py       # Detrending methods
│   │   └── utils.py            # Utility functions
│   ├── ccm/                 # CCM analysis module
│   │   ├── core.py             # Core CCM functions
│   │   └── optimization.py     # Parameter optimization
│   ├── surrogates/          # Surrogate testing module
│   │   ├── generators.py       # Surrogate generation methods
│   │   └── testing.py          # Significance testing
│   └── utils/               # General utilities
├── examples/                # Example scripts
│   ├── basic_ccm.py            # Basic CCM workflow
│   └── full_pipeline.py        # Complete analysis pipeline
├── data/                    # Example data
├── tests/                   # Unit tests (future)
└── README.md
```

## Key Concepts

### Modular Design

The package is designed to be modular and extensible:

- **Add new normalization methods**: Inherit from `Normalizer` base class
- **Add new detrending methods**: Add function to `detrending.py`
- **Add new surrogate types**: Add function to `generators.py`

### Paired Surrogate Testing

A critical feature for CCM analysis is that **both variables must be shuffled together** to preserve their instantaneous correlation while destroying temporal causal structure. This package implements this correctly:

```python
# Correct: Paired shuffling
x_surr, y_surr = generate_seasonal_pair_surrogates(x, y, n_surr=100)

# Incorrect: Independent shuffling (destroys correlation)
# x_surr = generate_seasonal_surrogates(x, n_surr=100)
# y_surr = generate_seasonal_surrogates(y, n_surr=100)
```

### Convergence Testing

CCM convergence is tested using a saturating curve fit:

```
ρ(L) = a*L / (K + L) + b
```

Where:
- `a`: amplitude (asymptotic increase)
- `K`: half-saturation constant
- `b`: baseline

Convergence criteria:
- R² > 0.7 (good fit)
- Tail slope < 0.001 (plateau)
- K < 0.8 * Lmax (convergence before end)
- ρ_conv > 0 (positive signal)

### Parameter Optimization

The package optimizes:
- **E** (embedding dimension): Dimension of reconstructed state space (2-6)
- **tau** (time lag): Lag for embedding, chosen to avoid autocorrelation (-3 to 0)
- **theta** (nonlinearity): S-Map nonlinearity parameter (0 = simplex, >0 = S-map)

## Examples

### 1. Different Normalization Methods

```python
from edmsystems.preprocessing import get_normalizer

# Z-score normalization
normalizer = get_normalizer('zscore')
normalized = normalizer.fit_transform(data['species'])
original = normalizer.inverse_transform(normalized)

# Robust normalization (resistant to outliers)
normalizer = get_normalizer('robust')
normalized = normalizer.fit_transform(data['species'])
```

### 2. Different Detrending Methods

```python
from edmsystems.preprocessing import detrend_stl, detrend_hp, detrend_loess

# STL: Keep seasonal signal
detrended = detrend_stl(data, period=12, keep_seasonal=True)

# HP filter: For monthly data
detrended = detrend_hp(data, lamb=129600)

# LOESS: Non-parametric
detrended = detrend_loess(data, frac=0.25)
```

### 3. Different Surrogate Types

```python
from edmsystems.surrogates import (
    generate_random_surrogates,
    generate_iaaft_surrogates,
    generate_seasonal_surrogates
)

# Random shuffling (destroys all structure)
surr = generate_random_surrogates(x, n_surr=100)

# IAAFT (preserves spectrum and distribution)
surr = generate_iaaft_surrogates(x, n_surr=100)

# Seasonal (preserves seasonal structure)
surr = generate_seasonal_surrogates(x, n_surr=100, period=12, mode='within_phase')
```

## Data Format

The package expects time series data as pandas DataFrames with:
- Datetime index (for temporal ordering)
- One column per variable
- A 'time' column (integer timesteps) for EDM analysis

Example:
```
Date        | time | SST   | species_A | species_B
------------|------|-------|-----------|----------
2000-01-01  |  0   | 15.2  | 100.5     | 45.2
2000-02-01  |  1   | 15.8  | 105.3     | 42.1
...
```

## Requirements

- Python ≥ 3.8
- numpy
- pandas
- scipy
- statsmodels
- matplotlib
- seaborn
- tqdm
- pyEDM (for CCM functionality)

## Citation

If you use this package in your research, please cite:

```
Falciani, J. (2025). EDMsystems: A modular package for Empirical Dynamical Modeling.
GitHub repository: https://github.com/yourusername/EDMsystems
```

## References

1. Sugihara, G., et al. (2012). Detecting causality in complex ecosystems. Science, 338(6106), 496-500.
2. Ye, H., et al. (2015). Distinguishing time-delayed causal interactions using convergent cross mapping. Scientific Reports, 5, 14750.
3. Deyle, E. R., & Sugihara, G. (2011). Generalized theorems for nonlinear state space reconstruction. PLoS ONE, 6(3), e18295.

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Contact

Jonathan Falciani - [your email]

Project Link: https://github.com/yourusername/EDMsystems
