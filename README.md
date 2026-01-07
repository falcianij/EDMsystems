# EDMsystems

A modular Python package for Empirical Dynamical Modeling (EDM) and Convergent Cross Mapping (CCM) analysis.

## Overview

EDMsystems provides a standardized, reproducible workflow for causal inference in dynamical systems using convergent cross mapping. The package implements EDM best practices including:

- **Automatic parameter optimization** (τ, E, Tp, θ)
- **Multiple surrogate testing methods** (twin/multivariate Fourier, random, circular, seasonal)
- **Random library sampling** for robust convergence testing
- **Parallel processing** support
- **Ground truth validation** tools
- **Comprehensive preprocessing** utilities

## Installation

### Clone the Repository

```bash
git clone https://github.com/falcianij/EDMsystems.git
cd EDMsystems
```

### Install Dependencies

```bash
# Install in editable mode for development
pip install -e .

# Or install specific dependencies
pip install -r requirements.txt
```

### Required Dependencies

- Python ≥ 3.8
- numpy ≥ 1.20.0
- pandas ≥ 1.3.0
- matplotlib ≥ 3.4.0
- scipy ≥ 1.7.0
- statsmodels ≥ 0.13.0
- pyEDM ≥ 1.14.0
- joblib ≥ 1.0.0
- tqdm ≥ 4.62.0
- seaborn ≥ 0.11.0

## Quick Start

### 1. Generate Test Data

```python
from edmsystems.testdata import make_test_dataframe, get_ground_truth_network

# Generate synthetic time series with datetime column
df = make_test_dataframe(n=500, seed=42, start_date='2000-01-01')

# Get ground truth causal network
truth_network = get_ground_truth_network()
```

### 2. Test a Single Pair

```python
from edmsystems.ccm import test_ccm_pair

# Extract variables
X = df['unidirectional_X'].values
Y = df['unidirectional_Y'].values

# Test for causality (X -> Y)
result = test_ccm_pair(
    X, Y,
    driver_name='unidirectional_X',
    target_name='unidirectional_Y',
    libSizes="50 500 50",       # Library sizes: start end increment
    sample=100,                  # Random samples per library size
    n_surrogates=99,            # Number of surrogates for testing
    surrogate_method='twin',    # Multivariate Fourier surrogates
    optimize_params=True,       # Auto-optimize tau, E, Tp
    n_jobs=-1,                  # Parallel processing
    verbose=True
)

print(f"Significant: {result['is_significant_twin']}")
print(f"p-value: {result['p_value_twin']:.4f}")
```

### 3. Run Workflow on Multiple Pairs

```python
from edmsystems.ccm import run_ccm_workflow
from edmsystems.ccm.analysis import summarize_results

# Define pairs to test
pairs = [
    ('unidirectional_X', 'unidirectional_Y'),
    ('bidirectional_X', 'bidirectional_Y'),
    ('independent_X', 'independent_Y'),
]

# Run CCM workflow
results = run_ccm_workflow(
    df,
    pairs=pairs,
    datetime_col='datetime',
    libSizes="50 500 25",
    sample=100,
    n_surrogates=99,
    surrogate_method='twin',
    optimize_params=True,
    n_jobs=-1,
    verbose=True
)

# Compare to ground truth
summary = summarize_results(results, truth_network, surrogate_method='twin', print_summary=True)
```

## Package Structure

```
EDMsystems/
├── edmsystems/
│   ├── ccm/                    # CCM analysis module
│   │   ├── core.py            # Core CCM functions
│   │   ├── parameters.py      # Parameter optimization
│   │   ├── workflow.py        # High-level workflow
│   │   └── analysis.py        # Result analysis & validation
│   ├── preprocessing/          # Preprocessing utilities
│   │   └── aggregation.py     # Time series aggregation
│   ├── surrogates/            # Surrogate generators
│   │   └── generators.py      # Multiple surrogate methods
│   └── testdata/              # Test data generation
│       └── generators.py      # Synthetic data with ground truth
├── vignettes/                  # Example workflows
│   └── standardized_ccm_workflow.py
├── sandbox/                    # Development notebooks
└── README.md
```

## Core Modules

### CCM Module (`edmsystems.ccm`)

**Parameter Optimization:**
```python
from edmsystems.ccm.parameters import optimize_parameters

params = optimize_parameters(
    X, Y,
    max_tau_lag=20,
    tau_threshold=0.1,      # ACF threshold
    max_E=10,
    max_Tp=8,
    optimize_theta=False,   # Optional S-map theta tuning
    verbose=True
)

# Returns: tau, E, Tp, theta + detailed results
```

**CCM Analysis:**
```python
from edmsystems.ccm import compute_ccm

summary, details = compute_ccm(
    dataFrame,
    columns='driver',
    target='target',
    E=3,
    Tp=-1,
    tau=-2,
    libSizes="50 500 25",
    sample=100,
    seed=42
)
```

**Ground Truth Comparison:**
```python
from edmsystems.ccm.analysis import compare_to_ground_truth, compute_performance_metrics

# Compare using specific surrogate method
comparison = compare_to_ground_truth(results, truth_network, surrogate_method='twin')
metrics = compute_performance_metrics(comparison)

print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")
```

### Surrogate Methods (`edmsystems.surrogates`)

**Available Methods:**

```python
from edmsystems.surrogates import (
    generate_twin_surrogates,           # Multivariate Fourier (GOLD STANDARD)
    generate_random_surrogates,         # Random shuffle
    generate_random_paired_surrogates,  # Paired shuffle
    generate_circular_surrogates,       # Circular shift
    generate_within_phase_surrogates    # Seasonal preserving
)

# Twin surrogates (preserves ACF and CCF)
X_surr, Y_surr = generate_twin_surrogates(X, Y, seed=42)

# Seasonal preserving (for monthly/yearly data)
X_surr, Y_surr = generate_within_phase_surrogates(X, Y, period=12, seed=42)
```

**Surrogate Properties:**

| Method | Preserves ACF | Preserves CCF | Best For |
|--------|---------------|---------------|----------|
| Twin (Multivariate Fourier) | ✓ (exact) | ✓ (exact) | Standard CCM testing |
| Random Paired | ✗ | ✓ (instantaneous) | Testing instantaneous vs lagged |
| Circular | ✓ | ✓ (shifted) | Breaking temporal alignment |
| Within-Phase | ✓ (seasonal) | ✓ (seasonal) | Seasonal/cyclic data |
| Random | ✗ | ✗ | Null baseline |

### Preprocessing (`edmsystems.preprocessing`)

```python
from edmsystems.preprocessing import (
    aggregate_temporal,
    fill_missing_values,
    align_time_series,
    create_lagged_features
)

# Aggregate to monthly means
df_monthly = aggregate_temporal(
    df,
    datetime_col='datetime',
    freq='MS',              # Month start
    agg_func='mean'
)

# Handle missing values
df_filled = fill_missing_values(
    df,
    method='interpolate',   # 'ffill', 'bfill', 'drop'
    limit=3                 # Max consecutive NaNs to fill
)

# Create lagged features
df_lagged = create_lagged_features(
    df,
    columns='temperature',
    lags=[1, 2, 3],
    drop_na=True
)
```

### Test Data (`edmsystems.testdata`)

**10 Synthetic Scenarios:**

```python
from edmsystems.testdata import make_test_dataframe, get_ground_truth_network

df = make_test_dataframe(n=500, seed=42)

# Scenarios included:
# 1. independent: No relationship
# 2. correlated: Correlation without causation
# 3. lagcorr: Lag-correlated (non-causal)
# 4. crosscorr: Cross-corr + autocorr (non-causal)
# 5. autocorr: Pure autocorrelated (non-causal)
# 6. seasonal_sync: Seasonal synchronous (non-causal)
# 7. seasonal_lag: Seasonal lagged (non-causal)
# 8. unidirectional: X → Y (TRUE CAUSAL)
# 9. bidirectional: X ↔ Y (TRUE CAUSAL)
# 10. weak: X → Y weak (TRUE CAUSAL)

# Get ground truth adjacency matrix
truth = get_ground_truth_network()
```

## Development & Local Editing

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/falcianij/EDMsystems.git
cd EDMsystems

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black flake8
```

### Editing the Code

The package is installed in **editable mode** (`pip install -e .`), which means:

1. **Changes are immediately available** - Edit any `.py` file and the changes are reflected on the next import
2. **No reinstallation needed** - Just restart your Python interpreter
3. **Source files are symlinked** - The installation points to your local directory

**Example workflow:**

```bash
# 1. Edit a file
vim edmsystems/ccm/workflow.py

# 2. Test your changes
python
>>> from edmsystems.ccm import test_ccm_pair
>>> # Your changes are now active!

# 3. If in Jupyter, use auto-reload
%load_ext autoreload
%autoreload 2
from edmsystems.ccm import test_ccm_pair  # Always uses latest code
```

### Project Workflow

```bash
# 1. Create a new branch
git checkout -b my-feature-branch

# 2. Make changes to code

# 3. Test your changes
python vignettes/standardized_ccm_workflow.py

# 4. Commit changes
git add edmsystems/
git commit -m "Add new feature"

# 5. Push to remote
git push -u origin my-feature-branch
```

### Code Style

Follow PEP 8 conventions:

```bash
# Format code with black
black edmsystems/

# Check style with flake8
flake8 edmsystems/ --max-line-length=100
```

## Complete Workflow Example

See `vignettes/standardized_ccm_workflow.py` for a comprehensive example:

```bash
cd EDMsystems
python vignettes/standardized_ccm_workflow.py
```

This demonstrates:
- Test data generation with datetime
- Parameter optimization (tau, E, Tp)
- Single pair analysis
- Batch processing on multiple pairs
- Surrogate significance testing
- Ground truth comparison
- Performance metrics (precision, recall, F1)
- Comprehensive visualization

## Key Features

### ✓ Automatic Parameter Optimization

Standard EDM procedure:
1. **τ (tau)**: Find where ACF crosses 0.1 threshold
2. **E**: Optimize for target attractor reconstruction
3. **Tp**: Optimize for cross-mapping skill
4. **θ (theta)**: Optional S-map nonlinearity parameter

### ✓ Random Library Sampling

Robust convergence testing via:
- Multiple random library windows per size
- Avoids temporal bias from fixed windows
- Averaging across realizations reduces variance

### ✓ Selective Pair Calculation

```python
# Test only specific pairs
pairs = [('X1', 'Y1'), ('X2', 'Y2')]
results = run_ccm_workflow(df, pairs=pairs)

# Or test all pairs
results = run_ccm_workflow(df, pairs=None)  # All combinations
```

### ✓ Parallel Processing

```python
results = run_ccm_workflow(
    df,
    n_jobs=-1,        # Parallel within each pair (surrogate generation)
    n_jobs_pairs=4    # Parallel across pairs (4 pairs simultaneously)
)
```

### ✓ Variable Library Sizes

```python
# Faster testing with fewer library sizes
results = test_ccm_pair(X, Y, libSizes="50 500 100")  # 5 sizes

# More detailed convergence curve
results = test_ccm_pair(X, Y, libSizes="50 500 10")   # 46 sizes
```

### ✓ Toggleable Output

```python
# Verbose mode (detailed progress)
result = test_ccm_pair(X, Y, verbose=True)

# Quiet mode (suppress output)
result = test_ccm_pair(X, Y, verbose=False)
```

## Output Format

The `run_ccm_workflow()` function returns a DataFrame with:

| Column | Description |
|--------|-------------|
| `driver` | Driver variable name |
| `target` | Target variable name |
| `tau` | Optimized time delay |
| `E` | Optimized embedding dimension |
| `Tp` | Optimized prediction horizon |
| `theta` | S-map localization parameter |
| `rho_original` | Cross-mapping skill at max library size |
| `convergent` | Boolean (positive slope) |
| `rho_surrogate_mean_{method}` | Mean rho of surrogates (per method) |
| `rho_surrogate_std_{method}` | Std dev of surrogate rhos (per method) |
| `p_value_{method}` | Empirical p-value (per method) |
| `is_significant_{method}` | Boolean (p < 0.01) (per method) |
| `percentile_95_{method}` | 95th percentile of surrogates (per method) |
| `percentile_99_{method}` | 99th percentile of surrogates (per method) |
| `n_failed_surrogates_{method}` | Count of failed surrogates (per method) |

## Advanced Usage

### Custom Surrogate Testing

```python
from edmsystems.ccm import test_ccm_pair

# Use different surrogate methods individually
result_twin = test_ccm_pair(X, Y, surrogate_method='twin')
result_circular = test_ccm_pair(X, Y, surrogate_method='circular')
result_seasonal = test_ccm_pair(X, Y, surrogate_method='within_phase', period=12)

# Test multiple surrogate methods simultaneously
result_multi = test_ccm_pair(
    X, Y,
    surrogate_method=['twin', 'within_phase', 'circular'],
    period=12
)

# Access method-specific results
print(f"Twin p-value: {result_multi['p_value_twin']:.4f}")
print(f"Within-phase p-value: {result_multi['p_value_within_phase']:.4f}")
print(f"Circular p-value: {result_multi['p_value_circular']:.4f}")
```

### S-map Theta Optimization

```python
# Enable S-map for nonlinear systems
result = test_ccm_pair(
    X, Y,
    optimize_theta=True,
    theta_range=[0, 0.01, 0.1, 0.3, 0.5, 1, 2, 3, 4, 5]
)

print(f"Optimal theta: {result['theta']}")
```

### Manual Parameter Control

```python
# Skip optimization, use specific parameters
result = test_ccm_pair(
    X, Y,
    tau=3,                  # Manually set tau
    E=4,                    # Manually set E
    Tp=-2,                  # Manually set Tp
    optimize_params=False   # Don't re-optimize
)
```

## Troubleshooting

### Import Errors After Editing

If changes aren't reflected:

```python
# Restart Python interpreter, or use reload:
import importlib
from edmsystems.ccm import workflow
importlib.reload(workflow)

# Or in Jupyter:
%load_ext autoreload
%autoreload 2
```

### Memory Issues with Large Datasets

```python
# Reduce library samples
results = run_ccm_workflow(df, sample=50)  # Instead of 100

# Reduce surrogates
results = run_ccm_workflow(df, n_surrogates=49)  # Instead of 99

# Limit library sizes
results = run_ccm_workflow(df, libSizes="100 400 50")  # Fewer points
```

### Slow Performance

```python
# Enable parallel processing
results = run_ccm_workflow(df, n_jobs=-1)  # Use all CPU cores

# Process pairs in parallel
results = run_ccm_workflow(df, n_jobs_pairs=4)  # 4 pairs at once

# Reduce computational load
results = run_ccm_workflow(
    df,
    sample=50,           # Fewer samples
    n_surrogates=49,     # Fewer surrogates
    libSizes="100 400 100"  # Fewer library sizes
)
```

## Citation

If you use this package in your research, please cite:

```
Falciani, J. (2025). EDMsystems: A modular package for Empirical Dynamical
Modeling and Convergent Cross Mapping. GitHub repository.
https://github.com/falcianij/EDMsystems
```

## License

[Add license information here]

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add feature'`)
6. Push to the branch (`git push origin feature-name`)
7. Create a Pull Request

## Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Email: [your.email@example.com]

## References

**Empirical Dynamical Modeling:**
- Sugihara, G., et al. (2012). Detecting causality in complex ecosystems. *Science*, 338(6106), 496-500.

**Convergent Cross Mapping:**
- Sugihara, G., et al. (1990). Nonlinear forecasting as a way of distinguishing chaos from measurement error in time series. *Nature*, 344(6268), 734-741.

**Surrogate Testing:**
- Prichard, D., & Theiler, J. (1994). Generating surrogate data for time series with several simultaneously measured variables. *Physical Review Letters*, 73(7), 951.

**S-map:**
- Sugihara, G. (1994). Nonlinear forecasting for the classification of natural time series. *Philosophical Transactions of the Royal Society A*, 348(1688), 477-495.
