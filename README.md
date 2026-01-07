# EDMsystems

Rigorous Empirical Dynamical Modeling (EDM) package for causal inference from time series data.

**Features:**
- **Rigorous parameter optimization** following EDM standards (ACF-based tau, std-based E and Tp)
- **Twin surrogate testing** (multivariate Fourier) preserving ACF and CCF
- **Parallel processing** across pairs using multiprocessing
- **Preprocessing** with inverse transformation capability
- **Ground truth comparison** with performance metrics
- **Test data generators** with known causal structure

Based on best practices from ecological forecasting and nonlinear dynamics research.

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/EDMsystems.git
cd EDMsystems

# Install package
pip install -e .

# Install dependencies
pip install numpy pandas matplotlib seaborn scipy statsmodels pyEDM tqdm
```

## Quick Start

```python
import numpy as np
from edmsystems.testdata import make_test_dataframe, get_ground_truth_network
from edmsystems.preprocessing import preprocess_for_ccm
from edmsystems.ccm import run_ccm_workflow, summarize_results

# Generate test data with known ground truth
df = make_test_dataframe(n=500, seed=42)
truth_network = get_ground_truth_network()

# Preprocess data (detrend + normalize)
from edmsystems.preprocessing import normalize_with_params, detrend_with_params

variables = [col for col in df.columns if col != 'time']
df_processed = df[['time']].copy()

for var in variables:
    detrended, _ = detrend_with_params(df[var].values, method='linear')
    normalized, _ = normalize_with_params(detrended, method='zscore')
    df_processed[var] = normalized

# Run CCM workflow with parallel processing
results = run_ccm_workflow(
    df_processed,
    auto_optimize=True,      # Auto-optimize parameters per pair
    libSizes="50 400 50",    # Library sizes: start end increment
    sample=50,               # Random samples per library size
    n_surrogates=99,         # Twin surrogates for testing
    n_jobs=64,               # Parallel jobs (CPU cores)
    verbose=True
)

# Compare to ground truth
summary = summarize_results(results, truth_network, print_summary=True)
```

## Complete Example

See `vignettes/ccm_workflow_example.py` for a complete workflow demonstration.

Run it with:
```bash
python vignettes/ccm_workflow_example.py
```

## Package Structure

```
EDMsystems/
├── edmsystems/
│   ├── ccm/
│   │   ├── parameters.py        # ACF/std-based parameter optimization
│   │   ├── workflow.py          # CCM workflow with multiprocessing
│   │   ├── analysis.py          # Ground truth comparison
│   │   └── core.py             # Core CCM functions
│   ├── surrogates/
│   │   ├── generators.py        # Twin/seasonal surrogate methods
│   │   └── testing.py          # Surrogate significance testing
│   ├── testdata/
│   │   └── generators.py        # Test data with ground truth
│   └── preprocessing/
│       ├── normalization.py     # Normalization methods
│       ├── detrending.py        # Detrending methods
│       ├── transforms.py        # Inverse transformation
│       └── utils.py            # Preprocessing utilities
├── vignettes/
│   └── ccm_workflow_example.py  # Complete example
└── README.md
```

## License

MIT License
