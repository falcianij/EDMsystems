"""
Basic CCM Analysis Example

This script demonstrates a simple CCM analysis workflow using the
modular EDMsystems package.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import EDMsystems modules
from edmsystems.preprocessing import (
    detrend_stl,
    normalize_dataframe,
    inverse_normalize_dataframe
)
from edmsystems.ccm import (
    auto_optimize_parameters,
    ccm_analysis,
    fit_ccm_curve
)
from edmsystems.surrogates import (
    generate_seasonal_pair_surrogates,
    test_significance
)


def main():
    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    print("Loading data...")

    # Load your data (example assumes CSV with datetime index)
    # data = pd.read_csv('data/your_data.csv', index_col='Date', parse_dates=True)

    # For this example, create synthetic data
    np.random.seed(42)
    n = 240  # 20 years of monthly data
    time = np.arange(n)

    # Create synthetic seasonal data with causal relationship
    driver = np.sin(2 * np.pi * time / 12) + 0.3 * np.random.randn(n)
    target = 0.7 * driver + np.sin(2 * np.pi * time / 12 + np.pi/4) + 0.3 * np.random.randn(n)

    data = pd.DataFrame({
        'time': time,
        'SST': driver,
        'species': target
    })

    print(f"Data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")

    # ============================================================
    # 2. PREPROCESSING
    # ============================================================
    print("\nPreprocessing data...")

    # Detrend using STL (keep seasonal signal)
    detrended = detrend_stl(
        data[['SST', 'species']],
        period=12,
        keep_seasonal=True
    )

    # Normalize
    normalized, normalizers = normalize_dataframe(
        detrended,
        method='zscore'
    )

    # Add time column back
    normalized['time'] = data['time'].values

    print("Preprocessing complete!")

    # ============================================================
    # 3. PARAMETER OPTIMIZATION
    # ============================================================
    print("\nOptimizing parameters...")

    params = auto_optimize_parameters(
        df=normalized,
        driver='SST',
        target='species',
        use_autocorr_tau=True
    )

    print(f"Optimal parameters:")
    print(f"  E = {params['best_E']}")
    print(f"  tau = {params['best_tau']}")
    print(f"  theta = {params['best_theta']:.2f}")
    print(f"  prediction skill (rho) = {params['best_rho']:.3f}")

    # ============================================================
    # 4. CCM ANALYSIS WITH CONVERGENCE
    # ============================================================
    print("\nRunning CCM analysis...")

    ccm_result = ccm_analysis(
        df=normalized,
        driver='SST',
        target='species',
        E=params['best_E'],
        tau=params['best_tau'],
        theta=params['best_theta'],
        Tp=1
    )

    print(f"\nCCM Results:")
    print(f"  CCM rho = {ccm_result['rho']:.3f}")
    print(f"  Converged rho = {ccm_result['rho_conv']:.3f}")
    print(f"  Converged = {ccm_result['convergence']}")

    # Plot convergence curve
    if ccm_result['fit'] is not None:
        plt.figure(figsize=(8, 5))
        lib_means = ccm_result['lib_means']

        plt.scatter(lib_means.iloc[:, 0], lib_means.iloc[:, 1],
                   alpha=0.6, label='Observed')

        # Plot fitted curve
        from edmsystems.ccm.core import saturating_curve
        fit = ccm_result['fit']
        L_grid = np.linspace(lib_means.iloc[:, 0].min(),
                           lib_means.iloc[:, 0].max(), 200)
        y_fit = saturating_curve(L_grid, fit['a'], fit['K'], fit['b'])

        plt.plot(L_grid, y_fit, 'r-', lw=2, label='Fit')
        plt.xlabel('Library Size')
        plt.ylabel('CCM ρ')
        plt.title(f"CCM Convergence (R² = {fit['R2']:.2f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('examples/ccm_convergence.png', dpi=150)
        print("Convergence plot saved to: examples/ccm_convergence.png")

    # ============================================================
    # 5. SIGNIFICANCE TESTING WITH SURROGATES
    # ============================================================
    print("\nTesting significance with seasonal surrogates...")

    n_surrogates = 100
    rho_surrogates = []

    for i in range(n_surrogates):
        # Generate paired surrogates (shuffles both variables together)
        x_surr, y_surr = generate_seasonal_pair_surrogates(
            normalized['SST'].values,
            normalized['species'].values,
            n_surr=1,
            period=12,
            mode='within_phase'
        )

        # Create surrogate dataframe
        df_surr = normalized.copy()
        df_surr['SST'] = x_surr[0]
        df_surr['species'] = y_surr[0]

        # Run CCM on surrogate (single library size for speed)
        try:
            from pyEDM import CCM
            result_surr = CCM(
                dataFrame=df_surr,
                columns='SST',
                target='species',
                libSizes=str(int(0.8 * len(df_surr))),
                sample=100,
                exclusionRadius=abs(params['best_E'] * params['best_tau']),
                E=params['best_E'],
                tau=params['best_tau'],
                Tp=1
            )
            rho_surrogates.append(result_surr['LibMeans'].iloc[0, 1])
        except:
            continue

    # Test significance
    significance = test_significance(
        observed=ccm_result['rho'],
        surrogates=np.array(rho_surrogates),
        alpha=0.05,
        tail='greater'
    )

    print(f"\nSignificance Testing:")
    print(f"  Observed rho = {significance['observed']:.3f}")
    print(f"  Surrogate mean = {significance['surr_mean']:.3f}")
    print(f"  Surrogate 95th percentile = {significance['surr_95p']:.3f}")
    print(f"  p-value = {significance['p_value']:.3f}")
    print(f"  Significant (α=0.05) = {significance['significant']}")

    # Plot surrogate distribution
    plt.figure(figsize=(8, 5))
    plt.hist(rho_surrogates, bins=30, density=True, alpha=0.7, color='gray',
            label='Surrogate distribution')
    plt.axvline(significance['observed'], color='red', linestyle='--',
               linewidth=2, label=f"Observed (ρ={significance['observed']:.3f})")
    plt.axvline(significance['surr_95p'], color='orange', linestyle=':',
               linewidth=2, label=f"95th percentile")
    plt.xlabel('CCM ρ')
    plt.ylabel('Density')
    plt.title(f"Surrogate Test (p={significance['p_value']:.3f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('examples/surrogate_test.png', dpi=150)
    print("Surrogate test plot saved to: examples/surrogate_test.png")

    # ============================================================
    # 6. INVERSE NORMALIZATION (if needed for predictions)
    # ============================================================
    print("\nDemonstrating inverse normalization...")

    # Get original scale back
    original_scale = inverse_normalize_dataframe(normalized[['SST', 'species']],
                                                 normalizers)

    print(f"Normalized range: [{normalized['species'].min():.2f}, {normalized['species'].max():.2f}]")
    print(f"Original range: [{original_scale['species'].min():.2f}, {original_scale['species'].max():.2f}]")

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
