"""
EDM-Standard CCM Analysis Example

This script demonstrates the rigorous EDM procedure for CCM analysis:

1. Choose τ > lag where ACF crosses 0.1
2. Given τ, choose E that best unfolds the attractor of Y (target)
3. Choose Tp that maximizes cross-mapping skill
4. Test significance using twin surrogates that preserve ACF and CCF
5. Use AUC (area under convergence curve) with 99th percentile threshold
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import EDMsystems modules
from edmsystems.preprocessing import (
    detrend_stl,
    normalize_dataframe
)
from edmsystems.ccm import (
    optimize_parameters_edm_standard,
    ccm_analysis
)
from edmsystems.surrogates import (
    generate_twin_iaaft_surrogates,
    empirical_p
)


def load_and_preprocess_data():
    """Load and preprocess data."""
    print("="*70)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*70)

    # Create synthetic data for demonstration
    np.random.seed(42)
    n = 240  # 20 years of monthly data
    time = np.arange(n)
    dates = pd.date_range('2000-01-01', periods=n, freq='M')

    # Environmental driver (SST)
    sst = 15 + 3 * np.sin(2 * np.pi * time / 12) + 0.5 * np.random.randn(n)

    # Species with causal response to SST
    species = 0.6 * sst + np.sin(2 * np.pi * time / 12 + np.pi/4) + 0.3 * np.random.randn(n)

    data = pd.DataFrame({
        'Date': dates,
        'SST': sst,
        'species': species
    })
    data.set_index('Date', inplace=True)

    print(f"Data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")

    # Detrend with STL (keep seasonal signal)
    print("\nDetrending with STL (period=12, keep_seasonal=True)...")
    detrended = detrend_stl(data, period=12, keep_seasonal=True)

    # Normalize with z-score
    print("Normalizing with z-score...")
    normalized, normalizers = normalize_dataframe(detrended, method='zscore')

    # Add time column
    normalized['time'] = np.arange(len(normalized))

    print("✓ Preprocessing complete!")

    return normalized, normalizers


def optimize_parameters(data, driver, target):
    """Optimize parameters following EDM standard procedure."""
    print("\n" + "="*70)
    print("STEP 2: PARAMETER OPTIMIZATION (EDM STANDARD PROCEDURE)")
    print("="*70)
    print(f"\nTesting causal link: {driver} → {target}")

    print("\nProcedure:")
    print("  1. Choose τ > lag where ACF crosses 0.1")
    print("  2. Given τ, choose E that best unfolds target attractor")
    print("  3. Choose Tp that maximizes cross-mapping skill")

    params = optimize_parameters_edm_standard(
        df=data,
        driver=driver,
        target=target,
        max_lag=10,
        acf_threshold=0.1,
        E_range=range(2, 11),
        Tp_range=range(0, 3),
        theta_range=np.linspace(0, 8, 17)
    )

    print(f"\n{'='*70}")
    print("OPTIMIZED PARAMETERS:")
    print(f"{'='*70}")
    print(f"  τ  = {params['tau']}")
    print(f"  E  = {params['E']}")
    print(f"  Tp = {params['Tp']}")
    print(f"  θ  = {params['theta']:.2f}")
    print(f"{'='*70}")

    return params


def run_ccm_analysis(data, driver, target, params):
    """Run CCM analysis with convergence testing."""
    print("\n" + "="*70)
    print("STEP 3: CCM ANALYSIS WITH CONVERGENCE TESTING")
    print("="*70)

    result = ccm_analysis(
        df=data,
        driver=driver,
        target=target,
        E=params['E'],
        tau=params['tau'],
        theta=params['theta'],
        Tp=params['Tp'],
        lib_frac_range=(0.05, 0.8),
        n_lib_sizes=10,
        n_samples=100
    )

    print(f"\nCCM Results:")
    print(f"  Final ρ      = {result['rho']:.3f}")
    print(f"  Converged ρ  = {result['rho_conv']:.3f}")
    print(f"  AUC          = {result['auc']:.1f}")
    print(f"  Converged    = {result['convergence']}")

    if result['fit']:
        print(f"\nCurve fit:")
        print(f"  R² = {result['fit']['R2']:.3f}")
        print(f"  K  = {result['fit']['K']:.1f}")
        print(f"  Tail slope = {result['fit']['slope_tail']:.6f}")

    # Plot convergence curve
    fig, ax = plt.subplots(figsize=(8, 5))
    lib_means = result['lib_means']

    ax.scatter(lib_means.iloc[:, 0], lib_means.iloc[:, 1],
              alpha=0.6, s=50, label='Observed')

    if result['fit']:
        from edmsystems.ccm.core import saturating_curve
        fit = result['fit']
        L_grid = np.linspace(lib_means.iloc[:, 0].min(),
                           lib_means.iloc[:, 0].max(), 200)
        y_fit = saturating_curve(L_grid, fit['a'], fit['K'], fit['b'])
        ax.plot(L_grid, y_fit, 'r-', lw=2, label='Saturating fit')

        if fit['K'] < fit['Lmax']:
            ax.axvline(fit['K'], color='gray', ls='--', alpha=0.7,
                      label=f"K={fit['K']:.0f}")

    ax.set_xlabel('Library Size', fontsize=12)
    ax.set_ylabel('CCM ρ', fontsize=12)
    ax.set_title(f'CCM Convergence: {driver} → {target}\n' +
                f'(AUC = {result["auc"]:.1f}, R² = {result["fit"]["R2"]:.2f})',
                fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('examples/edm_standard_convergence.png', dpi=150)
    print("\n✓ Convergence plot saved: examples/edm_standard_convergence.png")

    return result


def test_significance_with_twin_surrogates(data, driver, target, params, obs_auc, n_surr=100):
    """Test significance using twin IAAFT surrogates and AUC."""
    print("\n" + "="*70)
    print("STEP 4: SIGNIFICANCE TESTING WITH TWIN SURROGATES")
    print("="*70)

    print(f"\nGenerating {n_surr} twin IAAFT surrogates...")
    print("  - Preserves ACF of each variable (autocorrelation)")
    print("  - Destroys CCF (cross-correlation) and causal coupling")

    # Generate twin surrogates
    x_surr, y_surr = generate_twin_iaaft_surrogates(
        data[driver].values,
        data[target].values,
        n_surr=n_surr,
        verbose=True
    )

    print(f"\n✓ Generated {n_surr} surrogate pairs")
    print("\nRunning CCM on each surrogate pair...")

    auc_surr = []

    for i in range(n_surr):
        # Create surrogate dataframe
        df_surr = data.copy()
        df_surr[driver] = x_surr[i]
        df_surr[target] = y_surr[i]

        # Run CCM
        from pyEDM import CCM
        result_surr = CCM(
            dataFrame=df_surr,
            columns=driver,
            target=target,
            libSizes=f"{int(0.05*len(df_surr))} {int(0.8*len(df_surr))} {max(1, int(0.75*len(df_surr)/9))}",
            sample=100,
            exclusionRadius=abs(params['E'] * params['tau']),
            E=params['E'],
            tau=params['tau'],
            Tp=params['Tp']
        )

        # Calculate AUC for surrogate
        lib_sizes = result_surr['LibMeans'].iloc[:, 0].values
        rho_values = result_surr['LibMeans'].iloc[:, 1].values
        auc = np.trapz(rho_values, lib_sizes)
        auc_surr.append(auc)

        if (i+1) % 20 == 0:
            print(f"  Progress: {i+1}/{n_surr}")

    auc_surr = np.array(auc_surr)

    # Compute statistics
    p_value = empirical_p(obs_auc, auc_surr, tail='greater')
    surr_mean = np.mean(auc_surr)
    surr_99p = np.percentile(auc_surr, 99)
    surr_95p = np.percentile(auc_surr, 95)

    # Significance test (99th percentile threshold)
    significant_99 = obs_auc > surr_99p
    significant_95 = obs_auc > surr_95p

    print(f"\n{'='*70}")
    print("SIGNIFICANCE RESULTS:")
    print(f"{'='*70}")
    print(f"  Observed AUC        = {obs_auc:.1f}")
    print(f"  Surrogate mean AUC  = {surr_mean:.1f}")
    print(f"  Surrogate 95th %-ile = {surr_95p:.1f}")
    print(f"  Surrogate 99th %-ile = {surr_99p:.1f}")
    print(f"  p-value             = {p_value:.4f}")
    print(f"  ---")
    print(f"  Significant (α=0.05) = {significant_95}")
    print(f"  Significant (α=0.01) = {significant_99}")
    print(f"{'='*70}")

    # Plot surrogate distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(auc_surr, bins=30, density=True, alpha=0.7, color='gray',
           label=f'Twin IAAFT surrogates (n={n_surr})')
    ax.axvline(obs_auc, color='red', linestyle='--', linewidth=2,
              label=f'Observed AUC = {obs_auc:.1f}')
    ax.axvline(surr_99p, color='orange', linestyle=':', linewidth=2,
              label=f'99th percentile = {surr_99p:.1f}')
    ax.axvline(surr_95p, color='yellow', linestyle=':', linewidth=2,
              label=f'95th percentile = {surr_95p:.1f}')

    ax.set_xlabel('AUC (Area Under Convergence Curve)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Surrogate Test: {driver} → {target}\n' +
                f'(p = {p_value:.4f}, {"SIGNIFICANT" if significant_99 else "not significant"} at α=0.01)',
                fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('examples/edm_standard_surrogate_test.png', dpi=150)
    print("\n✓ Surrogate test plot saved: examples/edm_standard_surrogate_test.png")

    return {
        'p_value': p_value,
        'significant_95': significant_95,
        'significant_99': significant_99,
        'surr_mean': surr_mean,
        'surr_95p': surr_95p,
        'surr_99p': surr_99p,
        'auc_surr': auc_surr
    }


def main():
    print("\n" + "="*70)
    print("EDM-STANDARD CCM ANALYSIS")
    print("="*70)
    print("\nThis script demonstrates the rigorous EDM procedure:")
    print("  1. Parameter optimization (τ → E → Tp)")
    print("  2. CCM convergence analysis")
    print("  3. Twin IAAFT surrogate testing")
    print("  4. AUC-based significance (99th percentile)")

    # Step 1: Load and preprocess
    data, normalizers = load_and_preprocess_data()

    # Step 2: Optimize parameters
    driver = 'SST'
    target = 'species'

    params = optimize_parameters(data, driver, target)

    # Step 3: Run CCM
    ccm_result = run_ccm_analysis(data, driver, target, params)

    # Step 4: Test significance
    if ccm_result['convergence']:
        sig_result = test_significance_with_twin_surrogates(
            data, driver, target, params,
            obs_auc=ccm_result['auc'],
            n_surr=100
        )

        # Final interpretation
        print("\n" + "="*70)
        print("FINAL INTERPRETATION")
        print("="*70)

        if sig_result['significant_99']:
            print(f"\n✓ SIGNIFICANT CAUSAL EFFECT DETECTED:")
            print(f"  {driver} → {target}")
            print(f"\n  The CCM signal (AUC={ccm_result['auc']:.1f}) exceeds the")
            print(f"  99th percentile of twin surrogates ({sig_result['surr_99p']:.1f}).")
            print(f"  This indicates a genuine causal influence that cannot be")
            print(f"  explained by autocorrelation alone.")
            print(f"\n  Confidence level: p = {sig_result['p_value']:.4f}")
        else:
            print(f"\n✗ NO SIGNIFICANT CAUSAL EFFECT:")
            print(f"  {driver} → {target}")
            print(f"\n  The CCM signal (AUC={ccm_result['auc']:.1f}) does not exceed")
            print(f"  the 99th percentile threshold ({sig_result['surr_99p']:.1f}).")
            print(f"  The observed correlation could arise from autocorrelation")
            print(f"  and does not provide strong evidence for causality.")

        print(f"\n{'='*70}\n")

    else:
        print("\n" + "="*70)
        print("ANALYSIS INCOMPLETE")
        print("="*70)
        print("\n✗ CCM did not converge.")
        print("  Cannot test significance without convergence.")
        print("  Consider:")
        print("    - Longer time series")
        print("    - Different preprocessing")
        print("    - Alternative detrending methods")
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
