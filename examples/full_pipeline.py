"""
Full CCM Pipeline Example

This script demonstrates a complete CCM analysis workflow including:
1. Multiple preprocessing options
2. Parameter optimization
3. Pairwise CCM analysis
4. Significance testing
5. Result visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import permutations

# Import EDMsystems modules
from edmsystems.preprocessing import (
    detrend_stl,
    detrend_hp,
    detrend_linear,
    normalize_dataframe,
    get_normalizer
)
from edmsystems.ccm import auto_optimize_parameters, ccm_analysis
from edmsystems.surrogates import (
    generate_seasonal_pair_surrogates,
    test_significance
)


def load_data(filepath: str = None):
    """Load and prepare data."""
    if filepath is None:
        # Create synthetic multi-species data
        print("Creating synthetic data...")
        np.random.seed(42)
        n = 240  # 20 years monthly

        time = np.arange(n)
        dates = pd.date_range('2000-01-01', periods=n, freq='M')

        # Environmental driver
        sst = 15 + 3 * np.sin(2 * np.pi * time / 12) + 0.5 * np.random.randn(n)

        # Species with varying responses to SST
        species_a = 0.6 * sst + np.sin(2 * np.pi * time / 12) + 0.4 * np.random.randn(n)
        species_b = -0.4 * sst + np.cos(2 * np.pi * time / 12) + 0.3 * np.random.randn(n)
        species_c = 0.2 * sst + 0.5 * species_a + 0.3 * np.random.randn(n)

        data = pd.DataFrame({
            'Date': dates,
            'SST': sst,
            'species_A': species_a,
            'species_B': species_b,
            'species_C': species_c
        })
        data.set_index('Date', inplace=True)

    else:
        data = pd.read_csv(filepath, index_col='Date', parse_dates=True)

    return data


def preprocess_data(data: pd.DataFrame,
                   detrend_method: str = 'stl',
                   normalize_method: str = 'zscore'):
    """Preprocess time series data."""
    print(f"\nPreprocessing with {detrend_method} + {normalize_method}...")

    # Detrending options
    if detrend_method == 'stl':
        detrended = detrend_stl(data, period=12, keep_seasonal=True)
    elif detrend_method == 'hp':
        detrended = detrend_hp(data, lamb=129600)  # monthly
    elif detrend_method == 'linear':
        detrended = detrend_linear(data)
    else:
        detrended = data.copy()

    # Normalization
    normalized, normalizers = normalize_dataframe(
        detrended,
        method=normalize_method
    )

    # Add time column
    normalized['time'] = np.arange(len(normalized))

    return normalized, normalizers


def analyze_pairwise_ccm(data: pd.DataFrame,
                        species_cols: list,
                        n_surrogates: int = 100):
    """
    Perform pairwise CCM analysis for all species combinations.
    """
    print(f"\nAnalyzing {len(species_cols)} species...")

    # Generate all pairwise combinations
    interactions = list(permutations(species_cols, 2))
    print(f"Total interactions to test: {len(interactions)}")

    results = []

    for i, (driver, target) in enumerate(interactions, 1):
        print(f"\n[{i}/{len(interactions)}] Testing: {driver} -> {target}")

        # Step 1: Optimize parameters
        try:
            params = auto_optimize_parameters(
                df=data,
                driver=driver,
                target=target,
                use_autocorr_tau=True
            )

            E = params['best_E']
            tau = params['best_tau']
            theta = params['best_theta']

            print(f"  Optimal: E={E}, tau={tau}, theta={theta:.2f}")

            # Step 2: CCM with convergence
            ccm_result = ccm_analysis(
                df=data,
                driver=driver,
                target=target,
                E=E,
                tau=tau,
                theta=theta,
                Tp=1
            )

            print(f"  CCM rho = {ccm_result['rho']:.3f}, converged = {ccm_result['convergence']}")

            # Step 3: Significance testing (only if converged)
            if ccm_result['convergence']:
                rho_surr = []

                for _ in range(n_surrogates):
                    x_s, y_s = generate_seasonal_pair_surrogates(
                        data[driver].values,
                        data[target].values,
                        n_surr=1,
                        period=12,
                        mode='within_phase',
                        verbose=False
                    )

                    df_surr = data.copy()
                    df_surr[driver] = x_s[0]
                    df_surr[target] = y_s[0]

                    try:
                        from pyEDM import CCM
                        res = CCM(
                            dataFrame=df_surr,
                            columns=driver,
                            target=target,
                            libSizes=str(int(0.8 * len(df_surr))),
                            sample=100,
                            exclusionRadius=abs(E * tau),
                            E=E, tau=tau, Tp=1
                        )
                        rho_surr.append(res['LibMeans'].iloc[0, 1])
                    except:
                        continue

                sig = test_significance(ccm_result['rho'], np.array(rho_surr),
                                       alpha=0.05, tail='greater')

                print(f"  p-value = {sig['p_value']:.3f}, significant = {sig['significant']}")

                results.append({
                    'driver': driver,
                    'target': target,
                    'E': E,
                    'tau': tau,
                    'theta': theta,
                    'rho': ccm_result['rho'],
                    'rho_conv': ccm_result['rho_conv'],
                    'convergence': ccm_result['convergence'],
                    'p_value': sig['p_value'],
                    'significant': sig['significant'],
                    'surr_mean': sig['surr_mean'],
                    'ccm_norm': ccm_result['rho'] - sig['surr_mean']
                })

            else:
                results.append({
                    'driver': driver,
                    'target': target,
                    'E': E,
                    'tau': tau,
                    'theta': theta,
                    'rho': ccm_result['rho'],
                    'rho_conv': np.nan,
                    'convergence': False,
                    'p_value': np.nan,
                    'significant': False,
                    'surr_mean': np.nan,
                    'ccm_norm': np.nan
                })

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'driver': driver,
                'target': target,
                'E': np.nan,
                'tau': np.nan,
                'theta': np.nan,
                'rho': np.nan,
                'rho_conv': np.nan,
                'convergence': False,
                'p_value': np.nan,
                'significant': False,
                'surr_mean': np.nan,
                'ccm_norm': np.nan
            })

    return pd.DataFrame(results)


def plot_results(results: pd.DataFrame, output_dir: str = 'examples'):
    """Visualize CCM analysis results."""
    Path(output_dir).mkdir(exist_ok=True)

    # Filter to significant interactions
    sig_results = results[results['significant'] == True].copy()

    print(f"\n{len(sig_results)} significant interactions found")

    if len(sig_results) == 0:
        print("No significant interactions to plot.")
        return

    # 1. Heatmap of interaction strengths
    pivot = results.pivot(index='driver', columns='target', values='ccm_norm')
    pivot_sig = results.pivot(index='driver', columns='target', values='significant')

    # Mask non-significant
    masked = pivot.copy()
    masked[~pivot_sig.fillna(False)] = np.nan

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        masked,
        cmap='RdYlGn',
        center=0,
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Normalized CCM ρ'}
    )
    plt.title('Pairwise CCM Interaction Network (Significant Only)')
    plt.xlabel('Target (Driven)')
    plt.ylabel('Driver')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ccm_network.png', dpi=150)
    print(f"Network plot saved: {output_dir}/ccm_network.png")

    # 2. Summary statistics
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sig_results['rho'].hist(bins=20, edgecolor='black')
    plt.xlabel('CCM ρ')
    plt.ylabel('Frequency')
    plt.title('Distribution of Significant CCM Strengths')
    plt.axvline(sig_results['rho'].median(), color='red',
               linestyle='--', label=f"Median = {sig_results['rho'].median():.2f}")
    plt.legend()

    plt.subplot(1, 2, 2)
    sig_results['p_value'].hist(bins=20, edgecolor='black')
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.title('Distribution of p-values')
    plt.axvline(0.05, color='red', linestyle='--', label='α = 0.05')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/ccm_summary.png', dpi=150)
    print(f"Summary plot saved: {output_dir}/ccm_summary.png")


def main():
    print("="*60)
    print("Full CCM Pipeline Example")
    print("="*60)

    # 1. Load data
    data = load_data()
    print(f"\nData loaded: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")

    # 2. Preprocess
    processed, normalizers = preprocess_data(
        data,
        detrend_method='stl',
        normalize_method='zscore'
    )

    # 3. Define species to analyze
    species_cols = ['species_A', 'species_B', 'species_C', 'SST']

    # 4. Run pairwise CCM analysis
    results = analyze_pairwise_ccm(
        data=processed,
        species_cols=species_cols,
        n_surrogates=100
    )

    # 5. Save results
    results.to_csv('examples/ccm_results.csv', index=False)
    print(f"\nResults saved to: examples/ccm_results.csv")

    # 6. Visualize
    plot_results(results)

    # 7. Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total interactions tested: {len(results)}")
    print(f"Converged interactions: {results['convergence'].sum()}")
    print(f"Significant interactions: {results['significant'].sum()}")

    if results['significant'].sum() > 0:
        print("\nSignificant interactions:")
        sig = results[results['significant'] == True][
            ['driver', 'target', 'rho', 'p_value']
        ].sort_values('rho', ascending=False)
        print(sig.to_string(index=False))

    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)


if __name__ == "__main__":
    main()
