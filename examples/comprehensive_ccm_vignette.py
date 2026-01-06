"""
Comprehensive CCM Vignette: Test Data Analysis with Ground Truth Comparison

This vignette demonstrates the complete EDM workflow:
1. Generate test data with known ground truth relationships
2. Run parallelized CCM analysis on all variable pairs
3. Compare detected network to true network
4. Use multivariate Fourier surrogates (gold standard)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

from edmsystems.testdata import make_test_dataframe, get_ground_truth_network
from edmsystems.ccm import ccm_analysis, optimize_parameters_edm_standard
from edmsystems.surrogates import generate_multivariate_fourier_surrogates
from edmsystems.preprocessing import get_normalizer


# ============================================================================
# 1. GENERATE TEST DATA WITH KNOWN GROUND TRUTH
# ============================================================================

print("=" * 80)
print("COMPREHENSIVE EDM VIGNETTE: CCM WITH GROUND TRUTH VALIDATION")
print("=" * 80)
print()

print("Step 1: Generating test data with known causal relationships...")
print("-" * 80)

# Generate test dataframe (500 time points)
df = make_test_dataframe(n=500, seed=42)

print(f"Generated dataframe with {len(df)} time points")
print(f"Columns: {list(df.columns)}")
print()

# Get ground truth network
truth_network = get_ground_truth_network()

print("Ground truth causal network:")
print("-" * 80)
print("True causal edges:")
true_edges = []
for driver in truth_network.index:
    for target in truth_network.columns:
        if truth_network.loc[driver, target] == 1:
            true_edges.append(f"  {driver} -> {target}")

for edge in true_edges:
    print(edge)

print()
print(f"Total true edges: {len(true_edges)}")
print()


# ============================================================================
# 2. OPTIMIZE EDM PARAMETERS FOR EACH VARIABLE PAIR
# ============================================================================

print("Step 2: Optimizing EDM parameters for all variable pairs...")
print("-" * 80)

# Get all numeric columns (exclude datetime)
variables = [col for col in df.columns if col != 'datetime']

# Create all unique directed pairs
pairs = []
for driver in variables:
    for target in variables:
        if driver != target:
            pairs.append((driver, target))

print(f"Total variable pairs to analyze: {len(pairs)}")
print()


def optimize_pair_parameters(driver, target):
    """
    Optimize EDM parameters for a single driver-target pair.
    """
    try:
        result = optimize_parameters_edm_standard(
            df=df,
            driver=driver,
            target=target,
            acf_threshold=0.1,
            E_range=range(2, 11),
            Tp_range=range(1, 6),
            theta_range=[0, 1e-10, 3e-10, 1e-9, 3e-9, 1e-8, 3e-8, 1e-7],
            verbose=False
        )
        return {
            'driver': driver,
            'target': target,
            'tau': result['tau'],
            'E': result['E'],
            'Tp': result['Tp'],
            'theta': result['theta'],
            'rho': result['best_rho'],
        }
    except Exception as e:
        print(f"Warning: Failed to optimize {driver} -> {target}: {e}")
        return {
            'driver': driver,
            'target': target,
            'tau': -1,
            'E': 3,
            'Tp': 1,
            'theta': 0,
            'rho': np.nan,
        }


# Parallelize parameter optimization
print("Optimizing parameters in parallel...")
param_results = Parallel(n_jobs=-1, verbose=5)(
    delayed(optimize_pair_parameters)(driver, target)
    for driver, target in pairs
)

# Convert to dataframe
params_df = pd.DataFrame(param_results)
print()
print("Parameter optimization complete!")
print()
print("Sample of optimized parameters:")
print(params_df.head(10))
print()


# ============================================================================
# 3. RUN PARALLELIZED CCM ANALYSIS WITH SURROGATE TESTING
# ============================================================================

print("Step 3: Running CCM analysis with multivariate Fourier surrogates...")
print("-" * 80)


def run_ccm_with_surrogates(row):
    """
    Run CCM analysis with surrogate testing for a single pair.
    """
    driver = row['driver']
    target = row['target']
    tau = row['tau']
    E = row['E']
    Tp = row['Tp']
    theta = row['theta']

    try:
        # Normalize data
        normalizer = get_normalizer('zscore')
        x_norm = normalizer.fit_transform(df[driver].values)
        y_norm = normalizer.fit_transform(df[target].values)

        # Create normalized dataframe
        df_norm = pd.DataFrame({
            driver: x_norm,
            target: y_norm
        })

        # Run CCM analysis
        ccm_result = ccm_analysis(
            df=df_norm,
            driver=driver,
            target=target,
            E=E,
            tau=tau,
            Tp=Tp,
            theta=theta,
            lib_sizes=np.linspace(50, len(df_norm) - E * abs(tau), 10, dtype=int),
            n_samples=100,
            replace=False,
            verbose=False
        )

        # Generate multivariate Fourier surrogates (GOLD STANDARD)
        x_surr, y_surr = generate_multivariate_fourier_surrogates(
            x_norm, y_norm,
            n_surr=99,
            seed=42,
            verbose=False
        )

        # Run CCM on each surrogate pair
        surrogate_aucs = []
        for i in range(99):
            df_surr = pd.DataFrame({
                driver: x_surr[i],
                target: y_surr[i]
            })

            try:
                surr_result = ccm_analysis(
                    df=df_surr,
                    driver=driver,
                    target=target,
                    E=E,
                    tau=tau,
                    Tp=Tp,
                    theta=theta,
                    lib_sizes=np.linspace(50, len(df_surr) - E * abs(tau), 10, dtype=int),
                    n_samples=100,
                    replace=False,
                    verbose=False
                )
                surrogate_aucs.append(surr_result['auc'])
            except:
                pass

        # Calculate p-value (empirical)
        observed_auc = ccm_result['auc']
        if len(surrogate_aucs) > 0:
            p_value = (np.sum(np.array(surrogate_aucs) >= observed_auc) + 1) / (len(surrogate_aucs) + 1)
            auc_99th = np.percentile(surrogate_aucs, 99)
        else:
            p_value = np.nan
            auc_99th = np.nan

        return {
            'driver': driver,
            'target': target,
            'rho': ccm_result['rho'],
            'rho_conv': ccm_result['rho_conv'],
            'auc': observed_auc,
            'auc_99th': auc_99th,
            'p_value': p_value,
            'significant': p_value < 0.01 if not np.isnan(p_value) else False,
        }

    except Exception as e:
        print(f"Warning: CCM failed for {driver} -> {target}: {e}")
        return {
            'driver': driver,
            'target': target,
            'rho': np.nan,
            'rho_conv': np.nan,
            'auc': np.nan,
            'auc_99th': np.nan,
            'p_value': np.nan,
            'significant': False,
        }


# Run parallelized CCM analysis
print("Running CCM with surrogate testing in parallel (this may take a while)...")
ccm_results = Parallel(n_jobs=-1, verbose=5)(
    delayed(run_ccm_with_surrogates)(row)
    for _, row in params_df.iterrows()
)

# Convert to dataframe
results_df = pd.DataFrame(ccm_results)
print()
print("CCM analysis complete!")
print()


# ============================================================================
# 4. COMPARE DETECTED NETWORK TO GROUND TRUTH
# ============================================================================

print("Step 4: Comparing detected network to ground truth...")
print("-" * 80)

# Create detected adjacency matrix (based on significance)
detected_network = pd.DataFrame(0, index=variables, columns=variables)

for _, row in results_df.iterrows():
    if row['significant']:
        detected_network.loc[row['driver'], row['target']] = 1

# Calculate performance metrics
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

for driver in variables:
    for target in variables:
        if driver == target:
            continue

        true_edge = truth_network.loc[driver, target] == 1
        detected_edge = detected_network.loc[driver, target] == 1

        if true_edge and detected_edge:
            true_positives += 1
        elif not true_edge and detected_edge:
            false_positives += 1
        elif not true_edge and not detected_edge:
            true_negatives += 1
        elif true_edge and not detected_edge:
            false_negatives += 1

# Calculate metrics
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0

print("PERFORMANCE METRICS:")
print(f"  True Positives:  {true_positives}")
print(f"  False Positives: {false_positives}")
print(f"  True Negatives:  {true_negatives}")
print(f"  False Negatives: {false_negatives}")
print()
print(f"  Precision (PPV): {precision:.3f}")
print(f"  Recall (TPR):    {recall:.3f}")
print(f"  Specificity:     {specificity:.3f}")
print(f"  F1 Score:        {f1_score:.3f}")
print()

# Print detected edges
print("DETECTED SIGNIFICANT EDGES (p < 0.01):")
print("-" * 80)
sig_results = results_df[results_df['significant']].sort_values('auc', ascending=False)
for _, row in sig_results.iterrows():
    true_edge = truth_network.loc[row['driver'], row['target']] == 1
    status = "✓ TRUE POSITIVE" if true_edge else "✗ FALSE POSITIVE"
    print(f"  {row['driver']:20s} -> {row['target']:20s}  AUC={row['auc']:.3f}  p={row['p_value']:.4f}  {status}")

print()

# Print missed edges
print("MISSED TRUE EDGES (false negatives):")
print("-" * 80)
for edge in true_edges:
    driver, target = edge.strip().split(' -> ')
    detected = detected_network.loc[driver, target] == 1
    if not detected:
        row_data = results_df[(results_df['driver'] == driver) & (results_df['target'] == target)]
        if len(row_data) > 0:
            auc = row_data.iloc[0]['auc']
            p_val = row_data.iloc[0]['p_value']
            print(f"  {driver:20s} -> {target:20s}  AUC={auc:.3f}  p={p_val:.4f}")

print()


# ============================================================================
# 5. VISUALIZE RESULTS
# ============================================================================

print("Step 5: Creating visualizations...")
print("-" * 80)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 5A: Heatmap of ground truth network
ax = axes[0, 0]
sns.heatmap(truth_network, cmap='Reds', cbar=True, square=True,
            linewidths=0.5, ax=ax, vmin=0, vmax=1,
            cbar_kws={'label': 'Causal Edge'})
ax.set_title('Ground Truth Causal Network', fontsize=14, fontweight='bold')
ax.set_xlabel('Target', fontsize=12)
ax.set_ylabel('Driver', fontsize=12)

# 5B: Heatmap of detected network
ax = axes[0, 1]
sns.heatmap(detected_network, cmap='Blues', cbar=True, square=True,
            linewidths=0.5, ax=ax, vmin=0, vmax=1,
            cbar_kws={'label': 'Detected Edge'})
ax.set_title('Detected Network (CCM, p < 0.01)', fontsize=14, fontweight='bold')
ax.set_xlabel('Target', fontsize=12)
ax.set_ylabel('Driver', fontsize=12)

# 5C: Heatmap of AUC values
ax = axes[1, 0]
auc_matrix = pd.DataFrame(np.nan, index=variables, columns=variables)
for _, row in results_df.iterrows():
    auc_matrix.loc[row['driver'], row['target']] = row['auc']

sns.heatmap(auc_matrix, cmap='viridis', cbar=True, square=True,
            linewidths=0.5, ax=ax, vmin=0, vmax=auc_matrix.max().max(),
            cbar_kws={'label': 'CCM AUC'})
ax.set_title('CCM AUC Values (All Pairs)', fontsize=14, fontweight='bold')
ax.set_xlabel('Target', fontsize=12)
ax.set_ylabel('Driver', fontsize=12)

# 5D: Comparison scatter plot (AUC vs p-value)
ax = axes[1, 1]
results_df['is_true_edge'] = results_df.apply(
    lambda row: truth_network.loc[row['driver'], row['target']] == 1,
    axis=1
)

# Plot false edges (gray)
false_edges = results_df[~results_df['is_true_edge']]
ax.scatter(false_edges['auc'], -np.log10(false_edges['p_value']),
          alpha=0.5, s=30, c='lightgray', label='No true edge')

# Plot true edges (red)
true_edge_data = results_df[results_df['is_true_edge']]
ax.scatter(true_edge_data['auc'], -np.log10(true_edge_data['p_value']),
          alpha=0.8, s=80, c='red', marker='*', label='True causal edge')

# Add significance threshold line
ax.axhline(-np.log10(0.01), color='blue', linestyle='--', linewidth=2,
          label='p = 0.01 threshold')

ax.set_xlabel('CCM AUC', fontsize=12)
ax.set_ylabel('-log10(p-value)', fontsize=12)
ax.set_title('CCM Performance: AUC vs Significance', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/EDMsystems/examples/ccm_validation_results.png', dpi=300, bbox_inches='tight')
print("Saved visualization to: examples/ccm_validation_results.png")
print()

print("=" * 80)
print("VIGNETTE COMPLETE!")
print("=" * 80)
print()
print("Summary:")
print(f"  - Analyzed {len(pairs)} variable pairs")
print(f"  - Used multivariate Fourier surrogates (preserves ACF and CCF)")
print(f"  - Detected {true_positives} out of {len(true_edges)} true edges")
print(f"  - Overall F1 score: {f1_score:.3f}")
print()
