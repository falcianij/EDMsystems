"""
Standardized CCM Workflow Vignette

This script demonstrates the complete workflow for convergent cross mapping
analysis using the edmsystems package. It shows how to:

1. Generate test data with known ground truth
2. Run CCM analysis with parameter optimization
3. Test significance using surrogate methods
4. Compare detected network to ground truth
5. Visualize results

This workflow provides a standardized, reproducible approach for causal
inference in dynamical systems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from edmsystems.testdata import make_test_dataframe, get_ground_truth_network
from edmsystems.ccm import run_ccm_workflow, test_ccm_pair
from edmsystems.ccm.analysis import compare_to_ground_truth, compute_performance_metrics, summarize_results

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("STANDARDIZED CCM WORKFLOW VIGNETTE")
print("="*80)
print()

# ============================================================================
# STEP 1: Generate test data with known ground truth
# ============================================================================

print("STEP 1: Generating test data with known ground truth")
print("-"*80)

# Generate 500 time points of test data with datetime
df = make_test_dataframe(n=500, seed=42, start_date='2000-01-01', freq='D')

print(f"Generated dataframe with {len(df)} time points")
print(f"Columns: {list(df.columns)}")
print()

# Get ground truth network
truth_network = get_ground_truth_network()

print("Ground truth causal edges:")
true_edges = []
for driver in truth_network.index:
    for target in truth_network.columns:
        if truth_network.loc[driver, target] == 1:
            true_edges.append(f"  {driver} -> {target}")

for edge in true_edges:
    print(edge)

print(f"\nTotal true edges: {len(true_edges)}")
print()

# ============================================================================
# STEP 2: Example - Test a single pair with full workflow
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Example - Test a single causal pair")
print("="*80)

# Test unidirectional causal pair
X = df['unidirectional_X'].values
Y = df['unidirectional_Y'].values

result = test_ccm_pair(
    X, Y,
    driver_name='unidirectional_X',
    target_name='unidirectional_Y',
    libSizes="50 500 50",  # Reduced for faster demo
    sample=50,              # Reduced for faster demo
    n_surrogates=19,        # Reduced for faster demo
    surrogate_method='twin',
    optimize_params=True,
    optimize_theta=False,
    seed=42,
    verbose=True
)

print("\nResult summary:")
print(f"  Parameters: tau={result['tau']}, E={result['E']}, Tp={result['Tp']}")
print(f"  Rho (max lib): {result['rho_original']:.3f}")
print(f"  Rho (surrogates): {result['rho_surrogate_mean_twin']:.3f} ± {result['rho_surrogate_std_twin']:.3f}")
print(f"  p-value: {result['p_value_twin']:.4f}")
print(f"  Significant: {result['is_significant_twin']}")
print(f"  Convergent: {result['convergent']}")

# ============================================================================
# STEP 3: Run workflow on selected pairs
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Running CCM on selected pairs")
print("="*80)
print()

# Select a subset of pairs for demonstration
# In practice, you could test all pairs or specify custom pairs
test_pairs = [
    # True causal (should be detected)
    ('unidirectional_X', 'unidirectional_Y'),
    ('bidirectional_X', 'bidirectional_Y'),
    ('bidirectional_Y', 'bidirectional_X'),
    ('weak_X', 'weak_Y'),

    # Non-causal (should NOT be detected)
    ('independent_X', 'independent_Y'),
    ('correlated_X', 'correlated_Y'),
    ('autocorr_X', 'autocorr_Y'),
    ('seasonal_sync_X', 'seasonal_sync_Y'),
]

print(f"Testing {len(test_pairs)} pairs...")
print()

# Run workflow
results = run_ccm_workflow(
    df,
    pairs=test_pairs,
    datetime_col='datetime',
    libSizes="50 500 50",
    sample=50,
    n_surrogates=19,
    surrogate_method='twin',
    optimize_params=True,
    optimize_theta=False,
    exclusionRadius=0,
    n_jobs=-1,            # Parallel across pairs (use all cores)
    seed=42,
    verbose=False,        # Quiet mode
)

print("\nResults:")
print(results[['driver', 'target', 'tau', 'E', 'Tp', 'rho_original',
               'p_value_twin', 'is_significant_twin', 'convergent']])

# ============================================================================
# STEP 4: Compare to ground truth
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Comparing results to ground truth")
print("="*80)
print()

# Summarize results with ground truth comparison
summary = summarize_results(results, truth_network, surrogate_method='twin', print_summary=True)

# Get detailed comparison
comparison = compare_to_ground_truth(results, truth_network, surrogate_method='twin')

print("\nDetailed classifications:")
print(comparison[['driver', 'target', 'detected', 'true_edge', 'classification']])

# Show false positives
fp = comparison[comparison['classification'] == 'FP']
if len(fp) > 0:
    print("\nFalse positives (incorrectly detected as causal):")
    for _, row in fp.iterrows():
        print(f"  {row['driver']} -> {row['target']}")
else:
    print("\nNo false positives!")

# Show false negatives
fn = comparison[comparison['classification'] == 'FN']
if len(fn) > 0:
    print("\nFalse negatives (missed true causal edges):")
    for _, row in fn.iterrows():
        print(f"  {row['driver']} -> {row['target']}")
else:
    print("\nNo false negatives!")

# ============================================================================
# STEP 5: Visualize results
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Visualizing results")
print("="*80)
print()

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel A: Performance metrics
ax = axes[0, 0]
metrics = summary['performance_metrics']
metric_names = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
metric_values = [metrics['precision'], metrics['recall'],
                metrics['f1_score'], metrics['accuracy']]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylim([0, 1])
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Performance Metrics', fontsize=14, fontweight='bold')
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontweight='bold')

# Panel B: Confusion matrix
ax = axes[0, 1]
confusion = np.array([[metrics['TP'], metrics['FP']],
                     [metrics['FN'], metrics['TN']]])
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Detected', 'Not Detected'],
            yticklabels=['True Edge', 'No Edge'],
            cbar_kws={'label': 'Count'},
            ax=ax, vmin=0, linewidths=2, linecolor='black')
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual', fontsize=12, fontweight='bold')

# Panel C: Rho distribution (detected vs non-detected)
ax = axes[1, 0]
detected = results[results['is_significant_twin']]['rho_original'].values
not_detected = results[~results['is_significant_twin']]['rho_original'].values

if len(detected) > 0:
    ax.hist(detected, bins=10, alpha=0.7, label='Significant',
            color='green', edgecolor='black')
if len(not_detected) > 0:
    ax.hist(not_detected, bins=10, alpha=0.7, label='Not significant',
            color='red', edgecolor='black')

ax.set_xlabel('Rho (max lib)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Rho Distribution by Significance', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel D: P-values by true edge status
ax = axes[1, 1]
true_edge_p = comparison[comparison['true_edge']]['p_value'].values
no_edge_p = comparison[~comparison['true_edge']]['p_value'].values

positions = [1, 2]
data_to_plot = [true_edge_p[~np.isnan(true_edge_p)],
                no_edge_p[~np.isnan(no_edge_p)]]

bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                patch_artist=True, showmeans=True,
                boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=2),
                medianprops=dict(color='red', linewidth=2),
                meanprops=dict(marker='o', markerfacecolor='green', markersize=8),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))

ax.axhline(y=0.01, color='red', linestyle='--', linewidth=2, label='p = 0.01 threshold')
ax.set_xticklabels(['True Edge', 'No Edge'])
ax.set_ylabel('p-value', fontsize=12, fontweight='bold')
ax.set_title('P-value Distribution by Ground Truth', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/user/EDMsystems/vignettes/ccm_workflow_results.png',
            dpi=300, bbox_inches='tight')
print("Saved figure to: vignettes/ccm_workflow_results.png")

plt.show()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("VIGNETTE COMPLETE!")
print("="*80)
print()
print("This workflow demonstrated:")
print("  1. Test data generation with known ground truth")
print("  2. Parameter optimization (tau, E, Tp)")
print("  3. CCM with random library sampling")
print("  4. Twin surrogate significance testing")
print("  5. Ground truth comparison and validation")
print("  6. Comprehensive visualization")
print()
print("Key features of this standardized workflow:")
print("  ✓ Automatic parameter optimization")
print("  ✓ Multiple surrogate methods available")
print("  ✓ Parallel processing support")
print("  ✓ Optional theta optimization via S-map")
print("  ✓ Toggleable verbose output and plots")
print("  ✓ Variable library sizes for efficiency")
print("  ✓ Selective pair calculation")
print("  ✓ Ground truth validation")
print("="*80)
