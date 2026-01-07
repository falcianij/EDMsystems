"""
Comprehensive CCM Workflow Example

Demonstrates complete CCM analysis workflow with:
- Test data generation with known ground truth
- Preprocessing (detrending and normalization)
- Parameter optimization
- Parallel CCM analysis across pairs
- Surrogate testing
- Ground truth comparison
- Visualization

Based on: 011_ccm_analysis_norm_v3.ipynb and Surrogate Test 11-20-25.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from edmsystems.testdata import make_test_dataframe, get_ground_truth_network
from edmsystems.preprocessing import preprocess_for_ccm, inverse_preprocess
from edmsystems.ccm import run_ccm_workflow, summarize_results, compare_to_ground_truth

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("CCM WORKFLOW EXAMPLE")
print("="*80)
print()

# ============================================================================
# STEP 1: Generate test data with known ground truth
# ============================================================================

print("STEP 1: Generating test data...")
print("-"*80)

# Generate test data (10 scenarios, 500 time points)
df = make_test_dataframe(n=500, seed=42)
print(f"Generated data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Get ground truth network
truth_network = get_ground_truth_network()
print(f"\nGround truth network shape: {truth_network.shape}")
print(f"True causal edges: {truth_network.sum().sum()}")
print()

# ============================================================================
# STEP 2: Visualize raw time series
# ============================================================================

print("STEP 2: Visualizing time series...")
print("-"*80)

# Plot a subset of time series
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Independent (non-causal)
axes[0].plot(df['independent_X'], label='X', alpha=0.7)
axes[0].plot(df['independent_Y'], label='Y', alpha=0.7)
axes[0].set_title('Scenario 1: Independent (Non-causal)', fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Unidirectional causal (X → Y)
axes[1].plot(df['unidirectional_X'], label='X', alpha=0.7)
axes[1].plot(df['unidirectional_Y'], label='Y', alpha=0.7)
axes[1].set_title('Scenario 8: Unidirectional Causal (X → Y)', fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# Bidirectional causal (X ↔ Y)
axes[2].plot(df['bidirectional_X'], label='X', alpha=0.7)
axes[2].plot(df['bidirectional_Y'], label='Y', alpha=0.7)
axes[2].set_title('Scenario 9: Bidirectional Causal (X ↔ Y)', fontweight='bold')
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)
axes[2].set_xlabel('Time', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/user/EDMsystems/vignettes/time_series_examples.png', dpi=150)
print("  Saved: vignettes/time_series_examples.png")
plt.close()

# ============================================================================
# STEP 3: Preprocess data
# ============================================================================

print("\nSTEP 3: Preprocessing data...")
print("-"*80)

# Preprocess all variables (detrend + normalize)
preprocessed_data = {}
transform_params_dict = {}

variables = [col for col in df.columns if col != 'datetime']

for var in variables:
    processed, params = preprocess_for_ccm(
        df[var].values,
        detrend_method='linear',
        normalize_method='zscore'
    )
    preprocessed_data[var] = processed
    transform_params_dict[var] = params

# Create preprocessed dataframe
df_processed = pd.DataFrame(preprocessed_data)
df_processed.insert(0, 'time', range(len(df_processed)))

print(f"Preprocessed data shape: {df_processed.shape}")
print(f"Example stats (unidirectional_X): mean={np.mean(df_processed['unidirectional_X']):.3f}, std={np.std(df_processed['unidirectional_X']):.3f}")
print()

# ============================================================================
# STEP 4: Run CCM workflow
# ============================================================================

print("STEP 4: Running CCM workflow...")
print("-"*80)
print("Note: This uses multiprocessing for parallel analysis across pairs")
print()

# Define specific pairs to test (or use None for all permutations)
test_pairs = [
    # Causal pairs (should be detected)
    ('unidirectional_X', 'unidirectional_Y'),
    ('bidirectional_X', 'bidirectional_Y'),
    ('bidirectional_Y', 'bidirectional_X'),
    ('weak_causal_X', 'weak_causal_Y'),

    # Non-causal pairs (should NOT be detected)
    ('independent_X', 'independent_Y'),
    ('correlated_X', 'correlated_Y'),
    ('lagcorr_X', 'lagcorr_Y'),
]

# Run CCM workflow with parallel processing
results = run_ccm_workflow(
    df_processed,
    pairs=test_pairs,
    auto_optimize=True,  # Optimize parameters per pair
    libSizes="50 400 50",
    sample=50,           # Random samples per library size
    n_surrogates=99,     # Twin surrogates for significance testing
    n_jobs=4,            # Parallel jobs (adjust based on CPU cores)
    seed=42,
    verbose=True
)

print()
print("Results:")
print(results[['lib', 'target', 'E', 'tau', 'Tp', 'ccm_rho', 'convergence', 'p_rho', 'resolved_nonlinear']])
print()

# ============================================================================
# STEP 5: Compare to ground truth
# ============================================================================

print("STEP 5: Comparing to ground truth...")
print("-"*80)

summary = summarize_results(results, truth_network, print_summary=True)
comparison = compare_to_ground_truth(results, truth_network)

print("\nDetailed classifications:")
print(comparison[['lib', 'target', 'detected', 'true_edge', 'classification']])
print()

# ============================================================================
# STEP 6: Visualize results
# ============================================================================

print("STEP 6: Creating visualizations...")
print("-"*80)

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Panel A: Confusion Matrix
ax_cm = fig.add_subplot(gs[0, 0])
metrics = summary
cm = np.array([[metrics['TP'], metrics['FP']],
               [metrics['FN'], metrics['TN']]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
            xticklabels=['Detected', 'Not Detected'],
            yticklabels=['True Edge', 'No Edge'])
ax_cm.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
ax_cm.set_xlabel('Predicted', fontweight='bold')
ax_cm.set_ylabel('Actual', fontweight='bold')

# Panel B: Rho distribution (detected vs non-detected)
ax_rho = fig.add_subplot(gs[0, 1])
detected = results[results['resolved_nonlinear']]['ccm_rho'].values
not_detected = results[~results['resolved_nonlinear']]['ccm_rho'].values

if len(detected) > 0:
    ax_rho.hist(detected, bins=10, alpha=0.7, label='Significant', color='green', edgecolor='black')
if len(not_detected) > 0:
    ax_rho.hist(not_detected, bins=10, alpha=0.7, label='Not significant', color='red', edgecolor='black')

ax_rho.set_xlabel('CCM Rho', fontweight='bold')
ax_rho.set_ylabel('Count', fontweight='bold')
ax_rho.set_title('Rho Distribution by Significance', fontsize=12, fontweight='bold')
ax_rho.legend()
ax_rho.grid(True, alpha=0.3, axis='y')

# Panel C: P-values by true edge status
ax_pval = fig.add_subplot(gs[0, 2])
true_edge_p = comparison[comparison['true_edge']]['p_rho'].values
false_edge_p = comparison[~comparison['true_edge']]['p_rho'].values

if len(true_edge_p) > 0:
    ax_pval.hist(true_edge_p, bins=10, alpha=0.7, label='True edge', color='blue', edgecolor='black')
if len(false_edge_p) > 0:
    ax_pval.hist(false_edge_p, bins=10, alpha=0.7, label='False edge', color='orange', edgecolor='black')

ax_pval.axvline(0.05, color='red', linestyle='--', linewidth=2, label='p=0.05')
ax_pval.set_xlabel('P-value', fontweight='bold')
ax_pval.set_ylabel('Count', fontweight='bold')
ax_pval.set_title('P-value Distribution', fontsize=12, fontweight='bold')
ax_pval.legend()
ax_pval.grid(True, alpha=0.3, axis='y')

# Panel D-F: Example CCM curves (convergent and non-convergent)
# (This would require storing convergence curves - simplified here)

# Panel G: Performance metrics bar chart
ax_perf = fig.add_subplot(gs[1, :])
metrics_names = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
metrics_values = [metrics['precision'], metrics['recall'], metrics['f1_score'], metrics['accuracy']]
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

bars = ax_perf.bar(metrics_names, metrics_values, color=colors, edgecolor='black', linewidth=1.5)
ax_perf.set_ylim(0, 1.1)
ax_perf.set_ylabel('Score', fontweight='bold', fontsize=12)
ax_perf.set_title('Performance Metrics', fontsize=13, fontweight='bold')
ax_perf.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, metrics_values):
    height = bar.get_height()
    ax_perf.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# Panel H: Ground truth network heatmap
ax_gt = fig.add_subplot(gs[2, 0])
# Filter to only tested variables
tested_vars = sorted(set(results['lib'].unique()) | set(results['target'].unique()))
truth_subset = truth_network.loc[tested_vars, tested_vars]
sns.heatmap(truth_subset, cmap='RdYlGn', center=0, vmin=0, vmax=1,
            cbar_kws={'label': 'Edge'}, ax=ax_gt, square=True)
ax_gt.set_title('Ground Truth Network', fontsize=12, fontweight='bold')
ax_gt.set_xlabel('Target', fontweight='bold')
ax_gt.set_ylabel('Driver', fontweight='bold')

# Panel I: Detected network heatmap
ax_det = fig.add_subplot(gs[2, 1])
detected_matrix = pd.DataFrame(0, index=tested_vars, columns=tested_vars)
for _, row in results[results['resolved_nonlinear']].iterrows():
    if row['lib'] in tested_vars and row['target'] in tested_vars:
        detected_matrix.loc[row['lib'], row['target']] = 1

sns.heatmap(detected_matrix, cmap='RdYlGn', center=0, vmin=0, vmax=1,
            cbar_kws={'label': 'Edge'}, ax=ax_det, square=True)
ax_det.set_title('Detected Network', fontsize=12, fontweight='bold')
ax_det.set_xlabel('Target', fontweight='bold')
ax_det.set_ylabel('Driver', fontweight='bold')

# Panel J: Statistics text box
ax_stats = fig.add_subplot(gs[2, 2])
ax_stats.axis('off')

stats_text = f"""
SUMMARY STATISTICS

Total Pairs: {metrics['n_pairs']}
Convergent: {metrics['n_convergent']} ({metrics['pct_convergent']:.1f}%)
Significant: {metrics['n_significant']} ({metrics['pct_significant']:.1f}%)

True Positives: {metrics['TP']}
False Positives: {metrics['FP']}
False Negatives: {metrics['FN']}
True Negatives: {metrics['TN']}

Precision: {metrics['precision']:.3f}
Recall: {metrics['recall']:.3f}
F1 Score: {metrics['f1_score']:.3f}
Accuracy: {metrics['accuracy']:.3f}
"""

ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes,
             fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')

plt.suptitle('CCM Analysis Results', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('/home/user/EDMsystems/vignettes/ccm_results_summary.png', dpi=150, bbox_inches='tight')
print("  Saved: vignettes/ccm_results_summary.png")
plt.close()

# ============================================================================
# STEP 7: Demonstrate inverse transformation
# ============================================================================

print("\nSTEP 7: Demonstrating inverse transformation...")
print("-"*80)

# Take a sample prediction (simulated)
var_name = 'unidirectional_Y'
predictions_transformed = np.random.randn(10)  # Simulated predictions in transformed space

# Transform back to original scale
predictions_original = inverse_preprocess(predictions_transformed, transform_params_dict[var_name])

print(f"Predictions (transformed): {predictions_transformed[:5]}")
print(f"Predictions (original scale): {predictions_original[:5]}")
print("\n✓ Inverse transformation successful!")
print()

print("="*80)
print("WORKFLOW COMPLETE")
print("="*80)
print("\nResults saved to:")
print("  - vignettes/time_series_examples.png")
print("  - vignettes/ccm_results_summary.png")
print()
print("Key findings:")
print(f"  - Detected {metrics['n_significant']}/{metrics['n_pairs']} significant causal relationships")
print(f"  - Precision: {metrics['precision']:.3f} (how many detected edges are real?)")
print(f"  - Recall: {metrics['recall']:.3f} (how many real edges were detected?)")
print(f"  - F1 Score: {metrics['f1_score']:.3f} (harmonic mean of precision and recall)")
print()
