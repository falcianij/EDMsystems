"""
Analysis functions for CCM results.

Provides functions for ground truth comparison, performance metrics,
and result visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def compare_to_ground_truth(results_df: pd.DataFrame,
                           ground_truth: pd.DataFrame,
                           significance_col: str = 'resolved_nonlinear') -> pd.DataFrame:
    """
    Compare CCM results to ground truth network.

    Parameters
    ----------
    results_df : pd.DataFrame
        CCM results with 'lib', 'target', and significance column
    ground_truth : pd.DataFrame
        Ground truth adjacency matrix (rows=drivers, cols=targets)
    significance_col : str, default 'resolved_nonlinear'
        Column indicating significant detection

    Returns
    -------
    pd.DataFrame
        Results with ground truth comparison
    """
    comparison = results_df.copy()

    # Add ground truth column
    comparison['true_edge'] = False
    comparison['detected'] = comparison[significance_col]

    for idx, row in comparison.iterrows():
        driver = row['lib']
        target = row['target']

        # Check if edge exists in ground truth
        if driver in ground_truth.index and target in ground_truth.columns:
            comparison.loc[idx, 'true_edge'] = bool(ground_truth.loc[driver, target])

    # Classification
    comparison['classification'] = 'TN'  # True Negative (default)
    comparison.loc[comparison['detected'] & comparison['true_edge'], 'classification'] = 'TP'  # True Positive
    comparison.loc[comparison['detected'] & ~comparison['true_edge'], 'classification'] = 'FP'  # False Positive
    comparison.loc[~comparison['detected'] & comparison['true_edge'], 'classification'] = 'FN'  # False Negative

    return comparison


def compute_performance_metrics(comparison: pd.DataFrame) -> Dict:
    """
    Compute performance metrics from ground truth comparison.

    Parameters
    ----------
    comparison : pd.DataFrame
        Output from compare_to_ground_truth

    Returns
    -------
    dict
        Metrics: TP, FP, TN, FN, precision, recall, F1, accuracy
    """
    TP = np.sum(comparison['classification'] == 'TP')
    FP = np.sum(comparison['classification'] == 'FP')
    TN = np.sum(comparison['classification'] == 'TN')
    FN = np.sum(comparison['classification'] == 'FN')

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    return {
        'TP': int(TP),
        'FP': int(FP),
        'TN': int(TN),
        'FN': int(FN),
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'n_edges_detected': int(TP + FP),
        'n_edges_true': int(TP + FN)
    }


def summarize_results(results_df: pd.DataFrame,
                     ground_truth: Optional[pd.DataFrame] = None,
                     significance_col: str = 'resolved_nonlinear',
                     print_summary: bool = True) -> Dict:
    """
    Summarize CCM results.

    Parameters
    ----------
    results_df : pd.DataFrame
        CCM results
    ground_truth : pd.DataFrame or None
        Ground truth network for comparison
    significance_col : str
        Significance column name
    print_summary : bool
        Print summary to console

    Returns
    -------
    dict
        Summary statistics
    """
    n_pairs = len(results_df)
    n_significant = np.sum(results_df[significance_col])
    n_convergent = np.sum(results_df['convergence'])

    summary = {
        'n_pairs': n_pairs,
        'n_convergent': n_convergent,
        'n_significant': int(n_significant),
        'pct_convergent': n_convergent / n_pairs * 100 if n_pairs > 0 else 0,
        'pct_significant': n_significant / n_pairs * 100 if n_pairs > 0 else 0,
        'mean_rho': np.nanmean(results_df['ccm_rho'])
    }

    if ground_truth is not None:
        comparison = compare_to_ground_truth(results_df, ground_truth, significance_col)
        metrics = compute_performance_metrics(comparison)
        summary.update(metrics)

    if print_summary:
        print("="*70)
        print("CCM RESULTS SUMMARY")
        print("="*70)
        print(f"Total pairs tested: {summary['n_pairs']}")
        print(f"Convergent: {summary['n_convergent']} ({summary['pct_convergent']:.1f}%)")
        print(f"Significant: {summary['n_significant']} ({summary['pct_significant']:.1f}%)")
        print(f"Mean rho: {summary['mean_rho']:.3f}")

        if ground_truth is not None:
            print()
            print("Ground Truth Comparison:")
            print(f"  Precision: {summary['precision']:.3f}")
            print(f"  Recall: {summary['recall']:.3f}")
            print(f"  F1 Score: {summary['f1_score']:.3f}")
            print(f"  Accuracy: {summary['accuracy']:.3f}")
            print(f"  TP={summary['TP']}, FP={summary['FP']}, TN={summary['TN']}, FN={summary['FN']}")
        print("="*70)

    return summary


def create_adjacency_matrix(results_df: pd.DataFrame,
                           significance_col: str = 'resolved_nonlinear',
                           value_col: str = 'ccm_norm') -> pd.DataFrame:
    """
    Create adjacency matrix from CCM results.

    Parameters
    ----------
    results_df : pd.DataFrame
        CCM results
    significance_col : str
        Column for significance filtering
    value_col : str
        Column for edge weights (or 'binary' for 1/0)

    Returns
    -------
    pd.DataFrame
        Adjacency matrix
    """
    # Get unique variables
    all_vars = sorted(set(results_df['lib'].unique()) | set(results_df['target'].unique()))

    # Initialize matrix
    adj_matrix = pd.DataFrame(0.0, index=all_vars, columns=all_vars)

    # Fill in detected edges
    for _, row in results_df.iterrows():
        if row[significance_col]:
            driver = row['lib']
            target = row['target']

            if value_col == 'binary':
                adj_matrix.loc[driver, target] = 1.0
            else:
                adj_matrix.loc[driver, target] = row[value_col]

    return adj_matrix
