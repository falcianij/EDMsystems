"""
Analysis and validation functions for CCM results.

Provides tools for comparing detected causal networks to ground truth
and computing performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict


def compare_to_ground_truth(results_df: pd.DataFrame,
                           ground_truth: pd.DataFrame,
                           significance_col: str = 'is_significant') -> pd.DataFrame:
    """
    Compare detected causal network to ground truth.

    Parameters
    ----------
    results_df : pd.DataFrame
        CCM results with columns: driver, target, and significance indicator
    ground_truth : pd.DataFrame
        Ground truth adjacency matrix (index=drivers, columns=targets)
        Values of 1 indicate true causal edge, 0 indicates no causation
    significance_col : str, default 'is_significant'
        Column in results_df indicating significance

    Returns
    -------
    pd.DataFrame
        Comparison dataframe with columns:
        - driver, target: variable names
        - detected: boolean (detected as significant)
        - true_edge: boolean (true causal edge exists)
        - classification: 'TP', 'FP', 'TN', or 'FN'

    Examples
    --------
    >>> comparison = compare_to_ground_truth(results, truth_network)
    >>> print(comparison[comparison['classification'] == 'FP'])  # False positives
    """
    comparisons = []

    for _, row in results_df.iterrows():
        driver = row['driver']
        target = row['target']
        detected = row[significance_col]

        # Get ground truth (handle missing entries as no edge)
        try:
            true_edge = ground_truth.loc[driver, target] == 1
        except KeyError:
            true_edge = False

        # Classify
        if true_edge and detected:
            classification = 'TP'  # True Positive
        elif not true_edge and detected:
            classification = 'FP'  # False Positive
        elif not true_edge and not detected:
            classification = 'TN'  # True Negative
        else:  # true_edge and not detected
            classification = 'FN'  # False Negative

        comparisons.append({
            'driver': driver,
            'target': target,
            'detected': detected,
            'true_edge': true_edge,
            'classification': classification,
        })

    return pd.DataFrame(comparisons)


def compute_performance_metrics(comparison_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute performance metrics from comparison dataframe.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output from compare_to_ground_truth()

    Returns
    -------
    dict
        Performance metrics:
        - TP, FP, TN, FN: counts
        - precision (PPV): TP / (TP + FP)
        - recall (TPR, sensitivity): TP / (TP + FN)
        - specificity (TNR): TN / (TN + FP)
        - f1_score: harmonic mean of precision and recall
        - accuracy: (TP + TN) / (TP + TN + FP + FN)
        - mcc: Matthews correlation coefficient

    Examples
    --------
    >>> metrics = compute_performance_metrics(comparison)
    >>> print(f"F1 Score: {metrics['f1_score']:.3f}")
    """
    # Count classifications
    TP = (comparison_df['classification'] == 'TP').sum()
    FP = (comparison_df['classification'] == 'FP').sum()
    TN = (comparison_df['classification'] == 'TN').sum()
    FN = (comparison_df['classification'] == 'FN').sum()

    # Compute metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    # Matthews correlation coefficient
    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = numerator / denominator if denominator > 0 else 0

    return {
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'mcc': mcc,
    }


def summarize_results(results_df: pd.DataFrame,
                     ground_truth: Optional[pd.DataFrame] = None,
                     print_summary: bool = True) -> Dict:
    """
    Summarize CCM results with optional ground truth comparison.

    Parameters
    ----------
    results_df : pd.DataFrame
        CCM results dataframe
    ground_truth : pd.DataFrame or None
        Ground truth adjacency matrix (if available)
    print_summary : bool, default True
        Print formatted summary to console

    Returns
    -------
    dict
        Summary dictionary containing:
        - n_pairs_tested: number of pairs tested
        - n_significant: number of significant results
        - n_convergent: number showing convergence
        - If ground_truth provided: performance_metrics dict

    Examples
    --------
    >>> summary = summarize_results(results, truth_network)
    """
    n_pairs = len(results_df)
    n_significant = results_df['is_significant'].sum()
    n_convergent = results_df['convergent'].sum()

    summary = {
        'n_pairs_tested': n_pairs,
        'n_significant': n_significant,
        'n_convergent': n_convergent,
        'mean_rho': results_df['rho_mean'].mean(),
        'mean_auc': results_df['auc_original'].mean(),
    }

    if ground_truth is not None:
        comparison = compare_to_ground_truth(results_df, ground_truth)
        metrics = compute_performance_metrics(comparison)
        summary['performance_metrics'] = metrics

    if print_summary:
        print("\n" + "="*70)
        print("CCM RESULTS SUMMARY")
        print("="*70)
        print(f"Pairs tested: {n_pairs}")
        print(f"Significant results (p < 0.01): {n_significant} ({100*n_significant/n_pairs:.1f}%)")
        print(f"Convergent results: {n_convergent} ({100*n_convergent/n_pairs:.1f}%)")
        print(f"Mean rho: {summary['mean_rho']:.3f}")
        print(f"Mean AUC: {summary['mean_auc']:.2f}")

        if ground_truth is not None:
            print("\n" + "-"*70)
            print("PERFORMANCE vs GROUND TRUTH")
            print("-"*70)
            m = metrics
            print(f"True Positives:  {m['TP']}")
            print(f"False Positives: {m['FP']}")
            print(f"True Negatives:  {m['TN']}")
            print(f"False Negatives: {m['FN']}")
            print()
            print(f"Precision: {m['precision']:.3f}")
            print(f"Recall:    {m['recall']:.3f}")
            print(f"F1 Score:  {m['f1_score']:.3f}")
            print(f"Accuracy:  {m['accuracy']:.3f}")
            print(f"MCC:       {m['mcc']:.3f}")

        print("="*70 + "\n")

    return summary
