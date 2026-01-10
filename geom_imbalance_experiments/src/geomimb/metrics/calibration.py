"""Calibration and threshold tuning utilities."""

import numpy as np
from sklearn.calibration import calibration_curve
from typing import Dict, Tuple, Optional
import logging

from .classification import compute_metrics, apply_threshold

def tune_threshold_for_operating_point(
    scores: np.ndarray,
    y_true: np.ndarray,
    target_metric: str = 'precision',
    target_value: float = 0.95,
    constraint_type: str = 'min',
    optimize_metric: str = 'recall'
) -> Tuple[float, Dict]:
    """
    Tune threshold to achieve a target operating point.

    Parameters
    ----------
    scores : np.ndarray
        Decision scores
    y_true : np.ndarray
        True labels
    target_metric : str
        Metric to constrain (e.g., 'precision')
    target_value : float
        Target value for the metric
    constraint_type : str
        'min' for >= constraint, 'max' for <= constraint
    optimize_metric : str
        Metric to optimize subject to constraint

    Returns
    -------
    threshold : float
        Selected threshold
    info : dict
        Information about the search
    """
    # Get unique thresholds
    thresholds = np.unique(scores)
    n_thresh = len(thresholds)

    if n_thresh > 200:
        # Subsample if too many
        idx = np.linspace(0, n_thresh - 1, 200, dtype=int)
        thresholds = thresholds[idx]

    best_threshold = np.inf
    best_optimize_value = -np.inf if constraint_type == 'min' else np.inf
    feasible = False
    results = []

    for threshold in thresholds:
        y_pred = apply_threshold(scores, threshold)
        metrics = compute_metrics(y_true, y_pred)

        target_met = metrics[target_metric]
        optimize_val = metrics[optimize_metric]

        # Check constraint
        if constraint_type == 'min':
            constraint_satisfied = target_met >= target_value
        else:
            constraint_satisfied = target_met <= target_value

        results.append({
            'threshold': threshold,
            target_metric: target_met,
            optimize_metric: optimize_val,
            'feasible': constraint_satisfied
        })

        # Update best if constraint satisfied
        if constraint_satisfied:
            feasible = True
            if ((constraint_type == 'min' and optimize_val > best_optimize_value) or
                (constraint_type == 'max' and optimize_val < best_optimize_value)):
                best_optimize_value = optimize_val
                best_threshold = threshold

    info = {
        'feasible': feasible,
        'best_threshold': best_threshold,
        f'best_{optimize_metric}': best_optimize_value if feasible else 0.0,
        'n_thresholds_tried': len(thresholds),
        'results': results
    }

    if not feasible:
        logging.warning(f"No threshold achieves {target_metric} {constraint_type} {target_value}")
        # Return threshold that gives best target_metric
        target_values = [r[target_metric] for r in results]
        if constraint_type == 'min':
            best_idx = np.argmax(target_values)
        else:
            best_idx = np.argmin(target_values)
        best_threshold = results[best_idx]['threshold']
        info['fallback_threshold'] = best_threshold
        info[f'fallback_{target_metric}'] = results[best_idx][target_metric]

    return best_threshold, info

def calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> Dict:
    """
    Compute calibration metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities
    n_bins : int
        Number of calibration bins

    Returns
    -------
    dict
        Calibration metrics
    """
    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy='uniform'
    )

    # Expected calibration error (ECE)
    bin_counts = np.histogram(y_proba, bins=n_bins)[0]
    bin_weights = bin_counts / len(y_proba)

    # Filter out empty bins
    non_empty = bin_counts > 0
    if non_empty.sum() > 0:
        ece = np.sum(bin_weights[non_empty] *
                    np.abs(fraction_of_positives - mean_predicted_value[non_empty]))
    else:
        ece = 0.0

    # Maximum calibration error (MCE)
    if len(fraction_of_positives) > 0:
        mce = np.max(np.abs(fraction_of_positives - mean_predicted_value))
    else:
        mce = 0.0

    return {
        'ece': ece,
        'mce': mce,
        'fraction_of_positives': fraction_of_positives,
        'mean_predicted_value': mean_predicted_value,
        'n_bins': n_bins,
        'non_empty_bins': non_empty.sum() if non_empty.size > 0 else 0
    }

def compute_threshold_from_costs(
    pi: float,
    c10: float = 1.0,
    c01: float = 1.0
) -> float:
    """
    Compute theoretical threshold from costs and prevalence.

    For logit scale: threshold = log(c10/c01)

    Parameters
    ----------
    pi : float
        Prevalence
    c10 : float
        Cost of false positive
    c01 : float
        Cost of false negative

    Returns
    -------
    float
        Threshold in logit space
    """
    return np.log(c10 / c01)

def validate_operating_point(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_precision: float = 0.95
) -> Dict:
    """
    Validate if predictions meet operating point requirements.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predictions
    target_precision : float
        Target precision

    Returns
    -------
    dict
        Validation results
    """
    metrics = compute_metrics(y_true, y_pred)

    return {
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'meets_target': metrics['precision'] >= target_precision,
        'precision_gap': target_precision - metrics['precision']
    }