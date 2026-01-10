"""Model stability metrics across seeds and conditions."""

import numpy as np
from typing import List, Dict, Tuple
import logging

def coefficient_angle(u: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
    """
    Compute angle between two coefficient vectors.

    Parameters
    ----------
    u, v : np.ndarray
        Coefficient vectors

    Returns
    -------
    angle_rad : float
        Angle in radians
    angle_deg : float
        Angle in degrees
    """
    # Normalize vectors
    u_norm = u / np.linalg.norm(u)
    v_norm = v / np.linalg.norm(v)

    # Compute cosine similarity
    cos_sim = np.clip(np.dot(u_norm, v_norm), -1, 1)

    # Convert to angle
    angle_rad = np.arccos(cos_sim)
    angle_deg = np.degrees(angle_rad)

    return angle_rad, angle_deg

def compute_coefficient_stability(coefficients: List[np.ndarray]) -> Dict:
    """
    Compute stability metrics for a list of coefficient vectors.

    Parameters
    ----------
    coefficients : List[np.ndarray]
        List of coefficient vectors from different runs

    Returns
    -------
    dict
        Stability metrics including pairwise angles
    """
    n_models = len(coefficients)
    if n_models < 2:
        return {'n_models': n_models, 'angles_deg': [], 'angles_rad': []}

    # Compute all pairwise angles
    angles_rad = []
    angles_deg = []

    for i in range(n_models):
        for j in range(i + 1, n_models):
            angle_rad, angle_deg = coefficient_angle(coefficients[i], coefficients[j])
            angles_rad.append(angle_rad)
            angles_deg.append(angle_deg)

    angles_rad = np.array(angles_rad)
    angles_deg = np.array(angles_deg)

    return {
        'n_models': n_models,
        'angles_rad': angles_rad,
        'angles_deg': angles_deg,
        'mean_angle_deg': angles_deg.mean(),
        'std_angle_deg': angles_deg.std(),
        'max_angle_deg': angles_deg.max(),
        'min_angle_deg': angles_deg.min(),
    }

def compute_score_stability(
    scores_list: List[np.ndarray],
    return_per_sample: bool = False
) -> Dict:
    """
    Compute stability of scores across multiple runs.

    Parameters
    ----------
    scores_list : List[np.ndarray]
        List of score arrays from different runs (same samples)
    return_per_sample : bool
        If True, return per-sample variance

    Returns
    -------
    dict
        Stability metrics
    """
    if len(scores_list) < 2:
        return {
            'n_runs': len(scores_list),
            'mean_variance': 0.0,
            'std_variance': 0.0,
        }

    # Stack scores: shape (n_runs, n_samples)
    scores_array = np.array(scores_list)

    # Compute per-sample variance across runs
    per_sample_variance = np.var(scores_array, axis=0)

    result = {
        'n_runs': len(scores_list),
        'mean_variance': per_sample_variance.mean(),
        'std_variance': per_sample_variance.std(),
        'max_variance': per_sample_variance.max(),
        'min_variance': per_sample_variance.min(),
    }

    if return_per_sample:
        result['per_sample_variance'] = per_sample_variance

    return result

def compute_neff_stability(neff_values: List[float]) -> Dict:
    """
    Compute stability of effective sample size across runs.

    Parameters
    ----------
    neff_values : List[float]
        Effective sample sizes from different runs

    Returns
    -------
    dict
        Stability metrics
    """
    neff_array = np.array(neff_values)

    return {
        'n_runs': len(neff_values),
        'mean_neff': neff_array.mean(),
        'std_neff': neff_array.std(),
        'min_neff': neff_array.min(),
        'max_neff': neff_array.max(),
        'cv_neff': neff_array.std() / neff_array.mean() if neff_array.mean() > 0 else np.inf,
    }

def check_monotonicity(values: np.ndarray, x_values: np.ndarray = None) -> Dict:
    """
    Check if values are monotonic with respect to x_values.

    Parameters
    ----------
    values : np.ndarray
        Values to check
    x_values : np.ndarray, optional
        X values (if None, use indices)

    Returns
    -------
    dict
        Monotonicity information
    """
    if x_values is None:
        x_values = np.arange(len(values))

    # Sort by x_values
    sort_idx = np.argsort(x_values)
    x_sorted = x_values[sort_idx]
    y_sorted = values[sort_idx]

    # Check monotonicity
    diffs = np.diff(y_sorted)
    is_increasing = np.all(diffs >= 0)
    is_decreasing = np.all(diffs <= 0)
    is_strictly_increasing = np.all(diffs > 0)
    is_strictly_decreasing = np.all(diffs < 0)

    # Count violations
    increasing_violations = np.sum(diffs < 0)
    decreasing_violations = np.sum(diffs > 0)

    return {
        'is_monotonic': is_increasing or is_decreasing,
        'is_increasing': is_increasing,
        'is_decreasing': is_decreasing,
        'is_strictly_increasing': is_strictly_increasing,
        'is_strictly_decreasing': is_strictly_decreasing,
        'increasing_violations': int(increasing_violations),
        'decreasing_violations': int(decreasing_violations),
        'total_comparisons': len(diffs),
    }