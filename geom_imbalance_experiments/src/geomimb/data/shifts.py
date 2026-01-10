"""Dataset shift construction: label shift and concept drift."""

import numpy as np
from typing import Tuple
import logging

def make_label_shift_split(
    X: np.ndarray,
    y: np.ndarray,
    pi_target: float,
    n_target: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a label-shifted dataset by resampling with target prevalence.

    This preserves p(x|y) while changing p(y).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    pi_target : float
        Target prevalence for class 1
    n_target : int
        Target number of samples
    seed : int
        Random seed

    Returns
    -------
    X_resampled : np.ndarray
        Resampled features
    y_resampled : np.ndarray
        Resampled labels
    """
    rng = np.random.RandomState(seed)

    # Get class indices
    idx1 = np.where(y == 1)[0]
    idx0 = np.where(y == 0)[0]

    # Determine target class sizes
    n1_target = int(np.round(pi_target * n_target))
    n0_target = n_target - n1_target

    # Check if we need to sample with replacement
    replace1 = n1_target > len(idx1)
    replace0 = n0_target > len(idx0)

    if replace1 or replace0:
        logging.info(f"Sampling with replacement needed: "
                    f"n1_target={n1_target} vs available={len(idx1)}, "
                    f"n0_target={n0_target} vs available={len(idx0)}")

    # Sample indices
    sampled_idx1 = rng.choice(idx1, size=n1_target, replace=replace1)
    sampled_idx0 = rng.choice(idx0, size=n0_target, replace=replace0)

    # Combine indices
    all_indices = np.concatenate([sampled_idx0, sampled_idx1])

    # Extract samples
    X_resampled = X[all_indices]
    y_resampled = y[all_indices]

    # Shuffle
    shuffle_idx = rng.permutation(n_target)
    X_resampled = X_resampled[shuffle_idx]
    y_resampled = y_resampled[shuffle_idx]

    return X_resampled, y_resampled

def make_concept_drift_test(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    drift_params: dict = None
) -> np.ndarray:
    """
    Create concept-drifted test features with class-conditional shifts.

    Parameters
    ----------
    X : np.ndarray
        Original features
    y : np.ndarray
        Labels
    seed : int
        Random seed
    drift_params : dict, optional
        Custom drift parameters. If None, uses default from spec.

    Returns
    -------
    X_drift : np.ndarray
        Drifted features
    """
    rng = np.random.RandomState(seed)

    # Default drift parameters from spec
    if drift_params is None:
        drift_params = {
            'shift_dims': 3,  # Number of dimensions to shift
            'shift_class1': 0.8,  # Shift magnitude for class 1
            'shift_class0': -0.2,  # Shift magnitude for class 0
            'noise_std': 0.2,  # Additional noise std
        }

    X_drift = X.copy()

    # Class-conditional shifts (first drift_params['shift_dims'] dimensions)
    shift_dims = drift_params['shift_dims']

    # Apply shifts based on class
    class1_mask = y == 1
    class0_mask = y == 0

    # Shift for class 1
    X_drift[class1_mask, :shift_dims] += drift_params['shift_class1']

    # Shift for class 0
    X_drift[class0_mask, :shift_dims] += drift_params['shift_class0']

    # Add Gaussian noise to all dimensions
    noise = rng.normal(0, drift_params['noise_std'], size=X.shape)
    X_drift += noise

    logging.info(f"Applied concept drift: {class1_mask.sum()} class 1 samples, "
                f"{class0_mask.sum()} class 0 samples")

    return X_drift

def validate_shift_preservation(
    X_orig: np.ndarray,
    y_orig: np.ndarray,
    X_shift: np.ndarray,
    y_shift: np.ndarray,
    shift_type: str = 'label'
) -> dict:
    """
    Validate that the shift was applied correctly.

    Parameters
    ----------
    X_orig, y_orig : np.ndarray
        Original data
    X_shift, y_shift : np.ndarray
        Shifted data
    shift_type : str
        Type of shift ('label' or 'concept')

    Returns
    -------
    dict
        Validation metrics
    """
    results = {
        'shift_type': shift_type,
        'orig_prevalence': y_orig.mean(),
        'shift_prevalence': y_shift.mean(),
        'orig_size': len(y_orig),
        'shift_size': len(y_shift),
    }

    if shift_type == 'label':
        # For label shift, check that conditional means are preserved
        for c in [0, 1]:
            if (y_orig == c).sum() > 0 and (y_shift == c).sum() > 0:
                orig_mean = X_orig[y_orig == c].mean(axis=0)
                shift_mean = X_shift[y_shift == c].mean(axis=0)
                max_diff = np.abs(orig_mean - shift_mean).max()
                results[f'class{c}_mean_max_diff'] = max_diff

    elif shift_type == 'concept':
        # For concept drift, measure the shift magnitude
        for c in [0, 1]:
            if (y_orig == c).sum() > 0:
                orig_mean = X_orig[y_orig == c].mean(axis=0)
                drift_mean = X_shift[y_shift == c].mean(axis=0)
                shift_magnitude = np.linalg.norm(drift_mean - orig_mean)
                results[f'class{c}_drift_magnitude'] = shift_magnitude

    return results