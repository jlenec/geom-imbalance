"""
Mixture consistency testing for label shift validation
"""
import numpy as np
from typing import Tuple, Optional
from ..utils.ecdf import ecdf_from_samples, ecdf_eval


def compute_mixture_distance(
    S_window: np.ndarray,
    F0_ref: Tuple[np.ndarray, np.ndarray],
    F1_ref: Tuple[np.ndarray, np.ndarray],
    pi_grid_size: int = 401
) -> Tuple[float, float, Optional[float]]:
    """
    Compute d_u_star: minimum KS distance to mixture manifold.

    d_u_star = min_{pi in [0,1]} KS(F_u, F_mix(.; pi))

    Args:
        S_window: unlabeled scores in current window
        F0_ref: reference ECDF for class 0
        F1_ref: reference ECDF for class 1
        pi_grid_size: number of grid points for pi

    Returns:
        (d_u_star, pi_star, d_u_plugin) where:
        - d_u_star: minimum KS distance
        - pi_star: argmin pi
        - d_u_plugin: None (computed separately if BBSE is stable)
    """
    if len(S_window) == 0:
        return 0.0, 0.5, None

    # Compute F_u
    Fu_xs, Fu_cdf = ecdf_from_samples(S_window)

    # Get reference CDFs
    F0_xs, F0_cdf = F0_ref
    F1_xs, F1_cdf = F1_ref

    # Union support
    support = np.unique(np.concatenate([Fu_xs, F0_xs, F1_xs]))

    # Grid search over pi
    pi_grid = np.linspace(0.0, 1.0, pi_grid_size)

    min_ks = np.inf
    pi_star = 0.5

    for pi in pi_grid:
        # Compute KS distance for this pi
        max_diff = 0.0

        for s in support:
            Fu_s = ecdf_eval(Fu_xs, Fu_cdf, s)
            F0_s = ecdf_eval(F0_xs, F0_cdf, s)
            F1_s = ecdf_eval(F1_xs, F1_cdf, s)

            F_mix_s = (1 - pi) * F0_s + pi * F1_s

            diff = abs(Fu_s - F_mix_s)
            if diff > max_diff:
                max_diff = diff

        if max_diff < min_ks:
            min_ks = max_diff
            pi_star = pi

    return min_ks, pi_star, None


def compute_mixture_distance_at_pi(
    S_window: np.ndarray,
    F0_ref: Tuple[np.ndarray, np.ndarray],
    F1_ref: Tuple[np.ndarray, np.ndarray],
    pi: float
) -> float:
    """
    Compute KS distance to mixture at a specific pi (plug-in diagnostic).

    Args:
        S_window: unlabeled scores
        F0_ref: reference ECDF for class 0
        F1_ref: reference ECDF for class 1
        pi: prevalence value

    Returns:
        KS distance
    """
    if len(S_window) == 0:
        return 0.0

    # Compute F_u
    Fu_xs, Fu_cdf = ecdf_from_samples(S_window)

    # Get reference CDFs
    F0_xs, F0_cdf = F0_ref
    F1_xs, F1_cdf = F1_ref

    # Union support
    support = np.unique(np.concatenate([Fu_xs, F0_xs, F1_xs]))

    max_diff = 0.0
    for s in support:
        Fu_s = ecdf_eval(Fu_xs, Fu_cdf, s)
        F0_s = ecdf_eval(F0_xs, F0_cdf, s)
        F1_s = ecdf_eval(F1_xs, F1_cdf, s)

        F_mix_s = (1 - pi) * F0_s + pi * F1_s

        diff = abs(Fu_s - F_mix_s)
        if diff > max_diff:
            max_diff = diff

    return max_diff
