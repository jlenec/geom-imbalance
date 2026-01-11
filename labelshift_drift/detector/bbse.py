"""
Black Box Shift Estimation (BBSE) for binary classification
"""
import numpy as np
from typing import Tuple, Optional


def bbse_binary(
    p_u: np.ndarray,
    C: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    BBSE constrained least-squares solver for binary classification.

    Solves: min_{p in Delta^1} ||p_u - C @ p||_2^2

    For binary case, this reduces to 1D optimization.

    Args:
        p_u: empirical prediction frequencies [P(Yhat=0), P(Yhat=1)]
        C: 2x2 confusion matrix

    Returns:
        (pi_hat, p_dep) where:
        - pi_hat: estimated prevalence P(Y=1)
        - p_dep: [1-pi_hat, pi_hat]
    """
    # Extract confusion matrix elements
    # C[i,j] = P(Yhat=i | Y=j)
    fpr = C[1, 0]  # P(Yhat=1 | Y=0)
    tpr = C[1, 1]  # P(Yhat=1 | Y=1)

    # Observed rate of positive predictions
    p_u1 = p_u[1]

    # From p_u1 = (1-pi)*fpr + pi*tpr, solve for pi
    denom = tpr - fpr

    if abs(denom) >= 1e-12:
        pi_hat_unclipped = (p_u1 - fpr) / denom
        pi_hat = np.clip(pi_hat_unclipped, 0.0, 1.0)
    else:
        # Ill-conditioned case
        pi_hat = 0.5

    p_dep = np.array([1 - pi_hat, pi_hat])

    return pi_hat, p_dep


def compute_bbse_residual(
    p_u: np.ndarray,
    p_dep: np.ndarray,
    C: np.ndarray
) -> float:
    """
    Compute BBSE reconstruction residual.

    r_u = ||p_u - C @ p_dep||_1

    Args:
        p_u: observed prediction frequencies
        p_dep: estimated deployment prior
        C: confusion matrix

    Returns:
        L1 residual
    """
    p_rec = C @ p_dep
    r_u = np.sum(np.abs(p_u - p_rec))
    return r_u
