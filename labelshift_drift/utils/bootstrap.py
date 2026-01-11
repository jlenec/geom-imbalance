"""
Bootstrap utilities for calibration
"""
import numpy as np
from typing import Tuple


def bootstrap_confusion_matrix_det(
    S_ref: np.ndarray,
    Y_ref: np.ndarray,
    tau0: float,
    B: int = 200,
    seed: int = 42
) -> float:
    """
    Bootstrap the confusion matrix determinant to compute delta_C.

    Args:
        S_ref: reference scores
        Y_ref: reference labels
        tau0: reference threshold
        B: number of bootstrap replicates
        seed: random seed

    Returns:
        delta_C: 5th percentile of |det(C)|
    """
    rng = np.random.default_rng(seed)
    n = len(S_ref)

    abs_det_list = []

    for _ in range(B):
        # Sample with replacement
        indices = rng.choice(n, size=n, replace=True)
        S_b = S_ref[indices]
        Y_b = Y_ref[indices]

        # Compute confusion matrix
        C_b = compute_confusion_matrix(S_b, Y_b, tau0)

        # Store absolute determinant
        abs_det_list.append(abs(np.linalg.det(C_b)))

    delta_C = np.quantile(abs_det_list, 0.05)
    return delta_C


def compute_confusion_matrix(S: np.ndarray, Y: np.ndarray, tau0: float) -> np.ndarray:
    """
    Compute 2x2 confusion matrix at threshold tau0.

    C[i,j] = P(Yhat=i | Y=j) where Yhat = 1{S >= tau0}

    Args:
        S: scores
        Y: true labels
        tau0: threshold

    Returns:
        2x2 confusion matrix
    """
    Yhat = (S >= tau0).astype(int)

    # Count: n_ij = #(Y=j, Yhat=i)
    n00 = np.sum((Y == 0) & (Yhat == 0))
    n10 = np.sum((Y == 0) & (Yhat == 1))
    n01 = np.sum((Y == 1) & (Yhat == 0))
    n11 = np.sum((Y == 1) & (Yhat == 1))

    # Normalize to get conditional probabilities
    n0 = n00 + n10
    n1 = n01 + n11

    if n0 == 0 or n1 == 0:
        # Degenerate case
        return np.eye(2)

    C = np.array([
        [n00 / n0, n01 / n1],
        [n10 / n0, n11 / n1]
    ])

    return C
