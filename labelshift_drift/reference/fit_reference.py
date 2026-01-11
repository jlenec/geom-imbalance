"""
Fit reference model from validation data
"""
import numpy as np
from typing import Tuple
from ..config import ReferenceModel
from ..utils.ecdf import ecdf_from_samples
from ..utils.bootstrap import bootstrap_confusion_matrix_det, compute_confusion_matrix


def fit_reference_model(
    S_ref: np.ndarray,
    Y_ref: np.ndarray,
    tau0: float = 0.5,
    lambda_ewma: float = 0.1,
    bootstrap_B: int = 200,
    bootstrap_seed: int = 42
) -> ReferenceModel:
    """
    Fit reference model from labeled validation data.

    Args:
        S_ref: reference scores (probability scores)
        Y_ref: reference labels
        tau0: fixed reference threshold for BBSE
        lambda_ewma: EWMA smoothing parameter
        bootstrap_B: number of bootstrap replicates
        bootstrap_seed: seed for bootstrap

    Returns:
        ReferenceModel object with all components
    """
    # Compute reference prevalence
    pi_ref = np.mean(Y_ref)

    # Compute confusion matrix at tau0
    C_hat = compute_confusion_matrix(S_ref, Y_ref, tau0)

    # Bootstrap to get delta_C
    delta_C = bootstrap_confusion_matrix_det(S_ref, Y_ref, tau0, B=bootstrap_B, seed=bootstrap_seed)

    # Compute reference conditional CDFs
    S0_ref = S_ref[Y_ref == 0]
    S1_ref = S_ref[Y_ref == 1]

    F0_ref = ecdf_from_samples(S0_ref)
    F1_ref = ecdf_from_samples(S1_ref)

    # Placeholder thresholds (will be calibrated later)
    d_th = 0.0
    r_th = None
    pi_th = 0.0

    return ReferenceModel(
        tau0=tau0,
        pi_ref=pi_ref,
        C_hat=C_hat,
        delta_C=delta_C,
        F0_ref=F0_ref,
        F1_ref=F1_ref,
        d_th=d_th,
        r_th=r_th,
        pi_th=pi_th,
        lambda_ewma=lambda_ewma
    )
