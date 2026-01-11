"""
Threshold calibration on reference windows
"""
import numpy as np
from typing import List
from ..config import ReferenceModel
from .bbse import bbse_binary, compute_bbse_residual
from .mixture import compute_mixture_distance


def calibrate_thresholds(
    ref_model: ReferenceModel,
    S_ref_val: np.ndarray,
    Y_ref_val: np.ndarray,
    n_u: int,
    N_cal: int = 200,
    alpha_d: float = 0.01,
    alpha_r: float = 0.01,
    alpha_pi: float = 0.01,
    seed: int = 42
) -> ReferenceModel:
    """
    Calibrate thresholds on held-out reference windows.

    Args:
        ref_model: reference model (will be updated in place)
        S_ref_val: reference validation scores
        Y_ref_val: reference validation labels
        n_u: window size
        N_cal: number of calibration windows
        alpha_d: false alarm rate for d_u_star
        alpha_r: false alarm rate for r_u
        alpha_pi: false alarm rate for pi deviation
        seed: random seed

    Returns:
        Updated ReferenceModel with calibrated thresholds
    """
    rng = np.random.default_rng(seed)

    # Shuffle indices to create pseudo-stream
    n = len(S_ref_val)
    indices = np.arange(n)
    rng.shuffle(indices)

    S_shuffled = S_ref_val[indices]
    Y_shuffled = Y_ref_val[indices]

    # Extract calibration windows
    d_u_star_list = []
    r_u_list = []
    pi_hat_list = []
    pi_ewma_list = []

    # Initialize EWMA
    pi_ewma = ref_model.pi_ref

    for i in range(N_cal):
        start = i * n_u
        if start + n_u > n:
            break

        S_window = S_shuffled[start:start + n_u]
        Y_window = Y_shuffled[start:start + n_u]

        # Compute predictions at tau0
        Yhat_window = (S_window >= ref_model.tau0).astype(int)
        p_u = np.array([np.mean(Yhat_window == 0), np.mean(Yhat_window == 1)])

        # BBSE
        pi_hat, p_dep = bbse_binary(p_u, ref_model.C_hat)
        pi_hat_list.append(pi_hat)

        # Update EWMA
        pi_ewma = ref_model.lambda_ewma * pi_hat + (1 - ref_model.lambda_ewma) * pi_ewma
        pi_ewma_list.append(pi_ewma)

        # Mixture consistency
        d_u_star, pi_star, _ = compute_mixture_distance(
            S_window,
            ref_model.F0_ref,
            ref_model.F1_ref
        )
        d_u_star_list.append(d_u_star)

        # BBSE residual (if stable)
        detC = np.linalg.det(ref_model.C_hat)
        if abs(detC) >= ref_model.delta_C:
            r_u = compute_bbse_residual(p_u, p_dep, ref_model.C_hat)
            r_u_list.append(r_u)

    # Calibrate thresholds
    ref_model.d_th = np.quantile(d_u_star_list, 1 - alpha_d)

    if len(r_u_list) > 0:
        ref_model.r_th = np.quantile(r_u_list, 1 - alpha_r)
    else:
        ref_model.r_th = None

    # Pi threshold: |EWMA(pi) - pi_ref|
    pi_deviations = [abs(pe - ref_model.pi_ref) for pe in pi_ewma_list]
    ref_model.pi_th = np.quantile(pi_deviations, 1 - alpha_pi)

    return ref_model
