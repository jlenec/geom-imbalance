"""
Threshold adaptation under label shift
"""
import numpy as np


def compute_adapted_threshold(
    pi_ref: float,
    pi_dep: float,
    base_threshold: float = 0.5
) -> float:
    """
    Compute adapted threshold for label shift.

    For probability scores S = P_ref(Y=1|X), the adapted decision rule is:
        P_dep(Y=1|X) >= 0.5

    This translates to a threshold on S:
        tau(pi_dep) = (pi_ref * (1 - pi_dep)) / (pi_ref * (1 - pi_dep) + (1 - pi_ref) * pi_dep)

    Args:
        pi_ref: reference prevalence
        pi_dep: deployment prevalence
        base_threshold: base threshold (for probability scores, this is 0.5)

    Returns:
        Adapted threshold
    """
    numerator = pi_ref * (1 - pi_dep)
    denominator = pi_ref * (1 - pi_dep) + (1 - pi_ref) * pi_dep

    if denominator == 0:
        return base_threshold

    tau = numerator / denominator

    return tau
