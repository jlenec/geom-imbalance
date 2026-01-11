"""
Configuration dataclasses for drift detection
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class ReferenceModel:
    """Stores all reference information needed for drift detection"""
    # Fixed reference threshold for BBSE
    tau0: float

    # Reference prevalence
    pi_ref: float

    # Confusion matrix at tau0
    C_hat: np.ndarray  # 2x2 matrix

    # Conditioning floor
    delta_C: float

    # Reference conditional CDFs (xs_sorted, cdf_values)
    F0_ref: Tuple[np.ndarray, np.ndarray]
    F1_ref: Tuple[np.ndarray, np.ndarray]

    # Calibrated thresholds
    d_th: float
    r_th: Optional[float]
    pi_th: float

    # EWMA lambda
    lambda_ewma: float = 0.1


@dataclass
class WindowReport:
    """Report for a single window of drift detection"""
    window_idx: int
    timestamp: int

    # BBSE estimates
    pi_hat_bbse: float
    pi_ewma: float

    # Mixture consistency
    d_u_star: float
    pi_star: float
    d_u_plugin: Optional[float]

    # BBSE residual
    r_u: Optional[float]

    # Conditioning
    detC: float
    bbse_stable: bool

    # Controller state
    state: str

    # Operational threshold
    tau_operational: float
