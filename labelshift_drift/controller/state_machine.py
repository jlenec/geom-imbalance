"""
State machine controller for drift detection
"""
from typing import Tuple, Optional
from collections import deque
from ..config import ReferenceModel
from .threshold_adaptation import compute_adapted_threshold


class StateMachine:
    """
    State machine for drift detection with sustained trigger rule.

    States:
    - NORMAL: d_u_star <= d_th
    - PRIOR_SHIFT: |EWMA(pi) - pi_ref| > pi_th AND d_u_star <= d_th
    - DRIFT_SUSPECTED: d_u_star > d_th OR (bbse_stable AND r_u > r_th)
    """

    def __init__(self, ref_model: ReferenceModel):
        """
        Initialize state machine.

        Args:
            ref_model: reference model with thresholds
        """
        self.ref_model = ref_model

        # State
        self.state = "NORMAL"
        self.pi_ewma = ref_model.pi_ref

        # Sustained trigger: track last 3 violations
        self.violation_history = deque(maxlen=3)

        # Operational threshold
        self.tau_operational = ref_model.tau0
        self.tau_frozen = ref_model.tau0

    def update_ewma(self, pi_hat: float) -> float:
        """
        Update EWMA of prevalence estimate.

        Args:
            pi_hat: current window prevalence estimate

        Returns:
            Updated EWMA
        """
        self.pi_ewma = (
            self.ref_model.lambda_ewma * pi_hat +
            (1 - self.ref_model.lambda_ewma) * self.pi_ewma
        )
        return self.pi_ewma

    def update(
        self,
        d_u_star: float,
        r_u: Optional[float],
        pi_ewma: float,
        bbse_stable: bool
    ) -> Tuple[str, float]:
        """
        Update state and operational threshold.

        Args:
            d_u_star: mixture consistency statistic
            r_u: BBSE residual (None if not stable)
            pi_ewma: current EWMA prevalence
            bbse_stable: whether BBSE is stable

        Returns:
            (state, tau_operational)
        """
        self.pi_ewma = pi_ewma

        # Check violations
        viol_du = d_u_star > self.ref_model.d_th

        viol_ru = False
        if bbse_stable and r_u is not None and self.ref_model.r_th is not None:
            viol_ru = r_u > self.ref_model.r_th

        viol = viol_du or viol_ru

        # Update violation history
        self.violation_history.append(viol)

        # Sustained rule: 2 of last 3
        sustained_violation = sum(self.violation_history) >= 2

        # State logic
        if sustained_violation:
            self.state = "DRIFT_SUSPECTED"
            # Freeze threshold
            self.tau_operational = self.tau_frozen

        elif abs(pi_ewma - self.ref_model.pi_ref) > self.ref_model.pi_th and d_u_star <= self.ref_model.d_th:
            self.state = "PRIOR_SHIFT"
            # Adapt threshold
            self.tau_operational = compute_adapted_threshold(
                self.ref_model.pi_ref,
                pi_ewma
            )
            # Update frozen value for potential later use
            self.tau_frozen = self.tau_operational

        else:
            self.state = "NORMAL"
            # Keep baseline
            self.tau_operational = self.ref_model.tau0
            self.tau_frozen = self.tau_operational

        return self.state, self.tau_operational
