"""
Main drift detector for streaming deployment
"""
import numpy as np
import pandas as pd
from typing import List
from ..config import ReferenceModel, WindowReport
from .bbse import bbse_binary, compute_bbse_residual
from .mixture import compute_mixture_distance, compute_mixture_distance_at_pi
from ..controller.state_machine import StateMachine


class DriftDetector:
    """
    Drift detector for streaming scores with delayed labels.
    """

    def __init__(
        self,
        ref_model: ReferenceModel,
        n_u: int,
        step_u: int = None
    ):
        """
        Initialize drift detector.

        Args:
            ref_model: fitted and calibrated reference model
            n_u: unlabeled window size
            step_u: step size (default: n_u for non-overlapping windows)
        """
        self.ref_model = ref_model
        self.n_u = n_u
        self.step_u = step_u if step_u is not None else n_u

        # Initialize state machine
        self.state_machine = StateMachine(ref_model)

        # Storage
        self.reports: List[WindowReport] = []

    def process_stream(
        self,
        S_stream: np.ndarray
    ) -> pd.DataFrame:
        """
        Process a stream of scores.

        Args:
            S_stream: array of scores

        Returns:
            DataFrame of WindowReports
        """
        n_total = len(S_stream)
        window_idx = 0

        for start in range(0, n_total - self.n_u + 1, self.step_u):
            S_window = S_stream[start:start + self.n_u]

            report = self._process_window(S_window, window_idx, start)
            self.reports.append(report)

            window_idx += 1

        return self.to_dataframe()

    def _process_window(
        self,
        S_window: np.ndarray,
        window_idx: int,
        timestamp: int
    ) -> WindowReport:
        """
        Process a single window.

        Args:
            S_window: scores in window
            window_idx: window index
            timestamp: starting timestamp

        Returns:
            WindowReport
        """
        # Step 1: Compute predictions at tau0
        Yhat_window = (S_window >= self.ref_model.tau0).astype(int)
        p_u = np.array([np.mean(Yhat_window == 0), np.mean(Yhat_window == 1)])

        # Step 2: BBSE
        pi_hat_bbse, p_dep = bbse_binary(p_u, self.ref_model.C_hat)

        # Step 3: Update EWMA
        pi_ewma = self.state_machine.update_ewma(pi_hat_bbse)

        # Step 4: Mixture consistency
        d_u_star, pi_star, _ = compute_mixture_distance(
            S_window,
            self.ref_model.F0_ref,
            self.ref_model.F1_ref
        )

        # Step 5: BBSE stability
        detC = np.linalg.det(self.ref_model.C_hat)
        bbse_stable = abs(detC) >= self.ref_model.delta_C

        # Step 6: BBSE residual (if stable)
        r_u = None
        if bbse_stable:
            r_u = compute_bbse_residual(p_u, p_dep, self.ref_model.C_hat)

        # Step 7: Plug-in diagnostic (if stable)
        d_u_plugin = None
        if bbse_stable:
            d_u_plugin = compute_mixture_distance_at_pi(
                S_window,
                self.ref_model.F0_ref,
                self.ref_model.F1_ref,
                pi_hat_bbse
            )

        # Step 8: Update state machine
        state, tau_operational = self.state_machine.update(
            d_u_star=d_u_star,
            r_u=r_u,
            pi_ewma=pi_ewma,
            bbse_stable=bbse_stable
        )

        # Create report
        report = WindowReport(
            window_idx=window_idx,
            timestamp=timestamp,
            pi_hat_bbse=pi_hat_bbse,
            pi_ewma=pi_ewma,
            d_u_star=d_u_star,
            pi_star=pi_star,
            d_u_plugin=d_u_plugin,
            r_u=r_u,
            detC=detC,
            bbse_stable=bbse_stable,
            state=state,
            tau_operational=tau_operational
        )

        return report

    def to_dataframe(self) -> pd.DataFrame:
        """Convert reports to DataFrame"""
        if not self.reports:
            return pd.DataFrame()

        data = []
        for r in self.reports:
            data.append({
                'window_idx': r.window_idx,
                'timestamp': r.timestamp,
                'pi_hat_bbse': r.pi_hat_bbse,
                'pi_ewma': r.pi_ewma,
                'd_u_star': r.d_u_star,
                'pi_star': r.pi_star,
                'd_u_plugin': r.d_u_plugin,
                'r_u': r.r_u,
                'detC': r.detC,
                'bbse_stable': r.bbse_stable,
                'state': r.state,
                'tau_operational': r.tau_operational
            })

        return pd.DataFrame(data)
