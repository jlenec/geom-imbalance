"""
Stream simulator with delayed labels
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from ..utils.ecdf import ks_distance_between_empirical
from ..config import ReferenceModel


class DelayedLabelSimulator:
    """
    Simulates label arrival delay and performs delayed-label validation.
    """

    def __init__(
        self,
        ref_model: ReferenceModel,
        delay_D: int,
        n_l: int = 5000,
        n_min_pos: int = 50
    ):
        """
        Initialize delayed label simulator.

        Args:
            ref_model: reference model with conditional CDFs
            delay_D: label delay (samples)
            n_l: labeled window size
            n_min_pos: minimum positives for conditional drift test
        """
        self.ref_model = ref_model
        self.delay_D = delay_D
        self.n_l = n_l
        self.n_min_pos = n_min_pos

        # Buffer for delayed labels
        self.S_buffer = []
        self.Y_buffer = []
        self.t_buffer = []

    def process_stream_with_delay(
        self,
        S_stream: np.ndarray,
        Y_stream: np.ndarray
    ) -> pd.DataFrame:
        """
        Process stream with delayed labels.

        Args:
            S_stream: scores
            Y_stream: true labels (arrive with delay)

        Returns:
            DataFrame of labeled window reports
        """
        T = len(S_stream)
        reports = []

        # Accumulate delayed labels
        for t in range(T):
            arrival_time = t + self.delay_D

            if arrival_time <= T:
                self.S_buffer.append(S_stream[t])
                self.Y_buffer.append(Y_stream[t])
                self.t_buffer.append(t)

                # Check if we have enough for a labeled window
                if len(self.S_buffer) >= self.n_l:
                    report = self._process_labeled_window(arrival_time)
                    if report is not None:
                        reports.append(report)

                    # Slide window
                    self.S_buffer = self.S_buffer[self.n_l:]
                    self.Y_buffer = self.Y_buffer[self.n_l:]
                    self.t_buffer = self.t_buffer[self.n_l:]

        return pd.DataFrame(reports)

    def _process_labeled_window(
        self,
        arrival_time: int
    ) -> Optional[dict]:
        """
        Process a labeled window and compute conditional drift statistics.

        Args:
            arrival_time: when labels became available

        Returns:
            Report dict or None
        """
        S_window = np.array(self.S_buffer[:self.n_l])
        Y_window = np.array(self.Y_buffer[:self.n_l])

        # Separate by class
        S0 = S_window[Y_window == 0]
        S1 = S_window[Y_window == 1]

        # Compute conditional KS statistics
        d_l_0 = ks_distance_between_empirical(S0, self.ref_model.F0_ref) if len(S0) > 0 else np.nan

        d_l_1 = np.nan
        if len(S1) >= self.n_min_pos:
            d_l_1 = ks_distance_between_empirical(S1, self.ref_model.F1_ref)

        report = {
            'arrival_time': arrival_time,
            'original_time': self.t_buffer[0],
            'n_samples': len(S_window),
            'n_pos': len(S1),
            'd_l_0': d_l_0,
            'd_l_1': d_l_1
        }

        return report
