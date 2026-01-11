"""
Data generators for drift detection experiments
"""
import numpy as np
from typing import Tuple
from scipy.special import expit as sigmoid


def generate_gaussian_binary(
    n: int,
    pi: float,
    mu0: np.ndarray,
    mu1: np.ndarray,
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate binary classification data from Gaussian class-conditionals.

    X|Y=0 ~ N(mu0, I)
    X|Y=1 ~ N(mu1, I)

    Args:
        n: number of samples
        pi: prevalence P(Y=1)
        mu0: mean for class 0
        mu1: mean for class 1
        seed: random seed

    Returns:
        (X, Y) where X is (n, d) and Y is (n,)
    """
    rng = np.random.default_rng(seed)
    d = len(mu0)

    # Sample labels
    Y = rng.binomial(1, pi, size=n)

    # Sample features
    X = np.zeros((n, d))
    for i in range(n):
        if Y[i] == 0:
            X[i] = rng.normal(mu0, 1.0, size=d)
        else:
            X[i] = rng.normal(mu1, 1.0, size=d)

    return X, Y


class ScenarioGenerator:
    """
    Generates data for the 5 drift detection scenarios.
    """

    def __init__(
        self,
        d: int = 10,
        delta: float = 2.0,
        pi_ref: float = 0.02,
        seed: int = 42
    ):
        """
        Initialize scenario generator.

        Args:
            d: feature dimension
            delta: separation between class means
            pi_ref: reference prevalence
            seed: base random seed
        """
        self.d = d
        self.delta = delta
        self.pi_ref = pi_ref
        self.seed = seed

        # Define reference class means
        self.mu0 = np.zeros(d)
        self.mu1 = np.zeros(d)
        self.mu1[0] = delta  # Separation in first dimension

    def generate_reference(
        self,
        n_train: int,
        n_val: int
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Generate reference training and validation sets.

        Args:
            n_train: training set size
            n_val: validation set size

        Returns:
            ((X_train, Y_train), (X_val, Y_val))
        """
        X_train, Y_train = generate_gaussian_binary(
            n_train, self.pi_ref, self.mu0, self.mu1, self.seed
        )

        X_val, Y_val = generate_gaussian_binary(
            n_val, self.pi_ref, self.mu0, self.mu1, self.seed + 1
        )

        return (X_train, Y_train), (X_val, Y_val)

    def scenario_1_pure_label_shift(
        self,
        T: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scenario 1: Pure label shift (valid adaptation).

        pi(t) = 0.02 for t < 60k
        pi(t) = 0.06 for 60k <= t < 120k
        pi(t) = 0.01 for t >= 120k

        Args:
            T: total stream length

        Returns:
            (X, Y, pi_t) where pi_t is the true prevalence at each time
        """
        # Define prevalence schedule
        pi_schedule = np.zeros(T)
        pi_schedule[:60000] = 0.02
        pi_schedule[60000:120000] = 0.06
        pi_schedule[120000:] = 0.01

        # Generate data
        X_list = []
        Y_list = []

        batch_size = 10000
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            batch_len = end - start
            pi_batch = pi_schedule[start:end].mean()

            X_batch, Y_batch = generate_gaussian_binary(
                batch_len, pi_batch, self.mu0, self.mu1, self.seed + start
            )

            X_list.append(X_batch)
            Y_list.append(Y_batch)

        X = np.vstack(X_list)
        Y = np.concatenate(Y_list)

        return X, Y, pi_schedule

    def scenario_2_concept_drift(
        self,
        T: int,
        drift_time: int = 80000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scenario 2: Concept drift in p(x|y) (invalid for threshold adaptation).

        At t=80k, change positive class mean:
        mu1' = mu1 - [1.0, 0, 0, ..., 0]

        Args:
            T: total stream length
            drift_time: when drift occurs

        Returns:
            (X, Y, drift_indicator) where drift_indicator[t] = 1 if drifted
        """
        pi = self.pi_ref

        X_list = []
        Y_list = []

        # Before drift
        if drift_time > 0:
            X_before, Y_before = generate_gaussian_binary(
                drift_time, pi, self.mu0, self.mu1, self.seed + 1000
            )
            X_list.append(X_before)
            Y_list.append(Y_before)

        # After drift: reduce separability
        if T > drift_time:
            mu1_drifted = self.mu1.copy()
            mu1_drifted[0] -= 1.0  # Reduce separation

            X_after, Y_after = generate_gaussian_binary(
                T - drift_time, pi, self.mu0, mu1_drifted, self.seed + 2000
            )
            X_list.append(X_after)
            Y_list.append(Y_after)

        X = np.vstack(X_list)
        Y = np.concatenate(Y_list)

        drift_indicator = np.zeros(T)
        drift_indicator[drift_time:] = 1.0

        return X, Y, drift_indicator

    def scenario_3_score_mapping_drift(
        self,
        T: int,
        drift_time: int = 80000,
        a: float = 0.7,
        b: float = 0.4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Scenario 3: Score mapping drift (invalid).

        At t=80k, transform scores: S' = sigmoid(a * logit(S) + b)

        Args:
            T: total stream length
            drift_time: when drift occurs
            a: logit scale factor
            b: logit shift

        Returns:
            (X, Y, S_original, drift_indicator)
        """
        pi = self.pi_ref

        X, Y = generate_gaussian_binary(T, pi, self.mu0, self.mu1, self.seed + 3000)

        # Placeholder for original scores (to be computed after model fitting)
        S_original = np.zeros(T)

        drift_indicator = np.zeros(T)
        drift_indicator[drift_time:] = 1.0

        # Store transformation parameters as attributes for later use
        self.score_transform_a = a
        self.score_transform_b = b
        self.score_transform_time = drift_time

        return X, Y, S_original, drift_indicator

    def scenario_4_covariate_shift_benign(
        self,
        T: int,
        drift_time: int = 80000,
        shift_dim: int = 5,
        shift_amount: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scenario 4: Covariate shift in unused dimension (should NOT alarm).

        At t=80k, add shift_amount to feature shift_dim (which has zero weight).

        Args:
            T: total stream length
            drift_time: when shift occurs
            shift_dim: dimension to shift
            shift_amount: amount to shift

        Returns:
            (X, Y, shift_indicator)
        """
        pi = self.pi_ref

        X_list = []
        Y_list = []

        # Before shift
        if drift_time > 0:
            X_before, Y_before = generate_gaussian_binary(
                drift_time, pi, self.mu0, self.mu1, self.seed + 4000
            )
            X_list.append(X_before)
            Y_list.append(Y_before)

        # After shift: add to unused dimension
        if T > drift_time:
            X_after, Y_after = generate_gaussian_binary(
                T - drift_time, pi, self.mu0, self.mu1, self.seed + 5000
            )
            X_after[:, shift_dim] += shift_amount
            X_list.append(X_after)
            Y_list.append(Y_after)

        X = np.vstack(X_list)
        Y = np.concatenate(Y_list)

        shift_indicator = np.zeros(T)
        shift_indicator[drift_time:] = 1.0

        return X, Y, shift_indicator

    def scenario_5_ill_conditioned_C(
        self,
        n_train: int,
        n_val: int,
        tau0_quantile: float = 0.98
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], float]:
        """
        Scenario 5: Ill-conditioned confusion matrix.

        Force TPR ~= FPR by choosing very high tau0.

        Args:
            n_train: training set size
            n_val: validation set size
            tau0_quantile: quantile for tau0 (e.g., 0.98 for very high threshold)

        Returns:
            ((X_train, Y_train), (X_val, Y_val), tau0_bad)
        """
        # Generate reference data
        (X_train, Y_train), (X_val, Y_val) = self.generate_reference(n_train, n_val)

        # The caller will compute scores and set tau0 at this quantile
        # We just return the data and the quantile value as tau0_bad placeholder
        tau0_bad = tau0_quantile

        return (X_train, Y_train), (X_val, Y_val), tau0_bad


def apply_score_transform(
    S: np.ndarray,
    drift_time: int,
    a: float = 0.7,
    b: float = 0.4
) -> np.ndarray:
    """
    Apply score transformation after drift_time.

    S'[t] = sigmoid(a * logit(S[t]) + b) for t >= drift_time
    S'[t] = S[t] otherwise

    Args:
        S: original scores
        drift_time: when to start transformation
        a: logit scale
        b: logit shift

    Returns:
        Transformed scores
    """
    S_transformed = S.copy()

    # Clip for numerical stability
    S_clipped = np.clip(S[drift_time:], 1e-10, 1 - 1e-10)

    # Logit transform
    logit_S = np.log(S_clipped / (1 - S_clipped))

    # Apply transformation
    S_transformed[drift_time:] = sigmoid(a * logit_S + b)

    return S_transformed
