"""Synthetic dataset generation for Gaussian mixture experiments."""

import numpy as np
from typing import Tuple, Optional
import logging

from ..config import SYNTH_DIM, N_POOL

# Global pools for synthetic data (initialized once)
_X0_pool = None
_X1_pool = None

def initialize_pools(seed: int = 42) -> None:
    """
    Initialize global pools for synthetic data generation.

    Parameters
    ----------
    seed : int, default=42
        Random seed for pool generation
    """
    global _X0_pool, _X1_pool

    logging.info(f"Initializing synthetic data pools with seed {seed}")

    rng = np.random.RandomState(seed)

    # Class parameters
    d = SYNTH_DIM
    mu0 = np.zeros(d)
    mu1 = np.zeros(d)
    mu1[:5] = 1.0  # First 5 dimensions shift by +1
    Sigma = np.eye(d)  # Identity covariance

    # Generate pools
    _X0_pool = rng.multivariate_normal(mu0, Sigma, size=N_POOL)
    _X1_pool = rng.multivariate_normal(mu1, Sigma, size=N_POOL)

    logging.info(f"Pools initialized: X0 shape {_X0_pool.shape}, X1 shape {_X1_pool.shape}")

def get_synthetic_params() -> dict:
    """
    Get the parameters of the synthetic Gaussian mixture.

    Returns
    -------
    dict
        Dictionary with mu0, mu1, Sigma, and theoretical log-likelihood ratio info
    """
    d = SYNTH_DIM
    mu0 = np.zeros(d)
    mu1 = np.zeros(d)
    mu1[:5] = 1.0
    Sigma = np.eye(d)

    # Theoretical log-likelihood ratio for Gaussian case
    # LogRatio(x) = (x - mu0)^T Sigma^{-1} mu1 - 0.5 ||mu1 - mu0||^2_{Sigma^{-1}}
    # With Sigma = I and mu0 = 0:
    # LogRatio(x) = x^T mu1 - 0.5 ||mu1||^2 = sum(x[i]) for i in [0,4] - 2.5

    return {
        'mu0': mu0,
        'mu1': mu1,
        'Sigma': Sigma,
        'd': d,
        'theoretical_llr_coef': mu1,  # Coefficient of x in LogRatio
        'theoretical_llr_const': -0.5 * np.dot(mu1, mu1),  # Constant term
    }

def sample_synthetic_dataset(
    pi: float,
    n: int,
    seed: int,
    return_indices: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a synthetic dataset with specified prevalence.

    Parameters
    ----------
    pi : float
        Prevalence of class 1
    n : int
        Total number of samples
    seed : int
        Random seed
    return_indices : bool, default=False
        If True, also return the pool indices used

    Returns
    -------
    X : np.ndarray of shape (n, d)
        Feature matrix
    y : np.ndarray of shape (n,)
        Binary labels
    indices : dict (optional)
        Pool indices used for each class
    """
    if _X0_pool is None or _X1_pool is None:
        initialize_pools()

    rng = np.random.RandomState(seed)

    # Determine class sizes
    n1 = int(np.round(pi * n))
    n0 = n - n1

    # Sample indices from pools
    if n0 <= N_POOL and n1 <= N_POOL:
        # Sample without replacement
        idx0 = rng.choice(N_POOL, size=n0, replace=False)
        idx1 = rng.choice(N_POOL, size=n1, replace=False)
    else:
        # Need to sample with replacement
        logging.warning(f"Sampling with replacement: n0={n0}, n1={n1}, pool_size={N_POOL}")
        idx0 = rng.choice(N_POOL, size=n0, replace=True)
        idx1 = rng.choice(N_POOL, size=n1, replace=True)

    # Extract samples
    X0 = _X0_pool[idx0]
    X1 = _X1_pool[idx1]

    # Combine
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n0), np.ones(n1)])

    # Shuffle
    shuffle_idx = rng.permutation(n)
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    if return_indices:
        return X, y, {'idx0': idx0, 'idx1': idx1}
    return X, y

def compute_theoretical_boundary(pi: float, c10: float = 1.0, c01: float = 1.0) -> dict:
    """
    Compute the theoretical Bayes boundary for the synthetic Gaussian case.

    Parameters
    ----------
    pi : float
        Prevalence
    c10 : float
        Cost of false positive
    c01 : float
        Cost of false negative

    Returns
    -------
    dict
        Theoretical boundary parameters
    """
    params = get_synthetic_params()

    # For Gaussian with equal covariance:
    # Bayes rule: sum(x[i] * mu1[i]) >= log(c10/c01) - log(omega) + 0.5||mu1||^2
    # where omega = pi/(1-pi)

    omega = pi / (1 - pi)
    tau = np.log(c10 / c01) - np.log(omega) + 0.5 * np.dot(params['mu1'], params['mu1'])

    # Since mu1 has 1 in first 5 dims and 0 elsewhere:
    # Decision rule becomes: sum(x[0:5]) >= tau

    return {
        'tau': tau,
        'omega': omega,
        'decision_normal': params['mu1'],  # Normal to decision boundary
        'decision_offset': tau,
    }