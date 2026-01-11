"""
Empirical CDF utilities for mixture consistency testing
"""
import numpy as np
from typing import Tuple


def ecdf_from_samples(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical CDF from samples.

    Args:
        samples: 1D array of values

    Returns:
        Tuple of (xs_sorted, cdf_values_sorted) where:
        - xs_sorted: sorted unique values
        - cdf_values_sorted: CDF value at each sorted value
    """
    if len(samples) == 0:
        return np.array([]), np.array([])

    xs_sorted = np.sort(samples)
    n = len(xs_sorted)
    cdf_values = np.arange(1, n + 1) / n

    return xs_sorted, cdf_values


def ecdf_eval(xs_sorted: np.ndarray, cdf_sorted: np.ndarray, x_query: float) -> float:
    """
    Evaluate empirical CDF at a query point.

    Returns P(X <= x_query) by finding the rightmost index with xs_sorted[i] <= x_query.

    Args:
        xs_sorted: sorted sample values
        cdf_sorted: CDF values at each sorted sample
        x_query: query point

    Returns:
        CDF value at x_query
    """
    if len(xs_sorted) == 0:
        return 0.0

    if x_query < xs_sorted[0]:
        return 0.0
    if x_query >= xs_sorted[-1]:
        return cdf_sorted[-1]

    # Find rightmost index where xs_sorted[i] <= x_query
    idx = np.searchsorted(xs_sorted, x_query, side='right') - 1
    if idx < 0:
        return 0.0
    return cdf_sorted[idx]


def ks_distance_between_empirical(
    samples_a: np.ndarray,
    ecdf_b: Tuple[np.ndarray, np.ndarray]
) -> float:
    """
    Compute KS distance between two empirical CDFs.

    Args:
        samples_a: samples for first distribution
        ecdf_b: (xs_sorted, cdf_sorted) for second distribution

    Returns:
        Maximum absolute difference between the CDFs
    """
    if len(samples_a) == 0:
        return 0.0

    xs_b, cdf_b = ecdf_b

    # Compute ECDF of samples_a
    xs_a, cdf_a = ecdf_from_samples(samples_a)

    # Union of support points
    support = np.unique(np.concatenate([xs_a, xs_b]))

    # Evaluate both CDFs on union support
    max_diff = 0.0
    for s in support:
        F_a = ecdf_eval(xs_a, cdf_a, s)
        F_b = ecdf_eval(xs_b, cdf_b, s)
        diff = abs(F_a - F_b)
        if diff > max_diff:
            max_diff = diff

    return max_diff
