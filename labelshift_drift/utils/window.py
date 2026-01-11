"""
Window utilities for streaming data
"""
import numpy as np
from typing import Iterator, Tuple


def sliding_windows(
    data: np.ndarray,
    window_size: int,
    step_size: int
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Generate sliding windows from data stream.

    Args:
        data: 1D array of data
        window_size: size of each window
        step_size: step between windows

    Yields:
        (start_idx, window_data) tuples
    """
    n = len(data)
    start = 0

    while start + window_size <= n:
        window = data[start:start + window_size]
        yield start, window
        start += step_size
