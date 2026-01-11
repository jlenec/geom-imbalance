"""Random number generator utilities for reproducibility"""
import numpy as np


def set_seed(seed: int):
    """Set random seed for numpy"""
    np.random.seed(seed)


def get_rng(seed: int) -> np.random.Generator:
    """Get a numpy random generator with the given seed"""
    return np.random.default_rng(seed)
