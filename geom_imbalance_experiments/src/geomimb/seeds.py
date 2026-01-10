"""Seed management for reproducibility."""

import random
import numpy as np
import logging

# Global seed list for all experiments
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def set_global_seed(seed: int) -> None:
    """
    Set global random seed for all libraries.

    Parameters
    ----------
    seed : int
        Random seed value
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Log the seed setting
    logging.info(f"Global seed set to {seed}")

    # Note: scikit-learn models will use random_state parameter
    # XGBoost will use random_state parameter in model params

def get_model_seed(base_seed: int, model_name: str) -> int:
    """
    Generate a deterministic seed for a specific model.

    Parameters
    ----------
    base_seed : int
        Base seed from SEEDS list
    model_name : str
        Name of the model

    Returns
    -------
    int
        Deterministic seed for the model
    """
    # Simple hash to generate different but deterministic seeds
    return base_seed + hash(model_name) % 1000