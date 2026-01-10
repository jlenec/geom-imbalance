"""Real dataset loading and preprocessing."""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional
import logging

from ..config import N_REAL_TRAIN, N_REAL_TEST

def load_and_preprocess_breast_cancer(
    seed: int,
    test_size: float = 0.33,
    return_scaler: bool = False
) -> Dict:
    """
    Load and preprocess the breast cancer dataset.

    Parameters
    ----------
    seed : int
        Random seed for train-test split
    test_size : float, default=0.33
        Proportion of data to use for testing
    return_scaler : bool, default=False
        Whether to return the fitted scaler

    Returns
    -------
    dict
        Dictionary containing:
        - X_train, y_train: Training data
        - X_test, y_test: Test data
        - feature_names: Feature names
        - target_names: Class names
        - scaler: StandardScaler (if return_scaler=True)
        - metadata: Additional information
    """
    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Note: sklearn's breast cancer dataset has target:
    # 0 = malignant, 1 = benign
    # We want malignant=1 for our experiments, so flip
    y = 1 - y

    logging.info(f"Loaded breast cancer dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logging.info(f"Class distribution: {np.bincount(y)}")
    logging.info("Target mapping: malignant=1, benign=0")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Cap training size if needed
    if len(X_train) > N_REAL_TRAIN:
        idx = np.random.RandomState(seed).choice(len(X_train), N_REAL_TRAIN, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    # Cap test size if needed
    if len(X_test) > N_REAL_TEST:
        idx = np.random.RandomState(seed + 1).choice(len(X_test), N_REAL_TEST, replace=False)
        X_test = X_test[idx]
        y_test = y_test[idx]

    result = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': data.feature_names,
        'target_names': ['benign', 'malignant'],  # After flipping
        'metadata': {
            'original_shape': X.shape,
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'train_prevalence': y_train.mean(),
            'test_prevalence': y_test.mean(),
        }
    }

    if return_scaler:
        result['scaler'] = scaler

    return result

def get_dataset_info() -> Dict:
    """Get information about available real datasets."""
    return {
        'breast_cancer': {
            'name': 'Breast Cancer Wisconsin',
            'samples': 569,
            'features': 30,
            'classes': 2,
            'description': 'Binary classification: malignant (1) vs benign (0) tumors'
        }
    }