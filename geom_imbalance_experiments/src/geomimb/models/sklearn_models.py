"""Scikit-learn model wrappers with consistent interface."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from typing import Optional, Dict, Any
import logging

from ..config import LOGREG_PARAMS, GB_PARAMS, LOGIT_CLIP_MIN, LOGIT_CLIP_MAX

class BaseModelWrapper:
    """Base class for model wrappers with consistent interface."""

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        self.model = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'BaseModelWrapper':
        """Fit the model."""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        raise NotImplementedError

    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        """
        Predict logit scores (log-odds).

        Returns
        -------
        z : np.ndarray
            Logit scores where z = log(p/(1-p))
        """
        proba = self.predict_proba(X)[:, 1]
        # Clip probabilities for numerical stability
        proba_clipped = np.clip(proba, LOGIT_CLIP_MIN, LOGIT_CLIP_MAX)
        logits = np.log(proba_clipped) - np.log(1 - proba_clipped)
        return logits

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Get decision function values (defaults to logit)."""
        if hasattr(self.model, 'decision_function'):
            return self.model.decision_function(X)
        return self.predict_logit(X)

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self.model is not None:
            return self.model.get_params()
        return {}

class LogisticRegressionWrapper(BaseModelWrapper):
    """Wrapper for scikit-learn LogisticRegression."""

    def __init__(self, random_state: Optional[int] = None, **kwargs):
        super().__init__(random_state)
        # Merge default params with any overrides
        params = LOGREG_PARAMS.copy()
        params['random_state'] = random_state
        params.update(kwargs)
        self.model = LogisticRegression(**params)
        self.name = 'LogisticRegression'

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'LogisticRegressionWrapper':
        """Fit logistic regression."""
        logging.info(f"Fitting {self.name} with shape {X.shape}")
        if sample_weight is not None:
            # Normalize weights to have mean 1 (for numerical stability)
            sample_weight = sample_weight / sample_weight.mean()
            logging.info(f"Using sample weights: min={sample_weight.min():.3f}, "
                        f"max={sample_weight.max():.3f}, mean={sample_weight.mean():.3f}")

        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)

    def get_coefficients(self) -> np.ndarray:
        """Get the coefficient vector (for linear models only)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing coefficients")
        return self.model.coef_[0]  # Shape (n_features,)

    def get_intercept(self) -> float:
        """Get the intercept (for linear models only)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing intercept")
        return self.model.intercept_[0]

class GradientBoostingWrapper(BaseModelWrapper):
    """Wrapper for scikit-learn GradientBoostingClassifier."""

    def __init__(self, random_state: Optional[int] = None, **kwargs):
        super().__init__(random_state)
        # Merge default params with any overrides
        params = GB_PARAMS.copy()
        params['random_state'] = random_state
        params.update(kwargs)
        self.model = GradientBoostingClassifier(**params)
        self.name = 'GradientBoosting'

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'GradientBoostingWrapper':
        """Fit gradient boosting."""
        logging.info(f"Fitting {self.name} with shape {X.shape}")
        if sample_weight is not None:
            # Normalize weights
            sample_weight = sample_weight / sample_weight.mean()
            logging.info(f"Using sample weights: min={sample_weight.min():.3f}, "
                        f"max={sample_weight.max():.3f}, mean={sample_weight.mean():.3f}")

        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)

def compute_class_weights(y: np.ndarray, alpha: float = 1.0, normalize: bool = True) -> np.ndarray:
    """
    Compute sample weights based on class and scaling factor.

    Parameters
    ----------
    y : np.ndarray
        Class labels
    alpha : float
        Scaling factor for class 1 (class 0 gets weight 1.0)
    normalize : bool
        Whether to normalize weights to have mean 1

    Returns
    -------
    sample_weight : np.ndarray
        Weight for each sample
    """
    weights = np.ones_like(y, dtype=float)
    weights[y == 1] = alpha

    if normalize:
        weights = weights / weights.mean()

    return weights

def compute_effective_sample_size(weights: np.ndarray) -> float:
    """
    Compute effective sample size from weights.

    N_eff = N / E[w^2] where weights are normalized to have mean 1.

    Parameters
    ----------
    weights : np.ndarray
        Sample weights

    Returns
    -------
    float
        Effective sample size
    """
    # Normalize to mean 1
    weights_norm = weights / weights.mean()
    n = len(weights)
    return n / np.mean(weights_norm ** 2)