"""XGBoost model wrapper with fallback to sklearn GradientBoosting."""

import numpy as np
from typing import Optional, Dict, Any
import logging

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logging.warning("XGBoost not available, will use sklearn GradientBoosting as fallback")

from ..config import XGB_PARAMS
from .sklearn_models import BaseModelWrapper, GradientBoostingWrapper

class XGBoostWrapper(BaseModelWrapper):
    """Wrapper for XGBoost with fallback to sklearn."""

    def __init__(self, random_state: Optional[int] = None, **kwargs):
        super().__init__(random_state)

        if HAS_XGBOOST:
            # Use XGBoost
            params = XGB_PARAMS.copy()
            params['random_state'] = random_state
            params.update(kwargs)
            self.model = xgb.XGBClassifier(**params)
            self.name = 'XGBoost'
            self.is_xgb = True
        else:
            # Fallback to sklearn
            logging.info("Using GradientBoosting as XGBoost fallback")
            self._fallback = GradientBoostingWrapper(random_state=random_state, **kwargs)
            self.model = self._fallback.model
            self.name = 'GradientBoosting (XGB fallback)'
            self.is_xgb = False

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'XGBoostWrapper':
        """Fit the model."""
        if self.is_xgb:
            logging.info(f"Fitting {self.name} with shape {X.shape}")
            if sample_weight is not None:
                # Normalize weights
                sample_weight = sample_weight / sample_weight.mean()
                logging.info(f"Using sample weights: min={sample_weight.min():.3f}, "
                            f"max={sample_weight.max():.3f}, mean={sample_weight.mean():.3f}")

            # Fit with early stopping if validation set provided
            self.model.fit(X, y, sample_weight=sample_weight)
            self.is_fitted = True
        else:
            # Use fallback
            self._fallback.fit(X, y, sample_weight=sample_weight)
            self.is_fitted = self._fallback.is_fitted

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if self.is_xgb:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before prediction")
            return self.model.predict_proba(X)
        else:
            return self._fallback.predict_proba(X)

    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        """Predict logit scores."""
        if self.is_xgb:
            return super().predict_logit(X)
        else:
            return self._fallback.predict_logit(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Get decision function values."""
        if self.is_xgb:
            # XGBoost doesn't have decision_function, use logit
            return self.predict_logit(X)
        else:
            return self._fallback.decision_function(X)

    def get_feature_importance(self, importance_type: str = 'gain') -> np.ndarray:
        """
        Get feature importance scores.

        Parameters
        ----------
        importance_type : str
            Type of importance ('gain', 'weight', 'cover') for XGBoost

        Returns
        -------
        np.ndarray
            Feature importance scores
        """
        if self.is_xgb:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before accessing feature importance")

            # Get importance dict and convert to array
            importance_dict = self.model.get_booster().get_score(importance_type=importance_type)

            # Convert to array (handle missing features)
            n_features = self.model.n_features_in_
            importance = np.zeros(n_features)
            for key, value in importance_dict.items():
                # XGBoost uses f0, f1, ... format
                feature_idx = int(key[1:])
                importance[feature_idx] = value

            return importance
        else:
            # For sklearn GB, use feature_importances_
            return self.model.feature_importances_

def get_available_models() -> Dict[str, bool]:
    """Check which models are available."""
    return {
        'LogisticRegression': True,
        'GradientBoosting': True,
        'XGBoost': HAS_XGBOOST
    }