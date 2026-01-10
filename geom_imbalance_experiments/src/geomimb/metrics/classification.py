"""Classification metrics for evaluation."""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, confusion_matrix, precision_recall_fscore_support,
    roc_curve, precision_recall_curve
)
from typing import Dict, Tuple, Optional
import warnings

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    y_proba: Optional[np.ndarray] = None,
    c10: float = 1.0,
    c01: float = 1.0
) -> Dict:
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_score : np.ndarray, optional
        Decision scores (e.g., logits)
    y_proba : np.ndarray, optional
        Predicted probabilities for class 1
    c10 : float
        Cost of false positive
    c01 : float
        Cost of false negative

    Returns
    -------
    dict
        Dictionary of metrics
    """
    metrics = {}

    # Basic metrics from predictions
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n = len(y_true)

    # Rates
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity/Recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # 1 - Specificity
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 1.0  # Specificity

    # Precision, Recall, F1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

    metrics.update({
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'tpr': tpr,
        'fpr': fpr,
        'tnr': tnr,
        'fnr': 1 - tpr,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': (tp + tn) / n,
    })

    # Cost-weighted risk
    # R = c10 * P(yhat=1, y=0) + c01 * P(yhat=0, y=1)
    risk = (c10 * fp + c01 * fn) / n
    metrics['risk'] = risk

    # Metrics requiring scores/probabilities
    if y_score is not None or y_proba is not None:
        # Use probabilities for AUC if available, otherwise scores
        auc_input = y_proba if y_proba is not None else y_score

        # ROC-AUC
        if len(np.unique(y_true)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, auc_input)
        else:
            metrics['roc_auc'] = np.nan

    # Metrics requiring probabilities
    if y_proba is not None:
        # Ensure probabilities are in [0, 1]
        y_proba_clipped = np.clip(y_proba, 0, 1)

        # PR-AUC
        if len(np.unique(y_true)) == 2:
            metrics['pr_auc'] = average_precision_score(y_true, y_proba_clipped)
        else:
            metrics['pr_auc'] = np.nan

        # Brier score
        metrics['brier'] = brier_score_loss(y_true, y_proba_clipped)

        # Log loss
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics['logloss'] = log_loss(y_true, y_proba_clipped)

    return metrics

def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Apply threshold to scores to get predictions."""
    return (scores >= threshold).astype(int)

def logit_offset_correction(
    logits: np.ndarray,
    pi_train: float,
    pi_test: float,
    gamma: float = 0.0
) -> np.ndarray:
    """
    Apply logit offset correction for label shift.

    Parameters
    ----------
    logits : np.ndarray
        Training logits
    pi_train : float
        Training prevalence
    pi_test : float
        Test prevalence
    gamma : float
        log(c10/c01), default 0.0

    Returns
    -------
    np.ndarray
        Corrected logits
    """
    omega_train = pi_train / (1 - pi_train)
    omega_test = pi_test / (1 - pi_test)
    delta = np.log(omega_test / omega_train)

    return logits + delta

def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities."""
    # Numerically stable sigmoid
    return np.where(
        logits >= 0,
        1 / (1 + np.exp(-logits)),
        np.exp(logits) / (1 + np.exp(logits))
    )

def compute_optimal_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    c10: float = 1.0,
    c01: float = 1.0
) -> Tuple[float, Dict]:
    """
    Find threshold that minimizes cost-weighted risk.

    Parameters
    ----------
    scores : np.ndarray
        Decision scores
    y_true : np.ndarray
        True labels
    c10 : float
        Cost of false positive
    c01 : float
        Cost of false negative

    Returns
    -------
    threshold : float
        Optimal threshold
    info : dict
        Information about the search
    """
    # Try thresholds at unique score values
    thresholds = np.unique(scores)
    n_thresh = len(thresholds)

    if n_thresh > 200:
        # Subsample if too many unique values
        idx = np.linspace(0, n_thresh - 1, 200, dtype=int)
        thresholds = thresholds[idx]

    best_risk = float('inf')
    best_threshold = thresholds[0]
    risks = []

    for threshold in thresholds:
        y_pred = apply_threshold(scores, threshold)
        metrics = compute_metrics(y_true, y_pred, c10=c10, c01=c01)
        risk = metrics['risk']
        risks.append(risk)

        if risk < best_risk:
            best_risk = risk
            best_threshold = threshold

    return best_threshold, {
        'best_risk': best_risk,
        'thresholds_tried': len(thresholds),
        'risks': np.array(risks)
    }