"""Experiment 5: Concept drift control - offset fails, retraining helps."""

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from typing import List

from ..config import PI_TRAIN, N_SYNTH_TRAIN, N_SYNTH_TEST
from ..seeds import SEEDS, set_global_seed
from ..data.synthetic import sample_synthetic_dataset
from ..data.shifts import make_concept_drift_test
from ..models.sklearn_models import LogisticRegressionWrapper
from ..models.xgb_models import XGBoostWrapper
from ..metrics.classification import (
    compute_metrics, apply_threshold, logit_offset_correction, sigmoid
)
from ..metrics.calibration import compute_threshold_from_costs
from ..utils.io import save_results_csv, save_metadata
from ..utils.logging import log_experiment_start, get_experiment_logger

logger = get_experiment_logger('exp5')

def run_experiment(
    models: List[str] = ['LogisticRegression', 'XGBoost'],
    save_results: bool = True
) -> pd.DataFrame:
    """
    Run Experiment 5: Demonstrate that under concept drift, offset correction
    doesn't help but retraining does.

    Parameters
    ----------
    models : List[str]
        List of model names to run
    save_results : bool
        Whether to save results to CSV

    Returns
    -------
    pd.DataFrame
        Results dataframe
    """
    log_experiment_start("Experiment 5: Concept Drift Control")

    results = []
    dataset = 'synthetic'  # Use synthetic data for controlled drift

    # Fixed prevalence for this experiment (no label shift, only concept drift)
    pi_train = PI_TRAIN
    pi_test = PI_TRAIN  # Same prevalence, different conditionals

    # Progress bar
    total_runs = len(SEEDS) * len(models) * 3  # 3 methods
    pbar = tqdm(total=total_runs, desc="Exp5 Progress")

    for seed in SEEDS:
        set_global_seed(seed)
        logger.info(f"Processing seed: {seed}")

        # Generate original training data
        X_train, y_train = sample_synthetic_dataset(
            pi=pi_train, n=N_SYNTH_TRAIN, seed=seed
        )

        # Generate test data with same prevalence
        X_test_orig, y_test = sample_synthetic_dataset(
            pi=pi_test, n=N_SYNTH_TEST, seed=seed + 1000
        )

        # Apply concept drift to test data
        X_test_drift = make_concept_drift_test(X_test_orig, y_test, seed=seed + 2000)

        # Also create small drifted training set for retraining
        # (simulating having some labeled data from new distribution)
        n_drift_train = int(0.2 * N_SYNTH_TRAIN)  # 20% of original
        X_drift_train_orig, y_drift_train = sample_synthetic_dataset(
            pi=pi_train, n=n_drift_train, seed=seed + 3000
        )
        X_drift_train = make_concept_drift_test(
            X_drift_train_orig, y_drift_train, seed=seed + 4000
        )

        for model_name in models:
            # Train on original distribution
            if model_name == 'LogisticRegression':
                model_orig = LogisticRegressionWrapper(random_state=seed)
            elif model_name == 'XGBoost':
                model_orig = XGBoostWrapper(random_state=seed)
            else:
                continue

            model_orig.fit(X_train, y_train)

            # Get scores on drifted test set
            test_logits = model_orig.predict_logit(X_test_drift)
            test_proba = model_orig.predict_proba(X_test_drift)[:, 1]

            # Fixed costs for simplicity
            c10, c01 = 1.0, 1.0
            threshold_logit = compute_threshold_from_costs(pi_train, c10, c01)

            # Method 1: No correction
            y_pred_nocorr = apply_threshold(test_logits, threshold_logit)
            metrics_nocorr = compute_metrics(
                y_test, y_pred_nocorr, y_score=test_logits,
                y_proba=test_proba, c10=c10, c01=c01
            )

            results.append({
                'experiment': 'exp5',
                'dataset': dataset,
                'seed': seed,
                'model': model_name,
                'method': 'nocorr',
                'pi_train': pi_train,
                'pi_test': pi_test,
                'c10': c10,
                'c01': c01,
                **metrics_nocorr,
                'threshold': threshold_logit,
                'neff': None,
                'notes': 'no_correction'
            })
            pbar.update(1)

            # Method 2: Offset correction (should NOT help under concept drift)
            # Apply offset even though pi_train == pi_test (to show it doesn't help)
            test_logits_offset = logit_offset_correction(
                test_logits, pi_train, pi_test
            )
            test_proba_offset = sigmoid(test_logits_offset)
            y_pred_offset = apply_threshold(test_logits_offset, threshold_logit)

            metrics_offset = compute_metrics(
                y_test, y_pred_offset, y_score=test_logits_offset,
                y_proba=test_proba_offset, c10=c10, c01=c01
            )

            results.append({
                'experiment': 'exp5',
                'dataset': dataset,
                'seed': seed,
                'model': model_name,
                'method': 'offset',
                'pi_train': pi_train,
                'pi_test': pi_test,
                'c10': c10,
                'c01': c01,
                **metrics_offset,
                'threshold': threshold_logit,
                'neff': None,
                'notes': 'offset_correction'
            })
            pbar.update(1)

            # Method 3: Retrain on drifted data
            if model_name == 'LogisticRegression':
                model_retrain = LogisticRegressionWrapper(random_state=seed)
            else:
                model_retrain = XGBoostWrapper(random_state=seed)

            model_retrain.fit(X_drift_train, y_drift_train)

            # Evaluate retrained model
            test_logits_retrain = model_retrain.predict_logit(X_test_drift)
            test_proba_retrain = model_retrain.predict_proba(X_test_drift)[:, 1]
            y_pred_retrain = apply_threshold(test_logits_retrain, threshold_logit)

            metrics_retrain = compute_metrics(
                y_test, y_pred_retrain, y_score=test_logits_retrain,
                y_proba=test_proba_retrain, c10=c10, c01=c01
            )

            results.append({
                'experiment': 'exp5',
                'dataset': dataset,
                'seed': seed,
                'model': model_name,
                'method': 'retrain',
                'pi_train': pi_train,
                'pi_test': pi_test,
                'c10': c10,
                'c01': c01,
                **metrics_retrain,
                'threshold': threshold_logit,
                'neff': None,
                'notes': f'retrained_on_{n_drift_train}_samples'
            })
            pbar.update(1)

    pbar.close()

    # Convert to dataframe
    results_df = pd.DataFrame(results)

    # Log performance summary
    logger.info("\nConcept Drift Performance Summary:")
    for model_name in models:
        model_data = results_df[results_df['model'] == model_name]
        logger.info(f"\n{model_name}:")

        for method in ['nocorr', 'offset', 'retrain']:
            method_data = model_data[model_data['method'] == method]
            if len(method_data) > 0:
                mean_auc = method_data['roc_auc'].mean()
                mean_risk = method_data['risk'].mean()
                logger.info(f"  {method}: AUC={mean_auc:.3f}, Risk={mean_risk:.3f}")

        # Check if offset helps (it shouldn't)
        nocorr_auc = model_data[model_data['method'] == 'nocorr']['roc_auc'].mean()
        offset_auc = model_data[model_data['method'] == 'offset']['roc_auc'].mean()
        retrain_auc = model_data[model_data['method'] == 'retrain']['roc_auc'].mean()

        offset_improvement = offset_auc - nocorr_auc
        retrain_improvement = retrain_auc - nocorr_auc

        logger.info(f"  Offset improvement: {offset_improvement:.3f}")
        logger.info(f"  Retrain improvement: {retrain_improvement:.3f}")

    # Save results
    if save_results:
        save_results_csv(results_df, 'exp5_results.csv')
        save_metadata({
            'experiment': 'exp5',
            'dataset': dataset,
            'models': models,
            'pi_train': pi_train,
            'pi_test': pi_test,
            'drift_params': {
                'shift_dims': 3,
                'shift_class1': 0.8,
                'shift_class0': -0.2,
                'noise_std': 0.2
            },
            'n_seeds': len(SEEDS),
            'n_results': len(results_df)
        }, 'exp5_metadata.json')

    return results_df

if __name__ == '__main__':
    results = run_experiment()
    print(f"Experiment 5 completed with {len(results)} results")