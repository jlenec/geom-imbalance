"""Experiment 4: Operating point metrics move; offset restores."""

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from typing import List, Tuple

from ..config import PI_TRAIN, PI_TEST_GRID, N_REAL_TRAIN, N_REAL_TEST
from ..seeds import SEEDS, set_global_seed
from ..data.realdata import load_and_preprocess_breast_cancer
from ..data.shifts import make_label_shift_split
from ..models.sklearn_models import LogisticRegressionWrapper
from ..metrics.classification import (
    compute_metrics, apply_threshold, logit_offset_correction, sigmoid
)
from ..metrics.calibration import compute_threshold_from_costs
from ..utils.io import save_results_csv, save_metadata
from ..utils.logging import log_experiment_start, get_experiment_logger

logger = get_experiment_logger('exp4')

def run_experiment(
    cost_settings: List[Tuple[float, float]] = [(1.0, 1.0), (10.0, 1.0), (1.0, 10.0)],
    save_results: bool = True
) -> pd.DataFrame:
    """
    Run Experiment 4: Show that operating point metrics drift without correction
    but are restored with offset correction.

    Parameters
    ----------
    cost_settings : List[Tuple[float, float]]
        List of (c10, c01) cost settings to evaluate
    save_results : bool
        Whether to save results to CSV

    Returns
    -------
    pd.DataFrame
        Results dataframe
    """
    log_experiment_start("Experiment 4: Operating Point Metrics")

    results = []
    dataset = 'breast_cancer'  # Use real dataset for this experiment
    model_name = 'LogisticRegression'  # Focus on logistic regression

    # Progress bar
    total_runs = len(SEEDS) * len(PI_TEST_GRID) * len(cost_settings) * 2
    pbar = tqdm(total=total_runs, desc="Exp4 Progress")

    for seed in SEEDS:
        set_global_seed(seed)
        logger.info(f"Processing seed: {seed}")

        # Load data
        data = load_and_preprocess_breast_cancer(seed)
        X_train = data['X_train']
        y_train = data['y_train']
        X_test_base = data['X_test']
        y_test_base = data['y_test']

        # Apply label shift to training data
        X_train, y_train = make_label_shift_split(
            X_train, y_train, PI_TRAIN,
            min(N_REAL_TRAIN, len(y_train)), seed
        )

        # Train model once
        model = LogisticRegressionWrapper(random_state=seed)
        model.fit(X_train, y_train)

        # Evaluate on different test prevalences
        for pi_test in PI_TEST_GRID:
            # Create label-shifted test set
            n_test = min(N_REAL_TEST, len(y_test_base))
            X_test, y_test = make_label_shift_split(
                X_test_base, y_test_base, pi_test, n_test, seed + 2000
            )

            # Get base scores
            test_logits = model.predict_logit(X_test)
            test_proba = model.predict_proba(X_test)[:, 1]

            for c10, c01 in cost_settings:
                # Fixed threshold based on training distribution
                threshold_logit = compute_threshold_from_costs(PI_TRAIN, c10, c01)

                # Method 1: No correction (fixed threshold)
                y_pred_nocorr = apply_threshold(test_logits, threshold_logit)
                metrics_nocorr = compute_metrics(
                    y_test, y_pred_nocorr, y_score=test_logits,
                    y_proba=test_proba, c10=c10, c01=c01
                )

                results.append({
                    'experiment': 'exp4',
                    'dataset': dataset,
                    'seed': seed,
                    'model': model_name,
                    'method': 'nocorr',
                    'pi_train': PI_TRAIN,
                    'pi_test': pi_test,
                    'c10': c10,
                    'c01': c01,
                    **metrics_nocorr,
                    'threshold': threshold_logit,
                    'neff': None,
                    'notes': 'fixed_threshold'
                })
                pbar.update(1)

                # Method 2: With offset correction
                test_logits_corrected = logit_offset_correction(
                    test_logits, PI_TRAIN, pi_test
                )
                test_proba_corrected = sigmoid(test_logits_corrected)
                y_pred_offset = apply_threshold(test_logits_corrected, threshold_logit)

                metrics_offset = compute_metrics(
                    y_test, y_pred_offset, y_score=test_logits_corrected,
                    y_proba=test_proba_corrected, c10=c10, c01=c01
                )

                results.append({
                    'experiment': 'exp4',
                    'dataset': dataset,
                    'seed': seed,
                    'model': model_name,
                    'method': 'offset',
                    'pi_train': PI_TRAIN,
                    'pi_test': pi_test,
                    'c10': c10,
                    'c01': c01,
                    **metrics_offset,
                    'threshold': threshold_logit,
                    'neff': None,
                    'notes': 'offset_corrected'
                })
                pbar.update(1)

    pbar.close()

    # Convert to dataframe
    results_df = pd.DataFrame(results)

    # Log summary of operating point drift
    logger.info("\nOperating Point Drift Summary:")
    for (c10, c01), cost_group in results_df.groupby(['c10', 'c01']):
        logger.info(f"\nCost setting: c10={c10}, c01={c01}")

        for method in ['nocorr', 'offset']:
            method_data = cost_group[cost_group['method'] == method]

            # Show precision/recall at different prevalences
            for pi in PI_TEST_GRID:
                pi_data = method_data[method_data['pi_test'] == pi]
                if len(pi_data) > 0:
                    mean_prec = pi_data['precision'].mean()
                    mean_rec = pi_data['recall'].mean()
                    logger.info(f"  {method} @ pi={pi:.3f}: "
                               f"Precision={mean_prec:.3f}, Recall={mean_rec:.3f}")

    # Save results
    if save_results:
        save_results_csv(results_df, 'exp4_results.csv')
        save_metadata({
            'experiment': 'exp4',
            'dataset': dataset,
            'model': model_name,
            'cost_settings': cost_settings,
            'pi_train': PI_TRAIN,
            'pi_test_grid': PI_TEST_GRID,
            'n_seeds': len(SEEDS),
            'n_results': len(results_df)
        }, 'exp4_metadata.json')

    return results_df

if __name__ == '__main__':
    results = run_experiment()
    print(f"Experiment 4 completed with {len(results)} results")