"""Experiment 1: Label shift - offset works, no retraining needed."""

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple

from ..config import (
    PI_TRAIN, PI_TEST_GRID, COSTS_GRID, N_SYNTH_TRAIN, N_SYNTH_TEST,
    N_REAL_TRAIN, N_REAL_TEST, CALIBRATION_LABELED_FRAC
)
from ..seeds import SEEDS, set_global_seed
from ..data.synthetic import sample_synthetic_dataset
from ..data.realdata import load_and_preprocess_breast_cancer
from ..data.shifts import make_label_shift_split
from ..models.sklearn_models import LogisticRegressionWrapper
from ..models.xgb_models import XGBoostWrapper
from ..metrics.classification import (
    compute_metrics, apply_threshold, logit_offset_correction,
    sigmoid, compute_optimal_threshold
)
from ..metrics.calibration import compute_threshold_from_costs
from ..utils.io import save_results_csv, save_metadata
from ..utils.logging import log_experiment_start, get_experiment_logger

logger = get_experiment_logger('exp1')

def run_experiment(
    dataset_name: str = 'both',
    models: List[str] = ['LogisticRegression', 'XGBoost'],
    save_results: bool = True
) -> pd.DataFrame:
    """
    Run Experiment 1: Label shift with offset correction.

    Parameters
    ----------
    dataset_name : str
        'synthetic', 'breast_cancer', or 'both'
    models : List[str]
        List of model names to run
    save_results : bool
        Whether to save results to CSV

    Returns
    -------
    pd.DataFrame
        Results dataframe
    """
    log_experiment_start("Experiment 1: Label Shift Offset")

    results = []
    datasets = []

    if dataset_name in ['synthetic', 'both']:
        datasets.append('synthetic')
    if dataset_name in ['breast_cancer', 'both']:
        datasets.append('breast_cancer')

    # Progress bar
    total_runs = len(SEEDS) * len(datasets) * len(models) * len(PI_TEST_GRID) * len(COSTS_GRID) * 3
    pbar = tqdm(total=total_runs, desc="Exp1 Progress")

    for seed in SEEDS:
        set_global_seed(seed)

        for dataset in datasets:
            logger.info(f"Processing dataset: {dataset}, seed: {seed}")

            # Load/generate data
            if dataset == 'synthetic':
                X_train, y_train = sample_synthetic_dataset(
                    pi=PI_TRAIN, n=N_SYNTH_TRAIN, seed=seed
                )
                # Use different seed for base test data
                X_test_base, y_test_base = sample_synthetic_dataset(
                    pi=0.5, n=N_SYNTH_TEST * 2, seed=seed + 1000
                )
            else:  # breast_cancer
                data = load_and_preprocess_breast_cancer(seed)
                X_train = data['X_train']
                y_train = data['y_train']
                X_test_base = data['X_test']
                y_test_base = data['y_test']

                # Apply label shift to training data if needed
                if PI_TRAIN != y_train.mean():
                    X_train, y_train = make_label_shift_split(
                        X_train, y_train, PI_TRAIN,
                        min(N_REAL_TRAIN, len(y_train)), seed
                    )

            # Train models
            trained_models = {}
            for model_name in models:
                if model_name == 'LogisticRegression':
                    model = LogisticRegressionWrapper(random_state=seed)
                elif model_name == 'XGBoost':
                    model = XGBoostWrapper(random_state=seed)
                else:
                    continue

                model.fit(X_train, y_train)
                trained_models[model_name] = model

                # Get training logits
                train_logits = model.predict_logit(X_train)

            # Evaluate on different test prevalences
            for pi_test in PI_TEST_GRID:
                # Create label-shifted test set
                n_test = N_SYNTH_TEST if dataset == 'synthetic' else N_REAL_TEST
                X_test, y_test = make_label_shift_split(
                    X_test_base, y_test_base, pi_test, n_test, seed + 2000
                )

                # Also create small calibration set
                n_cal = int(CALIBRATION_LABELED_FRAC * n_test)
                X_cal, y_cal = make_label_shift_split(
                    X_test_base, y_test_base, pi_test, n_cal, seed + 3000
                )

                for model_name, model in trained_models.items():
                    # Get scores and probabilities
                    test_logits = model.predict_logit(X_test)
                    test_proba = model.predict_proba(X_test)[:, 1]

                    cal_logits = model.predict_logit(X_cal)

                    for c10, c01 in COSTS_GRID:
                        # Method 1: No correction
                        threshold_logit = compute_threshold_from_costs(PI_TRAIN, c10, c01)
                        y_pred_nocorr = apply_threshold(test_logits, threshold_logit)

                        metrics_nocorr = compute_metrics(
                            y_test, y_pred_nocorr, y_score=test_logits,
                            y_proba=test_proba, c10=c10, c01=c01
                        )

                        results.append({
                            'experiment': 'exp1',
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

                        # Method 2: Offset correction
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
                            'experiment': 'exp1',
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

                        # Method 3: Oracle threshold tuning
                        cal_logits_corrected = logit_offset_correction(
                            cal_logits, PI_TRAIN, pi_test
                        )
                        oracle_threshold, _ = compute_optimal_threshold(
                            cal_logits_corrected, y_cal, c10, c01
                        )
                        y_pred_oracle = apply_threshold(test_logits_corrected, oracle_threshold)

                        metrics_oracle = compute_metrics(
                            y_test, y_pred_oracle, y_score=test_logits_corrected,
                            y_proba=test_proba_corrected, c10=c10, c01=c01
                        )

                        results.append({
                            'experiment': 'exp1',
                            'dataset': dataset,
                            'seed': seed,
                            'model': model_name,
                            'method': 'oracle_threshold',
                            'pi_train': PI_TRAIN,
                            'pi_test': pi_test,
                            'c10': c10,
                            'c01': c01,
                            **metrics_oracle,
                            'threshold': oracle_threshold,
                            'neff': None,
                            'notes': f'tuned_on_{n_cal}_samples'
                        })
                        pbar.update(1)

    pbar.close()

    # Convert to dataframe
    results_df = pd.DataFrame(results)

    # Save results
    if save_results:
        save_results_csv(results_df, 'exp1_results.csv')
        save_metadata({
            'experiment': 'exp1',
            'datasets': datasets,
            'models': models,
            'pi_train': PI_TRAIN,
            'pi_test_grid': PI_TEST_GRID,
            'costs_grid': COSTS_GRID,
            'n_seeds': len(SEEDS),
            'n_results': len(results_df)
        }, 'exp1_metadata.json')

    return results_df

if __name__ == '__main__':
    results = run_experiment()
    print(f"Experiment 1 completed with {len(results)} results")