"""Experiment 2: AUC invariance demonstration."""

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from typing import List
from sklearn.metrics import roc_auc_score, average_precision_score

from ..config import PI_TRAIN, PI_TEST_GRID, N_SYNTH_TRAIN, N_SYNTH_TEST
from ..seeds import SEEDS, set_global_seed
from ..data.synthetic import sample_synthetic_dataset
from ..data.realdata import load_and_preprocess_breast_cancer
from ..data.shifts import make_label_shift_split
from ..models.sklearn_models import LogisticRegressionWrapper
from ..models.xgb_models import XGBoostWrapper
from ..utils.io import save_results_csv, save_metadata
from ..utils.logging import log_experiment_start, get_experiment_logger

logger = get_experiment_logger('exp2')

def run_experiment(
    dataset_name: str = 'both',
    models: List[str] = ['LogisticRegression', 'XGBoost'],
    save_results: bool = True
) -> pd.DataFrame:
    """
    Run Experiment 2: Demonstrate AUC invariance and PR-AUC dependence on prevalence.

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
    log_experiment_start("Experiment 2: AUC Invariance")

    results = []
    datasets = []

    if dataset_name in ['synthetic', 'both']:
        datasets.append('synthetic')
    if dataset_name in ['breast_cancer', 'both']:
        datasets.append('breast_cancer')

    # Progress bar
    total_runs = len(SEEDS) * len(datasets) * len(models) * len(PI_TEST_GRID)
    pbar = tqdm(total=total_runs, desc="Exp2 Progress")

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

                # Apply label shift to training data
                X_train, y_train = make_label_shift_split(
                    X_train, y_train, PI_TRAIN, len(y_train), seed
                )

            # Train models once
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

            # Evaluate on different test prevalences
            for pi_test in PI_TEST_GRID:
                # Create label-shifted test set
                n_test = N_SYNTH_TEST if dataset == 'synthetic' else min(10000, len(y_test_base))
                X_test, y_test = make_label_shift_split(
                    X_test_base, y_test_base, pi_test, n_test, seed + 2000
                )

                for model_name, model in trained_models.items():
                    # Get raw scores (logits) and probabilities
                    test_logits = model.predict_logit(X_test)
                    test_proba = model.predict_proba(X_test)[:, 1]

                    # Compute AUC using raw scores (should be invariant)
                    if len(np.unique(y_test)) == 2:  # Check both classes present
                        auc = roc_auc_score(y_test, test_logits)
                        pr_auc = average_precision_score(y_test, test_proba)
                    else:
                        auc = np.nan
                        pr_auc = np.nan
                        logger.warning(f"Only one class present for pi_test={pi_test}")

                    # Store results
                    results.append({
                        'experiment': 'exp2',
                        'dataset': dataset,
                        'seed': seed,
                        'model': model_name,
                        'method': 'nocorr',  # Using uncorrected scores
                        'pi_train': PI_TRAIN,
                        'pi_test': pi_test,
                        'c10': 1.0,
                        'c01': 1.0,
                        'auc': auc,
                        'pr_auc': pr_auc,
                        'precision': None,
                        'recall': None,
                        'f1': None,
                        'risk': None,
                        'threshold': None,
                        'neff': None,
                        'notes': 'auc_invariance_demo'
                    })
                    pbar.update(1)

                    # Log for debugging
                    logger.debug(f"Model: {model_name}, pi_test: {pi_test:.3f}, "
                               f"AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}")

    pbar.close()

    # Convert to dataframe
    results_df = pd.DataFrame(results)

    # Compute and log invariance check
    logger.info("\nAUC Invariance Check:")
    for (dataset, model), group in results_df.groupby(['dataset', 'model']):
        auc_values = group['auc'].values
        auc_range = np.nanmax(auc_values) - np.nanmin(auc_values)
        logger.info(f"{dataset} - {model}: AUC range = {auc_range:.4f} "
                   f"(min={np.nanmin(auc_values):.4f}, max={np.nanmax(auc_values):.4f})")

        pr_auc_values = group['pr_auc'].values
        pr_auc_range = np.nanmax(pr_auc_values) - np.nanmin(pr_auc_values)
        logger.info(f"{dataset} - {model}: PR-AUC range = {pr_auc_range:.4f} "
                   f"(min={np.nanmin(pr_auc_values):.4f}, max={np.nanmax(pr_auc_values):.4f})")

    # Save results
    if save_results:
        save_results_csv(results_df, 'exp2_results.csv')
        save_metadata({
            'experiment': 'exp2',
            'datasets': datasets,
            'models': models,
            'pi_train': PI_TRAIN,
            'pi_test_grid': PI_TEST_GRID,
            'n_seeds': len(SEEDS),
            'n_results': len(results_df)
        }, 'exp2_metadata.json')

    return results_df

if __name__ == '__main__':
    results = run_experiment()
    print(f"Experiment 2 completed with {len(results)} results")