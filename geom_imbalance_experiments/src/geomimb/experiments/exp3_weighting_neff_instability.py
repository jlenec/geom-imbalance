"""Experiment 3: Weighting reduces Neff and increases instability."""

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from typing import List
from sklearn.utils import resample

from ..config import PI_TRAIN, ALPHA_VALUES, N_SYNTH_TRAIN, N_REAL_TRAIN
from ..seeds import SEEDS, set_global_seed
from ..data.synthetic import sample_synthetic_dataset
from ..data.realdata import load_and_preprocess_breast_cancer
from ..data.shifts import make_label_shift_split
from ..models.sklearn_models import (
    LogisticRegressionWrapper, compute_class_weights,
    compute_effective_sample_size
)
from ..metrics.classification import compute_metrics
from ..metrics.stability import (
    coefficient_angle, compute_coefficient_stability,
    compute_score_stability
)
from ..utils.io import save_results_csv, save_metadata
from ..utils.logging import log_experiment_start, get_experiment_logger

logger = get_experiment_logger('exp3')

def run_experiment(
    dataset_name: str = 'both',
    save_results: bool = True
) -> pd.DataFrame:
    """
    Run Experiment 3: Demonstrate that weighting reduces effective sample size
    and increases model instability.

    Parameters
    ----------
    dataset_name : str
        'synthetic', 'breast_cancer', or 'both'
    save_results : bool
        Whether to save results to CSV

    Returns
    -------
    pd.DataFrame
        Results dataframe
    """
    log_experiment_start("Experiment 3: Weighting Neff and Instability")

    results = []
    datasets = []

    if dataset_name in ['synthetic', 'both']:
        datasets.append('synthetic')
    if dataset_name in ['breast_cancer', 'both']:
        datasets.append('breast_cancer')

    # Add undersample to alpha values (represented as alpha=0)
    alpha_values_extended = [0] + ALPHA_VALUES  # 0 means undersample

    # Progress bar
    total_runs = len(SEEDS) * len(datasets) * len(alpha_values_extended)
    pbar = tqdm(total=total_runs, desc="Exp3 Progress")

    # Store models for stability analysis
    models_by_config = {}

    # Fixed reference test set for stability evaluation
    reference_test_data = {}

    for dataset in datasets:
        logger.info(f"Processing dataset: {dataset}")

        # Create fixed reference test set (same for all seeds/alphas)
        if dataset == 'synthetic':
            X_ref, y_ref = sample_synthetic_dataset(
                pi=PI_TRAIN, n=10000, seed=99999
            )
        else:
            data = load_and_preprocess_breast_cancer(seed=99999)
            X_ref = data['X_test'][:5000]
            y_ref = data['y_test'][:5000]
            X_ref, y_ref = make_label_shift_split(
                X_ref, y_ref, PI_TRAIN, len(y_ref), seed=99999
            )

        reference_test_data[dataset] = (X_ref, y_ref)

        for seed in SEEDS:
            set_global_seed(seed)

            # Load/generate training data
            if dataset == 'synthetic':
                X_train, y_train = sample_synthetic_dataset(
                    pi=PI_TRAIN, n=N_SYNTH_TRAIN, seed=seed
                )
            else:
                data = load_and_preprocess_breast_cancer(seed)
                X_train = data['X_train']
                y_train = data['y_train']
                X_train, y_train = make_label_shift_split(
                    X_train, y_train, PI_TRAIN,
                    min(N_REAL_TRAIN, len(y_train)), seed
                )

            # Train variants
            for alpha in alpha_values_extended:
                if alpha == 0:
                    # Undersample to 50/50
                    method = 'undersample'
                    n_minority = np.sum(y_train == 1)
                    n_majority = np.sum(y_train == 0)

                    if n_minority < n_majority:
                        # Undersample majority class
                        idx_minority = np.where(y_train == 1)[0]
                        idx_majority = np.where(y_train == 0)[0]
                        idx_majority_sample = resample(
                            idx_majority, n_samples=n_minority,
                            random_state=seed
                        )
                        idx_keep = np.concatenate([idx_minority, idx_majority_sample])
                        X_train_used = X_train[idx_keep]
                        y_train_used = y_train[idx_keep]
                    else:
                        X_train_used = X_train
                        y_train_used = y_train

                    sample_weight = None
                    neff = len(y_train_used)
                else:
                    # Weighted training
                    method = f'weighted_alpha{alpha}'
                    X_train_used = X_train
                    y_train_used = y_train
                    sample_weight = compute_class_weights(
                        y_train, alpha=alpha, normalize=True
                    )
                    neff = compute_effective_sample_size(sample_weight)

                # Train model
                model = LogisticRegressionWrapper(random_state=seed)
                model.fit(X_train_used, y_train_used, sample_weight=sample_weight)

                # Store model for stability analysis
                config_key = (dataset, alpha)
                if config_key not in models_by_config:
                    models_by_config[config_key] = []
                models_by_config[config_key].append(model)

                # Evaluate on reference test set
                X_ref, y_ref = reference_test_data[dataset]
                test_logits = model.predict_logit(X_ref)
                test_proba = model.predict_proba(X_ref)[:, 1]
                y_pred = (test_proba >= 0.5).astype(int)

                metrics = compute_metrics(
                    y_ref, y_pred, y_score=test_logits, y_proba=test_proba
                )

                # Get coefficient for linear model
                coef = model.get_coefficients()

                results.append({
                    'experiment': 'exp3',
                    'dataset': dataset,
                    'seed': seed,
                    'model': 'LogisticRegression',
                    'method': method,
                    'alpha': alpha,
                    'pi_train': PI_TRAIN,
                    'pi_test': PI_TRAIN,  # Same as train for this exp
                    'c10': 1.0,
                    'c01': 1.0,
                    **metrics,
                    'threshold': 0.0,  # Logit threshold
                    'neff': neff,
                    'n_train': len(y_train),
                    'n_train_used': len(y_train_used),
                    'notes': f'alpha={alpha}'
                })
                pbar.update(1)

    pbar.close()

    # Compute stability metrics
    logger.info("\nComputing stability metrics...")
    stability_results = []

    for (dataset, alpha), models in models_by_config.items():
        if len(models) < 2:
            continue

        # Get coefficients
        coefficients = [m.get_coefficients() for m in models]

        # Coefficient stability
        coef_stability = compute_coefficient_stability(coefficients)

        # Score stability on reference set
        X_ref, y_ref = reference_test_data[dataset]
        scores = [m.predict_logit(X_ref) for m in models]
        score_stability = compute_score_stability(scores)

        # Add to results
        for angle_deg in coef_stability['angles_deg']:
            stability_results.append({
                'experiment': 'exp3',
                'dataset': dataset,
                'model': 'LogisticRegression',
                'alpha': alpha,
                'angle_deg': angle_deg,
                'angle_rad': angle_deg * np.pi / 180,
                'logit_variance': score_stability['mean_variance'],
                'metric_type': 'stability'
            })

    # Convert to dataframe
    results_df = pd.DataFrame(results)
    stability_df = pd.DataFrame(stability_results)

    # Merge stability metrics
    if len(stability_df) > 0:
        # Add average stability metrics to main results
        for idx, row in results_df.iterrows():
            dataset = row['dataset']
            alpha = row['alpha']

            # Get stability metrics for this config
            stab_data = stability_df[
                (stability_df['dataset'] == dataset) &
                (stability_df['alpha'] == alpha)
            ]

            if len(stab_data) > 0:
                results_df.loc[idx, 'mean_angle_deg'] = stab_data['angle_deg'].mean()
                results_df.loc[idx, 'logit_variance'] = stab_data['logit_variance'].iloc[0]

    # Save results
    if save_results:
        save_results_csv(results_df, 'exp3_results.csv')
        if len(stability_df) > 0:
            save_results_csv(stability_df, 'exp3_stability.csv')

        save_metadata({
            'experiment': 'exp3',
            'datasets': datasets,
            'alpha_values': alpha_values_extended,
            'pi_train': PI_TRAIN,
            'n_seeds': len(SEEDS),
            'n_results': len(results_df),
            'n_stability': len(stability_df)
        }, 'exp3_metadata.json')

    return results_df

if __name__ == '__main__':
    results = run_experiment()
    print(f"Experiment 3 completed with {len(results)} results")