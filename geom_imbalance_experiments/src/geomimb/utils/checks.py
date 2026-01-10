"""Validation and acceptance checks for experiments."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging

def check_auc_invariance(
    results: pd.DataFrame,
    tolerance: float = 0.01,
    group_cols: List[str] = ['dataset', 'model', 'seed']
) -> Dict[str, Any]:
    """
    Check that AUC is invariant across different test prevalences.

    Parameters
    ----------
    results : pd.DataFrame
        Results with columns including 'pi_test' and 'auc'
    tolerance : float
        Maximum allowed AUC range
    group_cols : List[str]
        Columns to group by

    Returns
    -------
    dict
        Check results
    """
    check_results = {
        'check_name': 'auc_invariance',
        'tolerance': tolerance,
        'passed': True,
        'violations': []
    }

    # Check which column name is used
    auc_col = 'auc' if 'auc' in results.columns else 'roc_auc'

    for group_keys, group_data in results.groupby(group_cols):
        auc_values = group_data.groupby('pi_test')[auc_col].mean()
        auc_range = auc_values.max() - auc_values.min()

        if auc_range > tolerance:
            check_results['passed'] = False
            violation = {
                'group': dict(zip(group_cols, group_keys)),
                'auc_range': auc_range,
                'auc_min': auc_values.min(),
                'auc_max': auc_values.max()
            }
            check_results['violations'].append(violation)

    check_results['n_violations'] = len(check_results['violations'])
    return check_results

def check_offset_improvement(
    results: pd.DataFrame,
    test_pi: float = 0.01,
    group_cols: List[str] = ['dataset', 'model', 'c10', 'c01']
) -> Dict[str, Any]:
    """
    Check that offset method improves over no correction at extreme imbalance.

    Parameters
    ----------
    results : pd.DataFrame
        Results with methods 'nocorr' and 'offset'
    test_pi : float
        Test prevalence to check
    group_cols : List[str]
        Columns to group by

    Returns
    -------
    dict
        Check results
    """
    check_results = {
        'check_name': 'offset_improvement',
        'test_pi': test_pi,
        'passed': True,
        'violations': []
    }

    # Filter to specific pi_test
    pi_results = results[results['pi_test'] == test_pi]

    for group_keys, group_data in pi_results.groupby(group_cols):
        nocorr_risk = group_data[
            group_data['method'] == 'nocorr'
        ]['risk'].mean()
        offset_risk = group_data[
            group_data['method'] == 'offset'
        ]['risk'].mean()

        if offset_risk > nocorr_risk:
            check_results['passed'] = False
            violation = {
                'group': dict(zip(group_cols, group_keys)),
                'nocorr_risk': nocorr_risk,
                'offset_risk': offset_risk,
                'risk_increase': offset_risk - nocorr_risk
            }
            check_results['violations'].append(violation)

    check_results['n_violations'] = len(check_results['violations'])
    return check_results

def check_neff_monotonicity(
    results: pd.DataFrame,
    alpha_col: str = 'alpha'
) -> Dict[str, Any]:
    """
    Check that effective sample size decreases monotonically with alpha.

    Parameters
    ----------
    results : pd.DataFrame
        Results from experiment 3
    alpha_col : str
        Column name for alpha values

    Returns
    -------
    dict
        Check results
    """
    check_results = {
        'check_name': 'neff_monotonicity',
        'passed': True,
        'violations': []
    }

    # Group by relevant columns
    for (dataset, model), group in results.groupby(['dataset', 'model']):
        # Get mean Neff for each alpha
        neff_by_alpha = group.groupby(alpha_col)['neff'].mean().sort_index()
        alphas = neff_by_alpha.index.values
        neffs = neff_by_alpha.values

        # Check monotonic decrease
        for i in range(1, len(neffs)):
            if neffs[i] > neffs[i-1]:
                check_results['passed'] = False
                violation = {
                    'dataset': dataset,
                    'model': model,
                    'alpha1': alphas[i-1],
                    'alpha2': alphas[i],
                    'neff1': neffs[i-1],
                    'neff2': neffs[i],
                    'increase': neffs[i] - neffs[i-1]
                }
                check_results['violations'].append(violation)

    check_results['n_violations'] = len(check_results['violations'])
    return check_results

def check_concept_drift_failure(
    results: pd.DataFrame,
    improvement_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Check that offset fails under concept drift while retraining helps.

    Parameters
    ----------
    results : pd.DataFrame
        Results from concept drift experiment
    improvement_threshold : float
        Minimum improvement for retraining to be considered helpful

    Returns
    -------
    dict
        Check results
    """
    check_results = {
        'check_name': 'concept_drift_response',
        'passed': True,
        'details': []
    }

    # Check which column name is used
    auc_col = 'auc' if 'auc' in results.columns else 'roc_auc'

    for (dataset, model), group in results.groupby(['dataset', 'model']):
        methods = group.groupby('method')[auc_col].mean()

        # Check offset doesn't help much
        if 'nocorr' in methods and 'offset' in methods:
            offset_improvement = methods['offset'] - methods['nocorr']
            if abs(offset_improvement) > improvement_threshold:
                check_results['details'].append({
                    'issue': 'offset_helps_too_much',
                    'dataset': dataset,
                    'model': model,
                    'improvement': offset_improvement
                })

        # Check retraining helps
        if 'nocorr' in methods and 'retrain' in methods:
            retrain_improvement = methods['retrain'] - methods['nocorr']
            if retrain_improvement < improvement_threshold:
                check_results['passed'] = False
                check_results['details'].append({
                    'issue': 'retrain_insufficient',
                    'dataset': dataset,
                    'model': model,
                    'improvement': retrain_improvement
                })

    return check_results

def run_all_checks(experiment_results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Run all acceptance checks on experiment results.

    Parameters
    ----------
    experiment_results : Dict[str, pd.DataFrame]
        Dictionary mapping experiment names to results DataFrames

    Returns
    -------
    dict
        All check results
    """
    all_checks = {}

    # Experiment 1 checks
    if 'exp1' in experiment_results:
        exp1_results = experiment_results['exp1']

        # AUC invariance
        auc_check = check_auc_invariance(exp1_results)
        all_checks['exp1_auc_invariance'] = auc_check
        logging.info(f"Exp1 AUC invariance: {'PASSED' if auc_check['passed'] else 'FAILED'}")

        # Offset improvement
        offset_check = check_offset_improvement(exp1_results)
        all_checks['exp1_offset_improvement'] = offset_check
        logging.info(f"Exp1 offset improvement: {'PASSED' if offset_check['passed'] else 'FAILED'}")

    # Experiment 3 checks
    if 'exp3' in experiment_results:
        exp3_results = experiment_results['exp3']
        neff_check = check_neff_monotonicity(exp3_results)
        all_checks['exp3_neff_monotonicity'] = neff_check
        logging.info(f"Exp3 Neff monotonicity: {'PASSED' if neff_check['passed'] else 'FAILED'}")

    # Experiment 5 checks
    if 'exp5' in experiment_results:
        exp5_results = experiment_results['exp5']
        drift_check = check_concept_drift_failure(exp5_results)
        all_checks['exp5_concept_drift'] = drift_check
        logging.info(f"Exp5 concept drift: {'PASSED' if drift_check['passed'] else 'FAILED'}")

    # Overall pass/fail
    all_checks['overall_passed'] = all(
        check.get('passed', True) for check in all_checks.values()
    )

    return all_checks