"""Plotting functions for all experiments."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging

from ..config import FIGURE_DPI, FIGURE_SIZE
from ..utils.io import save_figure

# Set matplotlib style
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100

def plot_auc_vs_pi(
    results: pd.DataFrame,
    title: str = 'AUC vs Test Prevalence',
    filename: str = 'auc_vs_pi.png'
) -> plt.Figure:
    """Plot AUC as a function of test prevalence."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Check which column name is used
    auc_col = 'auc' if 'auc' in results.columns else 'roc_auc'

    # Group by dataset and model
    for (dataset, model), group in results.groupby(['dataset', 'model']):
        # Get mean AUC for each pi_test
        pi_auc = group.groupby('pi_test')[auc_col].agg(['mean', 'std'])
        pi_values = pi_auc.index.values

        # Plot with error bars
        ax.errorbar(
            pi_values, pi_auc['mean'], yerr=pi_auc['std'],
            marker='o', capsize=5, label=f'{dataset} - {model}',
            linewidth=2, markersize=8
        )

    ax.set_xlabel('Test Prevalence ($\\pi_{test}$)')
    ax.set_ylabel('AUC')
    ax.set_xscale('log')
    ax.set_xlim(0.008, 0.6)
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)

    save_figure(fig, filename)
    return fig

def plot_risk_vs_pi(
    results: pd.DataFrame,
    cost_setting: Tuple[float, float] = (1.0, 1.0),
    title: str = 'Cost-Weighted Risk vs Test Prevalence',
    filename: str = 'risk_vs_pi.png'
) -> plt.Figure:
    """Plot risk as a function of test prevalence for different methods."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Filter for specific cost setting
    c10, c01 = cost_setting
    cost_results = results[
        (results['c10'] == c10) & (results['c01'] == c01)
    ]

    # Plot for each dataset
    datasets = cost_results['dataset'].unique()
    for i, dataset in enumerate(datasets[:2]):  # Max 2 subplots
        ax = axes[i]
        data = cost_results[cost_results['dataset'] == dataset]

        # Group by method
        for method in ['nocorr', 'offset', 'oracle_threshold']:
            if method not in data['method'].values:
                continue

            method_data = data[data['method'] == method]
            pi_risk = method_data.groupby('pi_test')['risk'].agg(['mean', 'std'])
            pi_values = pi_risk.index.values

            # Choose style
            styles = {
                'nocorr': {'color': 'red', 'linestyle': '--', 'label': 'No Correction'},
                'offset': {'color': 'blue', 'linestyle': '-', 'label': 'Offset Correction'},
                'oracle_threshold': {'color': 'green', 'linestyle': ':', 'label': 'Oracle Threshold'}
            }
            style = styles.get(method, {})

            ax.errorbar(
                pi_values, pi_risk['mean'], yerr=pi_risk['std'],
                marker='o', capsize=5, linewidth=2, markersize=6, **style
            )

        ax.set_xlabel('Test Prevalence ($\\pi_{test}$)')
        ax.set_ylabel('Cost-Weighted Risk')
        ax.set_xscale('log')
        ax.set_xlim(0.008, 0.6)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'{dataset}')

    plt.suptitle(f'{title} ($c_{{10}}$={c10}, $c_{{01}}$={c01})')
    plt.tight_layout()
    save_figure(fig, filename)
    return fig

def plot_neff_vs_alpha(
    results: pd.DataFrame,
    title: str = 'Effective Sample Size vs Weight Factor',
    filename: str = 'neff_vs_alpha.png'
) -> plt.Figure:
    """Plot effective sample size as a function of alpha (weight factor)."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Group by dataset and model
    for (dataset, model), group in results.groupby(['dataset', 'model']):
        # Get mean Neff for each alpha
        alpha_neff = group.groupby('alpha')['neff'].agg(['mean', 'std'])
        alpha_values = alpha_neff.index.values

        ax.errorbar(
            alpha_values, alpha_neff['mean'], yerr=alpha_neff['std'],
            marker='o', capsize=5, label=f'{dataset} - {model}',
            linewidth=2, markersize=8
        )

    ax.set_xlabel('Weight Factor ($\\alpha$)')
    ax.set_ylabel('Effective Sample Size ($N_{eff}$)')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)

    save_figure(fig, filename)
    return fig

def plot_coefficient_angles(
    results: pd.DataFrame,
    title: str = 'Coefficient Angle Distribution vs Weight Factor',
    filename: str = 'coef_angle_vs_alpha.png'
) -> plt.Figure:
    """Plot coefficient angle stability as boxplots."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Prepare data for boxplot
    alpha_values = sorted(results['alpha'].unique())
    angle_data = []

    for alpha in alpha_values:
        alpha_results = results[results['alpha'] == alpha]
        if 'angle_deg' in alpha_results.columns:
            angles = alpha_results['angle_deg'].values
            angle_data.append(angles[~np.isnan(angles)])

    # Create boxplot
    bp = ax.boxplot(angle_data, positions=alpha_values, widths=0.15)

    ax.set_xlabel('Weight Factor ($\\alpha$)')
    ax.set_ylabel('Pairwise Angle (degrees)')
    ax.set_xscale('log')
    ax.set_xticks(alpha_values)
    ax.set_xticklabels(alpha_values)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title(title)

    save_figure(fig, filename)
    return fig

def plot_logit_variance(
    results: pd.DataFrame,
    title: str = 'Logit Score Variance vs Weight Factor',
    filename: str = 'logit_var_vs_alpha.png'
) -> plt.Figure:
    """Plot logit score variance as a function of alpha."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Group by dataset and model
    for (dataset, model), group in results.groupby(['dataset', 'model']):
        # Get mean variance for each alpha
        alpha_var = group.groupby('alpha')['logit_variance'].agg(['mean', 'std'])
        alpha_values = alpha_var.index.values

        ax.errorbar(
            alpha_values, alpha_var['mean'], yerr=alpha_var['std'],
            marker='o', capsize=5, label=f'{dataset} - {model}',
            linewidth=2, markersize=8
        )

    ax.set_xlabel('Weight Factor ($\\alpha$)')
    ax.set_ylabel('Mean Logit Variance')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)

    save_figure(fig, filename)
    return fig

def plot_precision_recall_vs_pi(
    results: pd.DataFrame,
    title: str = 'Operating Point Metrics vs Test Prevalence',
    filename_base: str = 'operating_point'
) -> Tuple[plt.Figure, plt.Figure]:
    """Plot precision and recall as functions of test prevalence."""
    # Precision plot
    fig_prec, ax_prec = plt.subplots(figsize=FIGURE_SIZE)

    # Recall plot
    fig_rec, ax_rec = plt.subplots(figsize=FIGURE_SIZE)

    # Plot for each method
    for method in ['nocorr', 'offset']:
        method_data = results[results['method'] == method]

        # Precision
        pi_prec = method_data.groupby('pi_test')['precision'].agg(['mean', 'std'])
        pi_values = pi_prec.index.values

        style = {'nocorr': 'r--', 'offset': 'b-'}[method]
        label = {'nocorr': 'No Correction', 'offset': 'Offset Correction'}[method]

        ax_prec.errorbar(
            pi_values, pi_prec['mean'], yerr=pi_prec['std'],
            marker='o', capsize=5, label=label, linewidth=2, markersize=6
        )

        # Recall
        pi_rec = method_data.groupby('pi_test')['recall'].agg(['mean', 'std'])
        ax_rec.errorbar(
            pi_values, pi_rec['mean'], yerr=pi_rec['std'],
            marker='o', capsize=5, label=label, linewidth=2, markersize=6
        )

    # Add target line for precision
    ax_prec.axhline(y=0.95, color='black', linestyle=':', label='Target', linewidth=1)

    # Configure axes
    for ax, metric in [(ax_prec, 'Precision'), (ax_rec, 'Recall')]:
        ax.set_xlabel('Test Prevalence ($\\pi_{test}$)')
        ax.set_ylabel(metric)
        ax.set_xscale('log')
        ax.set_xlim(0.008, 0.6)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'{metric} vs Test Prevalence')

    save_figure(fig_prec, f'{filename_base}_precision.png')
    save_figure(fig_rec, f'{filename_base}_recall.png')

    return fig_prec, fig_rec

def plot_concept_drift_results(
    results: pd.DataFrame,
    title: str = 'Performance Under Concept Drift',
    filename: str = 'concept_drift.png'
) -> plt.Figure:
    """Plot performance comparison under concept drift."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Metrics to plot
    metrics = ['auc', 'risk']
    axes = [ax1, ax2]

    for ax, metric in zip(axes, metrics):
        # Group by method
        method_stats = results.groupby('method')[metric].agg(['mean', 'std'])

        # Sort methods for consistent ordering
        methods = ['nocorr', 'offset', 'retrain']
        labels = ['No Correction', 'Offset Correction', 'Retraining']
        colors = ['red', 'blue', 'green']

        x = np.arange(len(methods))
        means = [method_stats.loc[m, 'mean'] if m in method_stats.index else 0
                for m in methods]
        stds = [method_stats.loc[m, 'std'] if m in method_stats.index else 0
                for m in methods]

        bars = ax.bar(x, means, yerr=stds, capsize=10, color=colors, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel(metric.upper() if metric == 'auc' else 'Cost-Weighted Risk')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(title)
    plt.tight_layout()
    save_figure(fig, filename)
    return fig

def create_all_experiment_plots(results_dict: Dict[str, pd.DataFrame]) -> None:
    """Create all plots for all experiments."""
    logging.info("Creating all experiment plots...")

    # Experiment 1 plots
    if 'exp1' in results_dict:
        plot_auc_vs_pi(results_dict['exp1'], filename='exp1_auc_vs_pi.png')
        plot_risk_vs_pi(results_dict['exp1'], filename='exp1_risk_vs_pi.png')

    # Experiment 2 plots (combined AUC and PR-AUC)
    if 'exp2' in results_dict:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        data = results_dict['exp2']

        # Check which AUC column name is used
        auc_col_exp2 = 'auc' if 'auc' in data.columns else 'roc_auc'

        for metric, label, color in [(auc_col_exp2, 'AUC', 'blue'), ('pr_auc', 'PR-AUC', 'red')]:
            pi_metric = data.groupby('pi_test')[metric].agg(['mean', 'std'])
            ax.errorbar(
                pi_metric.index, pi_metric['mean'], yerr=pi_metric['std'],
                marker='o', label=label, color=color, linewidth=2
            )

        ax.set_xlabel('Test Prevalence ($\\pi_{test}$)')
        ax.set_ylabel('Score')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('AUC (invariant) vs PR-AUC (prevalence-dependent)')
        save_figure(fig, 'exp2_auc_prauc_comparison.png')

    # Experiment 3 plots
    if 'exp3' in results_dict:
        plot_neff_vs_alpha(results_dict['exp3'], filename='exp3_neff_vs_alpha.png')
        plot_coefficient_angles(results_dict['exp3'], filename='exp3_coef_angle_vs_alpha.png')
        plot_logit_variance(results_dict['exp3'], filename='exp3_logit_var_vs_alpha.png')

    # Experiment 4 plots
    if 'exp4' in results_dict:
        plot_precision_recall_vs_pi(results_dict['exp4'])

    # Experiment 5 plots
    if 'exp5' in results_dict:
        plot_concept_drift_results(results_dict['exp5'], filename='exp5_concept_drift.png')

    logging.info("All plots created successfully")