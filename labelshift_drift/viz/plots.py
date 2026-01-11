"""
Plotting functions for drift detection results
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional, Tuple


def plot_drift_detection_summary(
    df: pd.DataFrame,
    ref_model,
    pi_true: Optional[np.ndarray] = None,
    drift_indicator: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
):
    """
    Create 4-panel time series plot for drift detection.

    Panels:
    1. Prevalence estimates (pi_hat_bbse, pi_ewma, pi_true)
    2. Mixture consistency (d_u_star with threshold)
    3. BBSE residual (r_u with threshold, if applicable)
    4. Operational threshold over time

    Background colors show controller state.

    Args:
        df: DataFrame of WindowReports
        ref_model: ReferenceModel with thresholds
        pi_true: true prevalence at each timestamp (optional)
        drift_indicator: drift indicator at each timestamp (optional)
        figsize: figure size
        save_path: path to save figure (optional)
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # State colors
    state_colors = {
        'NORMAL': 'lightgreen',
        'PRIOR_SHIFT': 'lightyellow',
        'DRIFT_SUSPECTED': 'lightcoral'
    }

    # Add state background
    for ax in axes:
        _add_state_background(ax, df, state_colors)

    # Panel 1: Prevalence
    ax = axes[0]
    ax.plot(df['timestamp'], df['pi_hat_bbse'], label='BBSE estimate', alpha=0.7, marker='o', markersize=2)
    ax.plot(df['timestamp'], df['pi_ewma'], label='EWMA', linewidth=2)
    ax.axhline(ref_model.pi_ref, color='black', linestyle='--', label='Reference', linewidth=1.5)

    if pi_true is not None:
        # Downsample pi_true to match window timestamps
        pi_true_at_windows = pi_true[df['timestamp'].values]
        ax.plot(df['timestamp'], pi_true_at_windows, label='True', linestyle=':', linewidth=2, color='red')

    ax.set_ylabel('Prevalence π')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Prevalence Estimates Over Time')

    # Panel 2: Mixture consistency
    ax = axes[1]
    ax.plot(df['timestamp'], df['d_u_star'], label='$d_u^\\star$', linewidth=2, color='blue')
    ax.axhline(ref_model.d_th, color='red', linestyle='--', label='Threshold $d_{th}$', linewidth=1.5)

    if drift_indicator is not None:
        _add_drift_shading(ax, df, drift_indicator)

    ax.set_ylabel('$d_u^\\star$')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Mixture Consistency (Geometric Validation)')

    # Panel 3: BBSE residual
    ax = axes[2]
    if df['r_u'].notna().any():
        ax.plot(df['timestamp'], df['r_u'], label='$r_u$', linewidth=2, color='purple')
        if ref_model.r_th is not None:
            ax.axhline(ref_model.r_th, color='red', linestyle='--', label='Threshold $r_{th}$', linewidth=1.5)
    else:
        ax.text(0.5, 0.5, 'BBSE Residual Not Computed\n(Ill-conditioned C)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)

    ax.set_ylabel('BBSE Residual $r_u$')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('BBSE Reconstruction Residual')

    # Panel 4: Operational threshold
    ax = axes[3]
    ax.plot(df['timestamp'], df['tau_operational'], label='$\\tau_{operational}$', linewidth=2, color='darkgreen')
    ax.axhline(ref_model.tau0, color='black', linestyle='--', label='$\\tau_0$ (reference)', linewidth=1.5)

    ax.set_ylabel('Threshold τ')
    ax.set_xlabel('Time (sample index)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Operational Decision Threshold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')

    plt.show()


def _add_state_background(ax, df: pd.DataFrame, state_colors: dict):
    """Add colored background for controller states"""
    states = df['state'].values
    timestamps = df['timestamp'].values

    current_state = states[0]
    start_idx = 0

    for i in range(1, len(states)):
        if states[i] != current_state:
            # Draw rectangle for previous state
            if current_state in state_colors:
                ax.axvspan(timestamps[start_idx], timestamps[i-1],
                          alpha=0.2, color=state_colors[current_state], zorder=0)

            current_state = states[i]
            start_idx = i

    # Draw last state
    if current_state in state_colors:
        ax.axvspan(timestamps[start_idx], timestamps[-1],
                  alpha=0.2, color=state_colors[current_state], zorder=0)


def _add_drift_shading(ax, df: pd.DataFrame, drift_indicator: np.ndarray):
    """Add shading for true drift periods"""
    timestamps = df['timestamp'].values
    drift_at_windows = drift_indicator[timestamps]

    in_drift = False
    start_idx = 0

    for i in range(len(drift_at_windows)):
        if drift_at_windows[i] > 0 and not in_drift:
            start_idx = i
            in_drift = True
        elif drift_at_windows[i] == 0 and in_drift:
            ax.axvspan(timestamps[start_idx], timestamps[i-1],
                      alpha=0.1, color='red', zorder=0, label='True Drift' if start_idx == 0 else '')
            in_drift = False

    if in_drift:
        ax.axvspan(timestamps[start_idx], timestamps[-1],
                  alpha=0.1, color='red', zorder=0, label='True Drift')


def plot_policy_comparison(
    metrics: dict,
    save_path: Optional[str] = None
):
    """
    Create bar chart comparing three policies:
    - Policy A: Always BBSE adapt
    - Policy B: Controller (full)
    - Policy C: Never adapt

    Args:
        metrics: dict of {policy_name: {metric_name: value}}
        save_path: path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    policies = list(metrics.keys())
    metric_names = ['f1', 'min_recall']

    # Plot each metric
    for i, metric_name in enumerate(metric_names):
        ax = axes[i]
        values = [metrics[p].get(metric_name, 0) for p in policies]

        ax.bar(policies, values, color=['red', 'green', 'blue'], alpha=0.7)
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(f'{metric_name.replace("_", " ").title()} by Policy')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')

    plt.show()


def plot_state_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Create bar chart showing fraction of windows in each state.

    Args:
        df: DataFrame of WindowReports
        save_path: path to save figure
    """
    state_counts = df['state'].value_counts()
    state_fractions = state_counts / len(df)

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {'NORMAL': 'lightgreen', 'PRIOR_SHIFT': 'lightyellow', 'DRIFT_SUSPECTED': 'lightcoral'}
    state_colors_list = [colors.get(s, 'gray') for s in state_fractions.index]

    ax.bar(state_fractions.index, state_fractions.values, color=state_colors_list, edgecolor='black')
    ax.set_ylabel('Fraction of Windows')
    ax.set_title('Controller State Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    for i, (state, frac) in enumerate(state_fractions.items()):
        ax.text(i, frac + 0.01, f'{frac*100:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')

    plt.show()
