"""
Experiment driver script for drift detection experiments
"""
import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from labelshift_drift.simulation.data_generators import ScenarioGenerator, apply_score_transform
from labelshift_drift.reference.fit_reference import fit_reference_model
from labelshift_drift.detector.thresholds import calibrate_thresholds
from labelshift_drift.detector.drift_detector import DriftDetector
from labelshift_drift.viz.plots import (
    plot_drift_detection_summary,
    plot_state_distribution
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_scenario(config: dict, output_dir: str):
    """
    Run a single scenario experiment.

    Args:
        config: configuration dictionary
        output_dir: output directory for artifacts
    """
    print(f"\n{'='*60}")
    print(f"Running: {config['name']}")
    print(f"{'='*60}\n")

    # Create output directories
    fig_dir = os.path.join(output_dir, 'figures')
    metrics_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Set seed
    np.random.seed(config['seed'])

    # Initialize generator
    generator = ScenarioGenerator(
        d=config['d'],
        delta=config['delta'],
        pi_ref=config['pos_rate_ref'],
        seed=config['seed']
    )

    # Generate reference data
    print("Generating reference data...")
    (X_train, Y_train), (X_val, Y_val) = generator.generate_reference(
        config['n_ref_train'],
        config['n_ref_val']
    )

    # Train model
    print("Training logistic regression model...")
    model = LogisticRegression(max_iter=1000, random_state=config['seed'])
    model.fit(X_train, Y_train)

    # Get reference scores
    S_train = model.predict_proba(X_train)[:, 1]
    S_val = model.predict_proba(X_val)[:, 1]

    print(f"Reference AUC: {_compute_auc(S_val, Y_val):.4f}")

    # Fit reference model
    print("Fitting reference model...")
    ref_model = fit_reference_model(
        S_val,
        Y_val,
        tau0=config['tau0'],
        lambda_ewma=config['lambda_ewma'],
        bootstrap_B=config['bootstrap_B'],
        bootstrap_seed=config['seed']
    )

    print(f"  pi_ref = {ref_model.pi_ref:.4f}")
    print(f"  det(C) = {np.linalg.det(ref_model.C_hat):.4f}")
    print(f"  delta_C = {ref_model.delta_C:.4f}")

    # Calibrate thresholds
    print("Calibrating thresholds...")
    ref_model = calibrate_thresholds(
        ref_model,
        S_val,
        Y_val,
        n_u=config['n_u'],
        N_cal=config['N_cal'],
        alpha_d=config['alpha_d'],
        alpha_r=config['alpha_r'],
        alpha_pi=config['alpha_pi'],
        seed=config['seed']
    )

    print(f"  d_th = {ref_model.d_th:.4f}")
    print(f"  r_th = {ref_model.r_th:.4f}" if ref_model.r_th else "  r_th = None (BBSE unstable)")
    print(f"  pi_th = {ref_model.pi_th:.4f}")

    # Generate deployment data based on scenario
    print(f"\nGenerating deployment data for scenario {config['scenario']}...")

    scenario = config['scenario']
    T = config['T_deploy']

    pi_true = None
    drift_indicator = None

    if scenario == 1:
        # Pure label shift
        X_deploy, Y_deploy, pi_true = generator.scenario_1_pure_label_shift(T)
        S_deploy = model.predict_proba(X_deploy)[:, 1]

    elif scenario == 2:
        # Concept drift
        X_deploy, Y_deploy, drift_indicator = generator.scenario_2_concept_drift(
            T, drift_time=config['drift_time']
        )
        S_deploy = model.predict_proba(X_deploy)[:, 1]

    elif scenario == 3:
        # Score mapping drift
        X_deploy, Y_deploy, _, drift_indicator = generator.scenario_3_score_mapping_drift(
            T, drift_time=config.get('drift_time', 80000)
        )
        S_original = model.predict_proba(X_deploy)[:, 1]

        # Apply score transformation
        S_deploy = apply_score_transform(
            S_original,
            drift_time=config.get('drift_time', 80000),
            a=0.7,
            b=0.4
        )

    elif scenario == 4:
        # Covariate shift (benign)
        X_deploy, Y_deploy, drift_indicator = generator.scenario_4_covariate_shift_benign(
            T, drift_time=config.get('drift_time', 80000)
        )
        S_deploy = model.predict_proba(X_deploy)[:, 1]

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    print(f"  Deployment samples: {len(S_deploy)}")
    print(f"  Deployment pi: {np.mean(Y_deploy):.4f}")

    # Run drift detector
    print("\nRunning drift detector...")
    detector = DriftDetector(ref_model, n_u=config['n_u'], step_u=config['step_u'])
    df_reports = detector.process_stream(S_deploy)

    print(f"  Processed {len(df_reports)} windows")
    print(f"\nState distribution:")
    for state, count in df_reports['state'].value_counts().items():
        print(f"    {state}: {count} ({count/len(df_reports)*100:.1f}%)")

    # Save reports
    csv_path = os.path.join(metrics_dir, f"scenario_{scenario}_reports.csv")
    df_reports.to_csv(csv_path, index=False)
    print(f"\nSaved reports to {csv_path}")

    # Visualization
    print("\nGenerating plots...")

    # Main 4-panel plot
    fig_path = os.path.join(fig_dir, f"scenario_{scenario}_drift_detection.png")
    plot_drift_detection_summary(
        df_reports,
        ref_model,
        pi_true=pi_true,
        drift_indicator=drift_indicator,
        save_path=fig_path
    )
    print(f"  Saved figure to {fig_path}")

    # State distribution
    fig_path = os.path.join(fig_dir, f"scenario_{scenario}_state_distribution.png")
    plot_state_distribution(df_reports, save_path=fig_path)
    print(f"  Saved figure to {fig_path}")

    print(f"\nScenario {scenario} complete!\n")


def _compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUC from scores and labels"""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels, scores)


def main():
    parser = argparse.ArgumentParser(description='Run drift detection experiments')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output', type=str, default='artifacts', help='Output directory')
    parser.add_argument('--seed', type=int, default=None, help='Override random seed')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override seed if specified
    if args.seed is not None:
        config['seed'] = args.seed

    # Run scenario
    run_scenario(config, args.output)


if __name__ == '__main__':
    main()
