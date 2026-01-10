"""Input/output utilities for saving results and metadata."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any, Union, Optional
import subprocess

from ..config import OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, METADATA_DIR

def ensure_output_dirs() -> None:
    """Ensure all output directories exist."""
    for dir_path in [OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, METADATA_DIR]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def save_results_csv(
    results: pd.DataFrame,
    filename: str,
    subdir: str = 'tables'
) -> Path:
    """
    Save results DataFrame to CSV.

    Parameters
    ----------
    results : pd.DataFrame
        Results to save
    filename : str
        Filename (without path)
    subdir : str
        Subdirectory under outputs/

    Returns
    -------
    Path
        Path to saved file
    """
    ensure_output_dirs()
    filepath = Path(OUTPUT_DIR) / subdir / filename

    # Convert numpy types to Python types for better CSV compatibility
    results = results.copy()
    for col in results.columns:
        if results[col].dtype == np.bool_:
            results[col] = results[col].astype(bool)

    results.to_csv(filepath, index=False)
    logging.info(f"Saved results to {filepath}")
    return filepath

def load_results_csv(filename: str, subdir: str = 'tables') -> pd.DataFrame:
    """Load results from CSV."""
    filepath = Path(OUTPUT_DIR) / subdir / filename
    return pd.read_csv(filepath)

def save_metadata(
    metadata: Dict[str, Any],
    filename: str = 'run_metadata.json'
) -> Path:
    """
    Save experiment metadata to JSON.

    Parameters
    ----------
    metadata : dict
        Metadata to save
    filename : str
        Filename

    Returns
    -------
    Path
        Path to saved file
    """
    ensure_output_dirs()
    filepath = Path(METADATA_DIR) / filename

    # Add timestamp
    metadata['timestamp'] = datetime.now().isoformat()

    # Try to get git commit
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        metadata['git_commit'] = git_hash
    except:
        metadata['git_commit'] = 'unknown'

    # Convert numpy types
    metadata = convert_numpy_types(metadata)

    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Saved metadata to {filepath}")
    return filepath

def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def save_figure(
    fig,
    filename: str,
    dpi: int = 150,
    bbox_inches: str = 'tight'
) -> Path:
    """
    Save matplotlib figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Filename (without path)
    dpi : int
        Resolution
    bbox_inches : str
        Bounding box parameter

    Returns
    -------
    Path
        Path to saved file
    """
    ensure_output_dirs()
    filepath = Path(FIGURES_DIR) / filename

    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    logging.info(f"Saved figure to {filepath}")
    return filepath

def save_checks(checks: Dict[str, Any]) -> Path:
    """Save acceptance checks to JSON."""
    return save_metadata(checks, 'checks.json')

def create_paper_summary(
    all_results: pd.DataFrame,
    output_file: str = 'paper_summary.csv'
) -> pd.DataFrame:
    """
    Create paper-ready summary table from all results.

    Parameters
    ----------
    all_results : pd.DataFrame
        Combined results from all experiments
    output_file : str
        Output filename

    Returns
    -------
    pd.DataFrame
        Summary table
    """
    summary_rows = []

    # Group by dataset and model
    for (dataset, model), group in all_results.groupby(['dataset', 'model']):
        row = {
            'dataset': dataset,
            'model': model,
        }

        # AUC range across pi_test
        if 'auc' in group.columns:
            auc_values = group.groupby('pi_test')['auc'].mean()
            row['auc_min'] = auc_values.min()
            row['auc_max'] = auc_values.max()
            row['auc_range'] = auc_values.max() - auc_values.min()

        # Risk improvement at pi=0.01
        pi01_data = group[group['pi_test'] == 0.01]
        if len(pi01_data) > 0:
            for c10, c01 in [(1.0, 1.0), (10.0, 1.0), (1.0, 10.0)]:
                cost_data = pi01_data[
                    (pi01_data['c10'] == c10) & (pi01_data['c01'] == c01)
                ]
                if len(cost_data) > 0:
                    nocorr_risk = cost_data[
                        cost_data['method'] == 'nocorr'
                    ]['risk'].mean()
                    offset_risk = cost_data[
                        cost_data['method'] == 'offset'
                    ]['risk'].mean()
                    improvement = nocorr_risk - offset_risk
                    row[f'risk_improve_c{int(c10)}{int(c01)}'] = improvement

        # Neff values (from exp3 if available)
        if 'neff' in group.columns:
            for alpha in [1, 10, 50]:
                alpha_data = group[group.get('alpha', 1) == alpha]
                if len(alpha_data) > 0:
                    row[f'neff_alpha{alpha}'] = alpha_data['neff'].mean()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    save_results_csv(summary_df, output_file)

    return summary_df

def log_experiment_start(experiment_name: str) -> None:
    """Log the start of an experiment."""
    logging.info(f"\n{'='*60}")
    logging.info(f"Starting Experiment: {experiment_name}")
    logging.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"{'='*60}\n")