#!/usr/bin/env python
"""Run Experiment 1: Label shift offset correction."""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geomimb.experiments.exp1_label_shift_offset import run_experiment
from src.geomimb.utils.logging import setup_logging

def main():
    parser = argparse.ArgumentParser(description='Run Experiment 1: Label shift offset correction')
    parser.add_argument('--dataset', type=str, default='both',
                       choices=['synthetic', 'breast_cancer', 'both'],
                       help='Dataset to use')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['LogisticRegression', 'XGBoost'],
                       help='Models to run')
    parser.add_argument('--outdir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results')

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    # Update output directory if needed
    if args.outdir != 'outputs':
        import src.geomimb.config as config
        config.OUTPUT_DIR = args.outdir
        config.TABLES_DIR = f'{args.outdir}/tables'
        config.FIGURES_DIR = f'{args.outdir}/figures'
        config.METADATA_DIR = f'{args.outdir}/metadata'

    # Run experiment
    results = run_experiment(
        dataset_name=args.dataset,
        models=args.models,
        save_results=not args.no_save
    )

    print(f"\nExperiment 1 completed successfully!")
    print(f"Total results: {len(results)}")
    print(f"Results saved to: {args.outdir}/tables/exp1_results.csv")

if __name__ == '__main__':
    main()