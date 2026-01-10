#!/usr/bin/env python
"""Run Experiment 3: Weighting reduces Neff and increases instability."""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geomimb.experiments.exp3_weighting_neff_instability import run_experiment
from src.geomimb.utils.logging import setup_logging

def main():
    parser = argparse.ArgumentParser(description='Run Experiment 3: Weighting Neff and instability')
    parser.add_argument('--dataset', type=str, default='both',
                       choices=['synthetic', 'breast_cancer', 'both'],
                       help='Dataset to use')
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
        save_results=not args.no_save
    )

    print(f"\nExperiment 3 completed successfully!")
    print(f"Total results: {len(results)}")
    print(f"Results saved to: {args.outdir}/tables/exp3_results.csv")

if __name__ == '__main__':
    main()