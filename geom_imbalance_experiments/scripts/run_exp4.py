#!/usr/bin/env python
"""Run Experiment 4: Operating point metrics with offset correction."""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geomimb.experiments.exp4_operating_point_metrics import run_experiment
from src.geomimb.utils.logging import setup_logging

def main():
    parser = argparse.ArgumentParser(description='Run Experiment 4: Operating point metrics')
    parser.add_argument('--costs', type=float, nargs='+', action='append',
                       help='Cost settings as pairs (c10 c01). Can be specified multiple times.')
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

    # Parse cost settings
    if args.costs:
        cost_settings = []
        for cost_pair in args.costs:
            if len(cost_pair) == 2:
                cost_settings.append(tuple(cost_pair))
            else:
                parser.error("Each --costs must have exactly 2 values (c10 c01)")
    else:
        cost_settings = [(1.0, 1.0), (10.0, 1.0), (1.0, 10.0)]  # Default

    # Update output directory if needed
    if args.outdir != 'outputs':
        import src.geomimb.config as config
        config.OUTPUT_DIR = args.outdir
        config.TABLES_DIR = f'{args.outdir}/tables'
        config.FIGURES_DIR = f'{args.outdir}/figures'
        config.METADATA_DIR = f'{args.outdir}/metadata'

    # Run experiment
    results = run_experiment(
        cost_settings=cost_settings,
        save_results=not args.no_save
    )

    print(f"\nExperiment 4 completed successfully!")
    print(f"Total results: {len(results)}")
    print(f"Results saved to: {args.outdir}/tables/exp4_results.csv")

if __name__ == '__main__':
    main()