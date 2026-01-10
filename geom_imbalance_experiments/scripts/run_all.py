#!/usr/bin/env python
"""Run all experiments and generate all outputs."""

import argparse
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geomimb.utils.logging import setup_logging, log_experiment_start
from src.geomimb.utils.io import ensure_output_dirs, save_metadata, create_paper_summary
from src.geomimb.utils.checks import run_all_checks
from src.geomimb.plotting.plots import create_all_experiment_plots

# Import all experiments
from src.geomimb.experiments.exp1_label_shift_offset import run_experiment as run_exp1
from src.geomimb.experiments.exp2_auc_invariance import run_experiment as run_exp2
from src.geomimb.experiments.exp3_weighting_neff_instability import run_experiment as run_exp3
from src.geomimb.experiments.exp4_operating_point_metrics import run_experiment as run_exp4
from src.geomimb.experiments.exp5_concept_drift_control import run_experiment as run_exp5

def main():
    parser = argparse.ArgumentParser(description='Run all experiments')
    parser.add_argument('--outdir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--n_jobs', type=int, default=1,
                       help='Number of parallel jobs (not implemented yet)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--skip', type=str, nargs='+', default=[],
                       help='Experiments to skip (e.g., exp1 exp2)')

    args = parser.parse_args()

    # Setup logging
    log_file = os.path.join(args.outdir, 'experiment_log.txt')
    setup_logging(level=args.log_level, log_file=log_file)

    # Update output directory if needed
    if args.outdir != 'outputs':
        import src.geomimb.config as config
        config.OUTPUT_DIR = args.outdir
        config.TABLES_DIR = f'{args.outdir}/tables'
        config.FIGURES_DIR = f'{args.outdir}/figures'
        config.METADATA_DIR = f'{args.outdir}/metadata'

    # Ensure output directories exist
    ensure_output_dirs()

    # Track timing
    start_time = time.time()

    print("="*60)
    print("Running all experiments for:")
    print("'A Geometric Theory of Learning Under Class Imbalance'")
    print("="*60)

    # Store all results
    all_results = {}

    # Run experiments
    experiments = [
        ('exp1', run_exp1, "Experiment 1: Label shift offset correction"),
        ('exp2', run_exp2, "Experiment 2: AUC invariance demonstration"),
        ('exp3', run_exp3, "Experiment 3: Weighting Neff and instability"),
        ('exp4', run_exp4, "Experiment 4: Operating point metrics"),
        ('exp5', run_exp5, "Experiment 5: Concept drift control")
    ]

    for exp_name, run_func, description in experiments:
        if exp_name in args.skip:
            print(f"\nSkipping {exp_name} (as requested)")
            continue

        print(f"\n{'='*60}")
        print(f"Running {description}")
        print(f"{'='*60}")

        try:
            exp_start = time.time()
            results = run_func(save_results=True)
            all_results[exp_name] = results
            exp_duration = time.time() - exp_start
            print(f"\n{exp_name} completed in {exp_duration:.1f} seconds")
        except Exception as e:
            print(f"\nERROR in {exp_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Run all acceptance checks
    if all_results:
        print(f"\n{'='*60}")
        print("Running acceptance checks")
        print(f"{'='*60}")

        checks = run_all_checks(all_results)
        from src.geomimb.utils.io import save_checks
        save_checks(checks)

        # Print check summary
        print("\nAcceptance Check Summary:")
        for check_name, check_result in checks.items():
            if check_name == 'overall_passed':
                continue
            status = "PASSED" if check_result.get('passed', True) else "FAILED"
            print(f"  {check_name}: {status}")

        print(f"\nOverall: {'PASSED' if checks['overall_passed'] else 'FAILED'}")

    # Create all plots
    if all_results:
        print(f"\n{'='*60}")
        print("Creating plots")
        print(f"{'='*60}")

        try:
            create_all_experiment_plots(all_results)
            print("All plots created successfully")
        except Exception as e:
            print(f"ERROR creating plots: {str(e)}")
            import traceback
            traceback.print_exc()

    # Create paper summary table
    if all_results:
        print(f"\n{'='*60}")
        print("Creating paper summary table")
        print(f"{'='*60}")

        # Combine all results
        import pandas as pd
        combined_results = pd.concat(list(all_results.values()), ignore_index=True)
        summary_df = create_paper_summary(combined_results)
        print(f"Paper summary saved with {len(summary_df)} rows")

    # Final summary
    total_duration = time.time() - start_time
    print(f"\n{'='*60}")
    print("EXPERIMENT SUITE COMPLETED")
    print(f"{'='*60}")
    print(f"Total runtime: {total_duration/60:.1f} minutes")
    print(f"Output directory: {args.outdir}")
    print(f"Log file: {log_file}")

    # List generated files
    print("\nGenerated files:")
    for subdir in ['tables', 'figures', 'metadata']:
        dir_path = os.path.join(args.outdir, subdir)
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            if files:
                print(f"\n{subdir}:")
                for f in sorted(files):
                    print(f"  - {f}")

if __name__ == '__main__':
    main()