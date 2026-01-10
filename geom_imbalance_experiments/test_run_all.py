#!/usr/bin/env python
"""Quick test of all experiments with reduced parameters."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Override configuration for faster testing
import src.geomimb.config as config
config.SEEDS = [0, 1]  # Just 2 seeds
config.PI_TEST_GRID = [0.2, 0.05]  # Just 2 prevalences
config.N_SYNTH_TRAIN = 5000
config.N_SYNTH_TEST = 2000
config.N_REAL_TRAIN = 2000
config.N_REAL_TEST = 1000
config.ALPHA_VALUES = [1, 10]  # Just 2 alpha values

# Now import and run
from src.geomimb.utils.logging import setup_logging
from src.geomimb.utils.io import ensure_output_dirs
from src.geomimb.experiments.exp1_label_shift_offset import run_experiment as run_exp1
from src.geomimb.experiments.exp2_auc_invariance import run_experiment as run_exp2
from src.geomimb.experiments.exp3_weighting_neff_instability import run_experiment as run_exp3
from src.geomimb.experiments.exp4_operating_point_metrics import run_experiment as run_exp4
from src.geomimb.experiments.exp5_concept_drift_control import run_experiment as run_exp5
from src.geomimb.utils.checks import run_all_checks
from src.geomimb.plotting.plots import create_all_experiment_plots

setup_logging(level='INFO')
ensure_output_dirs()

print("Running quick test of all experiments...")
print("="*60)

all_results = {}

# Experiment 1
print("\nTEST: Experiment 1")
try:
    results1 = run_exp1(dataset_name='synthetic', models=['LogisticRegression'], save_results=True)
    all_results['exp1'] = results1
    print(f"[PASS] Exp1 completed with {len(results1)} results")
except Exception as e:
    print(f"[FAIL] Exp1 failed: {e}")

# Experiment 2
print("\nTEST: Experiment 2")
try:
    results2 = run_exp2(dataset_name='synthetic', models=['LogisticRegression'], save_results=True)
    all_results['exp2'] = results2
    print(f"[PASS] Exp2 completed with {len(results2)} results")
except Exception as e:
    print(f"[FAIL] Exp2 failed: {e}")

# Experiment 3
print("\nTEST: Experiment 3")
try:
    results3 = run_exp3(dataset_name='synthetic', save_results=True)
    all_results['exp3'] = results3
    print(f"[PASS] Exp3 completed with {len(results3)} results")
except Exception as e:
    print(f"[FAIL] Exp3 failed: {e}")

# Experiment 4
print("\nTEST: Experiment 4")
try:
    results4 = run_exp4(cost_settings=[(1.0, 1.0)], save_results=True)
    all_results['exp4'] = results4
    print(f"[PASS] Exp4 completed with {len(results4)} results")
except Exception as e:
    print(f"[FAIL] Exp4 failed: {e}")

# Experiment 5
print("\nTEST: Experiment 5")
try:
    results5 = run_exp5(models=['LogisticRegression'], save_results=True)
    all_results['exp5'] = results5
    print(f"[PASS] Exp5 completed with {len(results5)} results")
except Exception as e:
    print(f"[FAIL] Exp5 failed: {e}")

# Run checks
print("\nTEST: Acceptance checks")
try:
    checks = run_all_checks(all_results)
    overall_pass = checks.get('overall_passed', False)
    print(f"[{'PASS' if overall_pass else 'FAIL'}] Overall acceptance: {overall_pass}")
    for check_name, result in checks.items():
        if check_name != 'overall_passed':
            passed = result.get('passed', True)
            print(f"  - {check_name}: {'PASS' if passed else 'FAIL'}")
except Exception as e:
    print(f"[FAIL] Checks failed: {e}")

# Create plots
print("\nTEST: Plot generation")
try:
    create_all_experiment_plots(all_results)
    print("[PASS] All plots created")
except Exception as e:
    print(f"[FAIL] Plot generation failed: {e}")

print("\n" + "="*60)
print("Quick test complete!")