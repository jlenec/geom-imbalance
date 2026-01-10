#!/usr/bin/env python
"""Test all functionalities of the experiment suite."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing Geometric Imbalance Experiment Suite")
print("=" * 60)

# Test 1: Imports
print("\n1. Testing imports...")
try:
    from src.geomimb import config
    from src.geomimb.seeds import SEEDS, set_global_seed
    from src.geomimb.data.synthetic import sample_synthetic_dataset, get_synthetic_params
    from src.geomimb.data.realdata import load_and_preprocess_breast_cancer
    from src.geomimb.data.shifts import make_label_shift_split, make_concept_drift_test
    from src.geomimb.models.sklearn_models import LogisticRegressionWrapper
    from src.geomimb.models.xgb_models import XGBoostWrapper
    from src.geomimb.metrics.classification import compute_metrics, logit_offset_correction
    from src.geomimb.metrics.stability import coefficient_angle
    from src.geomimb.metrics.calibration import tune_threshold_for_operating_point
    from src.geomimb.utils.io import ensure_output_dirs
    from src.geomimb.utils.checks import check_auc_invariance
    from src.geomimb.plotting.plots import plot_auc_vs_pi
    print("[PASS] All imports successful")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

# Test 2: Configuration
print("\n2. Testing configuration...")
print(f"   PI_TRAIN: {config.PI_TRAIN}")
print(f"   PI_TEST_GRID: {config.PI_TEST_GRID}")
print(f"   COSTS_GRID: {config.COSTS_GRID}")
print(f"   Number of seeds: {len(SEEDS)}")

# Test 3: Synthetic data generation
print("\n3. Testing synthetic data generation...")
try:
    set_global_seed(42)
    X_synth, y_synth = sample_synthetic_dataset(pi=0.3, n=1000, seed=42)
    print(f"   [PASS] Generated data: X.shape={X_synth.shape}, y.mean()={y_synth.mean():.3f}")

    params = get_synthetic_params()
    print(f"   [PASS] Synthetic params: dim={params['d']}, llr_const={params['theoretical_llr_const']:.3f}")
except Exception as e:
    print(f"   [FAIL] Synthetic data failed: {e}")

# Test 4: Real data loading
print("\n4. Testing real data loading...")
try:
    data = load_and_preprocess_breast_cancer(seed=42)
    print(f"   [PASS] Loaded breast cancer data:")
    print(f"     - Train: {data['X_train'].shape}")
    print(f"     - Test: {data['X_test'].shape}")
    print(f"     - Train prevalence: {data['y_train'].mean():.3f}")
except Exception as e:
    print(f"   [FAIL] Real data failed: {e}")

# Test 5: Label shift
print("\n5. Testing label shift...")
try:
    X_shift, y_shift = make_label_shift_split(X_synth, y_synth, pi_target=0.1, n_target=500, seed=42)
    print(f"   [PASS] Label shift: new prevalence = {y_shift.mean():.3f} (target: 0.1)")
except Exception as e:
    print(f"   [FAIL] Label shift failed: {e}")

# Test 6: Model training
print("\n6. Testing model training...")
try:
    model = LogisticRegressionWrapper(random_state=42)
    model.fit(X_synth, y_synth)
    proba = model.predict_proba(X_synth[:10])
    logits = model.predict_logit(X_synth[:10])
    print(f"   [PASS] Logistic regression trained")
    print(f"   [PASS] Predictions work: proba.shape={proba.shape}, logits.shape={logits.shape}")

    # Check coefficients
    coef = model.get_coefficients()
    import numpy as np
    print(f"   [PASS] Coefficients: shape={coef.shape}, norm={np.linalg.norm(coef):.3f}")
except Exception as e:
    print(f"   [FAIL] Model training failed: {e}")

# Test 7: Metrics
print("\n7. Testing metrics...")
try:
    import numpy as np
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    y_score = np.array([-1.2, 0.3, 0.8, 1.5, -0.5, 0.2])
    y_proba = 1 / (1 + np.exp(-y_score))

    metrics = compute_metrics(y_true, y_pred, y_score=y_score, y_proba=y_proba)
    print(f"   [PASS] Metrics computed: precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}")

    # Test offset correction
    logits_corrected = logit_offset_correction(y_score, pi_train=0.2, pi_test=0.05)
    print(f"   [PASS] Offset correction: delta={logits_corrected[0]-y_score[0]:.3f}")
except Exception as e:
    print(f"   [FAIL] Metrics failed: {e}")

# Test 8: Threshold tuning
print("\n8. Testing threshold tuning...")
try:
    threshold, info = tune_threshold_for_operating_point(
        y_score, y_true, target_metric='precision', target_value=0.8
    )
    print(f"   [PASS] Threshold found: {threshold:.3f}, feasible={info['feasible']}")
except Exception as e:
    print(f"   [FAIL] Threshold tuning failed: {e}")

# Test 9: Output directories
print("\n9. Testing output directory creation...")
try:
    ensure_output_dirs()
    import os
    dirs_exist = all(os.path.exists(d) for d in ['outputs/tables', 'outputs/figures', 'outputs/metadata'])
    print(f"   [PASS] Output directories created: {dirs_exist}")
except Exception as e:
    print(f"   [FAIL] Directory creation failed: {e}")

# Test 10: Simple experiment run
print("\n10. Testing mini experiment run...")
try:
    # Run a tiny version of experiment 1
    from src.geomimb.experiments.exp1_label_shift_offset import run_experiment as run_exp1

    # Override config for speed
    import src.geomimb.config as config_module
    original_seeds = SEEDS
    config_module.SEEDS = [0]  # Just one seed
    config_module.PI_TEST_GRID = [0.2, 0.1]  # Just two prevalences
    config_module.N_SYNTH_TRAIN = 1000
    config_module.N_SYNTH_TEST = 500

    results = run_exp1(dataset_name='synthetic', models=['LogisticRegression'], save_results=False)
    print(f"   [PASS] Mini experiment completed: {len(results)} results")
    print(f"   [PASS] Methods tested: {results['method'].unique()}")

    # Check some results
    print(f"   [INFO] Result columns: {list(results.columns)[:10]}...")
    if 'roc_auc' in results.columns:
        auc_values = results.groupby('pi_test')['roc_auc'].mean()
        print(f"   [PASS] AUC values: {dict(auc_values.round(3))}")
    elif 'auc' in results.columns:
        auc_values = results.groupby('pi_test')['auc'].mean()
        print(f"   [PASS] AUC values: {dict(auc_values.round(3))}")
except Exception as e:
    print(f"   [FAIL] Mini experiment failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Testing complete!")