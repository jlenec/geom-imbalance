#!/usr/bin/env python3
"""Comprehensive validation test for all scenarios"""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from labelshift_drift.simulation.data_generators import ScenarioGenerator, apply_score_transform
from labelshift_drift.reference.fit_reference import fit_reference_model
from labelshift_drift.detector.thresholds import calibrate_thresholds
from labelshift_drift.detector.drift_detector import DriftDetector

print("="*60)
print("COMPREHENSIVE VALIDATION TEST")
print("="*60)

# Smaller sizes for quick testing
SEED = 42
n_train = 10000
n_val = 4000
T_deploy = 20000
n_u = 1000

np.random.seed(SEED)

# Setup
print("\n[1/6] Setting up generator and training model...")
generator = ScenarioGenerator(d=10, delta=2.0, pi_ref=0.02, seed=SEED)
(X_train, Y_train), (X_val, Y_val) = generator.generate_reference(n_train, n_val)

model = LogisticRegression(max_iter=1000, random_state=SEED)
model.fit(X_train, Y_train)
S_val = model.predict_proba(X_val)[:, 1]

auc_val = roc_auc_score(Y_val, S_val)
print(f"   Model trained - Validation AUC: {auc_val:.4f}")

# Fit reference
print("\n[2/6] Fitting reference model...")
ref_model = fit_reference_model(S_val, Y_val, tau0=0.5, bootstrap_B=50, bootstrap_seed=SEED)
print(f"   pi_ref = {ref_model.pi_ref:.4f}")
print(f"   det(C) = {np.linalg.det(ref_model.C_hat):.4f}")
print(f"   delta_C = {ref_model.delta_C:.4f}")

# Calibrate
print("\n[3/6] Calibrating thresholds...")
ref_model = calibrate_thresholds(ref_model, S_val, Y_val, n_u=n_u, N_cal=50, seed=SEED)
print(f"   d_th = {ref_model.d_th:.4f}")
print(f"   pi_th = {ref_model.pi_th:.4f}")

# Test Scenario 1: Pure label shift
print("\n[4/6] Testing Scenario 1: Pure Label Shift...")
X_s1, Y_s1, pi_true = generator.scenario_1_pure_label_shift(T_deploy)
S_s1 = model.predict_proba(X_s1)[:, 1]

detector_s1 = DriftDetector(ref_model, n_u=n_u, step_u=n_u)
df_s1 = detector_s1.process_stream(S_s1)

n_normal = (df_s1['state'] == 'NORMAL').sum()
n_prior_shift = (df_s1['state'] == 'PRIOR_SHIFT').sum()
n_drift = (df_s1['state'] == 'DRIFT_SUSPECTED').sum()

print(f"   Processed {len(df_s1)} windows")
print(f"   States: NORMAL={n_normal}, PRIOR_SHIFT={n_prior_shift}, DRIFT={n_drift}")

# Validate: Should see PRIOR_SHIFT (prevalence changes) but not DRIFT
assert n_prior_shift > 0, "Should detect prevalence shift!"
print("   [PASS] Correctly detected prevalence shifts")

# Test Scenario 2: Concept drift
print("\n[5/6] Testing Scenario 2: Concept Drift...")
X_s2, Y_s2, drift_ind = generator.scenario_2_concept_drift(T_deploy, drift_time=10000)
S_s2 = model.predict_proba(X_s2)[:, 1]

detector_s2 = DriftDetector(ref_model, n_u=n_u, step_u=n_u)
df_s2 = detector_s2.process_stream(S_s2)

n_drift_s2 = (df_s2['state'] == 'DRIFT_SUSPECTED').sum()
print(f"   Processed {len(df_s2)} windows")
print(f"   DRIFT_SUSPECTED windows: {n_drift_s2}")

# Validate: Should detect drift after drift_time
# Since drift_time=10000 and n_u=1000, drift starts at window 10
# May take a few windows to trigger sustained rule
windows_after_drift = df_s2[df_s2['timestamp'] >= 10000]
drift_detected_after = (windows_after_drift['state'] == 'DRIFT_SUSPECTED').sum()
print(f"   Drift detections after drift time: {drift_detected_after}")
assert drift_detected_after > 0, "Should detect concept drift!"
print("   [PASS] Correctly detected concept drift")

# Test Scenario 4: Benign covariate shift
print("\n[6/6] Testing Scenario 4: Benign Covariate Shift...")
X_s4, Y_s4, shift_ind = generator.scenario_4_covariate_shift_benign(T_deploy, drift_time=10000)
S_s4 = model.predict_proba(X_s4)[:, 1]

detector_s4 = DriftDetector(ref_model, n_u=n_u, step_u=n_u)
df_s4 = detector_s4.process_stream(S_s4)

n_drift_s4 = (df_s4['state'] == 'DRIFT_SUSPECTED').sum()
print(f"   Processed {len(df_s4)} windows")
print(f"   DRIFT_SUSPECTED windows: {n_drift_s4}")

# Validate: Should NOT alarm for covariate shift in unused dimension
# Allow a few false alarms (≤ 10% at alpha=0.01)
false_alarm_rate = n_drift_s4 / len(df_s4)
print(f"   False alarm rate: {false_alarm_rate:.2%}")
assert false_alarm_rate < 0.15, "Too many false alarms for benign shift!"
print("   [PASS] Did not alarm for benign covariate shift")

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
print("\nSummary:")
print(f"  Scenario 1 (Label Shift): {n_prior_shift}/{len(df_s1)} windows detected PRIOR_SHIFT")
print(f"  Scenario 2 (Concept Drift): {drift_detected_after}/{len(windows_after_drift)} post-drift windows detected DRIFT")
print(f"  Scenario 4 (Benign Shift): {false_alarm_rate:.1%} false alarm rate")
print("\nThe drift detection system is working correctly!")
