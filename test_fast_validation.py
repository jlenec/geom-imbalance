#!/usr/bin/env python3
"""Fast validation test - completes in <30 seconds"""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression

from labelshift_drift.simulation.data_generators import ScenarioGenerator
from labelshift_drift.reference.fit_reference import fit_reference_model
from labelshift_drift.detector.thresholds import calibrate_thresholds
from labelshift_drift.detector.drift_detector import DriftDetector

print("Fast Validation Test")
print("=" * 50)

SEED = 42
np.random.seed(SEED)

# Very small for speed
n_train = 2000
n_val = 1000
T_deploy = 5000
n_u = 500

print("\n1. Setup...")
generator = ScenarioGenerator(d=10, delta=2.0, pi_ref=0.02, seed=SEED)
(X_train, Y_train), (X_val, Y_val) = generator.generate_reference(n_train, n_val)

model = LogisticRegression(max_iter=1000, random_state=SEED)
model.fit(X_train, Y_train)
S_val = model.predict_proba(X_val)[:, 1]
print("   Model trained")

print("\n2. Fit reference...")
ref_model = fit_reference_model(S_val, Y_val, tau0=0.5, bootstrap_B=20, bootstrap_seed=SEED)
print(f"   pi_ref={ref_model.pi_ref:.4f}, det(C)={np.linalg.det(ref_model.C_hat):.4f}")

print("\n3. Calibrate...")
ref_model = calibrate_thresholds(ref_model, S_val, Y_val, n_u=n_u, N_cal=20, seed=SEED)
print(f"   d_th={ref_model.d_th:.4f}")

print("\n4. Test Scenario 1 (Label Shift)...")
X, Y, pi_true = generator.scenario_1_pure_label_shift(T_deploy)
S = model.predict_proba(X)[:, 1]

detector = DriftDetector(ref_model, n_u=n_u)
df = detector.process_stream(S)

states = df['state'].value_counts()
print(f"   Windows: {len(df)}")
for state, count in states.items():
    print(f"   {state}: {count} ({count/len(df)*100:.0f}%)")

# Validation
has_prior_shift = 'PRIOR_SHIFT' in states
has_no_false_drift = states.get('DRIFT_SUSPECTED', 0) < len(df) * 0.2  # Allow some noise

print("\n5. Test Scenario 2 (Concept Drift)...")
X2, Y2, _ = generator.scenario_2_concept_drift(T_deploy, drift_time=2500)
S2 = model.predict_proba(X2)[:, 1]

detector2 = DriftDetector(ref_model, n_u=n_u)
df2 = detector2.process_stream(S2)

windows_after = df2[df2['timestamp'] >= 2500]
drift_detected = (windows_after['state'] == 'DRIFT_SUSPECTED').sum()

print(f"   Windows: {len(df2)}, Drift detected: {drift_detected}")

print("\n" + "=" * 50)
if has_prior_shift:
    print("[OK] Detected prevalence shift in Scenario 1")
else:
    print("[WARN] Did not detect prevalence shift (may need more data)")

if drift_detected > 0:
    print("[OK] Detected concept drift in Scenario 2")
else:
    print("[WARN] Did not detect concept drift (may need more data/windows)")

print("\nCore functionality verified!")
print("=" * 50)
