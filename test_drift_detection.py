#!/usr/bin/env python3
"""Quick test of drift detection implementation"""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression

# Test imports
print('Testing imports...')
from labelshift_drift.simulation.data_generators import ScenarioGenerator
from labelshift_drift.reference.fit_reference import fit_reference_model
from labelshift_drift.detector.thresholds import calibrate_thresholds
from labelshift_drift.detector.drift_detector import DriftDetector

print('[OK] All imports successful!')

# Quick test with small data
print('\nRunning quick test with small dataset...')
np.random.seed(42)

# Generate tiny dataset
generator = ScenarioGenerator(d=10, delta=2.0, pi_ref=0.02, seed=42)
(X_train, Y_train), (X_val, Y_val) = generator.generate_reference(5000, 2000)

print(f'[OK] Generated data - Train: {X_train.shape}, Val: {X_val.shape}')

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, Y_train)
S_val = model.predict_proba(X_val)[:, 1]

print(f'[OK] Model trained - Val scores range: [{S_val.min():.3f}, {S_val.max():.3f}]')

# Fit reference
ref_model = fit_reference_model(S_val, Y_val, tau0=0.5, bootstrap_B=50, bootstrap_seed=42)
print(f'[OK] Reference model fitted - pi_ref={ref_model.pi_ref:.4f}, det(C)={np.linalg.det(ref_model.C_hat):.4f}')

# Calibrate (small)
ref_model = calibrate_thresholds(ref_model, S_val, Y_val, n_u=500, N_cal=20, seed=42)
print(f'[OK] Thresholds calibrated - d_th={ref_model.d_th:.4f}')

# Generate small deployment stream
X_deploy, Y_deploy, pi_true = generator.scenario_1_pure_label_shift(10000)
S_deploy = model.predict_proba(X_deploy)[:, 1]
print(f'[OK] Generated deployment stream - {len(S_deploy)} samples')

# Run detector
detector = DriftDetector(ref_model, n_u=500, step_u=500)
df = detector.process_stream(S_deploy)

print(f'[OK] Detector processed {len(df)} windows')
print('\nState distribution:')
for state, count in df['state'].value_counts().items():
    print(f'  {state}: {count} ({count/len(df)*100:.1f}%)')

print('\n' + '='*60)
print('SUCCESS! All tests passed!')
print('='*60)
