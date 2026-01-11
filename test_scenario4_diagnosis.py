#!/usr/bin/env python3
"""Diagnose Scenario 4: Check if covariate shift is truly benign"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from labelshift_drift.simulation.data_generators import ScenarioGenerator

print("Scenario 4 Diagnosis: Is the covariate shift truly benign?")
print("=" * 60)

SEED = 42
np.random.seed(SEED)

# Generate data
generator = ScenarioGenerator(d=10, delta=2.0, pi_ref=0.02, seed=SEED)
(X_train, Y_train), (X_val, Y_val) = generator.generate_reference(10000, 4000)

# Train model
model = LogisticRegression(max_iter=1000, random_state=SEED)
model.fit(X_train, Y_train)

# Check model weights
print("\n1. Model Coefficients:")
print(f"   Feature 0 (signal): {model.coef_[0][0]:.4f}")
print(f"   Feature 5 (shift):  {model.coef_[0][5]:.4f}")
print(f"   All coefficients: {model.coef_[0]}")

# Generate Scenario 4 data
X_s4, Y_s4, shift_ind = generator.scenario_4_covariate_shift_benign(
    20000, drift_time=10000, shift_dim=5, shift_amount=3.0
)

# Score before and after shift
S_before = model.predict_proba(X_s4[:10000])[:, 1]
S_after = model.predict_proba(X_s4[10000:])[:, 1]

print(f"\n2. Score Statistics:")
print(f"   Before shift: mean={S_before.mean():.4f}, std={S_before.std():.4f}")
print(f"   After shift:  mean={S_after.mean():.4f}, std={S_after.std():.4f}")
print(f"   Difference in means: {abs(S_after.mean() - S_before.mean()):.4f}")

# Check class-conditional scores
S0_before = S_before[Y_s4[:10000] == 0]
S1_before = S_before[Y_s4[:10000] == 1]
S0_after = S_after[Y_s4[10000:] == 0]
S1_after = S_after[Y_s4[10000:] == 1]

print(f"\n3. Class-Conditional Score Changes:")
print(f"   Class 0 before: mean={S0_before.mean():.4f}")
print(f"   Class 0 after:  mean={S0_after.mean():.4f}")
print(f"   Change: {abs(S0_after.mean() - S0_before.mean()):.4f}")
print()
print(f"   Class 1 before: mean={S1_before.mean():.4f}")
print(f"   Class 1 after:  mean={S1_after.mean():.4f}")
print(f"   Change: {abs(S1_after.mean() - S1_before.mean()):.4f}")

print("\n4. Interpretation:")
weight_dim5 = abs(model.coef_[0][5])
if weight_dim5 > 0.01:
    print(f"   The model has non-negligible weight ({weight_dim5:.4f}) on dimension 5")
    print(f"   A shift of +3.0 in this dimension WILL affect scores")
    print(f"   This is NOT a truly benign covariate shift!")
    print(f"   The detector is CORRECTLY flagging this as a geometric change")
else:
    print(f"   The model weight on dimension 5 is negligible ({weight_dim5:.4f})")
    print(f"   The shift should be benign")
    print(f"   False alarms may be due to sample size or calibration")

print("\n5. Recommendation:")
print("   For a truly benign covariate shift test:")
print("   - Use a dimension with exactly zero model weight")
print("   - Or use a feature NOT included in model training")
print("   - Or use larger sample sizes (n > 50k)")
print("=" * 60)
