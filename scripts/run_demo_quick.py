#!/usr/bin/env python3
"""
Quick demo with small data to show what figures will look like
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from labelshift_drift.simulation.data_generators import ScenarioGenerator, apply_score_transform
from labelshift_drift.reference.fit_reference import fit_reference_model
from labelshift_drift.detector.thresholds import calibrate_thresholds
from labelshift_drift.detector.drift_detector import DriftDetector
from labelshift_drift.viz.plots import plot_drift_detection_summary, plot_state_distribution

print("="*70)
print("QUICK DEMO - Generating Sample Figures")
print("="*70)

# Small data for speed
SEED = 42
n_train = 10000
n_val = 4000
T_deploy = 40000
n_u = 2000

np.random.seed(SEED)
os.makedirs("artifacts/demo_figures", exist_ok=True)
os.makedirs("artifacts/demo_metrics", exist_ok=True)

# Setup
print("\nSetting up...")
generator = ScenarioGenerator(d=10, delta=2.0, pi_ref=0.02, seed=SEED)
(X_train, Y_train), (X_val, Y_val) = generator.generate_reference(n_train, n_val)

model = LogisticRegression(max_iter=1000, random_state=SEED)
model.fit(X_train, Y_train)
S_val = model.predict_proba(X_val)[:, 1]

# Fit reference
print("Fitting reference model...")
ref_model = fit_reference_model(S_val, Y_val, tau0=0.5, bootstrap_B=50, bootstrap_seed=SEED)
ref_model = calibrate_thresholds(ref_model, S_val, Y_val, n_u=n_u, N_cal=50, seed=SEED)

# Run Scenario 1
print("\n[1/2] Running Scenario 1: Pure Label Shift...")
X_s1, Y_s1, pi_true = generator.scenario_1_pure_label_shift(T_deploy)
S_s1 = model.predict_proba(X_s1)[:, 1]

detector_s1 = DriftDetector(ref_model, n_u=n_u)
df_s1 = detector_s1.process_stream(S_s1)
df_s1.to_csv("artifacts/demo_metrics/scenario_1_reports.csv", index=False)

plot_drift_detection_summary(
    df_s1, ref_model, pi_true=pi_true,
    save_path="artifacts/demo_figures/scenario_1_drift_detection.png"
)
plot_state_distribution(df_s1, save_path="artifacts/demo_figures/scenario_1_state_distribution.png")

print(f"  Processed {len(df_s1)} windows")
state_counts = df_s1['state'].value_counts()
for state, count in state_counts.items():
    print(f"  {state}: {count} ({count/len(df_s1)*100:.1f}%)")

# Run Scenario 2
print("\n[2/2] Running Scenario 2: Concept Drift...")
X_s2, Y_s2, drift_ind = generator.scenario_2_concept_drift(T_deploy, drift_time=20000)
S_s2 = model.predict_proba(X_s2)[:, 1]

detector_s2 = DriftDetector(ref_model, n_u=n_u)
df_s2 = detector_s2.process_stream(S_s2)
df_s2.to_csv("artifacts/demo_metrics/scenario_2_reports.csv", index=False)

plot_drift_detection_summary(
    df_s2, ref_model, drift_indicator=drift_ind,
    save_path="artifacts/demo_figures/scenario_2_drift_detection.png"
)
plot_state_distribution(df_s2, save_path="artifacts/demo_figures/scenario_2_state_distribution.png")

print(f"  Processed {len(df_s2)} windows")
state_counts = df_s2['state'].value_counts()
for state, count in state_counts.items():
    print(f"  {state}: {count} ({count/len(df_s2)*100:.1f}%)")

# Generate comparison plot
print("\nGenerating comparison plot...")
fig, ax = plt.subplots(figsize=(10, 6))

scenarios = ['S1: Label Shift', 'S2: Concept Drift']
data = [df_s1, df_s2]

x = np.arange(len(scenarios))
width = 0.25

for i, df in enumerate(data):
    counts = df['state'].value_counts()
    total = len(df)
    normal_pct = counts.get('NORMAL', 0) / total * 100
    prior_pct = counts.get('PRIOR_SHIFT', 0) / total * 100
    drift_pct = counts.get('DRIFT_SUSPECTED', 0) / total * 100

    ax.bar(x[i] - width, normal_pct, width, label='NORMAL' if i==0 else '', color='lightgreen', edgecolor='black')
    ax.bar(x[i], prior_pct, width, label='PRIOR_SHIFT' if i==0 else '', color='gold', edgecolor='black')
    ax.bar(x[i] + width, drift_pct, width, label='DRIFT_SUSPECTED' if i==0 else '', color='lightcoral', edgecolor='black')

ax.set_ylabel('Percentage of Windows (%)', fontweight='bold')
ax.set_title('Controller State Distribution: Demonstration', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(scenarios)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("artifacts/demo_figures/demo_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig("artifacts/demo_figures/demo_comparison.pdf", bbox_inches='tight')
plt.close()

print("\n" + "="*70)
print("DEMO COMPLETE!")
print("="*70)
print("\nGenerated files in artifacts/demo_figures/:")
print("  - scenario_1_drift_detection.png (4-panel plot for label shift)")
print("  - scenario_2_drift_detection.png (4-panel plot for concept drift)")
print("  - scenario_1_state_distribution.png")
print("  - scenario_2_state_distribution.png")
print("  - demo_comparison.png (side-by-side comparison)")
print("\nThese demonstrate what the full results will look like!")
