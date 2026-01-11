#!/usr/bin/env python3
"""
Run all 5 drift detection scenarios and generate publication-ready figures and tables
"""
import os
import sys
import subprocess
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print("RUNNING ALL DRIFT DETECTION SCENARIOS")
print("="*70)

scenarios = [
    ("Scenario 1: Pure Label Shift", "configs/scenario_1_label_shift.yaml"),
    ("Scenario 2: Concept Drift", "configs/scenario_2_concept_drift.yaml"),
    ("Scenario 3: Score Mapping Drift", "configs/scenario_3_score_drift.yaml"),
    ("Scenario 4: Covariate Shift", "configs/scenario_4_covariate_shift.yaml"),
    ("Scenario 5: Ill-Conditioned C", "configs/scenario_5_ill_conditioned.yaml"),
]

failed = []

for i, (name, config_path) in enumerate(scenarios, 1):
    print(f"\n{'='*70}")
    print(f"[{i}/5] {name}")
    print(f"{'='*70}")

    cmd = [sys.executable, "scripts/run_simulation.py", "--config", config_path, "--seed", "42"]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\n[ERROR] {name} failed!")
        failed.append(name)
    else:
        print(f"\n[SUCCESS] {name} completed!")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

if failed:
    print(f"\nFailed scenarios ({len(failed)}):")
    for name in failed:
        print(f"  - {name}")
else:
    print("\nAll scenarios completed successfully!")

print(f"\nOutputs saved to:")
print(f"  - artifacts/figures/")
print(f"  - artifacts/metrics/")

print(f"\nNext: Run generate_paper_figures.py to create summary tables and plots")
