# Quick Start Guide: Drift Detection Experiments

## Installation (One-time)

```bash
# Install dependencies
py -3 -m pip install numpy scipy pandas scikit-learn matplotlib tqdm pyyaml jupyter
```

## Quick Test (2 minutes)

```bash
# Verify everything works
py -3 test_drift_detection.py
```

Expected output:
```
Testing imports...
[OK] All imports successful!
...
State distribution:
  PRIOR_SHIFT: 17 (85.0%)
  NORMAL: 3 (15.0%)

SUCCESS! All tests passed!
```

## Run Scenario 1: Pure Label Shift

```bash
py -3 scripts/run_simulation.py --config configs/scenario_1_label_shift.yaml
```

Outputs:
- `artifacts/metrics/scenario_1_reports.csv`
- `artifacts/figures/scenario_1_drift_detection.png` (4-panel plot)
- `artifacts/figures/scenario_1_state_distribution.png`

## Run Scenario 2: Concept Drift

```bash
py -3 scripts/run_simulation.py --config configs/scenario_2_concept_drift.yaml
```

Same output structure, different scenario results.

## Interactive Notebook

```bash
# Start Jupyter
py -3 -m jupyter notebook

# Open: notebooks/01_drift_detector_demo.ipynb
# Run all cells to see interactive demos of all scenarios
```

## Create Custom Scenario

1. Copy `configs/scenario_1_label_shift.yaml`
2. Modify parameters (T_deploy, drift_time, etc.)
3. Run:
```bash
py -3 scripts/run_simulation.py --config configs/my_scenario.yaml --seed 42
```

## Key Configuration Parameters

```yaml
# Data size
n_ref_train: 50000      # Training samples
n_ref_val: 20000        # Validation samples
T_deploy: 200000        # Deployment stream length

# Imbalance
pos_rate_ref: 0.02      # Reference prevalence (2%)

# Detector
n_u: 2000               # Window size
tau0: 0.5               # Fixed BBSE threshold
lambda_ewma: 0.1        # EWMA smoothing

# Calibration
N_cal: 200              # Calibration windows
alpha_d: 0.01           # False alarm rate for d_u_star

# Drift timing
drift_time: 80000       # When drift occurs (scenarios 2-4)

# Reproducibility
seed: 42                # Random seed
```

## Understanding the Output

### 4-Panel Plot
1. **Top**: Prevalence estimates (BBSE, EWMA, true)
2. **Second**: Mixture consistency d_u_star (with threshold)
3. **Third**: BBSE residual r_u (if stable)
4. **Bottom**: Operational threshold τ

Background colors show controller state:
- Green: NORMAL
- Yellow: PRIOR_SHIFT (adapting)
- Red: DRIFT_SUSPECTED (frozen)

### CSV Reports
Each row is a window with columns:
- `pi_hat_bbse`: BBSE prevalence estimate
- `pi_ewma`: Smoothed prevalence
- `d_u_star`: Mixture consistency metric
- `r_u`: BBSE residual
- `state`: Controller state
- `tau_operational`: Current threshold

## Common Commands

```bash
# Run with different seed
py -3 scripts/run_simulation.py --config configs/scenario_1_label_shift.yaml --seed 123

# Check Python version
py -3 --version

# List available packages
py -3 -m pip list | findstr "numpy\|scipy\|pandas\|sklearn"

# View output CSV
py -3 -c "import pandas as pd; df = pd.read_csv('artifacts/metrics/scenario_1_reports.csv'); print(df.head())"
```

## Troubleshooting

### ImportError
```bash
# Reinstall dependencies
py -3 -m pip install --upgrade numpy scipy pandas scikit-learn matplotlib
```

### Unicode errors in output
- Already fixed in test script
- If custom prints fail, use ASCII characters

### Out of memory
- Reduce `T_deploy` to 100000 or 50000
- Reduce `n_ref_train` to 25000

### Slow execution
- Reduce `N_cal` to 100
- Reduce `bootstrap_B` to 100
- Use smaller window size `n_u: 1000`

## Next Steps

1. ✓ Run quick test
2. ✓ Run scenario 1 & 2
3. Create configs for scenarios 3-5
4. Explore notebook interactively
5. Generate all figures for paper
6. Compile updated LaTeX paper

## File Locations

- **Source code**: `labelshift_drift/`
- **Experiments**: `scripts/run_simulation.py`
- **Configs**: `configs/*.yaml`
- **Outputs**: `artifacts/figures/` and `artifacts/metrics/`
- **Notebook**: `notebooks/01_drift_detector_demo.ipynb`
- **Test**: `test_drift_detection.py`
- **Paper**: `paper_with_experiments.tex` (updated with new section)

## Support

See detailed documentation in:
- `DRIFT_DETECTION_README.md` - Package documentation
- `IMPLEMENTATION_SUMMARY.md` - Complete implementation details
- Paper Section: "Drift Detection Under Delayed Labels"
