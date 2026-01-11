# Drift Detection Under Delayed Labels

This package implements the geometric theory-based drift detection system described in the paper "The Resampling Delusion: A Geometric Theory of Class Imbalance", Section: "Drift Detection Under Delayed Labels: Distinguishing Label Shift from Concept Drift".

## Overview

The system provides:
1. **Geometric validation test** using unlabeled scores to distinguish valid label shift from concept drift
2. **State-machine controller** handling the temporal gap between immediate unlabeled inference and delayed labeled confirmation
3. **BBSE (Black Box Shift Estimation)** for prevalence estimation without labels
4. **Mixture consistency testing** to validate the label-shift manifold assumption

## Installation

```bash
pip install -r labelshift_drift_requirements.txt
```

## Package Structure

```
labelshift_drift/
  ├── utils/          # ECDF, bootstrap, windowing utilities
  ├── reference/      # Reference model fitting
  ├── detector/       # BBSE, mixture testing, drift detection
  ├── controller/     # State machine and threshold adaptation
  ├── simulation/     # Data generators for 5 scenarios
  └── viz/            # Plotting functions
```

## Quick Start

### Run a Single Scenario

```bash
python scripts/run_simulation.py --config configs/scenario_1_label_shift.yaml
```

### Run the Jupyter Notebook Demo

```bash
jupyter notebook notebooks/01_drift_detector_demo.ipynb
```

## Scenarios

The package includes 5 experimental scenarios:

1. **Pure Label Shift** (valid adaptation): Prevalence changes over time, geometry stable
2. **Concept Drift** (invalid): Class-conditional distributions change
3. **Score Mapping Drift** (invalid): Model output transformation
4. **Covariate Shift (benign)**: Shift in unused dimensions (should NOT alarm)
5. **Ill-conditioned Confusion Matrix**: BBSE unstable, relies on d_u_star

## Key Components

### Reference Model Fitting

```python
from labelshift_drift.reference.fit_reference import fit_reference_model

ref_model = fit_reference_model(
    S_ref, Y_ref,
    tau0=0.5,
    lambda_ewma=0.1
)
```

### Threshold Calibration

```python
from labelshift_drift.detector.thresholds import calibrate_thresholds

ref_model = calibrate_thresholds(
    ref_model, S_val, Y_val,
    n_u=2000, N_cal=200
)
```

### Drift Detection

```python
from labelshift_drift.detector.drift_detector import DriftDetector

detector = DriftDetector(ref_model, n_u=2000)
df_reports = detector.process_stream(S_deploy)
```

### Visualization

```python
from labelshift_drift.viz.plots import plot_drift_detection_summary

plot_drift_detection_summary(
    df_reports, ref_model,
    pi_true=pi_true,
    drift_indicator=drift_indicator,
    save_path='figures/drift_detection.png'
)
```

## Controller States

- **NORMAL**: Mixture consistency holds, no prevalence shift detected
- **PRIOR_SHIFT**: Prevalence shifted but geometry stable → **adapt threshold**
- **DRIFT_SUSPECTED**: Mixture consistency violated → **freeze adaptation, await labels**

## Configuration

Experiments are configured via YAML files in `configs/`. Key parameters:

```yaml
# Data
n_ref_train: 50000
n_ref_val: 20000
T_deploy: 200000
pos_rate_ref: 0.02

# Detector
n_u: 2000          # window size
tau0: 0.5          # fixed reference threshold for BBSE
lambda_ewma: 0.1   # EWMA smoothing

# Calibration
N_cal: 200         # number of calibration windows
alpha_d: 0.01      # false alarm rate for d_u_star
alpha_r: 0.01      # false alarm rate for r_u
alpha_pi: 0.01     # false alarm rate for pi deviation
```

## Reproducibility

All experiments use fixed random seeds. To reproduce results:

```bash
python scripts/run_simulation.py --config configs/scenario_1_label_shift.yaml --seed 42
```

## Output Artifacts

Experiments generate:
- **CSV reports**: Window-by-window metrics in `artifacts/metrics/`
- **Figures**: 4-panel time series plots in `artifacts/figures/`
  - Prevalence estimates
  - Mixture consistency (d_u_star)
  - BBSE residual (r_u)
  - Operational threshold

## Citation

If you use this package, please cite:

```
Sudjianto, A. and Manokhin, V. (2025). The Resampling Delusion:
A Geometric Theory of Class Imbalance. GitHub repository.
```

## Theory Summary

The drift detector validates the **mixture identity**:

```
F^dep(s) = (1-π_dep) F_0^ref(s) + π_dep F_1^ref(s)
```

Under label shift, the deployment score CDF is a 1D mixture of reference conditional CDFs. The mixture consistency statistic d_u_star measures deviation from this manifold:

```
d_u_star = min_π KS(F_u, F_mix(·; π))
```

If d_u_star exceeds the calibrated threshold, the geometry has changed (concept drift), and threshold adaptation is unsafe.

## Contact

For questions or issues, please open a GitHub issue.
