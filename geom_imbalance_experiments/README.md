# Geometric Theory of Learning Under Class Imbalance - Experiments

This repository contains the experimental validation suite for the paper **"A Geometric Theory of Learning Under Class Imbalance"**.

## Overview

The experiments demonstrate that:
1. Under label shift, only threshold/logit-offset updates are needed (no retraining)
2. Ranking metrics (AUC) are invariant under prevalence changes
3. Reweighting/rebalancing reduces effective sample size and increases instability
4. Under concept drift, offset alone fails and retraining is necessary

## Installation

```bash
# Clone the repository
cd geom_imbalance_experiments

# Install in development mode
pip install -e .

# For SMOTE experiments (optional)
pip install -e ".[smote]"

# For running notebooks
pip install -e ".[dev]"
```

## Quick Start

Run all experiments:
```bash
python scripts/run_all.py
```

Or use the Jupyter notebook:
```bash
jupyter notebook notebooks/00_run_all_and_make_figures.ipynb
```

## Experiments

1. **Experiment 1**: Label shift - offset correction vs. no correction
2. **Experiment 2**: AUC invariance demonstration
3. **Experiment 3**: Weighting reduces effective sample size
4. **Experiment 4**: Operating point metrics with offset correction
5. **Experiment 5**: Concept drift control

## Output Structure

All results are saved in `outputs/`:
- `tables/`: CSV files with detailed results
- `figures/`: PNG plots for paper figures
- `metadata/`: Configuration and validation checks

## Paper Citation

If you use this code, please cite:
```bibtex
@article{geometric_imbalance_2024,
  title={A Geometric Theory of Learning Under Class Imbalance},
  author={},
  year={2024}
}
```