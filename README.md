# The Resampling Delusion: A Geometric Theory of Class Imbalance

This repository contains the implementation and experimental validation for our paper showing that class imbalance in machine learning is fundamentally a threshold selection problem, not a training problem.

## Key Findings

Our work challenges the widespread practice of reweighting/resampling for class imbalance:

1. **Geometric Invariance**: The discriminant geometry (log-likelihood ratio field) remains unchanged under label shift
2. **Threshold Suffices**: Simple logit offset correction achieves near-optimal performance without retraining
3. **Reweighting Harms**: Class weights reduce effective sample size and increase variance
4. **AUC Invariance**: Discriminative ability (AUC) is invariant to class prevalence

## Repository Structure

```
geom_imbalance_experiments/
├── src/
│   └── geomimb/
│       ├── config.py           # Global configuration
│       ├── data/               # Data generation and loading
│       │   ├── __init__.py
│       │   ├── synthetic.py    # Gaussian mixture data
│       │   └── real_data.py    # Real dataset loaders
│       ├── models/             # Model wrappers
│       │   ├── __init__.py
│       │   └── sklearn_models.py
│       ├── metrics/            # Evaluation metrics
│       │   ├── __init__.py
│       │   └── classification.py
│       ├── experiments/        # Main experiment scripts
│       │   ├── __init__.py
│       │   ├── exp1_label_shift_offset.py
│       │   ├── exp2_auc_invariance.py
│       │   ├── exp3_effective_sample_size.py
│       │   ├── exp4_operating_points.py
│       │   └── exp5_concept_drift.py
│       ├── plotting/           # Visualization
│       │   ├── __init__.py
│       │   └── plots.py
│       └── utils/              # Utilities
│           ├── __init__.py
│           ├── io.py
│           └── logging.py
├── outputs/
│   ├── figures/                # Generated plots
│   └── tables/                 # Result CSV files
├── scripts/
│   └── run_all_experiments.py # Run complete suite
└── notebooks/                  # Analysis notebooks
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/geom-imbalance.git
cd geom-imbalance

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running Experiments

### Run all experiments:
```bash
cd geom_imbalance_experiments
python scripts/run_all_experiments.py
```

### Run individual experiments:
```bash
# Experiment 1: Label shift with offset correction
python -m src.geomimb.experiments.exp1_label_shift_offset

# Experiment 2: AUC invariance
python -m src.geomimb.experiments.exp2_auc_invariance

# Experiment 3: Effective sample size
python -m src.geomimb.experiments.exp3_effective_sample_size

# Experiment 4: Operating points
python -m src.geomimb.experiments.exp4_operating_points

# Experiment 5: Concept drift
python -m src.geomimb.experiments.exp5_concept_drift
```

## Key Results

### Experiment 1: Label Shift
At extreme imbalance (π=0.01), simple offset correction reduces risk by **79%** (from 0.028 to 0.006).

### Experiment 2: AUC Invariance
AUC varies by only **0.0020** across test prevalences from 0.01 to 0.50, confirming geometric invariance.

### Experiment 3: Effective Sample Size
With moderate class weights (α=50), effective sample size drops to just **23%** of original.

### Experiment 4: Concept Drift
Under true concept drift, offset correction provides zero benefit while retraining improves AUC.

## Theoretical Framework

The key insight is that binary classification has two distinct components:

1. **Discriminant Geometry**: The log-likelihood ratio field Λ(x) = log[p(x|1)/p(x|0)]
2. **Operating Point**: The threshold τ selected based on prevalence π and costs c

Under label shift, only π changes while p(x|y) remains fixed. Therefore:
- The geometry Λ(x) is invariant
- Only the threshold needs adjustment: τ_new = τ_old + log(ω_old/ω_new)
- Retraining/reweighting unnecessarily modifies the invariant geometry

## Citation

If you use this code or theory in your research, please cite:
```bibtex
@article{sudjianto2024resampling,
  title={The Resampling Delusion: Why Class Imbalance Is a Threshold Problem, Not a Training Problem},
  author={Sudjianto, Agus},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

We welcome contributions! Please feel free to submit issues or pull requests.

## Contact

Agus Sudjianto - H2O.ai / UNC Charlotte
Email: [your-email]