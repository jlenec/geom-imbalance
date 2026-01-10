"""Global configuration for all experiments."""

# Experiment prevalences
PI_TRAIN = 0.2  # Training prevalence
PI_TEST_GRID = [0.5, 0.2, 0.1, 0.05, 0.01]  # Test prevalences

# Cost settings: (c10, c01) tuples
COSTS_GRID = [(1.0, 1.0), (10.0, 1.0), (1.0, 10.0)]

# Data sizes
N_SYNTH_TRAIN = 40000
N_SYNTH_TEST = 20000
N_REAL_TRAIN = 20000  # Cap at available if dataset is smaller
N_REAL_TEST = 10000

# Calibration
CALIBRATION_LABELED_FRAC = 0.02  # 2% labeled from test distribution

# Synthetic data parameters
SYNTH_DIM = 10  # Feature dimension
N_POOL = 200000  # Base pool size per class for synthetic data

# Model parameters
LOGREG_PARAMS = {
    'solver': 'lbfgs',
    'max_iter': 500,
    'C': 1.0,
    'fit_intercept': True,
    'random_state': None  # Will be set per seed
}

XGB_PARAMS = {
    'n_estimators': 400,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'reg_lambda': 1.0,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'random_state': None  # Will be set per seed
}

# Fallback for when XGBoost not available
GB_PARAMS = {
    'n_estimators': 400,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.9,
    'random_state': None  # Will be set per seed
}

# Output paths
OUTPUT_DIR = 'outputs'
TABLES_DIR = f'{OUTPUT_DIR}/tables'
FIGURES_DIR = f'{OUTPUT_DIR}/figures'
METADATA_DIR = f'{OUTPUT_DIR}/metadata'

# Numeric stability
EPSILON = 1e-6  # For logit clipping
LOGIT_CLIP_MIN = 1e-6
LOGIT_CLIP_MAX = 1 - 1e-6

# Plotting
FIGURE_DPI = 150
FIGURE_SIZE = (8, 6)
PLOT_STYLE = 'default'  # matplotlib style

# Experiment 3 specific
ALPHA_VALUES = [1, 5, 10, 20, 50]  # Weight scaling factors

# Parallelism
DEFAULT_N_JOBS = 1