import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Override config for faster testing
import src.geomimb.config as config
config.SEEDS = [0]  # Just 1 seed
config.N_SYNTH_TRAIN = 2000
config.N_SYNTH_TEST = 1000

from src.geomimb.utils.logging import setup_logging
from src.geomimb.experiments.exp5_concept_drift_control import run_experiment as run_exp5

setup_logging(level='INFO')

try:
    results = run_exp5(models=['LogisticRegression'], save_results=False)
    print(f"Success! Got {len(results)} results")
    print(f"Columns: {list(results.columns)}")
    print(f"First row:")
    print(results.iloc[0])
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()