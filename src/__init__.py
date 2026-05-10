# Import Interface Management for HELM

# Author:   Sharjeel Mustafa
# Created:  2026-04-30

# Objective: Manages what is exposed when importing the package, 
#            and provides a central place for imports across the package. 


# #############################
# IMPORTS
# #############################
# Builtin
import json
from pathlib import Path

# External
from torch import nn

# Relative
from .utils.validate import (validate_models, validate_dataset, 
                             validate_dir_structure, 
                             validate_global_config, 
                             validate_hparam_config)
from .tune import tune
from .train import train
from .test import test
from .utils.report import save_results
from .utils.cache import write_cache, read_cache, clear_cache

# #############################
# VARS, CONSTS, & SETUP
# #############################
TR_EPOCHS = -1
TU_EPOCHS = -1
N_TRIALS  = -1

# #############################
# FUNCTIONS: UTILITY
# #############################



# #############################
# FUNCTIONS: HELPER
# #############################
def _init_globals(root: Path):
    global TR_EPOCHS, TU_EPOCHS, N_TRIALS
    
    # Load global config
    path = root / 'config' / 'global.json'
    global_config = json.load(path.open('r'))
    
    TR_EPOCHS = int(global_config['TR_EPOCHS'])
    TU_EPOCHS = int(global_config['TU_EPOCHS'])
    N_TRIALS  = int(global_config['N_TRIALS'])


def _validate(dataset: dict):
    # Per instance internal validation logic
    if (TU_EPOCHS and N_TRIALS and TR_EPOCHS) == -1:
        raise ValueError("Global config not initialized. Call validate() first.")
    validate_dataset(dataset)


# #############################
# FUNCTIONS: MAIN
# #############################
def _validate_cache(root: Path, models: dict[str, nn.Module], datasets: list[str]) -> None:
    for model_name in models:
        for dataset_name in datasets:
            hparams = read_cache(root, model_name, dataset_name)
            if hparams is not None: continue
            print(f"No cache found for {model_name} on {dataset_name}.", ' ')
            print("This combination will be tuned and cached during execution.")


def validate(root: Path, models: dict[str, nn.Module], expr_n: int, 
             datasets: list[str], to_tune=True) -> None:
    # Allows the user to validate separately from the main pipeline, 
    # which avoids expensive runs, useful for debugging and testing.
    
    # General-global validation logic
    e_msg = "Experiment number must be non-negative."
    if expr_n < 0: raise ValueError(e_msg)
    _init_globals(root)
    validate_dir_structure(root)
    validate_models(models)
    validate_hparam_config(root, models)
    validate_global_config(root)
    
    # Validate dataset/model combinations if tuning is disabled
    if to_tune: return
    _validate_cache(root, models, datasets)


# #############################
# FUNCTIONS: INTERFACE
# #############################
def helm(root: Path, expr_n: int, timestamp: str,
         key: str, model: nn.Module, dataset: dict[str, object],
         to_tune: bool = True) -> dict[str, float]:
    
    # 0. Guard Clauses: Ensure validation and initialization
    _validate(dataset) # Ensure modelwise + dataset is valid
    _model = {key: model}
    
    # 1. Pipeline: Tuning
    # a. Attempt to read from cache if tuning is not explicitly forced
    hparams = read_cache(root, key, str(dataset['name'])) if not to_tune else None
    # b. If no cache hit, or tuning is forced, run tuning and write to cache
    if hparams is None:
        print(f"Running tuning for {key} on {dataset['name']}...")
        hparams = tune(root, key, _model, dataset, epochs=TU_EPOCHS, n_trials=N_TRIALS)
        write_cache(root, key, str(dataset['name']), hparams)
    else:
        print(f"Using cached hyperparameters for {key} on {dataset['name']}.")
    
    # 2. Pipeline: Training
    trmodel = train(model, hparams, dataset, epochs=TR_EPOCHS)
    
    # 3. Pipeline: Testing
    results = test(trmodel, dataset)
    
    # 5. Printing & Saving
    save_results(root, expr_n, timestamp, results)
    return results