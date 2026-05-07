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
def validate(root: Path, models: dict[str, nn.Module], 
             dataset: dict[str, object]) -> None:
    # General-global validation logic
    _init_globals(root)
    validate_dir_structure(root)
    validate_models(models)
    validate_hparam_config(root, models)
    validate_global_config(root)


# #############################
# FUNCTIONS: INTERFACE
# #############################
def helm(root: Path, key: str, model: nn.Module, dataset: dict[str, object], to_tune: bool = True):
    
    # 0. Guard Clauses: Ensure validation and initialization
    _validate(dataset)    # Ensure modelwise + dataset is valid
    _model = {key: model}
    
    # 1. Pipeline: Tuning or load cached hyperparameters
    hparams = tune(root, key, _model, dataset, epochs=TU_EPOCHS, n_trials=N_TRIALS)
    
    # 2. Pipeline: Training
    trmodel = train(model, hparams, dataset, epochs=TR_EPOCHS)
    
    # 3. Pipeline: Testing
    results = test(trmodel, dataset)
    
    # 5. Printing & Saving
    