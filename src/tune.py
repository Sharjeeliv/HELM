# Title

# Author:   Sharjeel Mustafa
# Created:  2026-04-30

# Objective: Short Summary. 


# #############################
# IMPORTS
# #############################
# Builtin
from pathlib import Path
import json

# External
import torch
import optuna
from optuna import Trial
from torch import nn

# Relative
from .utils.optuna import get_model, get_optimizer, get_trial_params
from .utils.utils import Stage
from .loop import loop


# #############################
# VARS, CONSTS, & SETUP
# #############################
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# #############################
# FUNCTIONS: HELPER
# #############################
def _objective(trial: Trial, key: str, models: dict, 
               dataset: dict, root: Path, epochs: int):
    
    # Retrieve trial hyperparameters
    hparams = get_trial_params(root, key, trial)
    
    # Omit invalid configurations:
    try: model = get_model(models, key, hparams, dataset)
    except RuntimeError: raise optuna.exceptions.TrialPruned()
    model.to(DEVICE)

    # Criterion and optimizer
    c = torch.nn.CrossEntropyLoss()
    o = get_optimizer(model, hparams)
    
    res = loop(model, c, o, dataset, epochs=epochs, 
               stage=Stage.TUNE, trial=trial)
    return res['l']


# #############################
# FUNCTIONS: INTERFACE
# #############################
def tune(root: Path, key: str, models: dict[str, nn.Module],
         dataset: dict, epochs: int, n_trials: int) -> dict:
    
    # Guard Clause
    err_string = f"Model {key} not found in models dictionary."
    if key not in models: raise ValueError(err_string)
    
    # Hyperparameter Tuning
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: _objective(trial, key, models, dataset, root, epochs),  n_trials=n_trials)
    return study.best_params
