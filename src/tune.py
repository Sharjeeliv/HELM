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

# Relative
from .utils.optuna import get_model, get_optimizer, get_trial_params
from .loop import loop


# #############################
# FUNCTIONS: HELPER
# #############################
def _objective(trial: Trial, models: dict, key: str, dataset: dict, kwargs: dict):
    
    # Retrieve trial hyperparameters
    hparams = get_trial_params(trial, key)
    
    # Omit invalid configurations:
    try: model = get_model(models, key, hparams, dataset)
    except RuntimeError: raise optuna.exceptions.TrialPruned()
    model.to(DEVICE)

    # Criterion and optimizer
    c = torch.nn.CrossEntropyLoss()
    o = get_optimizer(model, hparams)
    
    res = loop(model, c, o, dataset, epochs=TU_EPOCHS, kwargs={'trial': trial, **kwargs})
    return res['l']


# #############################
# FUNCTIONS: INTERFACE
# #############################
def tune(models: dict, key: str, dataset: dict, kwargs: dict = {}):
    
    # Guard Clause
    err_string = f"Model {key} not found in models dictionary."
    if key not in models: raise ValueError(err_string)
    
    # Hyperparameter Tuning
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: _objective(trial, models, key, dataset, kwargs),  n_trials=N_TRIALS)
    return study.best_params
