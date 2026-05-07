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
from optuna.trial import Trial
import torch

# Relative
from .utils import run_hook


# #############################
# VARS, CONSTS, & SETUP
# #############################
NON_MODEL_PARAMS = {'optimizer', 'lr', 'weight_decay'}


# #############################
# FUNCTIONS: UTILITY
# #############################
def _get_hparams(root, key):
    path = root / 'config' / 'hparam.json'
    hparam_config = json.load(path.open('r'))
    return hparam_config[key]


def trial_type(trial: Trial, name: str, config: dict):
    TYPE, START, END = 0, 1, 2
    ptype = config[TYPE]
    if ptype == 'log':     return trial.suggest_float(name, config[START], config[END], log=True)
    if ptype == 'int':     return trial.suggest_int(name, config[START], config[END])
    if ptype == 'flt':     return trial.suggest_float(name, config[START], config[END])
    if ptype == 'cat':     return trial.suggest_categorical(name, config[START:])
    raise ValueError(f"Unknown trial type: {ptype}")


def get_trial_params(root: Path, key: str, trial: Trial):
    params = _get_hparams(root, key)
    model_params = {}
    for param_name, param_range in params[key].items():
        suggest_func = trial_type(trial, param_name, param_range)
        model_params[param_name] = suggest_func
    return model_params


def get_model(models, key, hparams, dataset):
    run_hook(dataset, 'init', None)
    model_cls = models[key]
    # Filter non-model params
    model_params = {k: v for k, v in hparams.items() 
                    if k not in NON_MODEL_PARAMS}
    # Merge model and task params
    struct_params = {'input_dim':  dataset['input_dim'], 
                     'output_dim': dataset['output_dim']}
    final_params = {**model_params, **struct_params, **dataset['extra']}
    return model_cls(**final_params)


def get_optimizer(model, trial_params, forced_lr=0):
    # Unpack parameters
    optimizer_type = trial_params.get("optimizer", "Adam")
    lr = trial_params.get("lr", 1e-3)
    weight_decay = trial_params.get("weight_decay", 0.0)
    
    # Build optimizer for remaining models
    if forced_lr !=0: lr = forced_lr
    if optimizer_type == "SGD":
        return torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_type == "Adam":
        return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=lr, weight_decay=weight_decay)
    else: raise ValueError(f"Unknown optimizer type: {optimizer_type}")
