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

# Relative
from .utils.optuna import get_model, get_optimizer, get_trial_params
from .loop import loop
from .utils.utils import Stage


# #############################
# FUNCTIONS: INTERFACE
# #############################
def train(model, hparams, dataset, kwargs: dict):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, hparams)
    loop(model, criterion, optimizer, dataset, epochs=TR_EPOCHS, stage=Stage.TRAIN, kwargs={'full': False, **kwargs})
    return model