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
from torch import nn

# Relative
from .utils.optuna import get_model, get_optimizer, get_trial_params
from .loop import loop
from .utils.utils import Stage


# #############################
# FUNCTIONS: INTERFACE
# #############################
def train(model: nn.Module, hparams: dict, dataset: dict, epochs: int)-> nn.Module:
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, hparams)
    loop(model, criterion, optimizer, dataset, epochs=epochs, stage=Stage.TRAIN)
    return model