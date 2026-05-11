# Title

# Author:   Sharjeel Mustafa
# Created:  2026-04-30

# Objective: Short Summary. 


# #############################
# IMPORTS
# #############################
# Builtin
from pathlib import Path
from typing import Type
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
def train(model: nn.Module, hparams: dict, dataset: dict, epochs: int)-> tuple[nn.Module, object]:
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, hparams)
    res = loop(model, criterion, optimizer, dataset, epochs=epochs, stage=Stage.TRAIN)
    return model, res['history']