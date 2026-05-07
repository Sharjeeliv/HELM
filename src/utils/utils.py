# Title

# Author:   Sharjeel Mustafa
# Created:  2026-04-30

# Objective: Short Summary. 


# #############################
# IMPORTS
# #############################
# Builtin
import json
from pathlib import Path
from enum import Enum
from typing import Literal

# External
from torch import nn

# Relative


# #############################
# VARS, CONSTS, & SETUP
# #############################

# Define ENUM for train, test, and tune stages
class Stage(Enum):
    TRAIN = 'train'
    TEST  = 'test'
    TUNE  = 'tune'


# #############################
# FUNCTIONS: UTILITY
# #############################

# Early Stopping
# *****************************
class EarlyStopping:
    def __init__(self, patience=5, delta=1e-3):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter < self.patience: return
            self.early_stop = True

        else:
            self.best_score = score
            self.counter = 0

# Hook Runner
# *****************************
def run_hook(dataset: dict, hook_type: Literal['init', 'prop'],
             model: nn.Module|None = None):
    # Runs a custom hook function, modifies dataset in-place.
    hook = dataset['modelwise']['func'].get(hook_type)
    if hook: hook(dataset, model) if model else hook(dataset)

# Parameter loading/saving
# *****************************
def load_params(root: Path, key: str):
    params = json.load(open(root / 'config' / 'tuned.json'))
    print(f'Loaded parameters for {key}')
    return params[key]

def save_params(root: Path, key: str, model_params: dict):
    params = root / 'config' / 'tuned.json'
    res = json.load(open(params))
    res[key] = model_params
    with open(params, "w") as jf: json.dump(res, jf, indent=4)
    print(f'Saved parameters for {key}')
    
    
# logging and packaging
# *****************************
def epoch_print(epoch, train_loss, test_loss, test_acc, epoch_time, val=False):
    print(f"Epoch {epoch:02d} "
        f"| Train Loss: {train_loss:.4f} "
        f"| {'val' if val else 'Test'} Loss: {test_loss:.4f} "
        f"| Accuracy: {test_acc:.4f} "
        f"| Time: {epoch_time*10:.2f}s")
    

# Schema & Validation
# *****************************
def validate_dataset(dataset: dict, log=False):
    
    mapping = {
        'X': 'Feature Tensor of shape [N, in_dim]',
        'y': 'Label Tensor of shape [N]',
        'G': 'Adjacency [N, N] or Incidence [N, E]',
        'in_dim':  'Input dimension',
        'out_dim': 'Output dimension',
        'tr_mask': 'Training mask',
        'va_mask': 'Validation mask',
        'te_mask': 'Test mask'
    }
    
    required_keys = ['X', 'y', 'G', 'in_dim', 'out_dim', 'tr_mask', 'va_mask', 'te_mask']
    for key in required_keys:
        if key in dataset: continue
        raise ValueError(f"Dataset missing required key: '{key}'. Expected: {mapping[key]}")
        
    if 'mask' not in dataset: dataset['mask'] = None
    if log and 'extra' in dataset: print("Extra dataset parameters provided:", dataset['extra'])
