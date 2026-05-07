# TEST: Validation Functions

# Author:   Sharjeel Mustafa
# Created:  2026-05-04

# Objective: Unit tests for validation functions in src.utils.validate


# #############################
# IMPORTS
# #############################
# Builtin

# External
import pytest
import numpy as np
from torch import nn

# Relative


# #############################
# VARS, CONSTS, & SETUP
# #############################

# #############################
# FUNCTIONS: HELPER
# #############################

# *****************************
# HELPER: MODELS
# *****************************
class MockModel(nn.Module):
    def __init__(self, hidden_dim, lr, activation="relu"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.activation = activation

class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout, dataset=None):
        super().__init__()
        # These will be ignored: input_dim, output_dim, dataset, self
        # These must be in config: hidden_dim, dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.dataset = dataset

@pytest.fixture
def DummyModel():
    """Defines the class once and makes it available to all tests."""
    class _DummyModel(nn.Module):
        def __init__(self, in_dim, out_dim, hidden_dim=16):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, out_dim)
    return _DummyModel

@pytest.fixture
def models_dict():
    return {"GCN": GCN,
            "MockModel": MockModel}

# *****************************
# HELPER: CONFIGS
# *****************************
@pytest.fixture
def valid_hparam_config():
    return {
        "MockModel": {
            "hidden_dim": ["int", 16, 128],
            "lr": ["log", 1e-4, 1e-2],
            "activation": ["cat", "relu", "tanh"]
        },
        "GCN": {
            "hidden_dim": ["int", 16, 128],
            "dropout": ["flt", 0.0, 0.5]
        }
    }

# *****************************
# HELPER: DATASET
# *****************************
@pytest.fixture
def raw_dataset():
    """Provides a fresh dictionary of dummy data."""
    return {
        'name': 'DummyDataset',
        'X': np.random.rand(10, 5),
        'y': np.random.rand(10, 2),
        
        'input_dim': 5,
        'output_dim': 2,
        
        'te_mask': np.random.rand(10) < 0.5,
        'tr_mask': np.random.rand(10) < 0.8,
        'va_mask': np.random.rand(10) < 0.1,
        'encoder': lambda x: x,  # Dummy encoder function
        
        'modelwise': {
            'data': { 'extra_param': 42 },
            'func': {'init': lambda: None, 
                     'prop': lambda: None}
        }
    }