import pytest
import numpy as np
from torch import nn

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
def raw_dataset():
    """Provides a fresh dictionary of dummy data."""
    return {
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