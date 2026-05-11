# HELM - Full Pipeline Test

# Author:  Sharjeel Mustafa
# Created: 2026-05-10

# Objective: Integration test for the entire HELM pipeline using a simple MLP on the Cora dataset.
#            This test validates that all components (data loading, training, testing, and caching) 
#            work together as expected.


# #############################
# IMPORTS
# #############################
# Builtin
from pathlib import Path

# External
import torch
from torch import nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Relative
from src import validate, helm, clear_cache


# #############################
# VARS, CONSTS, & SETUP
# #############################
# Simple classifier for testing the full pipeline
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

MODELS = {'MLP': SimpleClassifier}


# #############################
# FUNCTIONS: HELPER
# #############################
def cora_loader():
    data_path = Path(__file__).parent.parent / 'fixtures' / 'cora'
    nodes = pd.read_csv(data_path / 'nodes.tsv', sep='\t')
    
    # Process features
    features = torch.tensor(nodes.iloc[:, :-1].values, dtype=torch.float)
    
    # Encode string labels
    encoder = LabelEncoder()
    tmp = nodes.iloc[:, -1]
    labels_encoded = encoder.fit_transform(tmp)
    y = torch.tensor(labels_encoded, dtype=torch.long)
    
    num_nodes = len(nodes)
    input_dim = features.shape[1]
    output_dim = len(encoder.classes_)
    
    # Split masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[:int(0.6 * num_nodes)] = True
    val_mask[int(0.6 * num_nodes):int(0.8 * num_nodes)] = True
    test_mask[int(0.8 * num_nodes):] = True

    return {
        'name': 'Cora',
        'X': features,
        'y': y,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'tr_mask': train_mask,
        'va_mask': val_mask,
        'te_mask': test_mask,
        'encoder': encoder,
        'modelwise': {
            'data': {},
            'func': {
                'init': None,
                'prop': None,
            }
        }
    }
    
    
# #############################
# FUNCTIONS: MAIN
# #############################
def test_full_pipeline(): 
    
    root = Path(__file__).parent
    dataset =  cora_loader()
    
    timestamp = "20260508_120000"
    key, expr_n, to_tune = "MLP", 1, True
    
    validate(root, MODELS, expr_n, [dataset], to_tune=to_tune)
    helm(root, expr_n, timestamp, key, MODELS[key], dataset, to_tune=to_tune)
    
    # Clear cache and results folder
    clear_cache(root)
