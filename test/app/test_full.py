# Title

# Author:   Full Name
# Created:  Date

# Objective: Short Summary. 


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
from src import validate, helm

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
# FUNCTIONS: UTILITY
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
# FUNCTIONS: HELPER
# #############################



# #############################
# FUNCTIONS: MAIN
# #############################
def test_full_pipeline(): 
    dataset =  cora_loader()
    root = Path(__file__).parent
    timestamp = "20260508_120000"
    key = "MLP"
    expr_n = 1
    to_tune = True
    
    print("Integration Test: Validating and Running Full Pipeline")
    print(root)
    
    validate(root, MODELS, expr_n, [dataset], to_tune=to_tune)
    helm(root, expr_n, timestamp, key, MODELS[key], dataset, to_tune=to_tune)


# #############################
# FUNCTIONS: INTERFACE
# #############################



# #############################
# UTILITY: SNIPPETS & NOTES
# #############################

# File template for python, paste into empty .py files
# Please remove these notes 'UTILITY: SNIPPETS & NOTES', and heading before comitting


# FILE HEADING BREAKDOWN
# Utility   - General purpose functions that are only used locally
#             If used in multiple places, move to dedicated utils.py or utils folder
# Helper    - Specific (non-general) functions that aid in the "main" task of the file
# Main      - Functions that correspond to the file's main algorithms or tasks
#             Note: A file should have limited main functions and do one type of thing!
#             In other words have a separation of concerns
# Interface - Functions that are designed to be called by other files, internals should
#             be kept private (i.e., start with an underscore), unless the functionality
#             is intended to be public, like some helpers or utils

# FILE HEADING SNIPPETS

# 1. Avoid adding any more first-level headings
# 2. Second-level headings can be added as follows:

# *****************************
# Heading Title
# *****************************

# Or like if directly below a first-level heading:

# Heading Title
# *****************************

# 3. Third-level headings can be added as follows:

# Heading Title
# -----------------------------
