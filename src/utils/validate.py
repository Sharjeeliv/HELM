# Title

# Author:   Sharjeel Mustafa
# Created:  2026-04-30

# Objective: Short Summary. 


# #############################
# IMPORTS
# #############################
# Builtin
from pathlib import Path
from typing import Callable

# External
from schema import Schema, And, Use
from torch import nn

# Relative


# #############################
# VARS, CONSTS, & SETUP
# #############################



# #############################
# FUNCTIONS: UTILITY
# #############################
def validate_models(models):
    """
    Validates the models dictionary to ensure it contains valid nn.Module instances.

    Args:
        models (dict): A dictionary where keys are model names (str) and values are nn.Module subclasses.

    Raises:
        ValueError: If any value in the models dictionary is not a subclass of nn.Module.
    """
    for name, model in models.items():
        if issubclass(model, nn.Module): continue
        raise ValueError(f"Model '{name}' is not a valid nn.Module subclass.")

import numpy as np


# Example dataset:
dataset = {
    'X': np.random.rand(100, 10),  # Feature matrix
    'y': np.random.randint(0, 2, size=(100,)),  # Optional keys
    'input_dim': 10,
    'output_dim': 2,
    'tr_mask': np.random.rand(100) < 0.8,  # 80% training mask
    'va_mask': np.random.rand(100) < 0.1,  # 10% validation mask
    'te_mask': np.random.rand(100) < 0.1,  # 10% testing mask
    'encoder': lambda x: x,  # Identity encoder for simplicity
    'modelwise': {
        'data': {'extra_param': 42},
        'func': {
            'init': 'hi',
            'prop': lambda: print("Model-specific propagation"),
        }
    }
}



def validate_dataset(dataset):
    """
    Validates the dataset dictionary to ensure it contains required keys and valid formats.
    Args:
        dataset (dict): A dictionary containing dataset information.
    Raises:
        ValueError: If any required key is missing.
    """
    
    def is_matrix(obj):
        return hasattr(obj, "shape") or hasattr(obj, "__array__")

    def _is_callable(obj):
        return callable(obj)

    dataset_schema = Schema({
        'X': is_matrix,
        'y': object,
        'input_dim': And(int, lambda n: n > 0),
        'output_dim': And(int, lambda n: n > 0),
        
        'tr_mask': object,
        'va_mask': object,
        'te_mask': object,
        'encoder': object,
        
        'modelwise': {
            'data': dict,
            'func': {
                'init': _is_callable,
                'prop': _is_callable,
            }
        }
    })
    dataset_schema.validate(dataset)
    

def validate_tuning_config(tuning_config):
    """
    Validates the tuning configuration to ensure it contains valid parameter specifications.

    Args:
        tuning_config (dict): A dictionary where keys are model names and values are parameter configurations.

    Raises:
        ValueError: If any parameter configuration is not in the expected format.
    """
    for model_name, params in tuning_config.items():
        for param_name, spec in params.items():
            
            # First element is string indicating distribution type, followed by values.
            base_msg = f"Parameter '{param_name}' in model '{model_name}'"
            if not isinstance(spec, list) or len(spec) < 2:
                raise ValueError(f"{base_msg} has an invalid specification.")
            
            dist_type = spec[0]
            # Categorical distrbution can have any number of values, but must have at least one.
            # All values must be strings, ints, or floats.
            if dist_type == "cat":
                for val in spec[1:]:
                    if isinstance(val, (str, int, float)): continue
                    raise ValueError(f"{base_msg} has invalid value '{val}'.")
            
            # Log, float, and int distributions must have exactly 3 elements: [dist_type, min_val, max_val]
            elif dist_type in ["log", "flt", "int"]:
                if len(spec) != 3:
                    raise ValueError(f"{base_msg} has invalid specification, must be [dist_type, min_val, max_val].") 
                if spec[1] >= spec[2]:
                    raise ValueError(f"{base_msg} has invalid range: min_val must be less than max_val.")
            
            # Log and float distributions must have numeric min and max values, while int distributions must have integer min and max values.
            elif dist_type in ["log", "flt"]:
                min_val, max_val = spec[1], spec[2]
                if not (isinstance(min_val, float) and isinstance(max_val, float)):
                    raise ValueError(f"{base_msg} must have float min and max values ('flt').")
                
            elif dist_type == "int":
                min_val, max_val = spec[1], spec[2]
                if not (isinstance(min_val, int) and isinstance(max_val, int)):
                    raise ValueError(f"{base_msg} must have integer min and max values ('int').")
                
            else:
                raise ValueError(f"{base_msg} has unknown distribution type '{dist_type}'.")
          
          
          
          
          
# Completed
def validate_config_dir(root: Path):
    """
    Validates that the configuration directory and required files exist.

    Args:
        root (Path): The root directory where the 'config' folder is expected to be located.

    Raises:
        FileNotFoundError: If the configuration directory or required files are missing.
    """
    config_dir = root / 'config'
    if not config_dir.exists():
        print(f"Expected Structure: ROOT/config/hparam.json and ROOT/config/global.json")
        raise FileNotFoundError(f"Configuration directory not found at {config_dir}")
    
    required_files = ['hparam.json', 'global.json']
    for filename in required_files:
        file_path = config_dir / filename
        if file_path.exists(): continue
        raise FileNotFoundError(f"Required configuration file '{filename}' not found at {file_path}")


# #############################
# FUNCTIONS: HELPER
# #############################



# #############################
# FUNCTIONS: MAIN
# #############################
if __name__ == "__main__":
    # Example usage
    try:
        validate_dataset(dataset)
        print("Dataset validation passed.")
    except ValueError as e:
        print(f"Dataset validation failed: {e}")


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
# -----------------------------s