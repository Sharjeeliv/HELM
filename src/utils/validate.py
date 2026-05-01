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
EXCLUDED_PARAMS = {'self', 'in_dim', 'out_dim', 'dataset'}


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


def validate_dataset(dataset):
    # --- Helper Functions ---
    def is_matrix(obj):
        return hasattr(obj, "shape") or hasattr(obj, "__array__")

    def _is_callable(obj):
        return callable(obj)

    # --- Custom Integrity Logic ---
    def check_integrity(d):
        """Cross-field checks for logical consistency."""
        # 1. Shape Consistency
        if len(d['X']) != len(d['y']):
            raise ValueError(f"X length ({len(d['X'])}) must match y length ({len(d['y'])})")
        
        
        # Contains NAN
        if np.isnan(d['X']).any() or np.isnan(d['y']).any():
            raise ValueError("Dataset contains NaN values, which are not allowed.")
        
        # Mask Overlap
        if (d['tr_mask'] & d['te_mask']).any():
            raise ValueError("Data leakage detected: tr_mask and te_mask overlap!")
        if (d['tr_mask'] & d['va_mask']).any():
            raise ValueError("Data leakage detected: tr_mask and va_mask overlap!")
        if (d['va_mask'] & d['te_mask']).any():
            raise ValueError("Data leakage detected: va_mask and te_mask overlap!")

        
        # 2. Mask Alignment
        n_samples = len(d['X'])
        for mask_name in ['tr_mask', 'va_mask', 'te_mask']:
            if len(d[mask_name]) != n_samples:
                raise ValueError(f"{mask_name} length must match X length ({n_samples})")

        # 3. Dimension Matching
        if d['X'].shape[1] != d['input_dim']:
            raise ValueError(f"X feature dim ({d['X'].shape[1]}) must match input_dim ({d['input_dim']})")

        return d

    # --- Combined Schema ---
    dataset_schema = Schema({
        'X': is_matrix,
        'y': is_matrix,
        'input_dim': And(int, lambda n: n > 0),
        'output_dim': And(int, lambda n: n > 0),
        'tr_mask': is_matrix,
        'va_mask': is_matrix,
        'te_mask': is_matrix,
        'encoder': object,
        'modelwise': {
            'data': dict,
            'func': {
                'init': _is_callable,
                'prop': _is_callable,
            }
        }
    }) # Runs after structural checks pass
    dataset_schema.validate(dataset)
    check_integrity(dataset) # Runs after schema validation passes, so we know all keys exist and have correct types

    

from inspect import signature



def validate_hparams():
    pass



def validate_tuning_config(tuning_config, models):
    """
    Validates the tuning configuration to ensure it contains valid parameter specifications.

    Args:
        tuning_config (dict): A dictionary where keys are model names and values are parameter configurations.

    Raises:
        ValueError: If any parameter configuration is not in the expected format.
    """
    
    
    # 1. Validate that tuning config keys match model names
    tuning_keys = set(tuning_config.keys())
    model_keys = set(models.keys())
    if not model_keys.issubset(tuning_keys):
        raise ValueError(f"Models contains keys not in tuning config: {model_keys - tuning_keys}")

    # 2. Validate each parameter specification
    #    For each model, the hparam keys should match the model parameters:
    for model_name, model_cls in models.items():
        model_params = set(signature(model_cls.__init__).parameters.keys()) - EXCLUDED_PARAMS
        tuning_params = set(tuning_config[model_name].keys())
        if not model_params.issubset(tuning_params):
            raise ValueError(f"Model '{model_name}' has parameters not in tuning config: {model_params - tuning_params}")



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
import numpy as np
dataset = {
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
        }}


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