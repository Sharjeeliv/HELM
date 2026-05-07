# Input Validation Functions for HELM

# Author:   Sharjeel Mustafa
# Created:  2026-04-30

# Objective: Validation functions to ensure integrity of inputs and
#            correct interfacing with the HELM framework


# #############################
# IMPORTS
# #############################
# Builtin
from pathlib import Path
from inspect import signature
import json

# External
from schema import Schema, And, Optional
from torch import nn
import numpy as np

# Relative


# #############################
# VARS, CONSTS, & SETUP
# #############################
EXCLUDED_PARAMS = {'self', 'input_dim', 'output_dim', 'dataset', # Data-driven parameters
                   'lr', 'optimizer', 'weight_decay'}            # Non-model parameters
REQUIRED_GLOBAL_CONFIG_KEYS = {'N_TRIALS', 'TR_EPOCH','TU_EPOCH', 'SEED'}

# #############################
# FUNCTIONS: HELPER
# #############################
def _is_callable(obj):
    return callable(obj)

def is_matrix(obj):
    return hasattr(obj, "shape") or hasattr(obj, "__array__")

def _check_data_integrity(d):
    """Cross-field checks for logical consistency."""
   
    # 1. Shape Consistency
    if len(d['X']) != len(d['y']):
        raise ValueError(f"X length ({len(d['X'])}) must match y length ({len(d['y'])})")
    
    # 2. Contains NAN
    if np.isnan(d['X']).any() or np.isnan(d['y']).any():
        raise ValueError("Dataset contains NaN values, which are not allowed.")
    
    # 3. Mask Overlap
    if (d['tr_mask'] & d['te_mask']).any():
        raise ValueError("Data leakage detected: tr_mask and te_mask overlap!")
    if (d['tr_mask'] & d['va_mask']).any():
        raise ValueError("Data leakage detected: tr_mask and va_mask overlap!")
    if (d['va_mask'] & d['te_mask']).any():
        raise ValueError("Data leakage detected: va_mask and te_mask overlap!")

    # 4. Mask Alignment
    n_samples = len(d['X'])
    for mask_name in ['tr_mask', 'va_mask', 'te_mask']:
        if len(d[mask_name]) == n_samples: continue
        raise ValueError(f"{mask_name} length must match X length ({n_samples})")

    # 5. Dimension Matching
    if d['X'].shape[1] != d['input_dim']:
        raise ValueError(f"X feature dim ({d['X'].shape[1]}) must match input_dim ({d['input_dim']})")


# #############################
# FUNCTIONS: MAIN
# #############################

# *****************************
# Models Validation
# *****************************
def validate_models(models: dict[str, nn.Module]) -> None:
    """Validates the models dictionary
    to ensure it contains valid nn.Module instances.

    Args:
        models (dict): A dictionary where keys are model names (str) and values are nn.Module subclasses.

    Raises:
        ValueError: If any value in the models dictionary is not a subclass of nn.Module.
    """
    for name, model in models.items():
        # 1. Determine if keys are strings
        if type(name) is not str:
            raise ValueError(f"Model name '{name}' is not a string.")
        # 2. Determine if values are classes (not instances)
        if not isinstance(model, type):
            raise ValueError(f"Model '{name}' is not a class.")
        # 3. Determine if values are subclasses of nn.Module
        if issubclass(model, nn.Module): continue
        raise ValueError(f"Model '{name}' is not a valid nn.Module subclass.")


# *****************************
# Dataset Validation
# *****************************
def validate_dataset(dataset: dict[str, object]) -> None:
    """Validates the dataset dictionary
    to ensure it contains all required fields with correct types and logical consistency.
    Args:
        dataset (dict): A dictionary containing the dataset and related information.
    Raises:
        SchemaError: If any required field is missing, has an incorrect type, or if there are logical inconsistencies in the data.
        
    The expected structure of the dataset dictionary is as follows:
        
        {
            'X': <matrix-like>,          # Feature matrix (2D array-like)
            'y': <matrix-like>,          # Target matrix (2D array-like)
            'input_dim': <int>,          # Number of input features (positive integer)
            'output_dim': <int>,         # Number of output targets (positive integer)
            'tr_mask': <matrix-like>,    # Boolean mask for training samples (1D array-like)
            'va_mask': <matrix-like>,    # Boolean mask for validation samples (1D array-like)
            'te_mask': <matrix-like>,    # Boolean mask for test samples (1D array-like)
            'encoder': <callable>,       # Function to encode the data (e.g., a feature encoder)
            'modelwise': {
                'data': <dict>,          # Additional data specific to the model (can be empty)
                'func': {
                    'init': <callable>,   # Function to initialize the model
                    'prop': <callable>    # Function to compute properties of the model
                }
            }
        """
    dataset_schema = Schema({
        'name': str,
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
                'init': Optional(_is_callable),
                'prop': Optional(_is_callable),
            }
        }
    })
    dataset_schema.validate(dataset)    # 1. Validate schema structure and types
    _check_data_integrity(dataset)      # 2. Validate data integrity and consistency


# *****************************
# Directory Structure Validation
# *****************************
def validate_dir_structure(root: Path):
    config_dir = root / 'config'
    if not config_dir.exists():
        raise FileNotFoundError(f"Configuration directory not found at {config_dir}")
    
    results_dir = root / 'results'
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found at {results_dir}")
    
    required_files = ['hparam.json', 'global.json']
    for filename in required_files:
        file_path = config_dir / filename
        if file_path.exists(): continue
        raise FileNotFoundError(f"Required configuration file '{filename}' not found at {file_path}")


# *****************************
# Config Validation
# *****************************
def _validate_model_hparams(model_hparam: dict, model: nn.Module) -> None: 
    """Validates the model hyperparameter keys
    in the tuning configuration match the model parameters."""
    model_params = set(signature(model.__init__).parameters.keys()) - EXCLUDED_PARAMS
    hparam_keys = set(model_hparam.keys())
    # Ensure all model parameters are present in hparam config, 
    # but allow extra keys for non-model parameters or metadata.
    if model_params.issubset(hparam_keys): return
    missing_params = model_params - hparam_keys
    raise ValueError(f"Model parameters {missing_params} are missing from hparam config.")
        

def _validate_hparam_values(model_hparam: dict, model_name: str) -> None:
    """Validates the values of the hyperparameters
    in the tuning configuration."""
    for param_name, spec in model_hparam.items():
        base_msg = f"Parameter '{param_name}' in model '{model_name}'"
        match spec:
            case ["cat", *vals] if vals and all(
                isinstance(v, (str, int, float)) for v in vals):            continue
            case [("log" | "flt"), float(low), float(high)] if low < high:  continue
            case ['int', int(low), int(high)] if low < high:                continue
            case [str() as dist, *_] if dist not in ["cat", "log", "flt", "int"]:
                raise ValueError(f"{base_msg} has unknown distribution type '{dist}'.")
            case _:
                raise ValueError(f"{base_msg} has an invalid specification.")
        

def validate_hparam_config(root: Path, models: dict[str, nn.Module]) -> None:
    """Validates the structure of the hyperparameter configuration."""
    
    # 0. Retrieve hparam config
    hparam_config = json.load(open(root / "config" / "hparam.json"))
    
    # 1. Validate tuning keys match model names
    if not set(models.keys()).issubset(set(hparam_config.keys())):
        extra_keys = set(models.keys()) - set(hparam_config.keys())
        raise ValueError(f"Models contains keys not in hparam config: {extra_keys}")
    
    for model_name, model_hparam in hparam_config.items():
        
        # 2. Validate model hparam keys match model parameters
        _validate_model_hparams(model_hparam, models[model_name])
        
        # 3. Validate each parameter specification
        #    - Categorical: ["cat", val1, val2, ...] where each val is str, int, or float
        #    - Log/Float/Int: ["log"/"flt"/"int", low, high] where low < high and 
        #      types match distribution type
        _validate_hparam_values(model_hparam, model_name)


def validate_global_config(root: Path): 
    """Validates the global configuration settings."""
    # Assuming the file exists
    global_config_path = root / 'config' / 'global.json'
    global_config = json.load(global_config_path.open())
    for key in REQUIRED_GLOBAL_CONFIG_KEYS:
        # 1. Check presence of required keys
        if key not in global_config:
            raise ValueError(f"Global config is missing required key: '{key}'")
        # 2. All values should be integers > 0
        value = global_config[key]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Global config key '{key}' must be a positive integer.")