# Testing for validation functions in src.utils.validate.

# Author:   Sharjeel Mustafa
# Created:  2024-05-01

# Objective: Test validation functions for dataset, models,
#            hyperparameters, and configuration directories.


# #############################
# IMPORTS
# #############################
# Builtin

# External
import json

import pytest
from schema import SchemaError
import numpy as np
from torch import nn

# Relative
from src.utils.validate import (validate_dataset, validate_models, 
                                validate_global_config, 
                                validate_dir_structure, 
                                validate_hparam_config)

# #############################
# FUNCTIONS: HELPER
# #############################
def create_json(name, data, tmp_path):
    """Helper to create a JSON file in the config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    file_path = config_dir / name
    file_path.write_text(json.dumps(data))
    return file_path

# #############################
# FUNCTIONS: MAIN
# #############################

# *****************************
# TEST: VALIDATE_DATASET
# *****************************
@pytest.mark.parametrize("key_path, new_value", [
    (["X"], None),
    (["te_mask"], "not a mask"),
    (["input_dim"], -1),
    (["output_dim"], 0),
    (["modelwise", "data"], "not a dict"),
    (["modelwise", "func", "init"], "not callable"),
])
def test_validate_dataset_errors(raw_dataset, key_path, new_value):
    # Traverse the dictionary to the nested key
    target = raw_dataset
    for key in key_path[:-1]: target = target[key]
    target[key_path[-1]] = new_value
    with pytest.raises(SchemaError): 
        validate_dataset(raw_dataset)

@pytest.mark.parametrize("missing_key", [
    "X", "y", "input_dim", "output_dim", "tr_mask", "modelwise"
])
def test_validate_dataset_missing_keys(raw_dataset, missing_key):
    del raw_dataset[missing_key]
    with pytest.raises(SchemaError):
        validate_dataset(raw_dataset)

def test_validate_dataset_success(raw_dataset):
    raw_dataset['tr_mask'] = raw_dataset['tr_mask'] & ~raw_dataset['te_mask']
    raw_dataset['va_mask'] = raw_dataset['va_mask'] & ~raw_dataset['te_mask'] & ~raw_dataset['tr_mask']
    validate_dataset(raw_dataset)

def test_mask_overlap(raw_dataset):
    """Ensure no sample is in both training and testing sets (leakage)."""
    # If overlap exists, check if it throws value error
    with pytest.raises(ValueError): validate_dataset(raw_dataset)
    # Remove overlap for a successful validation
    raw_dataset['tr_mask'] = raw_dataset['tr_mask'] & ~raw_dataset['te_mask']    
    overlap = raw_dataset['tr_mask'] & raw_dataset['te_mask']
    assert not overlap.any(), "Data leakage detected: masks overlap!"

def test_feature_dimension_mismatch(raw_dataset):
    """Ensure X's second dimension matches input_dim."""
    # If X has 10 features, input_dim must be 10
    raw_dataset['input_dim'] = raw_dataset['X'].shape[1] + 1
    with pytest.raises(ValueError): validate_dataset(raw_dataset)

def test_contains_nan(raw_dataset):
    """Ensure dataset doesn't contain NaN values that break training."""
    raw_dataset['X'][0, 0] = np.nan
    with pytest.raises(ValueError, match="contains NaN"):
        validate_dataset(raw_dataset)
        
        
# *****************************
# TEST: VALIDATE_MODELS
# *****************************
def test_validate_models_success(DummyModel):
    models = {
        'model1': DummyModel,
        'model2': DummyModel}
    assert validate_models(models) is None

@pytest.mark.parametrize("model_name, model_instance", [
    ("model1", "not a model"),
    (1, nn.Linear),
    ("model2", nn.Linear(5, 2))
])
def test_validate_models_errors(model_name, model_instance):
    models = {model_name: model_instance}
    with pytest.raises(ValueError): validate_models(models)


# *****************************
# TEST: VALIDATE_HPARAM_CONFIG
# *****************************
def test_validate_hparam_config_success(tmp_path, valid_hparam_config, models_dict):
    create_json("hparam.json", valid_hparam_config, tmp_path)
    validate_hparam_config(tmp_path, models_dict)

def test_validate_hparam_missing_model_key(tmp_path, models_dict):
    # Setup: Create the config directory and an empty hparam file
    create_json("hparam.json", {}, tmp_path)    
    with pytest.raises(ValueError):
        validate_hparam_config(tmp_path, models_dict)

def test_validate_model_hparams_missing_param(tmp_path, models_dict):
    data = {"MockModel": {"activation": ["cat", "a"]},
            "GCN": {"hidden_dim": ["int", 1, 10], "dropout": ["flt", 0.0, 0.5]}}
    create_json("hparam.json", data, tmp_path)
    
    with pytest.raises(ValueError):
        validate_hparam_config(tmp_path, models_dict)

@pytest.mark.parametrize("bad_spec", [
    ["int", 10, 5],            # low > high
    ["flt", 0.1, "0.5"],       # wrong type (str instead of float)
    ["cat"],                   # empty values
    ["log", 1e-4],             # missing high
    ["unknown", 1, 10],        # invalid dist name
    ["int", 1.5, 10.5],        # int dist with floats
])
def test_validate_hparam_values_failures(tmp_path, models_dict, bad_spec):
    data = {
        "MockModel": {
            "hidden_dim": bad_spec,
            "lr": ["log", 1e-4, 1e-2],
            "activation": ["cat", "relu"]
        },
        "GCN": {
            "hidden_dim": ["int", 1, 10], 
            "dropout": ["flt", 0.0, 0.5]}}
    create_json("hparam.json", data, tmp_path)
    
    with pytest.raises(ValueError):
        validate_hparam_config(tmp_path, models_dict)

# *****************************
# TEST: VALIDATE_GLOBAL_CONFIG
# *****************************
def test_validate_global_config_success(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "global.json"
    
    data = {"N_TRIALS": 10, "TR_EPOCH": 100, "TU_EPOCH": 50, "SEED": 42}
    config_file.write_text(json.dumps(data))
    validate_global_config(tmp_path)

@pytest.mark.parametrize("invalid_data", [
    {"N_TRIALS": 10, "TR_EPOCH": 100, "TU_EPOCH": 50},                      # Missing SEED
    {"N_TRIALS": 10, "TR_EPOCH": 100, "TU_EPOCH": 50, "SEED": -1},          # Negative value
    {"N_TRIALS": 10, "TR_EPOCH": 100, "TU_EPOCH": 50, "SEED": "not int"},   # Wrong type
    {"N_TRIALS": 0, "TR_EPOCH": 100, "TU_EPOCH": 50, "SEED": 42},           # Zero value
])
def test_validate_global_config_failures(tmp_path, invalid_data):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "global.json"
    config_file.write_text(json.dumps(invalid_data))
    with pytest.raises(ValueError): 
        validate_global_config(tmp_path)


# *****************************
# TEST: VALIDATE_DIR_STRUCTURE
# *****************************
def test_validate_dir_structure_success(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "config" / "hparam.json").touch()
    (tmp_path / "config" / "global.json").touch()
    validate_dir_structure(tmp_path)

def test_validate_dir_structure_missing_results(tmp_path):
    (tmp_path / "config").mkdir()
    with pytest.raises(FileNotFoundError, match="Results directory not found"):
        validate_dir_structure(tmp_path)

def test_validate_dir_structure_missing_file(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "config" / "hparam.json").touch()
    # missing global.json
    with pytest.raises(FileNotFoundError, match="Required configuration file 'global.json'"):
        validate_dir_structure(tmp_path)