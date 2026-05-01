import pytest

from src.utils.validate import validate_dataset
from schema import SchemaError


import pytest
import numpy as np
from schema import SchemaError

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
    for key in key_path[:-1]:
        target = target[key]
    
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
    # Schema.validate() returns the validated data, 
    # but your function doesn't return anything. 
    # Just checking it doesn't raise is enough.
    raw_dataset['tr_mask'] = raw_dataset['tr_mask'] & ~raw_dataset['te_mask']
    raw_dataset['va_mask'] = raw_dataset['va_mask'] & ~raw_dataset['te_mask'] & ~raw_dataset['tr_mask']
    validate_dataset(raw_dataset)

def test_mask_overlap(raw_dataset):
    """Ensure no sample is in both training and testing sets (leakage)."""
    
    # If overlap exists, check if it throws value error
    with pytest.raises(ValueError):
        validate_dataset(raw_dataset)
    
    
    # Remove overlap for a successful validation
    raw_dataset['tr_mask'] = raw_dataset['tr_mask'] & ~raw_dataset['te_mask']
    # remove overlap, in differnt way
    # raw_dataset['tr_mask'] = np.where(raw_dataset['te_mask'], False, raw_dataset['tr_mask'])
    
    overlap = raw_dataset['tr_mask'] & raw_dataset['te_mask']
    assert not overlap.any(), "Data leakage detected: masks overlap!"

def test_feature_dimension_mismatch(raw_dataset):
    """Ensure X's second dimension matches input_dim."""
    # If X has 10 features, input_dim must be 10
    raw_dataset['input_dim'] = raw_dataset['X'].shape[1] + 1
    with pytest.raises(ValueError): # Assuming your schema checks this
        validate_dataset(raw_dataset)

def test_contains_nan(raw_dataset):
    """Ensure dataset doesn't contain NaN values that break training."""
    raw_dataset['X'][0, 0] = np.nan
    # You might need to add a custom check for this in your validate_dataset
    with pytest.raises(ValueError, match="contains NaN"):
        validate_dataset(raw_dataset)