# Testing for caching functions in src.utils.report

# Author:   Sharjeel Mustafa
# Created:  2026-05-10

# Objective: Test caching functions for saving and loading hyperparameters.


# #############################
# IMPORTS
# #############################
# Builtin
import json

# External

# Relative
from src.utils.cache import _create_cache, write_cache, read_cache, clear_cache


# #############################
# FUNCTIONS: MAIN
# #############################
def test_write_cache_creates_file(tmp_path, cache_path, cache_params):
    # Action: Write to cache
    write_cache(tmp_path, "GCN", "Cora", cache_params)
    # Assert: Cache file should be created with correct content
    assert cache_path.exists()
    with cache_path.open('r') as f: data = json.load(f)
    assert data["GCN"]["Cora"] == cache_params


def test_write_cache_updates_existing_key(tmp_path, cache_params):
    # Setup: Write initial params
    write_cache(tmp_path, "GCN", "Cora", cache_params)
    # Action: Update same model/dataset with new params
    new_params = {"hidden_dim": 128}
    write_cache(tmp_path, "GCN", "Cora", new_params)
    # Assert: Should update existing entry, not create duplicate
    res = read_cache(tmp_path, "GCN", "Cora")
    assert res == new_params


def test_write_cache_preserves_other_entries(tmp_path, cache_params):
    # Setup: Write initial params for one model/dataset
    write_cache(tmp_path, "GCN", "Cora", cache_params)
    # Action: Write a different model
    other_params = {"layers": 3}
    write_cache(tmp_path, "GAT", "Cora", other_params)
    # Assert: Should contain both entries without overwriting
    assert read_cache(tmp_path, "GCN", "Cora") == cache_params
    assert read_cache(tmp_path, "GAT", "Cora") == other_params


def test_read_cache_missing_file(tmp_path):
    # Action & Assert: Should return None if cache file doesn't exist
    assert read_cache(tmp_path, "GCN", "Cora") is None


def test_read_cache_missing_key(tmp_path, cache_params):
    # Setup: Write cache for one model/dataset
    write_cache(tmp_path, "GCN", "Cora", cache_params)
    # Action & Assert: Should return None for missing model/dataset combinations
    assert read_cache(tmp_path, "GCN", "CiteSeer") is None
    assert read_cache(tmp_path, "GAT", "Cora") is None


def test_clear_cache(tmp_path, cache_path, cache_params):
    # Setup: Write to cache to ensure file exists
    write_cache(tmp_path, "GCN", "Cora", cache_params)
    assert cache_path.exists()
    # Action: Clear cache
    clear_cache(tmp_path)
    # Assert: Cache file should be deleted
    assert not cache_path.exists()


def test_clear_cache_no_file(tmp_path):
    # Should not raise error if file doesn't exist
    clear_cache(tmp_path)
    
    
def test_create_cache_idempotency(tmp_path, cache_path):
    # Action 1: Create cache file
    _create_cache(cache_path)
    cache_path.write_text(json.dumps({"test": 1}))

    # Action 2: Should not overwrite if exists
    _create_cache(cache_path)

    with cache_path.open('r') as f: data = json.load(f)
    assert data == {"test": 1}