# Testing for reporting functions in src.utils.report.

# Author:   Sharjeel Mustafa
# Created:  2026-05-08

# Objective: Test reporting functions for saving results to JSON and CSV.


# #############################
# IMPORTS
# #############################
# Builtin
import json

# External
import pytest

# Relative
from src.utils.report import _save_json, _make_csv, _save_csv, save_results


# #############################
# FUNCTIONS: MAIN
# #############################

# *****************************
# TEST: REPORT HELPERS
# *****************************
def test_save_json_logic(results_path, timestamp, mock_results):
    # 1. Test file saving and path creation
    _save_json(results_path, timestamp, mock_results)
    # Verify correct subdirs are created: 
    # results / experiments / expr_1 / timestamp / dataset_model.json
    expected_path = results_path / timestamp / "Cora_GCN.json"
    assert expected_path.exists()
    # 2. Verify content
    with open(expected_path, 'r') as f: saved_data = json.load(f)
    assert saved_data['timestamp'] == timestamp
    assert saved_data['dataset'] == "Cora"


def test_make_csv_header(results_path, mock_results):
    # 0. Ensure value error is raised if path is a directory
    # print(f"Testing _make_csv with directory path: {results_path}")
    with pytest.raises(ValueError):
        _make_csv(results_path, mock_results)
    # 1. Test header creation logic
    csv_path = results_path / "registry.csv"
    _make_csv(csv_path, mock_results)
    expected_path = results_path / "registry.csv"
    assert expected_path.exists()
    # 2. Verify header content
    with open(csv_path, 'r') as f: header = f.readline().strip().split(',')
    expected_header = ['timestamp', 'dataset', 'model', 'accuracy', 'f1', 'hidden_dim', 'lr']
    assert header == expected_header


def test_save_csv_appending(results_path, timestamp, mock_results):
    # 1. Call twice for initial save and append
    _save_csv(results_path, timestamp, mock_results)
    _save_csv(results_path, timestamp, mock_results)
    path = results_path / "registry.csv"
    with open(path, 'r') as f: lines = f.readlines()
    # 2. Verify both rows are present and correct
    assert len(lines) == 3 # Header + 2 rows
    last_row = lines[-1].strip().split(',')
    assert last_row[0] == timestamp
    assert last_row[1] == "Cora"
    # Accuracy from mock_results
    assert last_row[3] == "0.85"


# *****************************
# TEST: INTEGRATION (FULL)
# *****************************
def test_save_results_integration(tmp_path, timestamp, mock_results):
    expr_n = 1
    save_results(tmp_path, expr_n, timestamp, mock_results)
    # 1. Verify folder hierarchy: root / results / experiments / expr_1
    base_path = tmp_path / 'results' / 'experiments' / f"expr_{expr_n}"
    assert base_path.exists()
    # 2. Verify both files were created
    assert (base_path / "registry.csv").exists()
    assert (base_path / timestamp / "Cora_GCN.json").exists()


def test_save_results_validation_missing_key(tmp_path, timestamp):
    invalid_results = {"dataset": "Cora"}
    with pytest.raises(ValueError, match="Missing required key in results"):
        save_results(tmp_path, 1, timestamp, invalid_results)


def test_save_results_validation_type(tmp_path, timestamp):
    with pytest.raises(ValueError, match="Results must be a non-empty dictionary"):
        save_results(tmp_path, 1, timestamp, "not a dict") # type: ignore