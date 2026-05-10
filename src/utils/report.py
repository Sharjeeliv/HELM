# Title

# Author:   Sharjeel Mustafa
# Created:  2026-04-30

# Objective: Short Summary. 


# #############################
# IMPORTS
# #############################
# Builtin
import json
from pathlib import Path

# External
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix

# Relative


# #############################
# VARS, CONSTS, & SETUP
# #############################



# #############################
# FUNCTIONS: UTILITY
# #############################

# FUNCTIONS: PLOTTING
# *****************************
# def plot_confusion_matrix(labels, preds, class_names, title, root):
#     cm = confusion_matrix(labels, preds)
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=class_names, yticklabels=class_names)
#     plt.ylabel('Actual Classes')
#     plt.xlabel('Predicted Classes')
#     plt.title('Classification Confusion Matrix')
    
#     save_path = root / 'figures' / title
#     plt.savefig(save_path, bbox_inches='tight', dpi=300)
#     print(f"Confusion matrix saved to: {save_path}")
#     plt.close()
    
    
# FUNCTIONS: RESULT MANAGEMENT
# *****************************
def _save_json(base: Path, timestamp: str, data: dict):
    data['timestamp'] = timestamp
    m, d = data['model'], data['dataset']
    save_path = base / f"{timestamp}" / f"{d}_{m}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to: {save_path}")

def _make_csv(path: Path, data: dict):
    if path.exists() and not path.is_dir(): return
    if path.is_dir(): raise ValueError(f"Expected a file path, got directory: {path}")
    
    # Construct header based on metrics + hparams
    header = ['timestamp', 'dataset', 'model']
    header += list(data['metrics'].keys())
    header += list(data['hparams'].keys())
    # Write header if file doesn't exist
    with open(path, 'w') as f:
        f.write(','.join(header) + '\n')

def _save_csv(base: Path, timestamp: str, data: dict):
    # 1. Create CSV path and header if not exists
    m, d, t = data['model'], data['dataset'], timestamp
    csv_path = base / f"registry.csv"
    _make_csv(csv_path, data)
    
    # 2. Append results
    with open(csv_path, 'a') as f:
        row = [t, d, m]
        row += [str(data['metrics'][k]) for k in data['metrics']]
        row += [str(data['hparams'][k]) for k in data['hparams']]
        f.write(','.join(row) + '\n')
    print(f"Results appended to: {csv_path}")
    
    
def _validate_results(results: dict):
    # General structure
    if results is None or not isinstance(results, dict):
        raise ValueError("Results must be a non-empty dictionary.")
    required_keys = ['dataset', 'model', 'metrics', 'hparams', 'history']
    for key in required_keys:
        if key in results: continue
        raise ValueError(f"Missing required key in results: {key}")
    

def save_results(root: Path, expr_n: int, timestamp: str, results: dict):
    # 0. Validation, creation
    path = root / 'results' / 'experiments' / f"expr_{expr_n}"
    path.mkdir(parents=True, exist_ok=True)
    _validate_results(results)

    # 1. Save per result JSON
    _save_json(path, timestamp, results)
    
    # 2. Append to registry CSV
    _save_csv(path, timestamp, results)





# #############################
# FUNCTIONS: HELPER
# #############################



# #############################
# FUNCTIONS: MAIN
# #############################



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