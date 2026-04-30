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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Relative


# #############################
# VARS, CONSTS, & SETUP
# #############################



# #############################
# FUNCTIONS: UTILITY
# #############################

# FUNCTIONS: PLOTTING
# *****************************
def plot_confusion_matrix(labels, preds, class_names, title, root):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual Classes')
    plt.xlabel('Predicted Classes')
    plt.title('Classification Confusion Matrix')
    
    save_path = root / 'figures' / title
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()
    
    
# FUNCTIONS: RESULT MANAGEMENT
# *****************************
# Validate folder structure:
def validate_results_structure(root):
    # TODO: Remove this, combine with bottom
    results_dir = root / 'results'
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found at {results_dir}")
    
    # Validate subdirectories and create if missing
    for subdir in ['experiments', 'figures', 'summaries']:
        target_dir = results_dir / subdir
        if target_dir.exists(): continue
        target_dir.mkdir(parents=True)
        print(f"Created missing directory: {target_dir}")


def save_json_results(root, timestamp, dataset, expr, model, metrics, parameters, training_history):
    results_dir = root / 'results' / 'experiments' / expr / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    result_data = {
        "dataset": dataset,
        "model": model,
        "timestamp": timestamp,
        "metrics": metrics,
        "parameters": parameters,
        "training_history": training_history
    }
    
    save_path = results_dir / f"{dataset}_{model}.json"
    with open(save_path, 'w') as f:
        json.dump(result_data, f, indent=4)
    print(f"Results saved to: {save_path}")


def save_csv_results(root, timestamp, dataset, expr, model, metrics, parameters):
    csv_dir = root / 'results' / 'experiments' / expr
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = csv_dir / f"registry.csv"
    header = ['timestamp', 'dataset', 'model'] + list(metrics.keys()) + list(parameters.keys())
    
    # Write header if file doesn't exist
    if not csv_path.exists():
        with open(csv_path, 'w') as f:
            f.write(','.join(header) + '\n')
    
    # Append results
    with open(csv_path, 'a') as f:
        row = [timestamp, dataset, model] + [str(metrics[k]) for k in metrics] + [str(parameters[k]) for k in parameters]
        f.write(','.join(row) + '\n')
    print(f"Results appended to: {csv_path}")


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