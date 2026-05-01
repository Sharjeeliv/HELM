# Import Interface Management for HELM

# Author:   Sharjeel Mustafa
# Created:  2026-04-30

# Objective: Manages what is exposed when importing the package, 
#            and provides a central place for imports across the package. 


# #############################
# IMPORTS
# #############################
# Builtin
from pathlib import Path

# External

# Relative
from .src.utils.validate import validate_models, validate_dataset, validate_config_dir


# #############################
# VARS, CONSTS, & SETUP
# #############################



# #############################
# FUNCTIONS: UTILITY
# #############################



# #############################
# FUNCTIONS: HELPER
# #############################



# #############################
# FUNCTIONS: MAIN
# #############################



# #############################
# FUNCTIONS: INTERFACE
# #############################
def helm(root: Path, models, dataset):
    
    # 1. Validation
    # 1.1 Validate paths and folder structure
    validate_config_dir(root)
    # 1.2 Validate models and dataset formats
    
    # 2. Pipeline: Tuning
    
    # 3. Pipeline: Training
    
    # 4. Pipeline: Testing
    
    # 5. Printing & Saving
    
    