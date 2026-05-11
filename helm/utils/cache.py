# Cache utilities for HELM.

# Author:   Sharjeel Mustafa
# Created:  2026-05-07

# Objective: Cache utilities for HELM, including saving and loading of tuned hyperparameters.


# #############################
# IMPORTS
# #############################
# Builtin
from pathlib import Path
import json

# External

# Relative


# #############################
# VARS, CONSTS, & SETUP
# #############################
CACHE_FILE = 'cache.json'


# #############################
# FUNCTIONS: UTILITY
# #############################
def _create_cache(path: Path):
    if path.exists(): return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f: json.dump({}, f)


def write_cache(root: Path, key: str, dataset:str, hparams: dict):
    CACHE_PATH = root / 'cache' /CACHE_FILE
    _create_cache(CACHE_PATH)
    
    # Replace existing entry for key-dataset pair, if exists
    cache = json.load(CACHE_PATH.open('r'))
    cache[key] = cache.get(key, {})
    cache[key][dataset] = hparams
    with CACHE_PATH.open('w') as f: json.dump(cache, f)


def read_cache(root: Path, key: str, dataset: str):
    CACHE_PATH = root / 'cache' /CACHE_FILE
    if not CACHE_PATH.exists(): return None
    
    cache = json.load(CACHE_PATH.open('r'))
    res = cache.get(key, {}).get(dataset, None)
    
    string = f"found for model '{key}' and dataset '{dataset}'."
    if not res: print(f"No cache entry {string}")
    else:       print(f"Cache entry {string}")
    return res


def clear_cache(root: Path):
    CACHE_PATH = root / 'cache' /CACHE_FILE
    if CACHE_PATH.exists(): CACHE_PATH.unlink()
