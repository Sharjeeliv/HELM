This packages provides a template for boiler plate (hyper)graph learning including training, testing, tuning, etc. It handles much of the typical code, allowing the user to focus on novel changes to the pipline. To interace the user can call specific "pipes" and modify them if needed or call a general "pipeline" which abstracts away details for rapid usage and testing. The code itself is also designed to be modular and easy to modify or expand. 

The pipeline does not encompass data processing and model architecture. However, it does provide common utilities like printing results, early stopping implementation, caching, etc. It expects a particular input format, named files, and file structure.


All models must abide by the following interfaces. The initialization must include `in_dim, out_dim` to match the dataset. Additional needed variables (e.g., n_layers, hid_dim) must be defined in the corresponding params.json entry. This will be dynamically injected to the model at runtime.


```py
def __init__(self, in_dim, out_dim, ...)
```

To provide flexibility with model propagation, all models must implement their `forward` as follows:

```py

def forward(self, x: torch.Tensor, extra: dict):
    # unpack extra as needed
```
This allows for the model to take in specific data it may require (the user must ensure that it is packed into extra). Furthermore, it prevents overhead by only passing in objects as needed. This keeps all interfaces uniform while providing flexibility.


To ensure functionality across various datasets and models the following dataset scheme must be strictly followed. Additional, per-model data can be passed via the `extra` dictionary.


The dataset schema is as follows:
```py
dataset = {
    # 1. Primary Data
    'X': X,             # Features
    'y': y,             # Labels
    
    # 2. Structural Metadata
    'in_dim':  X.shape[1],
    'out_dim': output_dim,
    
    # 3. Mask Data
    'tr_mask': tr_mask,
    'va_mask': va_mask,
    'te_mask': te_mask,
    
    # 4. Utility Data
    'mask': None,       # Dynamically assigned
    'encoder': encoder, # For inverse transforms
    
    # 5. Model-Specific
    'extra': {
        'G': G,             # Example: Incidence/Adjacency
    }
}
```

Models can easily be passed to the pipeline by including relevant models in the `./models` folder and updating the `MODELS` variable in `./models/__init__.py`. Ensure that the key is how the model will be called and used throughout the pipeline, including the hyperparameter tuning config. The value should be the model **class** (not instance!) and inherit from nn.Module (i.e., PyTorch).


The models schema is as follows:
```py
MODELS = {
    'MODEL_NAME' : nn.Module,
    ...
}
```


Hyperparameter tuning can be easily managed via the hparams.json file. The following schema showcases how different types of search spaces can be encoded in the file. Allowing the user to easily change hyperparameters for runs without altering code or recompiling. Note: the H_PARAM_N must match the actual parameter name of the model/optimizer/criterion/etc.

The config/params.json schema is as follows:
```json
{
    "MODEL_NAME": {

        // Categorical Hyperparameters:
        // Format: [type, val_1, val_2, ..., val_n]
        // All values should be of the same type

        // Example of categorical hyperparameters
        "H_PARAM_1": ["cat", "val_1", "val_2", "val_3"],
        "H_PARAM_2": ["cat", 1, 2, 3],
        "H_PARAM_3": ["cat", 1.0, 2.0, 3.0],

        // Numerical Hyperparameters:
        // Format: [type, min_val, max_val]

        // Example of log hyperparameters
        "H_PARAM_4": ["log", 1e-4, 1e-2],

        // Example of int hyperparameters
        "H_PARAM_5": ["int", 5, 25],

        // Example of float hyperparameters
        "H_PARAM_6": ["flt", 0.5, 43.5],
    }
}
```