# **HELM: Hyper(graph) Experiment and Learning Manager**

**Author:** Sharjeel Mustafa
**Version:** 0.1.0 (Pre-release)
**Status:** Active Development 

HELM is a generic, config-driven pipeline for benchmarking graph and hypergraph machine learning models. Since hypergraphs generalise graphs, HELM operates on both naturally under a unified interface. It handles the full execution phase — validation, hyperparameter tuning, training, testing, and result storage — so researchers can focus on model development rather than pipeline boilerplate.

HELM is not a model library. It does not provide graph-specific layers, datasets, or preprocessing tools. It provides the infrastructure to run, compare, and record experiments consistently across arbitrary model architectures.

---

## Motivation

Benchmarking GNNs and HGNNs in practice requires re-implementing the same training loop, evaluation logic, hyperparameter search, and result storage for every new model or dataset. This leads to inconsistent comparisons, wasted effort, and results that are difficult to reproduce. HELM provides a single reusable pipeline that any conforming model can plug into, reducing experiment setup from days to hours.

---

## Features

- Config-driven hyperparameter tuning via Optuna
- Schema validation for datasets, models, and configs with informative error messages
- Introspection-based hyperparameter completeness checking
- Structured result storage (JSON per run, CSV registry per experiment)
- Early stopping support
- Modular pipeline stages: tune, train, test, evaluate
- Per-model data and function injection via the `modelwise` interface

---

## Current Limitations

HELM is under active development. The current version supports:

1. Transductive tasks only
2. Node classification only

Inductive learning, edge-level, hyperedge-level, and graph-level tasks are planned for future releases.

---

## Installation

HELM is not yet on PyPI. Install directly from GitHub:

```bash
pip install git+https://github.com/Sharjeeliv/HELM.git
```

> **Note:** PyTorch must be installed separately as installation is CUDA-version dependent.
> See [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### Updating

When a new version is pushed to GitHub, update your installation with:

```bash
pip install --force-reinstall git+https://github.com/Sharjeeliv/HELM.git
```

### Pinning to a Specific Commit

For reproducible experiments, pin to a specific commit:

```bash
pip install git+https://github.com/Sharjeeliv/HELM.git@<commit-hash>
```

---

## Project Structure

HELM expects the following structure in the **user's project**:

```
your_project/
    config/
        global.json       # Global training parameters
        hparam.json       # Hyperparameter search spaces
    results/              # Auto-populated by HELM
    models/               # Your model implementations
    data/                 # Your data processing code
    expr/                 # Your experiment scripts
```

See `schema.md` for the full schema specification of each configuration file.

---

## Required Configuration Files

### `config/global.json`

Global training parameters shared across all experiments:

```json
{
    "N_TRIALS": 50,
    "TR_EPOCH": 200,
    "TU_EPOCH": 100,
    "SEED": 42
}
```

### `config/hparam.json`

Hyperparameter search spaces per model. The key must match the model name in your `MODELS` dictionary:

```json
{
    "MODEL_NAME": {
        "hidden_dim":  ["int", 16, 128],
        "dropout":     ["flt", 0.0, 0.5],
        "lr":          ["log", 1e-4, 1e-2],
        "activation":  ["cat", "relu", "tanh"]
    }
}
```

Supported distribution types:

| Type  | Format                        | Description                        |
|-------|-------------------------------|------------------------------------|
| `int` | `["int", min, max]`           | Integer range, min < max           |
| `flt` | `["flt", min, max]`           | Float range, min < max             |
| `log` | `["log", min, max]`           | Log-scale float range, min < max   |
| `cat` | `["cat", val1, val2, ...]`    | Categorical, str/int/float values  |

---

## Model Interface

All models must conform to the following interface to be compatible with HELM.

### Initialisation

```python
def __init__(self, input_dim: int, output_dim: int, ...):
    pass
```

`input_dim` and `output_dim` are injected automatically from the dataset. All other parameters must have a corresponding entry in `hparam.json`.

### Forward Pass

```python
def forward(self, x: torch.Tensor, modelwise: dict) -> torch.Tensor:
    # Unpack model-specific data from modelwise['data'] as needed
    pass
```

### Registration

Models are registered in a dictionary and passed to HELM at runtime:

```python
from torch import nn

MODELS = {
    'GCN':  GCN,   # class, not instance
    'HGNN': HGNN,
}
```

---

## Dataset Schema

The dataset dictionary passed to HELM must conform to the following schema. See `schema.md` for full details.

```python
dataset = {
    # Primary data
    'X':          torch.Tensor,   # Node feature matrix [N, F]
    'y':          torch.Tensor,   # Labels [N]

    # Structural metadata
    'input_dim':  int,            # Number of input features (== X.shape[1])
    'output_dim': int,            # Number of output classes

    # Split masks
    'tr_mask':    torch.Tensor,   # Boolean training mask [N]
    'va_mask':    torch.Tensor,   # Boolean validation mask [N]
    'te_mask':    torch.Tensor,   # Boolean test mask [N]

    # Utility
    'encoder':    object,         # Label encoder for inverse transforms

    # Model-specific
    'modelwise': {
        'data': {                 # Any model-specific tensors or structures
            'G': torch.Tensor,    # Example: incidence or adjacency matrix
        },
        'func': {
            'init': callable,     # Optional: runs once before training
            'prop': callable,     # Optional: runs each iteration before forward
        }
    }
}
```

Masks must not overlap. HELM validates for data leakage before any training begins.

---

## Usage

### Basic Experiment

```python
from pathlib import Path
from time import time
from helm import helm

# 1. Define root (must contain config/ and results/)
root = Path(__file__).parent

# 2. Define models
from models import GCN, HGNN
MODELS = {'GCN': GCN, 'HGNN': HGNN}

# 3. Load and prepare your dataset (conforming to schema)
dataset = load_my_dataset()

# 4. Run the full pipeline
expr_n = 1 # The experiment number
timestamp = time.time()
for key, model in MODELS.items():
    helm(root=root, expr_n=expr_n timestamp=timestamp,
    key=key, model=model, dataset=dataset, to_tune=False)
```

### Running Individual Stages

```python
from helm.pipes.tune  import tune
from helm.pipes.train import train
from helm.pipes.test  import test

# Tune hyperparameters
_model = {key: model} # single key-model
hparams = tune(root, key, _model, dataset, epochs=TU_EPOCHS, n_trials=N_TRIALS)

# Train with best params
trmodel = get_model(_model, key, hparams, dataset)
trmodel, history = train(trmodel, hparams, dataset, epochs=TR_EPOCHS)
    

# Test
results = test(trmodel, dataset) 

# Results
metrics = {k: v for k, v in results.items() if k not in ['preds', 'labels']}
results = {'dataset': dataset['name'], 'model': key, 'metrics': metrics, 'hparams': hparams, 'history': history}

save_results(root, expr_n, timestamp, results)
return results
```

---

## Results Structure

HELM automatically writes results to `results/` in the following structure:

```
results/
    experiments/
        expr1/
            registry.csv              # Flat registry of all runs in this experiment
            2026-05-01T12-00-00/
                gcn_results.json
                hgnn_results.json
            2026-05-02T09-30-00/
                gcn_results.json
    summaries/
    figures/
```

### Per-run JSON format

```json
{
    "dataset":   "Cora",
    "model":     "GCN",
    "timestamp": "2026-05-01T12:00:00Z",
    "seed":      42,

    "metrics": {
        "accuracy":  0.82,
        "precision": 0.81,
        "recall":    0.80,
        "f1":        0.81
    },

    "hparams": {
        "hidden_dim": 64,
        "dropout":    0.3,
        "lr":         0.001
    },

    "history": [
        {"epoch": 1,   "loss": 0.95, "train_acc": 0.42, "val_acc": 0.40},
        {"epoch": 10,  "loss": 0.60, "train_acc": 0.71, "val_acc": 0.68}
    ]
}
```

---

## Validation

HELM runs a full preflight validation before any expensive computation starts:

1. **Directory structure** — confirms `config/` and `results/` exist with required files
2. **Global config** — validates types and value ranges
3. **Dataset schema** — validates structure, types, dimensions, NaN values, and mask overlap
4. **Models** — confirms all values are `nn.Module` subclasses (classes, not instances)
5. **Hyperparameter config** — confirms coverage, completeness against model signatures, and search space format

Any failure produces a specific, actionable error message before training begins.

---

## Roadmap

- Inductive learning support
- Edge-level and graph-level task support
- PyPI release
- Extended working examples
- Figures and summary generation utilities

---

## Contributing

Contributions are welcome. Please ensure any new model or data processing code conforms to the interfaces described above. See `schema.md` for the full specification.

If contributing to the pipeline itself, run the test suite before submitting:

```bash
pytest tests/
```

---

## Citation

If you use HELM in your research, please cite:

```bibtex
@misc{mustafa2026helm,
    title  = {HELM: Hyper(graph) Experiment and Learning Manager},
    author = {Mustafa, Sharjeel},
    year   = {2026},
    url    = {https://github.com/Sharjeeliv/HELM}
}
```

---

## License

MIT License. See `LICENSE` for details.