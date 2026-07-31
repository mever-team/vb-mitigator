# Extending VB-Mitigator

VB-Mitigator has **four** things you extend — datasets, methods, models, and
metrics — and they all use the **same registry pattern**: write one
self-contained, auto-discovered module, register it with a decorator, and select
it from a config. No central file to edit.

Run `vbm-list` at any time to see what's currently registered:

```bash
vbm-list
# datasets (11): biased_mnist, celeba, cifar10, ...
# methods (16):  badd, bb, debian, ...
# models (10):   efficientnet_b0, resnet18, ...
# metrics (7):   acc, acc_per_class, unb_bc_ba, ...
```

`vbm-train` / `vbm-eval` validate `DATASET.TYPE`, `MITIGATOR.TYPE`, `MODEL.TYPE`
and `METRIC` against these registries and fail early with the available list if
you mistype one.

## Add a dataset

Create `src/vbmitigator/datasets/my_dataset.py`:

```python
from vbmitigator.datasets.registry import register_dataset

@register_dataset("my_dataset")
def build_my_dataset(cfg):
    train_loader, train_set = ...            # your loaders
    return {
        "num_class": 10,
        "biases": ["my_bias"],               # one key per sensitive attribute
        "dataloaders": {"train": ..., "val": ..., "test": ...},
        "sets": {"train": train_set},        # only "train" is required
        "root": cfg.DATASET.MY_DATASET.ROOT,
        "target2name": {0: "cat", 1: "dog"},
        "ba_groups": [(0, 0), (1, 1)],       # optional (group-fairness metrics)
        "num_groups": 20,                    # optional (groupdro / di)
    }
```

Add a `CFG.DATASET.MY_DATASET` block in `config/defaults.py`, then run with
`DATASET.TYPE: my_dataset`. `get_dataset` validates the returned dict and reports
a clear error if a required key is missing. Batches your loaders yield must be
dicts with `inputs`, `targets`, `index`, and one key per bias name.

## Add a model

```python
from vbmitigator.models.registry import register_model

@register_model("my_net")
def build_my_net(num_classes, pretrained=False):
    return MyNet(num_classes, pretrained)
```

Select it with `MODEL.TYPE: my_net`. `forward(x)` should return `(logits, feat)`.
To support BAdd / MAVias, also implement `badd_forward` / `mavias_forward`
(see `models/resnet.py`). For torchvision backbones, map the `pretrained` bool
with `tv_weights(pretrained)`.

## Add a metric

```python
from vbmitigator.metrics.registry import register_metric

@register_metric("my_metric", performance="score", best="high")
def my_metric(data):
    # data has "targets", "predictions", and one array per bias attribute
    return {"score": ..., "aux": ...}
```

`performance` names the output key that drives best-checkpoint selection; `best`
is `"high"` or `"low"`. Select it with `METRIC: my_metric`.

## Add a method (mitigator)

Methods use an explicit registry (`mitigators/__init__.py`) rather than a
decorator, so their heavy optional dependencies stay **lazy** — a decorator would
force every method's module to import up front.

1. Create `src/vbmitigator/mitigators/my_method.py` with a class that subclasses
   `BaseTrainer` and overrides the hooks you need (commonly `_train_iter`, and
   optionally `_setup_models` / `_method_specific_setups`).
2. Register it:

   ```python
   _REGISTRY = {
       ...
       "my_method": ("my_method", "MyMethodTrainer"),
   }
   ```
3. Add its hyper-parameters under `CFG.MITIGATOR.MY_METHOD` in `config/defaults.py`.

Run it with `MITIGATOR.TYPE: my_method`.

## Testing your extension

The test suite registers a synthetic dataset and a toy model/metric through
these exact public APIs (see `tests/conftest.py` and `tests/test_*_registry.py`),
so they double as minimal, copy-pasteable examples.
```
