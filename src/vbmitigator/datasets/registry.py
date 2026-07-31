"""Dataset registry — the single extension point for datasets.

Adding a dataset takes **one self-contained module** under
``vbmitigator/datasets/`` that defines a builder decorated with
``@register_dataset("<name>")`` and returning the standard dataset dict::

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

The module is auto-discovered — no other file needs editing. Batches yielded by
the loaders must be dicts containing ``inputs``, ``targets``, ``index`` and one
key per name in ``biases``.
"""

import importlib
import pkgutil
import warnings

# Registered builders: name -> callable(cfg) -> dataset dict.
_REGISTRY = {}
_DISCOVERED = False

# Keys every builder must return (val/test loaders are optional but recommended).
REQUIRED_KEYS = ("num_class", "biases", "dataloaders", "sets", "root", "target2name")

# Package modules that are infrastructure, not datasets — skipped by discovery.
_NON_DATASET_MODULES = {"registry", "builder", "utils", "custom_transforms"}


def register_dataset(name):
    """Decorator registering ``fn`` as the builder for dataset ``name``."""

    def decorator(fn):
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not fn:
            raise ValueError(f"dataset '{name}' is already registered")
        _REGISTRY[name] = fn
        return fn

    return decorator


def _discover():
    """Import every dataset module once so its ``@register_dataset`` runs."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    import vbmitigator.datasets as pkg

    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name in _NON_DATASET_MODULES:
            continue
        try:
            importlib.import_module(f"vbmitigator.datasets.{info.name}")
        except Exception as e:  # a broken/optional module shouldn't kill discovery
            warnings.warn(f"dataset module '{info.name}' could not be imported: {e}")


def available_datasets():
    """Sorted list of registered dataset names."""
    _discover()
    return sorted(_REGISTRY)


def get_builder(name):
    """Return the builder registered for ``name`` (raising a clear error if absent)."""
    _discover()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {', '.join(available_datasets())}"
        )
    return _REGISTRY[name]


def validate_dataset(dataset, name):
    """Check a builder's output against the standard contract; return it unchanged."""
    if not isinstance(dataset, dict):
        raise TypeError(f"dataset '{name}' builder must return a dict, got {type(dataset)}")
    missing = [k for k in REQUIRED_KEYS if k not in dataset]
    if missing:
        raise KeyError(f"dataset '{name}' builder is missing keys: {missing}")
    dataloaders = dataset["dataloaders"]
    for split in ("train", "test"):
        if split not in dataloaders:
            raise KeyError(f"dataset '{name}' must provide a '{split}' dataloader")
    if "train" not in dataset["sets"]:
        raise KeyError(f"dataset '{name}' must provide sets['train']")
    return dataset


def needs_tag_loaders(cfg):
    """Whether the current run needs RAM tag dataloaders (MAVias / erm_tags)."""
    return cfg.MITIGATOR.TYPE in {"mavias", "erm_tags", "mhmavias"} or (
        cfg.METRIC == "wg_ovr_tags"
    )


def ram_transform(image_size):
    """Lazy proxy for RAM's tag-extraction transform.

    RAM (``recognize-anything``) is an optional dependency only needed by the
    tag-based methods, so it is imported on demand.
    """
    from ram import get_transform

    return get_transform(image_size=image_size)
