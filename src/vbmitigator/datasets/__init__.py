"""Datasets package.

Public API:
    get_dataset(cfg)              build the standard dict for cfg.DATASET.TYPE
    register_dataset(name)        decorator to add a dataset (see registry docs)
    available_datasets()          list registered dataset names
"""

from .builder import get_dataset
from .registry import available_datasets, register_dataset

__all__ = ["get_dataset", "register_dataset", "available_datasets"]
