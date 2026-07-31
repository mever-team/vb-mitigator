"""Dataset factory for VB-Mitigator.

Thin dispatcher over the dataset registry: ``get_dataset(cfg)`` looks up
``cfg.DATASET.TYPE`` among the registered builders (see
:mod:`vbmitigator.datasets.registry`) and validates the result against the
standard contract. Adding a dataset never touches this file.
"""

from .registry import available_datasets, get_builder, validate_dataset

__all__ = ["get_dataset", "available_datasets"]


def get_dataset(cfg):
    """Build and return the standard dataset dict for ``cfg.DATASET.TYPE``."""
    builder = get_builder(cfg.DATASET.TYPE)
    dataset = builder(cfg)
    return validate_dataset(dataset, cfg.DATASET.TYPE)
