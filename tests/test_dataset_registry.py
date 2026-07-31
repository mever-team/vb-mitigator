"""The dataset registry: discovery, dispatch, validation, and the one-file
extension contract."""

import pytest

from vbmitigator.config import CFG
from vbmitigator.datasets import available_datasets, get_dataset, register_dataset
from vbmitigator.datasets.registry import validate_dataset

REAL_DATASETS = {
    "biased_mnist",
    "fb_biased_mnist",
    "utkface",
    "waterbirds",
    "celeba",
    "imagenet9",
    "imagenet9m",
    "cifar10",
    "cifar100",
    "stanford_dogs",
    "urbancars",
}


def test_all_real_datasets_registered():
    assert REAL_DATASETS.issubset(set(available_datasets()))


def test_add_a_dataset_in_one_call():
    """Registering a builder is all it takes to make it available to get_dataset."""

    @register_dataset("toy_unit_dataset")
    def _build(cfg):
        return {
            "num_class": 2,
            "biases": ["b"],
            "dataloaders": {"train": [], "val": [], "test": []},
            "sets": {"train": [0, 1]},
            "root": ".",
            "target2name": {0: "a", 1: "b"},
        }

    assert "toy_unit_dataset" in available_datasets()

    cfg = CFG.clone()
    cfg.DATASET.TYPE = "toy_unit_dataset"
    out = get_dataset(cfg)
    assert out["num_class"] == 2
    assert out["biases"] == ["b"]


def test_unknown_dataset_raises_with_hint():
    cfg = CFG.clone()
    cfg.DATASET.TYPE = "does_not_exist"
    with pytest.raises(KeyError) as exc:
        get_dataset(cfg)
    assert "Available" in str(exc.value)


def test_validate_rejects_incomplete_builder():
    with pytest.raises(KeyError):
        validate_dataset({"num_class": 2}, "bad")  # missing required keys
    with pytest.raises(KeyError):
        validate_dataset(
            {
                "num_class": 2,
                "biases": ["b"],
                "dataloaders": {"train": []},  # no "test"
                "sets": {"train": []},
                "root": ".",
                "target2name": {},
            },
            "bad",
        )


def test_duplicate_registration_rejected():
    @register_dataset("dup_unit_dataset")
    def _one(cfg):
        return {}

    with pytest.raises(ValueError):

        @register_dataset("dup_unit_dataset")
        def _two(cfg):
            return {}
