"""Shared pytest fixtures.

Tests run fully on CPU against a tiny in-memory synthetic dataset, so they need
no downloads and finish in seconds.
"""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from vbmitigator.config import CFG
from vbmitigator.datasets import register_dataset

# The synthetic dataset used by the whole suite is registered through the *real*
# public extension API (@register_dataset), so the tests exercise the actual
# registry + get_dataset dispatch path — and double as the "how to add a
# dataset" example.
SYNTHETIC = "__synthetic__"


class _SyntheticBiasDataset(Dataset):
    """Tiny image dataset returning the standard batch-dict contract."""

    def __init__(self, n=32, num_class=4, image_size=32, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.inputs = torch.randn(n, 3, image_size, image_size, generator=g)
        self.targets = torch.randint(0, num_class, (n,), generator=g)
        # A "bias" attribute correlated with the target (0/1).
        self.bias = (self.targets % 2).long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return {
            "inputs": self.inputs[i],
            "targets": int(self.targets[i]),
            "unknown": int(self.bias[i]),
            "index": i,
        }


@register_dataset(SYNTHETIC)
def make_synthetic_dataset(cfg):
    """Tiny in-memory dataset registered like any real one."""
    num_class = 4
    train = _SyntheticBiasDataset(n=32, num_class=num_class, seed=1)
    val = _SyntheticBiasDataset(n=16, num_class=num_class, seed=2)
    test = _SyntheticBiasDataset(n=16, num_class=num_class, seed=3)
    bs = cfg.SOLVER.BATCH_SIZE
    return {
        "num_class": num_class,
        "num_groups": num_class * 2,
        "biases": ["unknown"],
        "dataloaders": {
            "train": DataLoader(train, batch_size=bs, shuffle=True),
            "val": DataLoader(val, batch_size=bs),
            "test": DataLoader(test, batch_size=bs),
        },
        "sets": {"train": train, "val": val, "test": test},
        "root": ".",
        "target2name": {i: f"class{i}" for i in range(num_class)},
        "ba_groups": [(i, i) for i in range(num_class)],
    }


@pytest.fixture
def tiny_cfg(tmp_path):
    """A minimal, frozen-elsewhere config for a 1-epoch CPU run."""
    cfg = CFG.clone()
    cfg.EXPERIMENT.GPU = "cpu"
    cfg.EXPERIMENT.SEED = 1
    cfg.EXPERIMENT.PROGRESS_BAR = False
    cfg.EXPERIMENT.CONFIG = "unit"
    cfg.DATASET.TYPE = SYNTHETIC
    cfg.MITIGATOR.TYPE = "erm"
    cfg.MODEL.TYPE = "resnet8"
    cfg.MODEL.PRETRAINED = False
    cfg.SOLVER.BATCH_SIZE = 16
    cfg.SOLVER.EPOCHS = 1
    cfg.SOLVER.TYPE = "SGD"
    cfg.SOLVER.SCHEDULER.TYPE = "MultiStepLR"
    cfg.METRIC = "acc"
    cfg.LOG.SAVE_CRITERION = "test"
    cfg.OUTPUT.DIR = str(tmp_path / "outputs")
    return cfg


@pytest.fixture
def patch_dataset():
    """Kept for signature compatibility.

    The synthetic dataset is registered at import time via ``@register_dataset``
    and selected through ``cfg.DATASET.TYPE``; trainers reach it via the real
    ``get_dataset`` dispatch, so no monkeypatching is needed.
    """
    return make_synthetic_dataset
