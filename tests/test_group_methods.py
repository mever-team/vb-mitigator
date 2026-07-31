"""Group-aware trainers after the _setup_dataset de-duplication:
groupdro/di no longer override _setup_dataset, and debian builds its second
loader from the existing dataset instead of rebuilding the whole dataset."""

import pytest

from vbmitigator.mitigators import get_trainer


def test_base_exposes_group_attrs(tiny_cfg, patch_dataset):
    tiny_cfg.freeze()
    t = get_trainer("erm")(tiny_cfg)
    # num_groups is now promoted into the base extraction.
    assert t.num_groups == 8  # synthetic dataset: num_class(4) * 2
    assert t.num_group == t.num_groups
    assert t.num_biases == 8 / 4


@pytest.mark.parametrize("method", ["groupdro", "di", "debian"])
def test_group_methods_train(tiny_cfg, patch_dataset, method):
    tiny_cfg.MITIGATOR.TYPE = method
    tiny_cfg.freeze()
    trainer = get_trainer(method)(tiny_cfg)
    trainer.train()  # must not raise after the refactor
    import os

    assert os.path.exists(os.path.join(trainer.run_dir, "best"))


def test_debian_second_loader_reuses_dataset(tiny_cfg, patch_dataset):
    tiny_cfg.MITIGATOR.TYPE = "debian"
    tiny_cfg.freeze()
    trainer = get_trainer("debian")(tiny_cfg)
    # The second loader must wrap the *same* dataset object (no rebuild).
    assert (
        trainer.second_train_loader.dataset
        is trainer.dataloaders["train"].dataset
    )
