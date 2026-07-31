"""CLI registry wiring: discovery command + config validation."""

import argparse
import json

import pytest

from vbmitigator.cli import _validate_cfg, list_main, registered_components
from vbmitigator.config import CFG


def test_registered_components_has_all_kinds():
    c = registered_components()
    assert set(c) == {"datasets", "methods", "models", "metrics"}
    assert "utkface" in c["datasets"]
    assert "erm" in c["methods"]
    assert "resnet18" in c["models"]
    assert "acc" in c["metrics"]


def test_list_main_json(capsys):
    list_main(["--json"])
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"datasets", "methods", "models", "metrics"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("DATASET.TYPE", "no_such_dataset"),
        ("MODEL.TYPE", "no_such_model"),
        ("METRIC", "no_such_metric"),
        ("MITIGATOR.TYPE", "no_such_method"),
    ],
)
def test_validate_cfg_rejects_unknown(field, value):
    cfg = CFG.clone()
    section, key = field.split(".") if "." in field else (None, field)
    if section:
        setattr(getattr(cfg, section), key, value)
    else:
        setattr(cfg, key, value)
    cfg.freeze()
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        _validate_cfg(cfg, parser)


def test_validate_cfg_accepts_defaults():
    cfg = CFG.clone()
    cfg.freeze()
    parser = argparse.ArgumentParser()
    _validate_cfg(cfg, parser)  # must not raise
