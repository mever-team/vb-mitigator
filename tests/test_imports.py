"""Package imports and the trainer registry."""

import pytest

import vbmitigator
from vbmitigator.mitigators import AVAILABLE_METHODS, get_trainer

# Methods whose modules only depend on the core install (no optional extras).
LIGHT_METHODS = [
    "erm",
    "sd",
    "groupdro",
    "lff",
    "end",
    "bb",
    "jtt",
    "di",
    "debian",
    "flac",
    "flacb",
    "badd",
    "softcon",
]


def test_version():
    assert isinstance(vbmitigator.__version__, str)


def test_registry_complete():
    assert set(LIGHT_METHODS).issubset(set(AVAILABLE_METHODS))
    assert "mavias" in AVAILABLE_METHODS


@pytest.mark.parametrize("name", LIGHT_METHODS)
def test_light_trainers_import(name):
    cls = get_trainer(name)
    assert callable(cls)


def test_unknown_method_raises():
    with pytest.raises(KeyError):
        get_trainer("does_not_exist")


def test_builders_import():
    from vbmitigator.datasets.builder import get_dataset  # noqa: F401
    from vbmitigator.metrics import available_metrics, get_metric  # noqa: F401
    from vbmitigator.models.builder import get_model  # noqa: F401
