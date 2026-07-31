"""Metric correctness and the metric registry."""

import numpy as np
import pytest

from vbmitigator.metrics import available_metrics, get_metric, get_metric_meta, register_metric

acc = get_metric("acc")
wg_ovr = get_metric("wg_ovr")


def test_acc_perfect_and_zero():
    perfect = {"predictions": np.array([0, 1, 2]), "targets": np.array([0, 1, 2])}
    assert acc(perfect)["accuracy"] == 100.0
    wrong = {"predictions": np.array([1, 0]), "targets": np.array([0, 1])}
    assert acc(wrong)["accuracy"] == 0.0


def test_registry_and_meta():
    assert {"acc", "wg_ovr", "unb_bc_ba", "wg_ovr_tags"}.issubset(set(available_metrics()))
    meta = get_metric_meta("acc")
    assert meta == {"best": "high", "performance": "accuracy"}
    assert get_metric_meta("unb_bc_ba")["performance"] == "unb_acc"


def test_wg_ovr_runs():
    data = {
        "predictions": np.array([0, 0, 1, 1, 0, 1]),
        "targets": np.array([0, 0, 1, 1, 0, 1]),
        "unknown": np.array([0, 1, 0, 1, 0, 1]),
        "ba_groups": [(0, 0), (1, 1)],
    }
    assert isinstance(wg_ovr(data), dict)


def test_add_a_metric_in_one_call():
    @register_metric("toy_unit_metric", performance="score", best="low")
    def _m(data):
        return {"score": float((data["predictions"] != data["targets"]).mean())}

    assert "toy_unit_metric" in available_metrics()
    assert get_metric_meta("toy_unit_metric") == {"best": "low", "performance": "score"}
    out = get_metric("toy_unit_metric")(
        {"predictions": np.array([1, 1]), "targets": np.array([0, 1])}
    )
    assert out["score"] == 0.5


def test_unknown_metric_raises_with_hint():
    with pytest.raises(KeyError) as exc:
        get_metric("does_not_exist")
    assert "Available" in str(exc.value)
