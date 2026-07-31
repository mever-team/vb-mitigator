"""Run-summary visualizations built from a predictions frame."""

import matplotlib.figure
import numpy as np
import pandas as pd

from vbmitigator.ui import viz


def _predictions():
    # 2 targets x 2 bias values; group (male, race=1) is deliberately worst.
    rng = np.random.default_rng(0)
    rows = []
    specs = {
        (0, 0): (30, 0.95),  # male, race0
        (0, 1): (30, 0.90),  # male, race1
        (1, 0): (30, 0.92),  # female, race0
        (1, 1): (30, 0.50),  # female, race1  <- worst
    }
    names = {0: "male", 1: "female"}
    for (t, b), (n, acc) in specs.items():
        for _ in range(n):
            correct = rng.random() < acc
            rows.append(
                {"target": t, "prediction": t if correct else 1 - t,
                 "target_name": names[t], "race": b}
            )
    return pd.DataFrame(rows)


def test_bias_columns():
    df = _predictions()
    assert viz.bias_columns(df) == ["race"]


def test_worst_group_is_lowest():
    df = _predictions()
    worst = viz.worst_group(df, "race")
    assert worst["target"] == 1 and worst["bias"] == 1
    assert worst["accuracy"] < 0.7


def test_subgroup_table_sorted():
    t = viz.subgroup_table(_predictions(), "race")
    assert list(t["accuracy"]) == sorted(t["accuracy"])  # ascending, worst first


def test_figures_render():
    df = _predictions()
    for fig in (
        viz.subgroup_accuracy_heatmap(df, "race"),
        viz.subgroup_count_heatmap(df, "race"),
        viz.confusion_matrix_fig(df),
    ):
        assert isinstance(fig, matplotlib.figure.Figure)


def test_per_class_accuracy_frame():
    pc = viz.per_class_accuracy(_predictions())
    assert set(pc.index) == {"male", "female"}
    assert "accuracy (%)" in pc.columns


def test_no_bias_columns_safe():
    df = pd.DataFrame({"target": [0, 1], "prediction": [0, 1], "target_name": ["a", "b"]})
    assert viz.bias_columns(df) == []
    # confusion matrix still works without a bias column
    assert isinstance(viz.confusion_matrix_fig(df), matplotlib.figure.Figure)
