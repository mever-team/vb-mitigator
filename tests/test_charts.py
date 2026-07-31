"""Workspace chart builders (consistent colours, overlay/bar shapes)."""

import altair as alt
import pandas as pd

from vbmitigator.ui import charts


def test_color_map_is_stable_and_unique():
    ids = ["a", "b", "c", "a"]
    cm = charts.run_color_map(ids)
    assert set(cm) == {"a", "b", "c"}
    assert cm == charts.run_color_map(ids)  # deterministic
    assert cm["a"] != cm["b"] != cm["c"]


def test_long_frame_and_smoothing():
    s1 = pd.Series([1.0, 0.5, 0.25], index=[1, 2, 3])
    s2 = pd.Series([2.0, 1.0], index=[1, 2])
    lf = charts.long_frame({"r1": s1, "r2": s2}, x_col="epoch")
    assert set(lf.columns) == {"epoch", "value", "run"}
    assert set(lf["run"]) == {"r1", "r2"}
    assert len(lf) == 5
    # smoothing keeps length but changes values
    lf_s = charts.long_frame({"r1": s1}, smooth=0.8)
    assert len(lf_s) == 3


def test_overlay_chart_has_consistent_colors():
    lf = charts.long_frame(
        {"r1": pd.Series([1.0, 0.5], index=[1, 2]),
         "r2": pd.Series([0.9, 0.4], index=[1, 2])}
    )
    cm = charts.run_color_map(["r1", "r2"])
    ch = charts.overlay_chart(lf, "loss", color_map=cm)
    assert isinstance(ch, alt.Chart)
    d = ch.to_dict()
    rng = d["encoding"]["color"]["scale"]["range"]
    assert rng == [cm["r1"], cm["r2"]]


def test_bar_compare_builds():
    df = pd.DataFrame({"run": ["r1", "r2"], "worst": [45.0, 88.0]})
    ch = charts.bar_compare(df, "worst", title="worst-group %")
    assert isinstance(ch, alt.Chart)


def test_empty_long_frame_safe():
    lf = charts.long_frame({"r1": pd.Series([], dtype=float)})
    assert lf.empty
    # overlay on empty frame still builds a chart object
    assert isinstance(charts.overlay_chart(lf, "x"), alt.Chart)
