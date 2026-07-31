"""Altair chart builders for the Workspace.

Pure altair + pandas (no Streamlit, no torch) so they are unit-testable. Every
run is assigned a stable colour that is reused across all panels, mirroring the
Weights & Biases feel.
"""

import altair as alt
import pandas as pd

# A 20-colour qualitative palette (distinct, legible on a light ground).
PALETTE = [
    "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#eeca3b", "#b279a2",
    "#ff9da6", "#9d755d", "#bab0ac", "#1f77b4", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#ff7f0e",
]


def run_color_map(run_ids):
    """Deterministic {run_id: hex} so colours are consistent across panels."""
    seen = list(dict.fromkeys(run_ids))  # unique, order-preserving
    return {rid: PALETTE[i % len(PALETTE)] for i, rid in enumerate(seen)}


def ema(series, factor):
    """Exponential-moving-average smoothing; ``factor`` in [0, 1), 0 = off."""
    if not factor or factor <= 0:
        return series
    return series.ewm(alpha=1 - min(float(factor), 0.99)).mean()


def long_frame(series_by_run, x_col="epoch", smooth=0.0):
    """Turn ``{run: Series-indexed-by-x}`` into a long df ``[x_col, value, run]``."""
    frames = []
    for run, s in series_by_run.items():
        if s is None or len(s) == 0:
            continue
        s2 = ema(s, smooth)
        frames.append(
            pd.DataFrame({x_col: list(s2.index), "value": list(s2.values), "run": run})
        )
    if not frames:
        return pd.DataFrame(columns=[x_col, "value", "run"])
    return pd.concat(frames, ignore_index=True)


def _color_encoding(runs, color_map, legend=None):
    domain = list(color_map) if color_map else sorted(set(runs))
    rng = [color_map[r] for r in domain] if color_map else None
    scale = alt.Scale(domain=domain, range=rng) if rng else alt.Undefined
    return alt.Color("run:N", scale=scale, legend=legend)


def overlay_chart(long_df, title, x_col="epoch", color_map=None, height=300):
    """A multi-run line chart for one metric (consistent colours + legend)."""
    return (
        alt.Chart(long_df)
        .mark_line(interpolate="monotone")
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_col),
            y=alt.Y("value:Q", title=None),
            color=_color_encoding(
                long_df["run"] if len(long_df) else [],
                color_map,
                legend=alt.Legend(title=None, orient="bottom", labelLimit=220),
            ),
            tooltip=[
                "run:N",
                alt.Tooltip(f"{x_col}:Q"),
                alt.Tooltip("value:Q", format=".4f"),
            ],
        )
        .properties(title=title, height=height)
        .interactive()
    )


def bar_compare(df, value_col, title=None, color_map=None):
    """A horizontal bar per run (e.g. worst-group accuracy across runs)."""
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{value_col}:Q", title=title or value_col),
            y=alt.Y("run:N", sort="-x", title=None),
            color=_color_encoding(df["run"] if len(df) else [], color_map, legend=None),
            tooltip=["run:N", alt.Tooltip(f"{value_col}:Q", format=".3f")],
        )
        .properties(height=max(120, 34 * max(1, len(df))))
    )
