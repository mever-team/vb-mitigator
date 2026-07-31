"""Visualizations built from a run's ``predictions.csv``.

Pure pandas + matplotlib (no torch), so they run inside the Streamlit process.
Every function takes the predictions DataFrame and returns either a matplotlib
Figure or a small DataFrame ready for ``st.bar_chart``.
"""

import matplotlib

matplotlib.use("Agg")  # headless / server-side rendering

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_RESERVED = ("index", "target", "prediction", "target_name")


def bias_columns(df):
    """Sensitive-attribute columns in a predictions frame."""
    return [c for c in df.columns if c not in _RESERVED]


def _target_labels(df):
    """Map target id -> display name using the target_name column if present."""
    if "target_name" in df.columns:
        return (
            df[["target", "target_name"]]
            .drop_duplicates()
            .set_index("target")["target_name"]
            .to_dict()
        )
    return {t: str(t) for t in sorted(df["target"].unique())}


def _with_correct(df):
    d = df.copy()
    d["correct"] = (d["target"] == d["prediction"]).astype(int)
    return d


def subgroup_table(df, bias):
    """Per-subgroup (target × bias) accuracy and sample count, worst first."""
    d = _with_correct(df)
    names = _target_labels(df)
    g = d.groupby(["target", bias])
    out = (
        g["correct"].mean().rename("accuracy").reset_index().assign(n=g.size().values)
    )
    out["group"] = out.apply(
        lambda r: f"{names.get(r['target'], r['target'])} · {bias}={int(r[bias])}", axis=1
    )
    return out.sort_values("accuracy").reset_index(drop=True)


def worst_group(df, bias):
    """The lowest-accuracy subgroup as a dict."""
    t = subgroup_table(df, bias)
    if t.empty:
        return None
    row = t.iloc[0]
    return {
        "group": row["group"],
        "target": int(row["target"]),
        "bias": int(row[bias]),
        "accuracy": float(row["accuracy"]),
        "n": int(row["n"]),
    }


def _heatmap(matrix, row_labels, col_labels, title, fmt, cmap, vmin=None, vmax=None,
             highlight=None):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title(title, fontsize=11, pad=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, fmt(val), ha="center", va="center", fontsize=10,
                        color="black")
            if highlight == (i, j):
                ax.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                  edgecolor="#d62728", lw=3)
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def subgroup_accuracy_heatmap(df, bias):
    """Accuracy per (target × bias) subgroup; worst cell outlined in red."""
    d = _with_correct(df)
    names = _target_labels(df)
    targets = sorted(df["target"].unique())
    biases = sorted(df[bias].unique())
    acc = np.full((len(targets), len(biases)), np.nan)
    worst = (None, np.inf)
    for i, t in enumerate(targets):
        for j, b in enumerate(biases):
            sel = (d["target"] == t) & (d[bias] == b)
            if sel.any():
                acc[i, j] = d.loc[sel, "correct"].mean()
                if acc[i, j] < worst[1]:
                    worst = ((i, j), acc[i, j])
    return _heatmap(
        acc,
        [names.get(t, t) for t in targets],
        [f"{bias}={b}" for b in biases],
        "Subgroup accuracy",
        fmt=lambda v: f"{v*100:.1f}%",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        highlight=worst[0],
    )


def subgroup_count_heatmap(df, bias):
    """Sample count per (target × bias) subgroup — reveals the bias structure."""
    names = _target_labels(df)
    targets = sorted(df["target"].unique())
    biases = sorted(df[bias].unique())
    counts = np.zeros((len(targets), len(biases)))
    for i, t in enumerate(targets):
        for j, b in enumerate(biases):
            counts[i, j] = int(((df["target"] == t) & (df[bias] == b)).sum())
    return _heatmap(
        counts,
        [names.get(t, t) for t in targets],
        [f"{bias}={b}" for b in biases],
        "Subgroup sample counts",
        fmt=lambda v: f"{int(v)}",
        cmap="Blues",
    )


def confusion_matrix_fig(df):
    """Confusion matrix of target vs prediction (row-normalized)."""
    names = _target_labels(df)
    classes = sorted(set(df["target"]) | set(df["prediction"]))
    m = np.zeros((len(classes), len(classes)))
    idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(df["target"], df["prediction"]):
        m[idx[t], idx[p]] += 1
    row_sums = m.sum(axis=1, keepdims=True)
    norm = np.divide(m, row_sums, out=np.zeros_like(m), where=row_sums > 0)
    fig = _heatmap(
        norm,
        [names.get(c, c) for c in classes],
        [names.get(c, c) for c in classes],
        "Confusion matrix (row-normalized)",
        fmt=lambda v: f"{v*100:.1f}%",
        cmap="Purples",
        vmin=0.0,
        vmax=1.0,
    )
    fig.axes[0].set_xlabel("predicted")
    fig.axes[0].set_ylabel("true")
    return fig


def per_class_accuracy(df):
    """DataFrame (class name -> accuracy %) for a bar chart."""
    d = _with_correct(df)
    names = _target_labels(df)
    acc = d.groupby("target")["correct"].mean() * 100
    acc.index = [names.get(t, t) for t in acc.index]
    return acc.rename("accuracy (%)").to_frame()
