"""Render sample-image montages for a run (executed as a subprocess).

Loads the run's dataset (needs torch + the data on disk), so it runs out-of-process
to keep the Streamlit UI torch-free. Usage::

    python -m vbmitigator.ui._montage --run <run_dir> --out <png> --mode overview
    python -m vbmitigator.ui._montage --run <run_dir> --out <png> --mode worst

``overview`` shows a few samples per (target × bias) subgroup; ``worst`` shows
misclassified examples from the lowest-accuracy subgroup.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _load_test_dataset(run_dir):
    from vbmitigator.config import CFG
    from vbmitigator.datasets import get_dataset

    cfg = CFG.clone()
    cfg.merge_from_file(os.path.join(run_dir, "config.yaml"))
    cfg.freeze()
    dataset = get_dataset(cfg)
    loaders = dataset["dataloaders"]
    return loaders["test"].dataset, dataset


def _to_image(tensor):
    """A normalized CxHxW tensor -> HxWxC array rescaled to [0,1] for display."""
    arr = tensor.detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr.clip(0, 1)


def _bias_col(df):
    reserved = ("index", "target", "prediction", "target_name")
    cols = [c for c in df.columns if c not in reserved]
    return cols[0] if cols else None


def _name(df, target):
    if "target_name" in df.columns:
        m = df.drop_duplicates("target").set_index("target")["target_name"].to_dict()
        return str(m.get(target, target))
    return str(target)


def _grid(images, titles, suptitle, ncols):
    n = len(images)
    nrows = max(1, (n + ncols - 1) // ncols)
    # constrained_layout reserves room for each per-image title, so the next
    # row's titles never overlap the row above; extra h_pad adds breathing room.
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 1.9, nrows * 2.3),
        layout="constrained",
    )
    fig.set_constrained_layout_pads(hspace=0.12, h_pad=0.12)
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        if title:
            ax.set_title(title, fontsize=8)
    fig.suptitle(suptitle, fontsize=12)
    return fig


def build_overview(run_dir, out, per_group=4, seed=0):
    test_set, _ = _load_test_dataset(run_dir)
    df = pd.read_csv(os.path.join(run_dir, "predictions.csv"))
    bias = _bias_col(df)
    rng = np.random.default_rng(seed)
    images, titles = [], []
    groups = sorted(df.groupby(["target", bias]).groups.keys())
    ncols = per_group
    for (t, b) in groups:
        sel = df[(df["target"] == t) & (df[bias] == b)]
        idxs = rng.choice(sel["index"].values, size=min(per_group, len(sel)), replace=False)
        for k, i in enumerate(idxs):
            images.append(_to_image(test_set[int(i)]["inputs"]))
            titles.append(f"{_name(df, t)} · {bias}={b}" if k == 0 else "")
        for _ in range(per_group - len(idxs)):
            images.append(np.ones((8, 8, 3)))
            titles.append("")
    fig = _grid(images, titles, "Samples per subgroup (target × bias)", ncols)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def build_worst(run_dir, out, n=8, seed=0):
    test_set, _ = _load_test_dataset(run_dir)
    df = pd.read_csv(os.path.join(run_dir, "predictions.csv"))
    bias = _bias_col(df)
    df["correct"] = (df["target"] == df["prediction"]).astype(int)
    acc = df.groupby(["target", bias])["correct"].mean()
    (t, b) = acc.idxmin()
    group_acc = acc.min()
    wrong = df[(df["target"] == t) & (df[bias] == b) & (df["correct"] == 0)]
    rng = np.random.default_rng(seed)
    idxs = rng.choice(wrong["index"].values, size=min(n, len(wrong)), replace=False) if len(wrong) else []
    images, titles = [], []
    for i in idxs:
        row = df[df["index"] == i].iloc[0]
        images.append(_to_image(test_set[int(i)]["inputs"]))
        titles.append(f"pred {_name(df, row['prediction'])}")
    suptitle = (
        f"Worst group: {_name(df, t)} · {bias}={b}  "
        f"(acc {group_acc*100:.1f}%) — misclassified examples"
    )
    fig = _grid(images or [np.ones((8, 8, 3))], titles or [""], suptitle, ncols=4)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render a run's sample montage.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["overview", "worst"], default="overview")
    args = parser.parse_args(argv)
    if args.mode == "overview":
        build_overview(args.run, args.out)
    else:
        build_worst(args.run, args.out)


if __name__ == "__main__":
    main()
