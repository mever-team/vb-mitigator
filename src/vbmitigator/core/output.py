"""Standardized run-output management.

Every run writes to a predictable tree::

    <OUTPUT.DIR>/<dataset>/<method>/<config>/<run_id>/
        config.yaml            # frozen config for the run
        out<seed>.log          # human-readable training log
        logs<seed>.csv         # per-epoch metrics (for plotting curves)
        metrics.json           # final / best metrics summary
        predictions.csv        # index, target, prediction, <bias attrs>
        best                   # best checkpoint (state dict)
        latest                 # last checkpoint
        train.events/          # tensorboard event files

``run_id`` is ``<timestamp>_seed<seed>`` so repeated runs of the same
configuration never clobber each other.
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

# Files that make up a run's output (used by the UI to preview a run).
RUN_FILES = [
    "config.yaml",
    "metrics.json",
    "predictions.csv",
]


def make_run_id(seed, when=None):
    when = when or datetime.now()
    return f"{when:%Y%m%d-%H%M%S}_seed{seed}"


def build_run_dir(cfg, run_id=None, create=True):
    """Return (and optionally create) the run directory for ``cfg``."""
    run_id = run_id or make_run_id(cfg.EXPERIMENT.SEED)
    run_dir = os.path.join(
        cfg.OUTPUT.DIR,
        str(cfg.DATASET.TYPE),
        str(cfg.MITIGATOR.TYPE),
        str(cfg.EXPERIMENT.CONFIG),
        run_id,
    )
    if create:
        os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_config(cfg, run_dir):
    """Dump the frozen config to ``config.yaml`` inside ``run_dir``."""
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(cfg.dump())


def save_metrics(metrics, run_dir):
    """Write a JSON summary of (numeric) metrics."""
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            clean[k] = v
        elif isinstance(v, (np.floating, np.integer)):
            clean[k] = v.item()
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)


def save_predictions(
    all_data, biases, run_dir, target2name=None, index=None, filename="predictions.csv"
):
    """Write a per-sample CSV with predictions, labels and sensitive attributes.

    ``all_data`` is the dict assembled during evaluation and must contain
    ``"targets"``, ``"predictions"`` and one entry per bias name in ``biases``.
    ``index`` is the true per-sample dataset index (loaders may shuffle, so this
    is what lets consumers map a row back to ``dataset[i]``); when omitted a
    plain 0..N-1 range is used.
    """
    cols = {
        "target": np.asarray(all_data["targets"]).reshape(-1),
        "prediction": np.asarray(all_data["predictions"]).reshape(-1),
    }
    if target2name:
        cols["target_name"] = [target2name.get(int(t), int(t)) for t in cols["target"]]
    for b in biases:
        if b in all_data:
            cols[b] = np.asarray(all_data[b]).reshape(-1)
    df = pd.DataFrame(cols)
    if index is not None:
        df.insert(0, "index", np.asarray(index).reshape(-1))
        df = df.set_index("index")
    else:
        df.index.name = "index"
    path = os.path.join(run_dir, filename)
    df.to_csv(path)
    return path
