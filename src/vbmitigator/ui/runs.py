"""Filesystem helpers for the UI: discover configs and runs, load summaries.

These are deliberately free of any Streamlit dependency so they can be unit
tested and reused.
"""

import glob
import json
import os
import subprocess
import sys

import pandas as pd
import yaml


def registered_components(python_exe=None):
    """Return the registered {datasets, methods, models, metrics} names.

    Fetched from a short-lived subprocess so the (torch-free) UI process never
    has to import torch just to display the lists. Returns ``{}`` on failure.
    """
    exe = python_exe or sys.executable
    try:
        result = subprocess.run(
            [exe, "-c", "from vbmitigator.cli import list_main; list_main(['--json'])"],
            capture_output=True,
            text=True,
            timeout=120,
            env=_torch_child_env(),
        )
        return json.loads(result.stdout)
    except Exception:
        return {}


def _torch_child_env():
    """Env for a child that imports torch.

    Streamlit imports numpy (via pandas), which can leave ``MKL_THREADING_LAYER``
    set to ``INTEL`` — incompatible with torch's libgomp and fatal in the child.
    Forcing ``GNU`` avoids the clash.
    """
    return {**os.environ, "MKL_THREADING_LAYER": "GNU"}


def render_montage(run_dir, mode="overview", python_exe=None):
    """Render a sample-image montage for a run (best-effort).

    Runs the torch-dependent renderer out-of-process; needs ``predictions.csv``,
    ``config.yaml`` and the dataset's data on disk. Returns the PNG path, a
    cached one if already rendered, or ``None`` on failure.
    """
    # The "v2" suffix invalidates montages cached before the layout fix.
    out = os.path.join(run_dir, f"_montage_{mode}_v2.png")
    if os.path.exists(out):
        return out
    if not os.path.exists(os.path.join(run_dir, "predictions.csv")):
        return None
    exe = python_exe or sys.executable
    try:
        subprocess.run(
            [exe, "-m", "vbmitigator.ui._montage", "--run", run_dir, "--out", out,
             "--mode", mode],
            capture_output=True,
            text=True,
            timeout=600,
            env=_torch_child_env(),
        )
        return out if os.path.exists(out) else None
    except Exception:
        return None


def load_predictions_full(run_dir):
    """Load the full ``predictions.csv`` (or None)."""
    return load_predictions(run_dir, nrows=None)


# --------------------------------------------------------------------------- #
# Config discovery
# --------------------------------------------------------------------------- #
def discover_configs(configs_dir):
    """Return a sorted list of experiment configs found under ``configs_dir``.

    Expects the layout ``configs/<dataset>/<method>/<name>.yaml``.
    Each entry is a dict: ``{dataset, method, name, path}``.
    """
    out = []
    for path in glob.glob(os.path.join(configs_dir, "*", "*", "*.yaml")):
        rel = os.path.relpath(path, configs_dir)
        parts = rel.split(os.sep)
        if len(parts) != 3:
            continue
        dataset, method, fname = parts
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "name": os.path.splitext(fname)[0],
                "path": path,
            }
        )
    return sorted(out, key=lambda d: (d["dataset"], d["method"], d["name"]))


# --------------------------------------------------------------------------- #
# Run discovery
# --------------------------------------------------------------------------- #
def discover_runs(output_dir):
    """Return runs under ``output_dir`` = ``<dataset>/<method>/<config>/<run_id>/``.

    Each entry: ``{dataset, method, config, run_id, path, mtime}``, newest first.
    """
    runs = []
    pattern = os.path.join(output_dir, "*", "*", "*", "*")
    for path in glob.glob(pattern):
        if not os.path.isdir(path):
            continue
        rel = os.path.relpath(path, output_dir).split(os.sep)
        if len(rel) != 4:
            continue
        dataset, method, config, run_id = rel
        runs.append(
            {
                "dataset": dataset,
                "method": method,
                "config": config,
                "run_id": run_id,
                "path": path,
                "mtime": os.path.getmtime(path),
            }
        )
    return sorted(runs, key=lambda r: r["mtime"], reverse=True)


# --------------------------------------------------------------------------- #
# Run inspection
# --------------------------------------------------------------------------- #
def list_run_files(run_dir):
    """List files inside a run directory with their sizes (bytes)."""
    files = []
    for name in sorted(os.listdir(run_dir)):
        full = os.path.join(run_dir, name)
        if os.path.isfile(full):
            files.append({"name": name, "size": os.path.getsize(full)})
    return files


def load_metrics(run_dir):
    """Load ``metrics.json`` if present, else ``{}``."""
    path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def read_progress(run_dir):
    """Load a run's live ``progress.json`` (written during training), else {}."""
    if not run_dir:
        return {}
    path = os.path.join(run_dir, "progress.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def load_step_losses(run_dir):
    """Load the full per-step training-loss log (train_steps.csv), or None."""
    if not run_dir:
        return None
    path = os.path.join(run_dir, "train_steps.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception:
        return None


def load_config(run_dir):
    """Load the frozen ``config.yaml`` as a dict (or ``{}``)."""
    path = os.path.join(run_dir, "config.yaml")
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def load_curves(run_dir):
    """Load the per-epoch ``logs*.csv`` as a DataFrame indexed by epoch (or None)."""
    matches = sorted(glob.glob(os.path.join(run_dir, "logs*.csv")))
    if not matches:
        return None
    try:
        df = pd.read_csv(matches[0])
    except Exception:
        return None
    if "epoch" in df.columns:
        df = df.set_index("epoch")
    # keep only numeric columns for plotting
    return df.select_dtypes(include="number")


def load_predictions(run_dir, nrows=200):
    """Load a preview of ``predictions.csv`` (or None)."""
    path = os.path.join(run_dir, "predictions.csv")
    if os.path.exists(path):
        try:
            return pd.read_csv(path, nrows=nrows)
        except Exception:
            return None
    return None


def summarize_run(run_dir):
    """One-stop summary for the UI."""
    return {
        "files": list_run_files(run_dir),
        "metrics": load_metrics(run_dir),
        "config": load_config(run_dir),
        "curves": load_curves(run_dir),
        "predictions": load_predictions(run_dir),
    }
