"""Output/run-directory manager."""

import json
import os

import numpy as np

from vbmitigator.config import CFG
from vbmitigator.core.output import (
    build_run_dir,
    save_config,
    save_metrics,
    save_predictions,
)


def _cfg(tmp_path):
    cfg = CFG.clone()
    cfg.OUTPUT.DIR = str(tmp_path / "outputs")
    cfg.DATASET.TYPE = "utkface"
    cfg.MITIGATOR.TYPE = "badd"
    cfg.EXPERIMENT.CONFIG = "race"
    cfg.EXPERIMENT.SEED = 3
    return cfg


def test_run_dir_structure(tmp_path):
    cfg = _cfg(tmp_path)
    run_dir = build_run_dir(cfg, run_id="rid", create=True)
    rel = os.path.relpath(run_dir, cfg.OUTPUT.DIR)
    assert rel == os.path.join("utkface", "badd", "race", "rid")
    assert os.path.isdir(run_dir)


def test_save_config_and_metrics(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.freeze()
    run_dir = build_run_dir(cfg, run_id="rid")
    save_config(cfg, run_dir)
    save_metrics({"test_accuracy": 91.5, "note": "ok", "arr": np.float64(1.0)}, run_dir)

    assert os.path.exists(os.path.join(run_dir, "config.yaml"))
    with open(os.path.join(run_dir, "metrics.json")) as f:
        m = json.load(f)
    assert m["test_accuracy"] == 91.5
    assert m["arr"] == 1.0


def test_save_predictions_columns(tmp_path):
    import pandas as pd

    cfg = _cfg(tmp_path)
    run_dir = build_run_dir(cfg, run_id="rid")
    all_data = {
        "targets": np.array([0, 1, 1, 0]),
        "predictions": np.array([0, 1, 0, 0]),
        "race": np.array([1, 0, 1, 0]),
    }
    path = save_predictions(
        all_data, ["race"], run_dir, target2name={0: "male", 1: "female"}
    )
    df = pd.read_csv(path)
    assert set(["index", "target", "prediction", "target_name", "race"]).issubset(
        df.columns
    )
    assert len(df) == 4
