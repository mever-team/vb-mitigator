"""End-to-end smoke test: a 1-epoch run on synthetic data produces the
standardized output artifacts."""

import os

import pytest

from vbmitigator.mitigators import get_trainer


def _run(cfg, method):
    cfg.MITIGATOR.TYPE = method
    cfg.freeze()
    trainer = get_trainer(method)(cfg)
    trainer.train()
    return trainer.run_dir


@pytest.mark.parametrize("method", ["erm", "sd"])
def test_training_produces_outputs(tiny_cfg, patch_dataset, method):
    run_dir = _run(tiny_cfg, method)

    # Standardized output tree exists at <dataset>/<method>/<config>/<run_id>.
    assert os.path.basename(os.path.dirname(os.path.dirname(run_dir))) == method
    for fname in ("config.yaml", "metrics.json", "predictions.csv", "best"):
        assert os.path.exists(os.path.join(run_dir, fname)), f"missing {fname}"


def test_predictions_csv_contents(tiny_cfg, patch_dataset):
    import pandas as pd

    run_dir = _run(tiny_cfg, "erm")
    df = pd.read_csv(os.path.join(run_dir, "predictions.csv"))
    for col in ("target", "prediction", "unknown"):
        assert col in df.columns
    assert len(df) == 16  # size of the synthetic test split


def test_metrics_json_has_accuracy(tiny_cfg, patch_dataset):
    import json

    run_dir = _run(tiny_cfg, "erm")
    with open(os.path.join(run_dir, "metrics.json")) as f:
        m = json.load(f)
    assert "test_accuracy" in m
