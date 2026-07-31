"""Regression tests for review fixes: eval checkpoint loading, scheduler=None,
isolated loggers, and silent metrics."""

import logging
import os

import numpy as np
import torch

from vbmitigator.core.utils import load_checkpoint
from vbmitigator.metrics import get_metric
from vbmitigator.mitigators import get_trainer

wg_ovr = get_metric("wg_ovr")
wg_ovr_std = get_metric("wg_ovr_std")


def _eval_cfg(train_cfg, model_path):
    cfg = train_cfg.clone()
    cfg.defrost()
    cfg.EXPERIMENT.EVAL = True
    cfg.EXPERIMENT.SEED = 2  # different run_id than the training run
    cfg.MODEL.PATH = model_path
    cfg.freeze()
    return cfg


def test_scheduler_none_trains(tiny_cfg, patch_dataset):
    tiny_cfg.SOLVER.SCHEDULER.TYPE = "None"
    tiny_cfg.freeze()
    trainer = get_trainer("erm")(tiny_cfg)
    trainer.train()  # must not raise (constant-LR fallback)
    assert os.path.exists(os.path.join(trainer.run_dir, "best"))


def test_eval_loads_checkpoint(tiny_cfg, patch_dataset):
    tiny_cfg.freeze()
    trainer = get_trainer("erm")(tiny_cfg)
    trainer.train()
    train_run = trainer.run_dir

    # Evaluate pointing MODEL.PATH at the training run directory.
    ev = get_trainer("erm")(_eval_cfg(trainer.cfg, train_run))
    ev.eval()

    # The evaluated model must equal the saved "best" weights, proving eval
    # actually loaded them (rather than scoring a fresh/untrained model).
    best = load_checkpoint(os.path.join(train_run, "best"))["model"]
    for key, value in ev.model.state_dict().items():
        assert torch.equal(value.cpu(), best[key].cpu())

    # Eval writes the standardized artifacts to its own run dir.
    assert os.path.exists(os.path.join(ev.run_dir, "predictions.csv"))
    assert os.path.exists(os.path.join(ev.run_dir, "metrics.json"))


def test_resolve_checkpoint_variants(tiny_cfg, patch_dataset, tmp_path):
    tiny_cfg.freeze()
    trainer = get_trainer("erm")(tiny_cfg)
    trainer.train()

    # direct file
    best_file = os.path.join(trainer.run_dir, "best")
    assert trainer._resolve_checkpoint(best_file) == best_file
    # directory -> best
    assert trainer._resolve_checkpoint(trainer.run_dir) == best_file
    # bare tag relative to run_dir
    assert trainer._resolve_checkpoint("best") == best_file
    # missing -> None
    assert trainer._resolve_checkpoint(str(tmp_path / "nope")) is None
    assert trainer._resolve_checkpoint("") is None


def test_isolated_loggers(tiny_cfg, patch_dataset):
    root_handlers_before = len(logging.getLogger().handlers)
    tiny_cfg.freeze()
    t1 = get_trainer("erm")(tiny_cfg)

    c2 = t1.cfg.clone()
    c2.defrost()
    c2.EXPERIMENT.SEED = 2
    c2.freeze()
    t2 = get_trainer("erm")(c2)

    # Root logger is never touched, and each run has exactly one file handler.
    assert len(logging.getLogger().handlers) == root_handlers_before
    assert t1.logger is not t2.logger
    assert len(t1.logger.handlers) == 1
    assert len(t2.logger.handlers) == 1


def test_progress_json_written(tiny_cfg, patch_dataset):
    import json

    import pandas as pd

    tiny_cfg.freeze()
    trainer = get_trainer("erm")(tiny_cfg)
    trainer.train()

    prog = json.load(open(os.path.join(trainer.run_dir, "progress.json")))
    assert prog["status"] == "finished"
    assert prog["total_epochs"] == 1

    # full per-step loss log (every step, not a capped window)
    steps = pd.read_csv(os.path.join(trainer.run_dir, "train_steps.csv"))
    assert {"step", "epoch", "loss"}.issubset(steps.columns)
    assert len(steps) >= 1


def test_predictions_have_true_index(tiny_cfg, patch_dataset):
    import pandas as pd

    tiny_cfg.freeze()
    trainer = get_trainer("erm")(tiny_cfg)
    trainer.train()
    df = pd.read_csv(os.path.join(trainer.run_dir, "predictions.csv"))
    # index column present and covers every test sample exactly once
    assert "index" in df.columns
    assert sorted(df["index"]) == list(range(16))


def test_metrics_are_silent(capsys):
    data = {
        "targets": np.array([0, 1, 1, 0]),
        "predictions": np.array([0, 1, 0, 0]),
        "unknown": np.array([0, 1, 0, 1]),
    }
    wg_ovr(dict(data))
    wg_ovr_std(dict(data))
    assert capsys.readouterr().out == ""
