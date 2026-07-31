"""Command-line entry points for training, evaluation and discovery.

Installed as console scripts::

    vbm-train --cfg configs/utkface/badd/race.yaml
    vbm-eval  --cfg configs/utkface/badd/race.yaml --model best
    vbm-list                 # what datasets / methods / models / metrics exist
"""

import argparse
import json
import os

import torch.backends.cudnn as cudnn

from vbmitigator.config import CFG
from vbmitigator.datasets import available_datasets
from vbmitigator.metrics import available_metrics
from vbmitigator.mitigators import AVAILABLE_METHODS, get_trainer
from vbmitigator.models import available_models

cudnn.benchmark = True


def registered_components():
    """Return the registered {datasets, methods, models, metrics} names."""
    return {
        "datasets": list(available_datasets()),
        "methods": list(AVAILABLE_METHODS),
        "models": list(available_models()),
        "metrics": list(available_metrics()),
    }


def _build_cfg(args):
    cfg = CFG.clone()
    if args.cfg:
        cfg.merge_from_file(args.cfg)
        # Use the YAML stem as the <config> level of the output tree, unless the
        # config already set EXPERIMENT.CONFIG explicitly.
        if cfg.EXPERIMENT.CONFIG == "default":
            cfg.EXPERIMENT.CONFIG = os.path.splitext(os.path.basename(args.cfg))[0]
    if args.opts:
        cfg.merge_from_list(args.opts)
    if args.seed is not None:
        cfg.EXPERIMENT.SEED = args.seed
    if args.epoch_steps is not None:
        cfg.EXPERIMENT.EPOCH_STEPS = args.epoch_steps
    return cfg


def _validate_cfg(cfg, parser):
    """Fail early with a clear message if a config selects an unregistered part."""
    checks = [
        ("DATASET.TYPE", cfg.DATASET.TYPE, available_datasets()),
        ("MITIGATOR.TYPE", cfg.MITIGATOR.TYPE, AVAILABLE_METHODS),
        ("MODEL.TYPE", cfg.MODEL.TYPE, available_models()),
        ("METRIC", cfg.METRIC, available_metrics()),
    ]
    for field, value, allowed in checks:
        if value not in allowed:
            parser.error(
                f"Unknown {field} '{value}'. Available: {', '.join(sorted(allowed))}"
            )


def _common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--cfg",
        type=str,
        default="",
        help="path to a YAML config (optional; you can instead pass overrides "
        "like DATASET.TYPE utkface MITIGATOR.TYPE erm)",
    )
    parser.add_argument("--seed", type=int, default=None, help="override EXPERIMENT.SEED")
    parser.add_argument(
        "--epoch_steps", type=int, default=None, help="limit epochs per invocation"
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="override any config value, e.g. SOLVER.LR 0.01",
    )
    return parser


def train_main(argv=None):
    parser = _common_parser("Train a bias-mitigation method.")
    args = parser.parse_args(argv)
    cfg = _build_cfg(args)
    cfg.EXPERIMENT.EVAL = False
    cfg.freeze()
    _validate_cfg(cfg, parser)
    trainer = get_trainer(cfg.MITIGATOR.TYPE)(cfg)
    trainer.train()


def eval_main(argv=None):
    parser = _common_parser("Evaluate a trained model.")
    parser.add_argument(
        "--model", type=str, default=None, help="checkpoint tag/path to load (optional)"
    )
    args = parser.parse_args(argv)
    cfg = _build_cfg(args)
    cfg.EXPERIMENT.EVAL = True
    if args.model is not None:
        cfg.MODEL.PATH = args.model
    cfg.freeze()
    _validate_cfg(cfg, parser)
    trainer = get_trainer(cfg.MITIGATOR.TYPE)(cfg)
    trainer.eval()


def list_main(argv=None):
    parser = argparse.ArgumentParser(
        description="List registered datasets, methods, models and metrics."
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)
    components = registered_components()
    if args.json:
        print(json.dumps(components, indent=2))
        return
    for kind, names in components.items():
        print(f"{kind} ({len(names)}):")
        print("  " + ", ".join(names))


if __name__ == "__main__":  # pragma: no cover
    train_main()
