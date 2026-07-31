# VB-Mitigator

[![Docs](https://img.shields.io/badge/docs-readthedocs.io-0e8f86)](https://vb-mitigator.readthedocs.io/)
[![CI](https://github.com/gsarridis/vb-mitigator/actions/workflows/ci.yml/badge.svg)](https://github.com/gsarridis/vb-mitigator/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Visual Bias Mitigator** — an open-source framework to implement, run, and
evaluate methods that mitigate bias in computer-vision models.

It provides a collection of established bias-mitigation methods, standard bias
benchmarks, a standardized experiment/output layout, and a Streamlit dashboard
for launching and inspecting runs.

> 📖 **Full documentation (guides for all users):**
> **[vb-mitigator.readthedocs.io](https://vb-mitigator.readthedocs.io/)** —
> what visual bias is, a getting-started walkthrough, using the app, reading the
> results, and the dataset/method catalog.

---

## Highlights

- 🚀 **16 mitigation methods** behind a single config switch (`MITIGATOR.TYPE`).
- 🗂️ **11 vision bias benchmarks** behind a single switch (`DATASET.TYPE`).
- 🧩 **Extensible by design** — add a method or dataset by dropping in one file
  and one registry entry (see [Extending VB-Mitigator](#extending-vb-mitigator)).
- 📁 **Standardized outputs** — every run lands in
  `outputs/<dataset>/<method>/<config>/<run_id>/` with a frozen `config.yaml`,
  `metrics.json`, per-sample `predictions.csv`, checkpoints, and TensorBoard curves.
- 📊 **UI** — `vbm-ui` launches a dashboard to start runs and browse/summarize results.
- ✅ **Tested & CI'd** — a fast CPU test suite and GitHub Actions.

## Installation

```bash
# Python >= 3.10; a virtual environment is recommended
conda create -n vb-mitigator python=3.11 && conda activate vb-mitigator

# core install (editable)
pip install -e .

# optional extras
pip install -e ".[ui]"       # Streamlit dashboard
pip install -e ".[mavias]"   # tag-based methods (MAVias / erm_tags): RAM, ollama, transformers
pip install -e ".[dev]"      # tests + linting
```

## Quickstart

Train a method on a dataset by pointing at a YAML config:

```bash
vbm-train --cfg configs/utkface/badd/race.yaml
```

Everything for that run is written to
`outputs/utkface/badd/race/<timestamp>_seed1/`:

```
outputs/utkface/badd/race/20260728-171200_seed1/
├── config.yaml        # exact, frozen configuration
├── metrics.json       # final/best metrics
├── predictions.csv    # per-sample: index, target, prediction, <bias attrs>
├── logs1.csv          # per-epoch metrics (training curves)
├── out1.log           # human-readable log
├── best / latest      # checkpoints
└── train.events/      # TensorBoard events
```

Evaluate a trained checkpoint:

```bash
vbm-eval --cfg configs/utkface/badd/race.yaml --model best
```

Override any config value inline:

```bash
vbm-train --cfg configs/utkface/erm/race.yaml SOLVER.LR 0.01 SOLVER.EPOCHS 50
```

## The UI

```bash
vbm-ui            # or:  python -m vbmitigator.ui
```

The dashboard has two tabs:

- **Launch** — pick a dataset → method → configuration, tweak the seed/overrides,
  and start a training run in the background. Live job logs are shown inline.
- **Runs** — browse every run in the output tree, preview its files, and read a
  ready-made summary (final metrics + training curves + a predictions preview).

### Running the UI on a remote server (VS Code Remote / SSH)

`vbm-ui` runs headless by default and disables Streamlit's browser auto-open,
XSRF/CORS, and file-watcher — the settings that otherwise make Streamlit show a
**blank page** through a forwarded port. Just run it and open the forwarded port
locally:

```bash
vbm-ui --server.port 8501
```

Then open the port in your **local** browser — click the `http://localhost:8501`
link Streamlit prints, or use VS Code's **Ports** panel (it auto-forwards the
port; if not, add it manually). Any `--server.*` flag you pass overrides the
defaults.

## Methods & datasets

**Methods** (`MITIGATOR.TYPE`): `erm`, `flac`, `flacb`, `badd`, `mavias`,
`maviasb`, `groupdro`, `debian`, `di`, `sd`, `lff`, `bb`, `end`, `jtt`,
`softcon`, `erm_tags`.

**Datasets** (`DATASET.TYPE`): `biased_mnist`, `fb_biased_mnist`, `utkface`,
`waterbirds`, `celeba`, `urbancars`, `imagenet9`, `imagenet9m`, `cifar10`,
`cifar100`, `stanford_dogs`.

> Datasets are **not** bundled. Set each dataset's `ROOT` in its config (or via a
> CLI override) to point at your local copy.

## Extending VB-Mitigator

**Add a method.** Create `src/vbmitigator/mitigators/my_method.py` with a class
that subclasses `BaseTrainer` (override `_train_iter` and any setup hooks), then
register it in `src/vbmitigator/mitigators/__init__.py`:

```python
_REGISTRY = {
    ...
    "my_method": ("my_method", "MyMethodTrainer"),
}
```

Add defaults for its hyper-parameters in `src/vbmitigator/config/defaults.py`
(`CFG.MITIGATOR.MY_METHOD = CN(); ...`) and you can run it via any config.

**Add a dataset.** Drop **one self-contained module** in
`src/vbmitigator/datasets/` with a builder decorated with `@register_dataset`.
It's auto-discovered — no central file to edit:

```python
# src/vbmitigator/datasets/my_dataset.py
from vbmitigator.datasets.registry import register_dataset

@register_dataset("my_dataset")
def build_my_dataset(cfg):
    train_loader, train_set = ...  # your loaders
    return {
        "num_class": 10,
        "biases": ["my_bias"],              # one key per sensitive attribute
        "dataloaders": {"train": ..., "val": ..., "test": ...},
        "sets": {"train": train_set},       # only "train" is required
        "root": cfg.DATASET.MY_DATASET.ROOT,
        "target2name": {0: "cat", 1: "dog"},
        "ba_groups": [(0, 0), (1, 1)],      # optional (group-fairness metrics)
        "num_groups": 20,                   # optional (groupdro / di)
    }
```

Add its config block in `config/defaults.py` and it's runnable via
`DATASET.TYPE: my_dataset`. `get_dataset` validates the returned dict against the
contract and gives a clear error if a key is missing. Batches yielded by your
loaders must be dicts with `inputs`, `targets`, `index`, and one key per bias
name.

**Add a model.** Same pattern — one auto-discovered decorator on a builder
`(num_classes, pretrained) -> nn.Module` anywhere under `models/`:

```python
from vbmitigator.models.registry import register_model

@register_model("my_net")
def build_my_net(num_classes, pretrained=False):
    return MyNet(num_classes, pretrained)
```

Select it with `MODEL.TYPE: my_net`. To support BAdd/MAVias, also implement
`badd_forward` / `mavias_forward` on the module (see `models/resnet.py`).

**Add a metric.** One decorator on a function `data_dict -> {key: value}`:

```python
from vbmitigator.metrics.registry import register_metric

@register_metric("my_metric", performance="score", best="high")
def my_metric(data):
    return {"score": ...}
```

`performance` names the output key that drives best-checkpoint selection; `best`
is `"high"` or `"low"`. Select it with `METRIC: my_metric`.

Datasets, methods, models, and metrics all use the same registry pattern, so
extending any of them is the same one-file, no-central-edit workflow.

## Reproducing published results

The ad-hoc launcher scripts (SLURM/cluster jobs, sweeps) are **not** part of the
user-facing API. Reproducibility scripts live under `scripts/` and are documented
in [`scripts/README.md`](scripts/README.md).

## Development

```bash
pip install -e ".[dev,ui]"
python -m pytest        # fast, CPU-only
ruff check src tests
```

> Use `python -m pytest`, not a bare `pytest`. If you have conda's `base`
> environment active alongside your venv, a bare `pytest` may resolve to conda's
> binary (a different interpreter without this package installed) and fail with
> `ModuleNotFoundError: No module named 'vbmitigator'`. `python -m pytest`
> always uses the active environment's interpreter. The test suite also puts
> `src/` on `sys.path`, so it runs as long as the dependencies are installed.

## Citations

```bibtex
@article{sarridis2024flac,
  title={FLAC: Fairness-aware representation learning by suppressing attribute-class associations},
  author={Sarridis, Ioannis and Koutlis, Christos and Papadopoulos, Symeon and Diou, Christos},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, year={2024}, publisher={IEEE}
}
@article{sarridis2024badd,
  title={BAdd: Bias Mitigation through Bias Addition},
  author={Sarridis, Ioannis and Koutlis, Christos and Papadopoulos, Symeon and Diou, Christos},
  journal={arXiv preprint arXiv:2408.11439}, year={2024}
}
@article{sarridis2024mavias,
  title={MAVias: Mitigate any Visual Bias},
  author={Sarridis, Ioannis and Koutlis, Christos and Papadopoulos, Symeon and Diou, Christos},
  journal={arXiv preprint arXiv:2412.06632}, year={2024}
}
```

**Maintainer:** Ioannis Sarridis (gsarridis@iti.gr)

## Acknowledgments

Supported by the EU Horizon Europe projects **MAMMOth** (GA 101070285) and
**ELIAS** (GA 101120237).

## License

MIT — see [LICENSE](LICENSE).
