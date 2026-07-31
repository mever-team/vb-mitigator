# Overview

VB-Mitigator turns a bias-mitigation experiment into a single config switch.

## Architecture

```
config (YAML)  ->  vbm-train  ->  get_trainer(cfg.MITIGATOR.TYPE)(cfg)
                                        |
                                        +-- BaseTrainer
                                              |-- get_dataset(cfg)   # datasets/builder.py
                                              |-- get_model(cfg)     # models/builder.py
                                              |-- metrics            # metrics/
                                              +-- outputs/<dataset>/<method>/<config>/<run_id>/
```

- **`vbmitigator.config`** — `yacs` defaults; each experiment YAML overrides a subset.
- **`vbmitigator.datasets`** — dataset loaders + a `get_dataset` factory returning a
  standard dict (`num_class`, `biases`, `dataloaders`, `sets`, `target2name`,
  `root`, `ba_groups`).
- **`vbmitigator.models`** — model zoo + `get_model` factory.
- **`vbmitigator.mitigators`** — trainers; a lazy registry maps `MITIGATOR.TYPE`
  to a trainer class.
- **`vbmitigator.metrics`** — accuracy / worst-group / bias-conflicting metrics.
- **`vbmitigator.core`** — utilities and the standardized run-output manager.
- **`vbmitigator.ui`** — the Streamlit dashboard.

## Outputs

Every run is written to `outputs/<dataset>/<method>/<config>/<run_id>/`, with a
frozen `config.yaml`, a `metrics.json` summary, a per-sample `predictions.csv`
(index, target, prediction, and every sensitive attribute — ready for
downstream fairness analysis), per-epoch `logs*.csv` for training curves,
checkpoints, and TensorBoard events.
