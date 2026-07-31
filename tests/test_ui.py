"""UI filesystem helpers (config/run discovery, summaries, jobs)."""

import json
import os

from vbmitigator.ui import jobs, runs


def _make_config(configs_dir, dataset, method, name):
    d = os.path.join(configs_dir, dataset, method)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{name}.yaml")
    with open(path, "w") as f:
        f.write(f"DATASET:\n  TYPE: '{dataset}'\nMITIGATOR:\n  TYPE: '{method}'\n")
    return path


def _make_run(output_dir, dataset, method, config, run_id):
    d = os.path.join(output_dir, dataset, method, config, run_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.yaml"), "w") as f:
        f.write("DATASET:\n  TYPE: 'x'\n")
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump({"test_accuracy": 88.0, "best_performance": 88.0}, f)
    with open(os.path.join(d, "predictions.csv"), "w") as f:
        f.write("index,target,prediction,unknown\n0,1,1,0\n1,0,1,1\n")
    with open(os.path.join(d, "logs1.csv"), "w") as f:
        f.write("epoch,lr,train_cls_loss,test_accuracy\n1,0.1,0.9,50.0\n2,0.1,0.5,88.0\n")
    return d


def test_discover_configs(tmp_path):
    cdir = str(tmp_path / "configs")
    _make_config(cdir, "utkface", "badd", "race")
    _make_config(cdir, "celeba", "erm", "blonde")
    found = runs.discover_configs(cdir)
    assert len(found) == 2
    assert {c["dataset"] for c in found} == {"utkface", "celeba"}


def test_discover_and_summarize_runs(tmp_path):
    odir = str(tmp_path / "outputs")
    run_dir = _make_run(odir, "utkface", "badd", "race", "20260101-000000_seed1")
    found = runs.discover_runs(odir)
    assert len(found) == 1
    assert found[0]["method"] == "badd"

    summary = runs.summarize_run(run_dir)
    assert summary["metrics"]["test_accuracy"] == 88.0
    assert summary["curves"] is not None
    assert "test_accuracy" in summary["curves"].columns
    assert summary["predictions"] is not None
    assert {f["name"] for f in summary["files"]} >= {
        "config.yaml",
        "metrics.json",
        "predictions.csv",
    }


def test_jobs_empty(tmp_path):
    assert jobs.list_jobs(str(tmp_path / "outputs")) == []


def test_build_command_cfg_optional():
    # config-based launch includes --cfg
    cmd = jobs.build_command("configs/utkface/erm/race.yaml", ["EXPERIMENT.SEED", "1"])
    assert "--cfg" in cmd and "configs/utkface/erm/race.yaml" in cmd
    # config-less launch (registry dropdowns) omits --cfg, keeps overrides
    cmd2 = jobs.build_command("", ["DATASET.TYPE", "utkface", "MITIGATOR.TYPE", "erm"])
    assert "--cfg" not in cmd2
    assert "DATASET.TYPE" in cmd2 and "utkface" in cmd2
    # tqdm terminal bar disabled by default (UI has its own live bar)
    assert cmd2[cmd2.index("EXPERIMENT.PROGRESS_BAR") + 1] == "False"
    # ...but caller can override it
    cmd3 = jobs.build_command("", ["EXPERIMENT.PROGRESS_BAR", "True"])
    assert cmd3.count("EXPERIMENT.PROGRESS_BAR") == 1


def test_job_run_dir_and_progress(tmp_path):
    import json

    # a run dir with a live progress.json
    run_dir = tmp_path / "outputs" / "utkface" / "erm" / "race" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "progress.json").write_text(
        json.dumps({"status": "training", "epoch": 2, "total_epochs": 5,
                    "step": 10, "total_steps": 40, "recent_losses": [0.9, 0.5]})
    )
    # a job whose log announces that run dir
    log = tmp_path / "job.log"
    log.write_text(f"some output\nVBM_RUN_DIR={run_dir}\nmore output\n")
    job = {"log": str(log)}

    assert jobs.job_run_dir(job) == str(run_dir)
    prog = runs.read_progress(str(run_dir))
    assert prog["epoch"] == 2 and prog["total_steps"] == 40
    assert runs.read_progress(str(tmp_path / "nope")) == {}


def test_registered_components_subprocess():
    # Runs under pytest with numpy already imported — the exact condition that
    # triggers the MKL_THREADING_LAYER clash the helper guards against.
    components = runs.registered_components()
    assert set(components) == {"datasets", "methods", "models", "metrics"}
    assert len(components["datasets"]) == 11
    assert "erm" in components["methods"]


def test_app_module_parses():
    import ast

    import vbmitigator.ui as ui_pkg

    app_path = os.path.join(os.path.dirname(ui_pkg.__file__), "app.py")
    ast.parse(open(app_path).read())


def test_app_renders(tmp_path):
    """Both views of the dashboard run headlessly without raising."""
    import pytest

    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

    import vbmitigator.ui as ui_pkg

    app_path = os.path.join(os.path.dirname(ui_pkg.__file__), "app.py")
    at = AppTest.from_file(app_path, default_timeout=120).run()
    assert not at.exception
    assert at.session_state["view"] == "Launch"  # programmatic nav wired up

    # Point outputs at an empty dir and open the Workspace (fast + deterministic).
    for ti in at.sidebar.text_input:
        if ti.label and "Output" in ti.label:
            ti.set_value(str(tmp_path))
    at.session_state["view"] = "Workspace"
    at.run()
    assert not at.exception
