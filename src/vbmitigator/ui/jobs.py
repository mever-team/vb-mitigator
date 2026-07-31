"""Background training-job management for the UI.

Launches ``vbm-train`` as a detached subprocess and tracks it via a small JSON
registry plus a per-job log file, so runs survive Streamlit reruns.
"""

import json
import os
import signal
import subprocess
import sys
from datetime import datetime

JOBS_DIRNAME = ".vbm_jobs"


def _jobs_dir(output_dir):
    d = os.path.join(output_dir, JOBS_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return d


def launch_training(cfg_path, output_dir, extra_opts=None):
    """Start a training run for ``cfg_path`` in the background.

    Returns the job dict. The child runs ``python -m vbmitigator.cli --cfg ...``
    with stdout/stderr redirected to a log file under ``<output_dir>/.vbm_jobs``.
    """
    jobs_dir = _jobs_dir(output_dir)
    job_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_path = os.path.join(jobs_dir, f"{job_id}.log")
    cmd = build_command(cfg_path, extra_opts)

    # Streamlit imports numpy (via pandas), which can set MKL_THREADING_LAYER to
    # INTEL — incompatible with torch's libgomp and fatal in the child. Force GNU
    # so training launched from the UI actually starts.
    env = {**os.environ, "MKL_THREADING_LAYER": "GNU"}
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # detach from Streamlit's process group
            env=env,
        )
    job = {
        "job_id": job_id,
        "pid": proc.pid,
        "cfg": cfg_path,
        "cmd": " ".join(cmd),
        "log": log_path,
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(jobs_dir, f"{job_id}.json"), "w") as f:
        json.dump(job, f, indent=2)
    return job


def build_command(cfg_path, extra_opts=None):
    """The `python -m vbmitigator.cli ...` command for a UI-launched run.

    ``--cfg`` is optional. The tqdm terminal progress bar is disabled (the UI has
    its own live bar, and tqdm's carriage-return output floods the log file)
    unless the caller overrides EXPERIMENT.PROGRESS_BAR.
    """
    opts = list(extra_opts or [])
    if "EXPERIMENT.PROGRESS_BAR" not in opts:
        opts += ["EXPERIMENT.PROGRESS_BAR", "False"]
    cmd = [sys.executable, "-m", "vbmitigator.cli"]
    if cfg_path:
        cmd += ["--cfg", cfg_path]
    cmd += opts
    return cmd


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def list_jobs(output_dir):
    """Return known jobs (newest first), each annotated with a live ``status``."""
    jobs_dir = _jobs_dir(output_dir)
    jobs = []
    for name in sorted(os.listdir(jobs_dir), reverse=True):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(jobs_dir, name)) as f:
            job = json.load(f)
        job["status"] = "running" if _pid_alive(job.get("pid", -1)) else "finished"
        jobs.append(job)
    return jobs


def job_run_dir(job):
    """Recover the run directory a job is writing to (printed as VBM_RUN_DIR=)."""
    path = job.get("log")
    if not path or not os.path.exists(path):
        return None
    found = None
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("VBM_RUN_DIR="):
                found = line.split("=", 1)[1].strip()
    return found


def read_job_log(job, tail=200):
    """Return the last ``tail`` lines of a job's log file."""
    path = job.get("log")
    if not path or not os.path.exists(path):
        return ""
    with open(path, errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-tail:])


def stop_job(job):
    """Terminate a running job's process group."""
    pid = job.get("pid")
    if pid and _pid_alive(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
