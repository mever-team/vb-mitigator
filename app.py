import os
import re
import subprocess
import threading
import uuid
import atexit

import yaml
from flask import Flask, jsonify, request, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "my_datasets")
MITIGATORS_DIR = os.path.join(BASE_DIR, "mitigators")

app = Flask(__name__)

# in-memory registry of running/finished processes
RUNS = {}
RUNS_LOCK = threading.Lock()

FOLDERS = {
    "data": DATA_DIR,
    "mitigator": MITIGATORS_DIR,
}


def safe_name(name):
    """Reject anything that looks like a path traversal attempt."""
    if not name or "/" in name or "\\" in name or ".." in name:
        return None
    return name


def list_py_files(folder):
    if not os.path.isdir(folder):
        return []
    out = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(".py") and f != "__init__.py":
            out.append(f[:-3])
    return out


def list_yaml_files(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.endswith((".yaml", ".yml")))


# ---------- selection lists ----------

@app.route("/api/list/<kind>")
def api_list(kind):
    if kind not in FOLDERS:
        return jsonify({"error": "unknown kind"}), 400
    return jsonify({"items": list_py_files(FOLDERS[kind])})


# ---------- py file editing ----------

@app.route("/api/file/<kind>/<name>", methods=["GET"])
def get_file(kind, name):
    if kind not in FOLDERS:
        return jsonify({"error": "unknown kind"}), 400
    name = safe_name(name)
    if name is None:
        return jsonify({"error": "invalid name"}), 400
    path = os.path.join(FOLDERS[kind], name + ".py")
    if not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    with open(path, "r") as fh:
        content = fh.read()
    return jsonify({"content": content})


@app.route("/api/file/<kind>/<name>", methods=["POST"])
def save_file(kind, name):
    if kind not in FOLDERS:
        return jsonify({"error": "unknown kind"}), 400
    name = safe_name(name)
    if name is None:
        return jsonify({"error": "invalid name"}), 400
    content = request.json.get("content", "")
    path = os.path.join(FOLDERS[kind], name + ".py")
    if not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    with open(path, "w") as fh:
        fh.write(content)
    return jsonify({"ok": True})


# ---------- config files ----------

@app.route("/api/configs")
def api_configs():
    mitigator = request.args.get("mitigator", "")
    data = request.args.get("data", "")
    CONFIGS_DIR = os.path.join(BASE_DIR, "configs", data, mitigator)
    items = []
    for fname in list_yaml_files(CONFIGS_DIR):
        match = True
        items.append({"name": fname, "match": match})
    return jsonify({"items": items})


@app.route("/api/config/<name>", methods=["GET"])
def get_config(name):
    name = safe_name(name)
    mitigator = request.args.get("mitigator", "")
    data = request.args.get("data", "")
    if name is None: return jsonify({"error": "invalid name"}), 400
    path = os.path.join(BASE_DIR, "configs", data, mitigator, name)
    if not os.path.isfile(path): return jsonify({"error": "not found"}), 404
    with open(path, "r") as fh:
        content = fh.read()
    return jsonify({"content": content})


@app.route("/api/config/<name>", methods=["POST"])
def save_config(name):
    name = safe_name(name)
    mitigator = request.args.get("mitigator", "")
    data = request.args.get("data", "")
    if name is None: return jsonify({"error": "invalid name"}), 400
    content = request.json.get("content", "")
    try: yaml.safe_load(content)
    except yaml.YAMLError as exc: return jsonify({"error": "invalid yaml", "detail": str(exc)}), 400
    path = os.path.join(BASE_DIR, "configs", data, mitigator, name)
    with open(path, "w") as fh:
        fh.write(content)
    return jsonify({"ok": True})


NEW_CFG_TEMPLATE = """EXPERIMENT:
  NAME: ""
  TAG: ""
  PROJECT: ""
DATASET:
  TYPE: ""
  BIASES: []
MITIGATOR:
  TYPE: ""
SOLVER:
  BATCH_SIZE: 128
  EPOCHS: 20
  LR: 0.001
  TYPE: "Adam"
  WEIGHT_DECAY: 0.0
  SCHEDULER:
    LR_DECAY_STAGES: []
    LR_DECAY_RATE: 0.1
MODEL:
  TYPE: ""
METRIC: ""
"""


@app.route("/api/config/new", methods=["POST"])
def new_config():
    name = safe_name(request.json.get("name", ""))
    if name is None:
        return jsonify({"error": "invalid name"}), 400
    if not name.endswith((".yaml", ".yml")):
        name += ".yaml"
    path = os.path.join(CONFIGS_DIR, name)
    if os.path.exists(path):
        return jsonify({"error": "already exists"}), 400
    with open(path, "w") as fh:
        fh.write(NEW_CFG_TEMPLATE)
    return jsonify({"ok": True, "name": name})


# ---------- run tools.train ----------

def _reader_thread(run_id, proc):
    """Reads raw output (bytes) and appends it as-is, \\r included."""
    buf = []
    while True:
        chunk = proc.stdout.read(1)
        if chunk == "" and proc.poll() is not None:
            break
        if chunk:
            buf.append(chunk)
            with RUNS_LOCK:
                RUNS[run_id]["output"] = "".join(buf)
    proc.stdout.close()
    with RUNS_LOCK:
        RUNS[run_id]["running"] = False
        RUNS[run_id]["returncode"] = proc.returncode


@app.route("/api/run/train/<name>", methods=["POST"])
def run_train(name):
    name = safe_name(name)
    mitigator = request.args.get("mitigator", "")
    data = request.args.get("data", "")
    if name is None: return jsonify({"error": "invalid name"}), 400
    full_path = os.path.join(BASE_DIR, "configs", data, mitigator, name)
    if not os.path.isfile(full_path):  return jsonify({"error": "config not found"}), 404
    run_id = uuid.uuid4().hex
    cmd = ["python", "-m", "tools.train", "--cfg", full_path]
    print(" ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with RUNS_LOCK:
        RUNS[run_id] = {
            "proc": proc,
            "output": "",
            "running": True,
            "returncode": None,
            "cmd": " ".join(cmd),
        }
    t = threading.Thread(target=_reader_thread, args=(run_id, proc), daemon=True)
    t.start()
    return jsonify({"run_id": run_id, "cmd": " ".join(cmd)})

@app.route("/api/run/<run_id>/stop", methods=["POST"])
def stop_run(run_id):
    with RUNS_LOCK: run = RUNS.get(run_id)
    if run is None: return jsonify({"error": "unknown run"}), 404
    proc = run["proc"]
    if proc.poll() is None:
        proc.terminate()
        try: proc.wait(timeout=5)
        except subprocess.TimeoutExpired: proc.kill()
    return jsonify({"ok": True})

@app.route("/api/run/<run_id>")
def run_status(run_id):
    with RUNS_LOCK:
        run = RUNS.get(run_id)
        if run is None:
            return jsonify({"error": "unknown run"}), 404
        return jsonify(
            {
                "output": run["output"],
                "running": run["running"],
                "returncode": run["returncode"],
                "cmd": run["cmd"],
            }
        )


@app.route("/")
def index():
    return render_template("index.html")

@atexit.register
def stop_all_runs():
    with RUNS_LOCK:
        procs = [
            run["proc"]
            for run in RUNS.values()
            if "proc" in run and run["proc"].poll() is None
        ]
    for proc in procs:
        try: proc.terminate()
        except Exception:  pass
    for proc in procs:
        try: proc.wait(timeout=5)
        except Exception:
            try: proc.kill()
            except Exception: pass


if __name__ == "__main__":
    app.run(debug=False, port=5000)
