# Get started

This page takes you from nothing to a first result. You'll mostly use the
**app** — the command line is only needed once, to install.

## 1. Install

You need Python 3.10+ . A virtual environment (conda or venv) is recommended.

```bash
# create and activate an environment
conda create -n vb-mitigator python=3.11 && conda activate vb-mitigator

# install VB-Mitigator with the app
pip install -e ".[ui]"
```

That's it. Three commands become available:

| Command | What it does |
|---|---|
| `vbm-ui` | open the web dashboard (what most people use) |
| `vbm-train` | train a run from the terminal |
| `vbm-list` | list the available datasets, methods, models and metrics |

:::{admonition} Optional extras
:class: note
- `pip install -e ".[mavias]"` adds the tag-based methods (MAVias / erm_tags),
  which need a few heavier libraries.
- `pip install -e ".[dev]"` adds the test tools.
:::

## 2. Add a dataset

VB-Mitigator does **not** ship image data — you point it at a copy on your disk.
For the examples we use **UTKFace** (face photos labelled with gender and race).
Download it and place it so the folder looks like:

```
data/utkface/UTKFace/<images>.jpg
```

Every dataset has a configurable location; if you keep the default layout above,
things work out of the box. (Datasets without local data still let you explore
the app — you just won't be able to train on them.)

## 3. Open the app

```bash
vbm-ui
```

Your browser opens the dashboard. Running on a remote server? See the
[remote note](using-the-app.md#running-on-a-remote-server) — one extra flag and
you're set.

## 4. Run your first experiment

In the app:

1. On the **Launch** view, pick a **Dataset** (`utkface`), a **Method** (start
   with `erm` — the plain baseline), and a **Configuration**.
2. Click **🚀 Start training**. The run trains in the background.
3. Click **Track it live in Workspace ▶** to watch the loss curve and progress
   bar update in real time.
4. When it finishes, open the run and go to the **Analysis** tab to see the
   subgroup heatmap and the worst-group example images.

Now do it again with a **mitigation** method (e.g. `badd` or `flac`) and
**compare** the two runs in the Workspace. You should see the worst group climb.

That whole loop is walked through in [Using the app](using-the-app.md), and how
to read what you see is in [Understanding results](understanding-results.md).

## Prefer the terminal?

Everything the app does, the CLI does too:

```bash
# train the plain baseline
vbm-train --cfg configs/utkface/erm/race.yaml

# train a mitigation method
vbm-train --cfg configs/utkface/badd/race.yaml

# evaluate a saved checkpoint
vbm-eval --cfg configs/utkface/badd/race.yaml --model best
```

Results are written to `outputs/<dataset>/<method>/<config>/<run_id>/` — the app
reads that same folder, so terminal and app runs appear together.
