# Using the app

Launch it with:

```bash
vbm-ui
```

The dashboard has two areas, chosen with the pills at the top: **Launch** and
**Workspace**. A sidebar on the left holds global settings (where your configs
and outputs live) and lists everything that's registered.

## Launch — start a run

Pick what to train and press go:

1. **Dataset → Method → Configuration** dropdowns. (No config files? The app
   falls back to letting you pick a dataset, method, model and metric directly.)
2. **Seed** and an optional **overrides** box for quick tweaks
   (e.g. `SOLVER.EPOCHS 5`).
3. **🚀 Start training** launches the run in the background, then offers a
   **Track it live in Workspace ▶** button.

You never wait on a frozen screen — training runs as a separate process, and the
app stays responsive.

## Workspace — compare everything at a glance

The Workspace is modelled on tools like Weights & Biases. It has two parts:

**The runs panel (left).** Every run appears with a coloured dot, a show/hide
checkbox, and a status badge (🟢 running · ✅ done · ✖️ crashed). That colour is
the run's identity — it's reused in every chart. Toggle runs on/off to control
what's plotted; search to filter; **open** to drill into one.

**The metric panels (right).** For every quantity your runs logged (losses,
accuracies, worst-group, …) you get a chart that **overlays all visible runs**
with a shared legend, tooltips and zoom. Controls let you:

- **filter metrics** by name,
- **smooth** noisy curves,
- choose the number of **columns**, and
- **auto-refresh** so charts update live while a run trains.

There's also a **Metrics · across runs** panel that bars up overall vs
worst-group accuracy for each visible run — the quickest way to see which method
wins where it matters.

## Drilling into a run

Click **open** on any run for its detail view, with four tabs:

- **Charts** — this run's curves (and a live progress bar if it's still training).
- **Overview** — the exact configuration used and the final metrics.
- **Analysis** — *the fairness story*: the subgroup accuracy heatmap (worst cell
  in red), the subgroup counts, and **real example images** of who the model
  gets wrong. This is the part general-purpose tools don't give you.
- **Logs** — the raw training log.

How to interpret all of this is covered in
[Understanding results](understanding-results.md).

## Watching a run live

While a run trains, its detail **Charts** tab (and the Workspace, if
auto-refresh is on) update every couple of seconds: a real progress bar
(`epoch 3/10 · step 40/85`), the current loss, and the loss curve growing in
real time. You can **⏹ Stop** a run from its detail view.

## Running on a remote server

If you run `vbm-ui` on a remote machine (SSH / cluster) and view it on your
laptop, the app already starts in a browser-friendly, forwarding-safe mode.
Just forward the port and open it locally:

```bash
vbm-ui --server.port 8501
```

Then open `http://localhost:8501` on your laptop (your IDE's port-forwarding, or
an SSH tunnel `ssh -N -L 8501:localhost:8501 <host>`). If the machine that trains
is a different node than the one your browser reaches, forward from the login
node to the compute node.
