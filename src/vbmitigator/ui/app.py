"""VB-Mitigator Streamlit dashboard.

Run with::

    vbm-ui
    # or
    python -m vbmitigator.ui

Features
--------
* **Launch**     pick a config (or components) and start a training run.
* **Workspace**  a W&B-style overlay of all runs: a runs panel with per-run
                 colours + show/hide, and a grid of metric panels comparing
                 every visible run; drill into a run for its charts, config and
                 our fairness dashboard (subgroup heatmaps + worst-group faces).
"""

import base64
import os
import re
import time

import pandas as pd
import streamlit as st

from vbmitigator.ui import charts, jobs, runs, viz

_LOGO = os.path.join(os.path.dirname(__file__), "assets", "logo.png")


def _logo_uri():
    try:
        with open(_LOGO, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except OSError:
        return None


_LOGO_URI = _logo_uri()

st.set_page_config(
    page_title="VB-Mitigator",
    page_icon=_LOGO if os.path.exists(_LOGO) else "🧪",
    layout="wide",
)


def _altair(container, chart):
    """Render an Altair chart full-width across Streamlit versions.

    ``use_container_width`` is being removed in favour of ``width='stretch'``;
    older versions only accept the former. Try the new arg, then the old, then a
    plain call. The chart carries its own explicit height, so it never collapses.
    """
    for kwargs in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            container.altair_chart(chart, **kwargs)
            return
        except Exception:
            continue


@st.cache_data(show_spinner="Loading registered components…")
def _components():
    return runs.registered_components()


def _sidebar():
    if hasattr(st, "logo"):
        try:
            st.logo(_LOGO)  # top-left app branding
        except Exception:
            pass
    if os.path.exists(_LOGO):
        st.sidebar.image(_LOGO, width=56)
    st.sidebar.title("VB-Mitigator")
    configs_dir = st.sidebar.text_input("Configs directory", value="configs")
    output_dir = st.sidebar.text_input("Output directory", value="outputs")
    st.sidebar.caption(
        "Runs are stored as `<output>/<dataset>/<method>/<config>/<run_id>/`."
    )
    components = _components()
    if components:
        with st.sidebar.expander("Registered components", expanded=False):
            for kind in ("datasets", "methods", "models", "metrics"):
                names = components.get(kind, [])
                st.markdown(f"**{kind}** ({len(names)})")
                st.caption(", ".join(names) if names else "—")
    return configs_dir, output_dir, components


def _check_overrides(extra, components):
    """Warn if a KEY VALUE override selects a component that isn't registered."""
    field_to_kind = {
        "DATASET.TYPE": "datasets",
        "MITIGATOR.TYPE": "methods",
        "MODEL.TYPE": "models",
        "METRIC": "metrics",
    }
    tokens = extra.split()
    for i in range(0, len(tokens) - 1, 2):
        key, value = tokens[i], tokens[i + 1]
        kind = field_to_kind.get(key)
        if kind and components.get(kind) and value not in components[kind]:
            st.warning(
                f"`{value}` is not a registered {kind[:-1]}. "
                f"Available: {', '.join(components[kind])}"
            )


def _launch_from_registry(output_dir, components):
    """Config-less launch: pick components from the registries directly."""
    st.caption(
        "No config files found — build a run from the registered components instead."
    )
    if not components:
        st.info("Registered components are unavailable; check the install.")
        return
    c1, c2 = st.columns(2)
    dataset = c1.selectbox("Dataset", components.get("datasets", []))
    method = c2.selectbox("Method", components.get("methods", []))
    model = c1.selectbox("Model", components.get("models", []))
    metric = c2.selectbox("Metric", components.get("metrics", []))
    seed = st.number_input("Seed", min_value=0, value=1, step=1)
    extra = st.text_input("Extra overrides (space separated KEY VALUE)", value="")
    if st.button("🚀 Start training", type="primary"):
        opts = [
            "OUTPUT.DIR", output_dir,
            "DATASET.TYPE", dataset,
            "MITIGATOR.TYPE", method,
            "MODEL.TYPE", model,
            "METRIC", metric,
            "EXPERIMENT.SEED", str(int(seed)),
            *(extra.split() if extra.strip() else []),
        ]
        job = jobs.launch_training("", output_dir, extra_opts=opts)
        st.success(f"Launched `{job['job_id']}`.")
        st.button("Track it live in Workspace ▶", type="primary", on_click=_goto,
                  args=("Workspace",), key="goto_runs_reg")


def _launch_tab(configs_dir, output_dir, components):
    st.header("Launch a run")
    cfgs = runs.discover_configs(configs_dir)
    if not cfgs:
        _launch_from_registry(output_dir, components)
        return

    datasets = sorted({c["dataset"] for c in cfgs})
    dataset = st.selectbox("Dataset", datasets)
    methods = sorted({c["method"] for c in cfgs if c["dataset"] == dataset})
    method = st.selectbox("Method", methods)
    named = [
        c for c in cfgs if c["dataset"] == dataset and c["method"] == method
    ]
    name = st.selectbox("Configuration", [c["name"] for c in named])
    chosen = next(c for c in named if c["name"] == name)

    with st.expander("Config file", expanded=False):
        with open(chosen["path"]) as f:
            st.code(f.read(), language="yaml")

    col1, col2 = st.columns([1, 3])
    with col1:
        seed = st.number_input("Seed", min_value=0, value=1, step=1)
    with col2:
        extra = st.text_input(
            "Extra overrides (space separated KEY VALUE)",
            value="",
            help="e.g. SOLVER.EPOCHS 5 EXPERIMENT.GPU cpu",
        )

    if extra.strip():
        _check_overrides(extra, components)

    if st.button("🚀 Start training", type="primary"):
        opts = ["EXPERIMENT.SEED", str(int(seed))]
        if extra.strip():
            opts += extra.split()
        job = jobs.launch_training(
            chosen["path"], output_dir, extra_opts=["OUTPUT.DIR", output_dir, *opts]
        )
        st.success(f"Launched `{job['job_id']}`.")
        st.button("Track it live in Workspace ▶", type="primary", on_click=_goto,
                  args=("Workspace",), key="goto_runs_cfg")


def _render_live(run_dir, status):
    """Live progress bar + metrics + loss curves for a run (polls progress.json)."""
    prog = runs.read_progress(run_dir)
    if not prog:
        st.caption("Waiting for the run to start…")
        return
    epoch, total_ep = prog.get("epoch", 0), max(1, prog.get("total_epochs", 1))
    step, total_st = prog.get("step", 0), prog.get("total_steps", 0)
    within = (step / total_st) if total_st else 0.0
    done = prog.get("status") == "finished" or status == "finished"
    frac = 1.0 if done else min(1.0, max(0.0, (epoch - 1 + within) / total_ep))
    st.progress(
        frac,
        text=(
            "✅ finished"
            if done
            else f"epoch {epoch}/{total_ep} · step {step}/{total_st}"
        ),
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Epoch", f"{epoch}/{total_ep}")
    if prog.get("train_loss") is not None:
        c2.metric("Train loss", f"{prog['train_loss']:.4f}")
    metrics = prog.get("metrics", {})
    test_acc = metrics.get("test_overall", metrics.get("test_accuracy"))
    if test_acc is not None:
        c3.metric("Test metric", f"{test_acc:.3f}")

    steps = runs.load_step_losses(run_dir)
    if steps is not None and "loss" in steps.columns:
        st.caption(f"Training loss — all {len(steps)} steps")
        st.line_chart(steps.set_index("step")["loss"])

    curves = runs.load_curves(run_dir)
    if curves is not None and not curves.empty:
        cols = [c for c in curves.columns if "loss" in c or "acc" in c or "overall" in c]
        if cols:
            st.caption("Per-epoch metrics")
            st.line_chart(curves[cols])

    if done:
        st.success("Run finished — reselect it above to open the full fairness dashboard.")


if hasattr(st, "fragment"):
    _live_fragment = st.fragment(run_every=2.0)(_render_live)
else:  # pragma: no cover - older streamlit
    _live_fragment = _render_live


def _live_job_map(output_dir):
    """Map normalized run_dir -> its (alive/finished) launch job."""
    out = {}
    for job in jobs.list_jobs(output_dir):
        rd = jobs.job_run_dir(job)
        if rd:
            out[os.path.normpath(rd)] = job
    return out


@st.cache_data(show_spinner="Rendering sample images…")
def _montage(run_dir, mode):
    return runs.render_montage(run_dir, mode)


def _headline_metrics(metrics):
    """Overall + worst-group + the fairness gap between them."""
    overall = next((metrics[k] for k in ("test_overall", "test_accuracy") if k in metrics), None)
    worst = metrics.get("test_worst_group_accuracy")
    cols = st.columns(3)
    if overall is not None:
        cols[0].metric("Overall accuracy", f"{overall*100:.1f}%")
    if worst is not None:
        cols[1].metric("Worst-group accuracy", f"{worst*100:.1f}%")
    if overall is not None and worst is not None:
        gap = (overall - worst) * 100
        cols[2].metric("Fairness gap", f"{gap:.1f} pts", delta=f"-{gap:.1f}",
                       delta_color="inverse")


def _fairness_section(df):
    biases = viz.bias_columns(df)
    if not biases:
        st.caption("This run has no sensitive-attribute annotations to break down.")
        return
    bias = biases[0] if len(biases) == 1 else st.selectbox("Sensitive attribute", biases)

    worst = viz.worst_group(df, bias)
    if worst:
        st.error(
            f"**Worst-performing subgroup:** {worst['group']} — "
            f"**{worst['accuracy']*100:.1f}%** accuracy over {worst['n']} samples.",
            icon="⚠️",
        )

    left, right = st.columns(2)
    with left:
        st.pyplot(viz.subgroup_accuracy_heatmap(df, bias))
    with right:
        st.pyplot(viz.subgroup_count_heatmap(df, bias))
    st.caption(
        "Left: accuracy per subgroup (worst outlined in red). "
        "Right: how many samples each subgroup has — uneven counts reveal the bias."
    )




def _short(r):
    return f"{r['dataset']}/{r['method']}/{r['config']}/{r['run_id']}"


def _run_headline(run_dir):
    """(overall, worst_group) accuracies from metrics.json, or (None, None)."""
    m = runs.load_metrics(run_dir)
    overall = next((m[k] for k in ("test_overall", "test_accuracy") if k in m), None)
    return overall, m.get("test_worst_group_accuracy")


_BADGE = {"running": "🟢", "finished": "✅", "crashed": "✖️"}
_EVAL_KEYS = ("test", "val", "overall", "worst", "acc", "unb", "ba_", "bc_", "std")
_TRAIN_KEYS = ("loss", "lr", "norm")


def _run_status(r):
    if r["running"]:
        return "running"
    if runs.load_metrics(r["path"]) or runs.read_progress(r["path"]).get("status") == "finished":
        return "finished"
    return "crashed" if r["job"] else "finished"


def _annotate(all_runs, output_dir):
    jobmap = _live_job_map(output_dir)
    for r in all_runs:
        r["job"] = jobmap.get(os.path.normpath(r["path"]))
        r["running"] = bool(r["job"] and r["job"]["status"] == "running")
        r["status"] = _run_status(r)
        r["label"] = _short(r)
    all_runs.sort(key=lambda r: not r["running"])
    return all_runs


def _workspace(output_dir):
    all_runs = _annotate(runs.discover_runs(output_dir), output_dir)
    if not all_runs:
        st.info(f"No runs yet under `{output_dir}`. Start one from the **Launch** view.")
        return

    # Drill into one run?
    detail = st.session_state.get("detail")
    if detail:
        run = next((r for r in all_runs if r["path"] == detail), None)
        if run:
            _run_detail(run)
            return
        st.session_state.detail = None

    # Stable colours over ALL runs (a run keeps its colour when toggled).
    color_map = charts.run_color_map(sorted(r["label"] for r in all_runs))
    left, main = st.columns([1, 3], gap="medium")
    with left:
        _runs_panel(all_runs, color_map)
    with main:
        visible = [r for r in all_runs if st.session_state.get(f"vis::{r['path']}", False)]
        _panels(visible, color_map)


def _runs_panel(all_runs, color_map):
    n_run = sum(r["running"] for r in all_runs)
    st.markdown(f"#### Runs · {len(all_runs)}")
    st.caption(f"🟢 {n_run} running · {len(all_runs) - n_run} done")
    query = st.text_input("Search runs", "", key="run_search", placeholder="filter by name")
    shown = [r for r in all_runs if query.lower() in r["label"].lower()]

    c1, c2 = st.columns(2)
    if c1.button("Show all"):
        for r in shown:
            st.session_state[f"vis::{r['path']}"] = True
        st.rerun()
    if c2.button("Hide all"):
        for r in shown:
            st.session_state[f"vis::{r['path']}"] = False
        st.rerun()

    if len(shown) > 60:
        st.caption(f"Showing 60 of {len(shown)} — refine the search.")
        shown = shown[:60]

    for i, r in enumerate(shown):
        st.session_state.setdefault(f"vis::{r['path']}", i < 10)
        swatch = (
            f"<div style='width:12px;height:12px;border-radius:3px;margin-top:6px;"
            f"background:{color_map[r['label']]}'></div>"
        )
        sw, cb, op = st.columns([0.12, 0.66, 0.22])
        sw.markdown(swatch, unsafe_allow_html=True)
        cb.checkbox(
            f"{_BADGE[r['status']]} {r['method']}/{r['config']}/{r['run_id']}",
            key=f"vis::{r['path']}",
        )
        if op.button("open", key=f"open::{r['path']}"):
            st.session_state.detail = r["path"]
            st.rerun()


def _panels(visible, color_map):
    if not visible:
        st.info("Toggle some runs on the left to plot them here.")
        return
    top = st.columns([2, 1, 1])
    mquery = top[0].text_input("Filter metrics (regex)", "", key="metric_search")
    smooth = top[1].slider("Smoothing", 0.0, 0.95, 0.0, 0.05, key="ws_smooth")
    ncols = top[2].select_slider("Columns", options=[1, 2, 3], value=2, key="ws_cols")
    live = st.checkbox(
        "🔴 Auto-refresh while running",
        value=any(r["running"] for r in visible),
        key="ws_live",
    )

    curves = {r["label"]: runs.load_curves(r["path"]) for r in visible}
    curves = {k: v for k, v in curves.items() if v is not None and not v.empty}
    metrics = sorted({c for v in curves.values() for c in v.columns})
    if mquery:
        try:
            metrics = [m for m in metrics if re.search(mquery, m, re.I)]
        except re.error:
            pass

    sections = {"Evaluation": [], "Training": [], "Other": []}
    for m in metrics:
        ml = m.lower()
        if any(k in ml for k in _EVAL_KEYS):
            sections["Evaluation"].append(m)
        elif any(k in ml for k in _TRAIN_KEYS):
            sections["Training"].append(m)
        else:
            sections["Other"].append(m)

    for section, mets in sections.items():
        if not mets:
            continue
        st.subheader(section)
        cols = st.columns(ncols)
        for i, m in enumerate(mets):
            series = {lab: c[m] for lab, c in curves.items() if m in c.columns}
            lf = charts.long_frame(series, x_col="epoch", smooth=smooth)
            _altair(cols[i % ncols], charts.overlay_chart(lf, m, x_col="epoch", color_map=color_map))

    # per-step training loss overlay
    step_series = {}
    for r in visible:
        s = runs.load_step_losses(r["path"])
        if s is not None and "loss" in s.columns:
            step_series[r["label"]] = s.set_index("step")["loss"]
    if step_series:
        st.subheader("Training loss · per step")
        lf = charts.long_frame(step_series, x_col="step", smooth=smooth)
        _altair(st, charts.overlay_chart(lf, "loss", x_col="step", color_map=color_map))

    _fairness_compare(visible, color_map)

    if live and any(r["running"] for r in visible):
        time.sleep(3)
        st.rerun()


def _fairness_compare(visible, color_map):
    rows = []
    for r in visible:
        overall, worst = _run_headline(r["path"])
        if overall is None and worst is None:
            continue
        rows.append(
            {
                "run": r["label"],
                "overall": None if overall is None else round(overall * 100, 1),
                "worst": None if worst is None else round(worst * 100, 1),
            }
        )
    if not rows:
        return
    st.subheader("Metrics · across runs")
    dfc = pd.DataFrame(rows)
    c1, c2 = st.columns(2)
    _altair(c1, charts.bar_compare(dfc, "worst", "worst-group accuracy (%)", color_map))
    _altair(c2, charts.bar_compare(dfc, "overall", "overall accuracy (%)", color_map))
    st.caption(
        "Compare subgroup robustness, not just averages — open a run for its "
        "subgroup heatmaps and worst-group faces."
    )


def _run_detail(run):
    if st.button("← Back to workspace"):
        st.session_state.detail = None
        st.rerun()
    st.markdown(f"## {_BADGE[run['status']]} `{run['label']}`")
    tab_charts, tab_over, tab_fair, tab_logs = st.tabs(
        ["Charts", "Overview", "Analysis", "Logs"]
    )

    with tab_charts:
        if run["running"]:
            st.info("Training in progress — updates live.")
            _live_fragment(run["path"], "running")
            if st.button("⏹ Stop run"):
                jobs.stop_job(run["job"])
                st.warning("Sent stop signal.")
        else:
            cm = charts.run_color_map([run["label"]])
            curve = runs.load_curves(run["path"])
            if curve is not None and not curve.empty:
                cols = st.columns(2)
                for i, m in enumerate(curve.columns):
                    lf = charts.long_frame({run["label"]: curve[m]}, x_col="epoch")
                    _altair(cols[i % 2], charts.overlay_chart(lf, m, x_col="epoch", color_map=cm))
            steps = runs.load_step_losses(run["path"])
            if steps is not None and "loss" in steps.columns:
                lf = charts.long_frame(
                    {run["label"]: steps.set_index("step")["loss"]}, x_col="step"
                )
                _altair(st, charts.overlay_chart(lf, "train loss · per step", x_col="step", color_map=cm))

    with tab_over:
        m = runs.load_metrics(run["path"])
        if m:
            st.markdown("**Final metrics**")
            st.json(m)
        st.markdown("**Config**")
        st.json(runs.load_config(run["path"]))

    with tab_fair:
        df = runs.load_predictions_full(run["path"])
        if df is not None and not df.empty:
            _headline_metrics(runs.load_metrics(run["path"]))
            _fairness_section(df)
            with st.expander("🖼️  Sample images", expanded=False):
                overview = _montage(run["path"], "overview")
                worst_png = _montage(run["path"], "worst")
                if overview:
                    st.image(overview, caption="Samples per subgroup")
                if worst_png:
                    st.image(worst_png, caption="Worst-group misclassifications")
                if not overview and not worst_png:
                    st.caption("No images available for this run.")
        else:
            st.caption("No predictions to analyze yet.")

    with tab_logs:
        if run["job"]:
            st.text(jobs.read_job_log(run["job"]))
        for name in sorted(os.listdir(run["path"])):
            if name.startswith("out") and name.endswith(".log"):
                with open(os.path.join(run["path"], name), errors="replace") as f:
                    st.text(f.read()[-5000:])


def _goto(view):
    """Switch the active view (usable as a button on_click callback)."""
    st.session_state.view = view


def _nav():
    """Top navigation rendered as tab-like pills (programmatically switchable)."""
    st.session_state.setdefault("view", "Launch")
    options = ["Launch", "Workspace"]
    if hasattr(st, "segmented_control"):
        st.segmented_control(
            "nav", options, key="view", label_visibility="collapsed"
        )
    else:  # pragma: no cover - older streamlit
        st.radio("nav", options, horizontal=True, key="view",
                 label_visibility="collapsed")
    return st.session_state.get("view") or "Launch"


_SPLASH = """
<style>
@keyframes vbmspin {to {transform: rotate(360deg);}}
@keyframes vbmfade {from {opacity: 0;} to {opacity: 1;}}
@keyframes vbmslide {0% {left: -40%;} 100% {left: 110%;}}
@keyframes vbmpulse {0%,100% {opacity:.55;} 50% {opacity:1;}}
.vbm-splash {position: fixed; inset: 0; z-index: 99999; display: flex;
  flex-direction: column; align-items: center; justify-content: center; gap: 20px;
  background: radial-gradient(1100px 640px at 50% 28%, #123038 0%, #0a0e13 72%);
  color: #eaf6f4; font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  animation: vbmfade .5s ease;}
.vbm-logo {width: 104px; height: 104px; border-radius: 50%; object-fit: cover;
  box-shadow: 0 0 0 4px #15353b, 0 10px 34px rgba(0,0,0,.55);
  animation: vbmpulse 2.4s ease-in-out infinite;}
.vbm-logo-emoji {font-size: 60px; animation: vbmpulse 2s ease-in-out infinite;}
.vbm-title {font-size: 34px; font-weight: 800; letter-spacing: -.02em;
  background: linear-gradient(90deg, #ffffff, #58e6d6);
  -webkit-background-clip: text; background-clip: text; color: transparent;}
.vbm-ring {width: 52px; height: 52px; border-radius: 50%;
  border: 4px solid #1c3b42; border-top-color: #35c2b6;
  animation: vbmspin .9s linear infinite;}
.vbm-bar {width: 240px; height: 4px; border-radius: 4px; background: #16303620;
  border: 1px solid #1c3b42; overflow: hidden; position: relative;}
.vbm-bar::after {content: ""; position: absolute; top: 0; left: -40%; height: 100%;
  width: 40%; background: linear-gradient(90deg, transparent, #35c2b6, transparent);
  animation: vbmslide 1.15s ease-in-out infinite;}
.vbm-sub {color: #8fb3ad; font-size: 12.5px; letter-spacing: .18em; text-transform: uppercase;}
</style>
<div class="vbm-splash">
  %%LOGO%%
  <div class="vbm-title">VB-Mitigator</div>
  <div class="vbm-ring"></div>
  <div class="vbm-bar"></div>
  <div class="vbm-sub">Loading registries…</div>
</div>
"""


def _boot():
    """Show a one-time fancy splash while the registries load (cached after)."""
    if st.session_state.get("booted"):
        return
    logo_html = (
        f'<img class="vbm-logo" src="{_LOGO_URI}" alt="VB-Mitigator logo">'
        if _LOGO_URI
        else '<div class="vbm-logo-emoji">🧪</div>'
    )
    splash = st.empty()
    splash.markdown(_SPLASH.replace("%%LOGO%%", logo_html), unsafe_allow_html=True)
    _components()  # warms the cached registry lookup (the slow first call)
    st.session_state.booted = True
    splash.empty()


def main():
    _boot()
    configs_dir, output_dir, components = _sidebar()
    if _nav() == "Launch":
        _launch_tab(configs_dir, output_dir, components)
    else:
        _workspace(output_dir)


main()
