# VB-Mitigator

```{image} _static/logo.png
:alt: VB-Mitigator logo
:width: 150px
:align: center
```

<p style="text-align:center; font-size:1.25em; color:#555; margin-top:0.4em">
<b>Find the bias your accuracy score hides — and fix it.</b>
</p>

A model can look great on average and still fail badly for specific groups of
people. **VB-Mitigator** is an open toolkit that makes those hidden failures
*visible* and gives you proven methods to *reduce* them — through a friendly
point-and-click app, no deep-learning expertise required.

:::{admonition} The one-minute version
:class: tip

Train an image model, and VB-Mitigator shows you not just *"91% accurate"* but
*"…yet only 46% accurate on one subgroup."* Then it lets you switch on a
bias-mitigation method and watch that gap close — all from a web app you launch
with one command.
:::

## Why it matters

Imagine a model that predicts a person's gender from a photo. On paper it scores
well. But if the training data accidentally links *gender* with *skin tone*, the
model may quietly learn that shortcut — and then misread people whose
appearance doesn't match the shortcut. Averages hide this; **subgroups reveal
it.**

VB-Mitigator was built to make that reality easy to see and easy to act on:

- 👁️ **See the bias.** Per-subgroup accuracy heatmaps, worst-group callouts, and
  real example images of who the model gets wrong.
- 🛠️ **Reduce the bias.** 16 published mitigation methods, switchable with a click.
- ⚖️ **Compare fairly.** Judge methods on their *worst* group, not just the average.
- 🖥️ **Use it without code.** A web dashboard launches, monitors and compares
  runs live.

```{image} _static/before_after.png
:alt: ERM vs BAdd — worst-group accuracy
:width: 480px
:align: center
```

<p style="text-align:center; color:#777">
A real example on faces: a plain model leaves one subgroup far behind (red);
a mitigation method lifts it back up (green).
</p>

## Who is this for?

- **Researchers** comparing bias-mitigation methods on standard benchmarks.
- **Practitioners & students** who want to *understand* model fairness hands-on.
- **Anyone** who has heard "AI can be biased" and wants to see and address it
  concretely.

## Start here

- 🚀 **[Get started](getting-started.md)** — install it and run your first experiment from the app.
- 💡 **[What is visual bias?](introduction.md)** — a plain-English tour of the problem.
- 🖥️ **[Using the app](using-the-app.md)** — launch, watch training live, and compare runs.
- 📊 **[Reading the results](understanding-results.md)** — heatmaps, worst-group, the fairness gap.
- 📚 **[Catalog](catalog.md)** — the datasets, methods, models and metrics in plain English.

```{toctree}
:maxdepth: 2
:hidden:
:caption: For everyone

introduction
getting-started
using-the-app
understanding-results
catalog
faq
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: For developers

overview
extending
api
```

---

*Supported by the EU Horizon Europe projects **MAMMOth** (GA 101070285) and
**ELIAS** (GA 101120237). Released under the MIT License.*
