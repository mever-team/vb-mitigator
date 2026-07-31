# What is visual bias?

*No machine-learning background needed for this page.*

## A model that's "accurate" can still be unfair

Machine-learning models learn by example. Show a model thousands of labelled
photos and it will find whatever patterns best predict the label — **including
patterns you never intended it to use.**

Here's the classic trap. Suppose we train a model to predict a person's
**gender** from a face photo. If, in our training photos, gender happens to be
correlated with **skin tone** (a *sensitive attribute*), the model can take a
shortcut: instead of learning what actually distinguishes the classes, it leans
on skin tone. It still scores well on average — because the shortcut works for
most of the data — but it quietly fails for the people who don't fit the
shortcut.

That failure is **bias**, and a single accuracy number will never show it to you.

## Groups, subgroups, and the "worst group"

To see bias, we stop looking at one big average and instead split the data into
**subgroups** — combinations of the thing we're predicting (the *target*) and
the sensitive attribute:

| | sensitive = A | sensitive = B |
|---|---|---|
| **class 1** | subgroup 1·A | subgroup 1·B |
| **class 2** | subgroup 2·A | subgroup 2·B |

A fair model performs *similarly well across all subgroups*. A biased model has
one or more subgroups that lag far behind. The lowest-scoring one is the
**worst group**, and its accuracy is the number that really tells you whether
the model is trustworthy for everyone.

:::{admonition} Overall vs. worst group — a real example
:class: note
On a faces benchmark, a plain model scored **72% overall** — but only **46%**
on its worst subgroup. The average looked fine; one group was barely better than
a coin flip.
:::

## What "mitigation" means

**Bias mitigation** is a set of techniques that discourage the model from taking
those shortcuts, so it works more evenly across subgroups. Researchers have
proposed many such methods. They differ in *how* they do it (reweighting hard
examples, adding fairness-aware losses, using an auxiliary "bias" model, and so
on), but they share one goal: **raise the worst group without wrecking the
average.**

VB-Mitigator ships 16 of these published methods and lets you apply any of them
by changing a single setting.

## What VB-Mitigator gives you

1. **A way to see it** — subgroup accuracy heatmaps (the worst cell outlined in
   red), a headline "fairness gap", and real example images of the faces/objects
   the model gets wrong.
2. **A way to fix it** — one-click mitigation methods on standard benchmarks.
3. **A way to compare fairly** — put several runs side by side and rank them by
   worst-group performance, not just the average.
4. **A way to do all this without code** — a web dashboard (see
   [Using the app](using-the-app.md)).

```{image} _static/subgroup_heatmap.png
:alt: Subgroup accuracy heatmap with the worst group outlined
:width: 380px
:align: center
```

<p style="text-align:center; color:#777">
Reading this at a glance: rows are the classes, columns the sensitive attribute,
each cell an accuracy. The red-outlined cell is the worst group.
</p>

Ready to try it? Head to [Get started](getting-started.md).
