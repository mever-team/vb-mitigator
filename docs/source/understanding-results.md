# Understanding the results

Every run produces the same set of artifacts and views. Here's how to read them,
from the big picture down to individual images.

## The headline: overall vs. worst group

At the top of a run you'll see up to three numbers:

- **Overall accuracy** — how often the model is right, averaged over everyone.
- **Worst-group accuracy** — accuracy on the *weakest* subgroup.
- **Fairness gap** — the distance between them.

A small gap is good (the model treats subgroups evenly). A large gap is the
warning sign: the average is being propped up by the easy groups.

> Rule of thumb: **judge a model by its worst group, not its average.** Two
> models with the same overall accuracy can be very different in fairness.

## The subgroup accuracy heatmap

```{image} _static/subgroup_heatmap.png
:alt: Subgroup accuracy heatmap
:width: 380px
:align: center
```

- **Rows** = the classes you're predicting (e.g. *male* / *female*).
- **Columns** = the sensitive attribute (e.g. *race = 0/1*).
- **Each cell** = accuracy for that subgroup; **green is good, red is poor.**
- The **red outline** marks the worst group.

If one cell is much redder than the others, the model has a blind spot for that
combination of class and attribute — exactly the kind of bias mitigation aims to
remove.

## The subgroup counts heatmap

Right next to it, a counts heatmap shows *how many* samples fall in each
subgroup. Very uneven counts are often the *source* of the bias: the model sees
far more of some combinations than others during training, so it over-fits to
them.

## The example images

The **Analysis** tab shows real pictures:

- **Samples per subgroup** — what each subgroup actually looks like.
- **Worst-group misclassifications** — real examples the model got wrong, each
  labelled with what it *predicted*. This turns an abstract percentage into
  something concrete: these are the people (or objects) the model is failing.

## Comparing methods

In the Workspace, turn on two or more runs to overlay their curves, and read the
**Metrics · across runs** bars to compare overall and worst-group accuracy
directly.

```{image} _static/before_after.png
:alt: Before/after comparison of worst-group accuracy
:width: 460px
:align: center
```

A good mitigation result looks like the green bars above: the **worst group rises
substantially** while the overall accuracy stays high (or even improves). If a
method raises the average but not the worst group, it isn't really fixing the
bias.

## Where the numbers come from

Each finished run saves, in its output folder:

- `metrics.json` — the final headline numbers.
- `predictions.csv` — one row per test sample: its true label, the prediction,
  and its sensitive attribute(s). Every heatmap and bar on the Analysis tab is
  computed from this file, so you can also open it in your own tools (e.g. for a
  deeper fairness audit).
- `logs<seed>.csv` / `train_steps.csv` — the per-epoch and per-step curves.
- `best` / `latest` — the saved model checkpoints.
- `config.yaml` — the exact settings, for reproducibility.
