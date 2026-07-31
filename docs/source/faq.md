# FAQ & glossary

## Frequently asked questions

**Do I need to know machine learning to use this?**
No. To *use the app* — launch runs, read the fairness views, compare methods —
you only need to install it once and point it at a dataset. Understanding the
methods deeply is optional and covered in the papers they cite.

**Do I need a GPU?**
A GPU makes training much faster, but small datasets/models run on a CPU too
(just slower). The app itself is lightweight and runs anywhere.

**Where does my data go? Is anything uploaded?**
Nothing is uploaded. VB-Mitigator runs entirely on your machine; datasets stay
in the local folder you point it at, and results are written next to them under
`outputs/`.

**Which method is best?**
There's no universal winner — it depends on the dataset and the bias. That's the
whole point of the tool: run a few and **compare them by worst-group accuracy**
in the Workspace.

**My model is 90%+ accurate. Why should I care about subgroups?**
Because a high average can hide a subgroup that performs terribly. If your model
will be used on real people, the worst group is what determines whether it's
fair and safe. See [What is visual bias?](introduction.md).

**Can I add my own dataset / method / model / metric?**
Yes — each is a one-file, drop-in addition. See
[Extending VB-Mitigator](extending.md).

**The charts/images don't appear for a run.**
The subgroup charts need the run's `predictions.csv`; the example images
additionally need the dataset's image files on disk. Older runs created before a
feature was added may lack these — just re-run.

**Training I launched from the app didn't start / the log is empty.**
Make sure the dataset's data is present at the configured location. On clusters,
confirm the app and the training process can see the same filesystem.

## Glossary

Target
: The thing the model predicts (e.g. gender, digit, bird type).

Sensitive attribute
: A property we don't want the model to rely on (e.g. race, background colour).
Also called the *bias attribute*.

Subgroup
: A combination of a target class and a sensitive-attribute value
(e.g. *female · race=1*).

Worst group
: The subgroup with the lowest accuracy — the headline fairness number.

Fairness gap
: Overall accuracy minus worst-group accuracy. Smaller is fairer.

Bias mitigation
: Techniques that stop a model from taking shortcuts based on the sensitive
attribute, so it performs more evenly across subgroups.

ERM
: "Empirical Risk Minimisation" — ordinary training with no mitigation; the
baseline that reveals the bias.

Run
: One training job with a chosen dataset, method, model and settings. Everything
about it lives in one output folder.

Checkpoint
: A saved copy of the model's weights (`best` = the best-scoring epoch,
`latest` = the final one).
