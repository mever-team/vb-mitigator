# Catalog

Everything you can pick in VB-Mitigator, in plain English. To see exactly what's
available in your install, run `vbm-list` or open the sidebar's *Registered
components* panel.

## Datasets

The benchmark images you train on. Each one has a known *sensitive attribute* so
bias can be measured. **Data is not bundled** — you point the tool at your own
copy (see [Get started](getting-started.md)).

| Dataset | Predict… | Sensitive attribute (the bias) |
|---|---|---|
| **biased_mnist** | the digit | the background/foreground colour |
| **fb_biased_mnist** | the digit | two colours (foreground + background) |
| **utkface** | gender | race |
| **celeba** | an attribute (e.g. blonde) | gender |
| **waterbirds** | bird type (land/water) | the background scene |
| **urbancars** | the car | background + a co-occurring object |
| **cifar10 / cifar100** | the object class | tag-based / spurious cues |
| **stanford_dogs** | the dog breed | tag-based cues |
| **imagenet9** | 9 super-classes | background (Background Challenge) |
| **imagenet9m** | a configurable subset | a *synthetic*, tunable bias (jpeg/resize) |

## Methods (mitigators)

Pick **erm** for the honest, unmitigated baseline; pick any other to *reduce*
bias. They're all published methods — the short idea is given here.

| Method | In one line |
|---|---|
| **erm** | Plain training — the baseline that shows the bias. |
| **erm_tags** | Baseline that also uses auto-generated image tags. |
| **flac** | Suppresses associations between the label and the sensitive attribute. |
| **flacb** | A "blind" FLAC variant that needs no bias labels at train time. |
| **badd** | Adds bias features into the model so it stops relying on them. |
| **mavias** | "Mitigate Any Visual Bias" — discovers biases from image tags + an LLM. |
| **maviasb** | A bias-capturing-classifier variant of MAVias. |
| **groupdro** | Optimises for the *worst* group directly (distributionally robust). |
| **debian** | Learns a bias-discovery network and reweights against it. |
| **di** | Domain-Independent training — a separate head per bias group. |
| **sd** | Spectral Decoupling — a loss term that curbs shortcut learning. |
| **lff** | "Learning from Failure" — up-weights examples a biased model gets wrong. |
| **bb** | BiasBalance — balances the influence of bias-aligned samples. |
| **end** | EnD — disentangles the sensitive attribute from the representation. |
| **jtt** | "Just Train Twice" — find hard cases, then retrain up-weighting them. |
| **softcon** | Soft contrastive learning against the bias. |

:::{admonition} Which should I try first?
:class: tip
Run **erm** to see the bias, then compare it against **badd**, **flac**,
**groupdro** or **lff**. Whichever lifts the *worst group* most on your data
wins.
:::

## Models

The neural-network backbone. Smaller = faster; larger = usually more accurate.

- **simple_conv** — a tiny CNN, great for the MNIST-style datasets.
- **resnet8 / resnet20 / resnet32** — small CIFAR-style ResNets.
- **resnet18 / resnet34 / resnet50** — standard ResNets (the usual choice for
  faces/objects).
- **efficientnet_b0**, **vit_b_16** — an EfficientNet and a Vision Transformer,
  for stronger backbones.

Set it per run with `MODEL.TYPE`; enable ImageNet-pretrained weights with
`MODEL.PRETRAINED True`.

## Metrics

How a run is scored and how the "best" checkpoint is chosen.

| Metric | Measures |
|---|---|
| **acc** | Overall accuracy. |
| **acc_per_class** | Accuracy broken down per class. |
| **wg_ovr** | **Worst-group** and overall accuracy (the fairness workhorse). |
| **wg_ovr_std** | Worst-group/overall plus the spread across subgroups. |
| **wg_ovr_analytic** | An analytic variant of the worst-group metric. |
| **unb_bc_ba** | Unbiased / bias-conflicting / bias-aligned accuracies. |
| **wg_ovr_tags** | Worst-group evaluation driven by image tags. |

For a fairness study, **wg_ovr** is the standard choice.
