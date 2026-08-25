# Publications

VB-Mitigator brings together methods introduced in the following papers. If the
tool is useful in your work, please **cite the framework and the specific
method(s) you used** (see [How to cite](#how-to-cite)).

---

## VB-Mitigator (the framework)

The framework itself — the benchmarks, the mitigation-method implementations, the
standardized outputs, and the app documented here.

- **Paper:** [arXiv:2507.18348](https://arxiv.org/abs/2507.18348) (2025) —
  *VB-Mitigator: An open-source framework for evaluating and advancing visual
  bias mitigation.*

```bibtex
@article{sarridis2025vbmitigator,
  title={VB-Mitigator: An open-source framework for evaluating and advancing visual bias mitigation},
  author={Sarridis, Ioannis and Koutlis, Christos and Papadopoulos, Symeon and Diou, Christos},
  journal={arXiv preprint arXiv:2507.18348},
  year={2025}
}
```

---

## FLAC — Fairness-Aware representation learning by suppressing attribute-class associations

Learns representations that **suppress the statistical association** between the
target label and the sensitive attribute, so the model stops relying on the
protected attribute as a shortcut.

- **In the tool:** `MITIGATOR.TYPE: flac` (and the blind variant `flacb`).
- **Venue:** IEEE Transactions on Pattern Analysis and Machine Intelligence
  (TPAMI), 47(2):1148–1160, 2024.
- **Paper:** [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10737139).

```bibtex
@article{sarridis2024flac,
  title={FLAC: Fairness-aware representation learning by suppressing attribute-class associations},
  author={Sarridis, Ioannis and Koutlis, Christos and Papadopoulos, Symeon and Diou, Christos},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={47},
  number={2},
  pages={1148--1160},
  year={2024},
  publisher={IEEE}
}
```

---

## BAdd — Bias Mitigation through Bias Addition

**Adds** bias features into the model during training so the classifier is forced
to become invariant to them, rather than exploiting them.

- **In the tool:** `MITIGATOR.TYPE: badd`.
- **Venue:** IEEE/CVF International Conference on Computer Vision Workshops
  (ICCVW), 2025, pp. 7723–7732.
- **Paper:** [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11375491).

```bibtex
@inproceedings{sarridis2025badd,
  title={BAdd: Bias Mitigation through Bias Addition},
  author={Sarridis, Ioannis and Koutlis, Christos and Papadopoulos, Symeon and Diou, Christos},
  booktitle={2025 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  pages={7723--7732},
  year={2025},
  organization={IEEE}
}
```

---

## MAVias — Mitigate Any Visual Bias

Discovers biases automatically from **open-vocabulary image tags** (with an LLM to
judge relevance), then mitigates them — so you don't need to know or annotate the
sensitive attribute in advance.

- **In the tool:** `MITIGATOR.TYPE: mavias` (and `maviasb`). Needs the optional
  `[mavias]` extra (`pip install -e ".[mavias]"`).
- **Venue:** IEEE/CVF International Conference on Computer Vision (ICCV), 2025,
  pp. 1271–1281.
- **Paper:** [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11444940).

```bibtex
@inproceedings{sarridis2025mavias,
  title={MAVias: Mitigate Any Visual Bias},
  author={Sarridis, Ioannis and Koutlis, Christos and Papadopoulos, Symeon and Diou, Christos},
  booktitle={2025 IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={1271--1281},
  year={2025},
  organization={IEEE}
}
```

---

## How to cite

If you use VB-Mitigator, please cite the **framework** plus the **method(s)** you
ran. For example, a study using BAdd and MAVias would cite VB-Mitigator, BAdd and
MAVias. All BibTeX entries above are ready to copy.

---

*Maintainer: Ioannis Sarridis (gsarridis@iti.gr). Work supported by the EU Horizon
Europe projects MAMMOth (GA 101070285) and ELIAS (GA 101120237).*
